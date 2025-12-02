"""
Learnable Parameters (15 total):
    Direct (10):
        0: fc_detune      - Filter cavity detuning [Hz]
        1: inj_sqz        - Injected squeezing [dB]  
        2: inj_lss        - Injection loss [fraction]
        3: arm_power      - Arm cavity power [W]
        4: sec_detune     - SEC detuning [rad]
        5: ifo_omc_mm     - IFO-OMC mode mismatch [fraction]  ** mode collapse risk **
        6: sqz_omc_mm     - SQZ-OMC mode mismatch [fraction]  ** mode collapse risk **
        7: fc_mm          - Filter cavity mismatch [fraction] ** mode collapse risk **
        8: lo_angle       - Local oscillator / homodyne angle [rad]
        9: phase_noise    - Phase noise RMS [rad]
    
    Squeezing Angles (5):
        sqz_angle_0 through sqz_angle_4 in [0, π]

Unlearnable (removed - phase information destroyed in PSD):
    - ifo_omc_mm_phase
    - sqz_omc_mm_phase  
    - fc_mm_phase
"""

import numpy as np
import h5py
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiASDDataLoaderV9:
    """
    DataLoader for v9 multi-ASD training.
    Removes sin/cos encoded phases - only loads learnable parameters.
    Uses sin(2θ)/cos(2θ) encoding for squeezing angles (π-periodic).
    """
    def __init__(self, file_path, num=200001):
        train_end = int(0.8 * num)
        val_end = int(0.9 * num)
        
        with h5py.File(file_path, 'r') as hf:
            # Load all 10 ASD curves
            self.qasd = torch.from_numpy(
                hf['Simulated_ASD']['QASD'][:train_end]
            ).float()
            self.qasd_val = torch.from_numpy(
                hf['Simulated_ASD']['QASD'][train_end:val_end]
            ).float()
            
            # Load encoded parameters - only take first 10 (direct params)
            # Indices 10-15 were sin/cos encoded phases - skip them
            params_enc = torch.from_numpy(
                hf['Simulated_Params']['Parameters_Encoded'][:val_end]
            ).float()
            
            # Load squeezing angles (5 values) in radians
            sqz_raw = torch.from_numpy(
                hf['Simulated_Params/SQZ_Angles'][:val_end]
            ).float()
        
        # Only use direct parameters (0-9), skip sin/cos (10-15)
        params_direct = params_enc[:, :10]
        
        # Normalize direct parameters to [0, 1]
        self.direct_min = params_direct.min(dim=0, keepdim=True).values
        self.direct_max = params_direct.max(dim=0, keepdim=True).values
        self.direct_range = torch.clamp(self.direct_max - self.direct_min, min=1e-8)
        params_direct_norm = (params_direct - self.direct_min) / self.direct_range
        
        # Encode squeezing angles with sin(2θ)/cos(2θ) for π-periodicity
        # This maps θ=0 and θ=π to the same point
        sqz_sin = torch.sin(2 * sqz_raw)  # (N, 5)
        sqz_cos = torch.cos(2 * sqz_raw)  # (N, 5)
        
        # Interleave: [sin0, cos0, sin1, cos1, ...]
        sqz_encoded = torch.zeros(val_end, 10)
        for i in range(5):
            sqz_encoded[:, 2*i] = sqz_sin[:, i]
            sqz_encoded[:, 2*i + 1] = sqz_cos[:, i]
        
        # Split train/val
        self.params_train = params_direct_norm[:train_end]
        self.params_val = params_direct_norm[train_end:val_end]
        self.sqz_train = sqz_encoded[:train_end]
        self.sqz_val = sqz_encoded[train_end:val_end]
        
        # Store raw angles for evaluation
        self.sqz_raw_val = sqz_raw[train_end:val_end]
        
        # Store dimensions
        self.num_direct = 10
        self.num_angle_sincos = 10  # 5 angles × 2 (sin, cos)
        self.num_angles = 5
        self.num_params = 15  # 10 direct + 5 angles
        
        # Indices of mode-collapse prone parameters (for variance regularization)
        self.mode_collapse_indices = [5, 6, 7]  # ifo_omc_mm, sqz_omc_mm, fc_mm
        
        print(f"Loaded {train_end} train, {val_end - train_end} val samples")
        print(f"Direct params: {self.num_direct}, Angle sin/cos: {self.num_angle_sincos}")
        print(f"Mode collapse indices: {self.mode_collapse_indices}")
    
    def batch(self, device, start, stop):
        """Get batch of (ASDs, direct_params, angle_sincos)"""
        return (
            self.qasd[start:stop].to(device),
            self.params_train[start:stop].to(device),
            self.sqz_train[start:stop].to(device)
        )
    
    def denormalize_direct(self, pred_norm):
        """Denormalize direct parameters."""
        return pred_norm.cpu() * self.direct_range + self.direct_min
    
    def decode_angles(self, sincos_pred):
        """
        Decode sin(2θ)/cos(2θ) predictions back to angles in [0, π].
        
        Args:
            sincos_pred: (B, 10) tensor of [sin0, cos0, sin1, cos1, ...]
        Returns:
            angles: (B, 5) tensor in [0, π]
        """
        if isinstance(sincos_pred, torch.Tensor):
            sincos_pred = sincos_pred.cpu()
        
        angles = torch.zeros(sincos_pred.shape[0], 5)
        for i in range(5):
            sin_val = sincos_pred[:, 2*i]
            cos_val = sincos_pred[:, 2*i + 1]
            # atan2 gives [-π, π], divide by 2 gives [-π/2, π/2]
            # mod π gives [0, π]
            angle_2x = torch.atan2(sin_val, cos_val)
            angles[:, i] = (angle_2x / 2) % np.pi
        
        return angles


class EncoderBlock(nn.Module):
    """Transformer encoder block."""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x_norm = self.ln1(x)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


class MultiASDEncoderV9(nn.Module):
    """
    Multi-ASD Transformer for LIGO parameter estimation.
    
    Outputs:
        - 10 direct parameters (sigmoid → [0,1])
        - 10 angle sin/cos values (tanh → [-1,1], 5 angles × 2)
    
    Sin/Cos Encoding:
        Squeezing angles θ ∈ [0, π] have π-periodicity in the quantum noise.
        sin(2θ)/cos(2θ) encoding maps θ=0 and θ=π to the same point (0, 1).
        Decoding: θ = arctan2(sin, cos) / 2 mod π
    """
    
    def __init__(self, d_model=256, num_heads=16, num_layers=7,
                 d_ff=1024, dropout=0.22, num_freq_bins=1024):
        super().__init__()
        
        self.d_model = d_model
        self.num_freq_bins = num_freq_bins
        
        # Learnable frequency positional embedding
        # Helps model learn relationships between different frequency regions
        # Shape: (1, 1, 1024) - broadcasts across batch and all 10 ASDs
        self.freq_pos_embed = nn.Parameter(torch.randn(1, 1, num_freq_bins) * 0.02)
        
        # ASD projection (after adding frequency embedding)
        self.asd_proj = nn.Linear(num_freq_bins, d_model)
        
        # Positional embeddings for ASDs
        self.pos_embed = nn.Parameter(torch.randn(1, 10, d_model) * 0.02)
        self.type_embed = nn.Parameter(torch.randn(1, 2, d_model) * 0.02)
        self.angle_embed = nn.Parameter(torch.randn(1, 5, d_model) * 0.02)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Transformer
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        
        # === OUTPUT HEADS ===
        
        # Head for direct parameters (10) - output in [0,1]
        self.direct_head = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 10),
        )
        
        # Head for squeezing angles - outputs sin(2θ), cos(2θ) pairs
        # Uses tanh for [-1, 1] output range
        self.angle_head = nn.Sequential(
            nn.Linear(d_model * 2, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 2),  # sin, cos pair for each angle
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, 10, 1024) - 10 ASD spectra
        Returns:
            direct_params: (B, 10) - direct physical parameters [0,1]
            sqz_angles_sincos: (B, 10) - sin(2θ)/cos(2θ) encoded angles [-1,1]
        """
        B = x.shape[0]
        device = x.device
        
        # Add frequency positional embedding before projection
        # x: (B, 10, 1024) + freq_pos_embed: (1, 1, 1024) -> (B, 10, 1024)
        x = x + self.freq_pos_embed
        
        # Project ASDs to model dimension
        x = self.asd_proj(x)  # (B, 10, d_model)
        
        # Add embeddings
        x = x + self.pos_embed
        
        type_ids = torch.zeros(10, dtype=torch.long, device=device)
        type_ids[5:] = 1
        x = x + self.type_embed[:, type_ids, :]
        
        angle_ids = torch.arange(5, device=device).repeat(2)
        x = x + self.angle_embed[:, angle_ids, :]
        
        # Prepend CLS
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 11, d_model)
        
        # Transformer
        for block in self.encoder_blocks:
            x = block(x)
        x = self.ln_final(x)
        
        # Extract features
        cls_out = x[:, 0]      # (B, d_model)
        asd_out = x[:, 1:]     # (B, 10, d_model)
        
        # === DIRECT PARAMETERS ===
        direct_params = torch.sigmoid(self.direct_head(cls_out))  # (B, 10) in [0,1]
        
        # === SQUEEZING ANGLES (sin/cos encoded for π-periodicity) ===
        sqz_angles_sincos = []
        for i in range(5):
            fds_feat = asd_out[:, i]
            fis_feat = asd_out[:, i + 5]
            combined = torch.cat([fds_feat, fis_feat], dim=-1)
            sincos_pred = torch.tanh(self.angle_head(combined))  # (B, 2) in [-1, 1]
            sqz_angles_sincos.append(sincos_pred)
        sqz_angles_sincos = torch.cat(sqz_angles_sincos, dim=-1)  # (B, 10)
        
        return direct_params, sqz_angles_sincos
