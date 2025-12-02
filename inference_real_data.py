"""
inference_real_data_v9.py - Test trained V9 model on real LIGO data

Outputs 15 learnable parameters (excludes unlearnable phase parameters).

Usage:
    python inference_real_data_v9.py --model best_model_v9.pt --data real_data.hdf5 --training_data Samples_TrainV9_noisy.hdf5

Expected data format:
    - HDF5 file with 'Simulated_ASD/QASD' shape (10, 1024) or (N, 10, 1024)
    - 10 ASDs: 5 FDS + 5 FIS configurations
    - Frequency array optional at 'Simulated_Params/Frequency'
"""

import numpy as np
import h5py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse

from model.model_final_v9 import * 
from train_v9 import * 


# =============================================================================
# NORMALIZATION STATS
# =============================================================================

class NormalizationStatsV9:
    """Load normalization statistics from training data."""
    def __init__(self, training_data_path):
        with h5py.File(training_data_path, 'r') as hf:
            params_enc = hf['Simulated_Params']['Parameters_Encoded'][:]
        
        # Direct parameters (first 10 only)
        params_direct = params_enc[:, :10]
        self.direct_min = params_direct.min(axis=0)
        self.direct_max = params_direct.max(axis=0)
        self.direct_range = self.direct_max - self.direct_min
        self.direct_range = np.where(self.direct_range == 0, 1, self.direct_range)
        
        print(f"Loaded normalization stats from {training_data_path}")
        print(f"  Direct params range: {self.direct_min} to {self.direct_max}")
    
    def denormalize_direct(self, pred_norm):
        """Convert normalized [0,1] predictions to physical units."""
        if isinstance(pred_norm, torch.Tensor):
            pred_norm = pred_norm.cpu().numpy()
        return pred_norm * self.direct_range + self.direct_min
    
    def decode_angles(self, sincos_pred):
        """
        Decode sin(2θ)/cos(2θ) predictions back to angles in [0, π].
        
        Args:
            sincos_pred: (B, 10) array of [sin0, cos0, sin1, cos1, ...]
        Returns:
            angles: (B, 5) array in [0, π]
        """
        if isinstance(sincos_pred, torch.Tensor):
            sincos_pred = sincos_pred.cpu().numpy()
        
        B = sincos_pred.shape[0]
        angles = np.zeros((B, 5))
        for i in range(5):
            sin_val = sincos_pred[:, 2*i]
            cos_val = sincos_pred[:, 2*i + 1]
            # atan2 gives [-π, π], divide by 2 gives [-π/2, π/2]
            # mod π gives [0, π]
            angle_2x = np.arctan2(sin_val, cos_val)
            angles[:, i] = (angle_2x / 2) % np.pi
        
        return angles

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_real_data(data_path):
    """Load real LIGO data from HDF5."""
    with h5py.File(data_path, 'r') as hf:
        # Try different possible paths
        if 'Simulated_ASD' in hf and 'QASD' in hf['Simulated_ASD']:
            qasd = hf['Simulated_ASD']['QASD'][:]
        elif 'QASD' in hf:
            qasd = hf['QASD'][:]
        else:
            raise KeyError(f"Could not find QASD data. Available keys: {list(hf.keys())}")
        
        # Try to load frequency
        freq = None
        if 'Simulated_Params' in hf and 'Frequency' in hf['Simulated_Params']:
            freq = hf['Simulated_Params']['Frequency'][:]
        elif 'Frequency' in hf:
            freq = hf['Frequency'][:]
        
    print(f"Loaded real data: QASD shape = {qasd.shape}")
    return qasd, freq


def preprocess_real_data(qasd):
    """
    Preprocess real data to match training format.
    
    Args:
        qasd: Input ASD array, shape (10, N) or (B, 10, N)
        target_freq_bins: Expected number of frequency bins (1024)
    
    Returns:
        torch.Tensor of shape (B, 10, 1024)
    """
    qasd = np.array(qasd, dtype=np.float32)
    
    # Handle single sample vs batch
    if qasd.ndim == 2:
        qasd = qasd[np.newaxis, ...]  # (10, N) -> (1, 10, N)
    
    return torch.from_numpy(qasd)
# =============================================================================
# INFERENCE
# =============================================================================

def run_inference(model, qasd_tensor, norm_stats, device='cuda'):
    """Run model inference and denormalize outputs."""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        qasd_tensor = qasd_tensor.to(device)
        direct_pred, angles_sincos_pred = model(qasd_tensor)
    
    # Denormalize direct params
    direct_phys = norm_stats.denormalize_direct(direct_pred)
    
    # Decode sin/cos angles to [0, π]
    angles_phys = norm_stats.decode_angles(angles_sincos_pred)
    
    return {
        'direct': direct_phys,
        'sqz_angles': angles_phys
    }


def format_results(results, sample_idx=0):
    """Format results as a readable dictionary matching alog format."""
    
    direct = results['direct'][sample_idx]
    angles = results['sqz_angles'][sample_idx]
    
    return {
        # Filter Cavity
        'FC detuning [Hz]': direct[0],
        
        # Squeezing  
        'Injected Squeezing [dB]': direct[1],
        'Injection loss [fraction]': direct[2],
        'Phase noise [rad]': direct[9],
        
        # Interferometer
        'Arm Power [W]': direct[3],
        'SEC detuning [rad]': direct[4],
        'LO angle [rad]': direct[8],
        
        # Mode Mismatches (magnitudes only - phases are unlearnable)
        'IFO-OMC mismatch [fraction]': direct[5],
        'SQZ-OMC mismatch [fraction]': direct[6],
        'SQZ-FC mismatch [fraction]': direct[7],
        
        # Squeezing Angles
        'sqz_angle_0 [rad]': angles[0],
        'sqz_angle_1 [rad]': angles[1],
        'sqz_angle_2 [rad]': angles[2],
        'sqz_angle_3 [rad]': angles[3],
        'sqz_angle_4 [rad]': angles[4],
    }


def print_results_alog_style(results_dict):
    """Print results in LIGO alog table format."""
    
    arm_power = results_dict['Arm Power [W]']
    sec_detune = results_dict['SEC detuning [rad]']
    ifo_mm = results_dict['IFO-OMC mismatch [fraction]']
    inj_sqz = results_dict['Injected Squeezing [dB]']
    inj_loss = results_dict['Injection loss [fraction]']
    sqz_mm = results_dict['SQZ-OMC mismatch [fraction]']
    fc_mm = results_dict['SQZ-FC mismatch [fraction]']
    fc_detune = results_dict['FC detuning [Hz]']
    phase_noise = results_dict['Phase noise [rad]']
    
    print("\n")
    print("+" + "-"*40 + "+")
    print(f"| {'Arm Power':<25} | {arm_power/1e3:.1f} kW{' '*5} |")
    print(f"| {'SEC detuning':<25} | {np.degrees(sec_detune):.3f}°{' '*4} |")
    print(f"| {'IFO-OMC mismatch':<25} | {ifo_mm*100:.2f} %{' '*6} |")
    print(f"| {'Injected Squeezing':<25} | {inj_sqz:.2f} dB{' '*3} |")
    print(f"| {'Injection loss':<25} | {inj_loss*100:.2f} %{' '*6} |")
    print(f"| {'SQZ-OMC mismatch':<25} | {sqz_mm*100:.2f} %{' '*6} |")
    print(f"| {'SQZ-FC mismatch':<25} | {fc_mm*100:.2f} %{' '*6} |")
    print(f"| {'FC detuning':<25} | {fc_detune:.2f} Hz{' '*4} |")
    print(f"| {'Phase noise':<25} | {phase_noise*1e3:.2f} mrad{' '*3} |")
    print("+" + "-"*40 + "+")
    print(f"| {'Squeezing Angles':<25} |{' '*13} |")
    print("+" + "-"*40 + "+")
    for i in range(5):
        angle = results_dict[f'sqz_angle_{i} [rad]']
        print(f"| {'  θ_' + str(i):<25} | {np.degrees(angle):.2f}°{' '*5} |")
    print("+" + "-"*40 + "+")


def visualize_results(qasd, freq, results_dict, sample_idx=0, save_path=None):
    """Visualize input ASDs and predicted parameters."""
    
    qasd = np.array(qasd)
    if qasd.ndim == 2:
        qasd = qasd[np.newaxis, ...]
    
    # Create frequency array if not provided
    if freq is None:
        freq = np.geomspace(10, 8000, qasd.shape[-1]) #np.logspace(np.log10(10), np.log10(8000), qasd.shape[-1])
    
    fig = plt.figure(figsize=(14, 5))
    
    # Plot FDS
    ax1 = fig.add_subplot(1, 2, 1)
    for i in range(5):
        ax1.semilogx(freq, qasd[sample_idx, i], label=f'FDS θ_{i}', alpha=0.8)
    ax1.set_xlabel('Frequency [Hz]')
    ax1.set_ylabel('QASD Ratio')
    ax1.set_title('Frequency-Dependent Squeezing')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([10, 8000])
    
    # Plot FIS
    ax2 = fig.add_subplot(1, 2, 2)
    for i in range(5, 10):
        ax2.semilogx(freq, qasd[sample_idx, i], label=f'FIS θ_{i-5}', alpha=0.8)
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel('QASD Ratio')
    ax2.set_title('Frequency-Independent Squeezing')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([10, 8000])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run V9 inference on real LIGO data')
    parser.add_argument('--model', type=str, required=False, default='best_model_v9_0.pt',
                        help='Path to trained model .pt file')
    parser.add_argument('--data', type=str, required=False, default=r'C:\Users\User\repos\Transformer\Sample_Generation\Structed_Data_08_08.hdf5',
                        help='Path to real data HDF5')
    parser.add_argument('--training_data', type=str, required=False, default=r'C:\Users\User\repos\Transformer\Sample_Generation\Samples_TrainV9_noisy.hdf5', 
                        help='Path to training data HDF5 (for normalization stats)')
    parser.add_argument('--device', type=str, default='cuda', 
                        help='Device (cuda or cpu)')
    parser.add_argument('--save_fig', type=str, default='results.png', 
                        help='Path to save visualization (optional)')
    
    args = parser.parse_args()
    
    # Load model
    print(f"\nLoading model from {args.model}...")
    model = MultiASDEncoderV9(
        d_model=256,
        num_heads=16,
        num_layers=8,
        d_ff=2048,
        dropout=0.22,
        num_freq_bins=1024
    )
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    
    # Load normalization stats
    norm_stats = NormalizationStatsV9(args.training_data)
    
    # Load and preprocess real data
    qasd, freq = load_real_data(args.data)
    qasd_tensor = preprocess_real_data(qasd)
    
    # Run inference
    results = run_inference(model, qasd_tensor, norm_stats, device=args.device)
    
    # Format and print results
    results_dict = format_results(results, sample_idx=0)
    print_results_alog_style(results_dict)
    
    # Visualize if requested
    if args.save_fig:
        visualize_results(qasd, freq, results_dict, sample_idx=0, save_path=args.save_fig)
    
    return results_dict

if __name__ == '__main__':
    main()