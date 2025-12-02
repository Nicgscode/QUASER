"""
train_v9.py - Training script for Multi-ASD Transformer V9

Sin/Cos Encoding for π-Periodic Angles:
    Squeezing angles have π-periodicity (θ and θ+π produce identical ASDs).
    Using sin(2θ)/cos(2θ) encoding maps θ=0 and θ=π to the same point,
    eliminating the artificial discontinuity at boundaries.

Variance Regularization:
    Applied to indices [5, 6, 7] = ifo_omc_mm, sqz_omc_mm, fc_mm
    These parameters are highly degenerate and prone to mode collapse.
    
    Formula: var_loss = -λ * mean(var(predictions[:, idx]))
    This encourages the model to maintain prediction variance rather than
    collapsing to the mean.
"""

import numpy as np
import h5py
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model.model_final_v9 import MultiASDDataLoaderV9, MultiASDEncoderV9


def variance_regularization_loss(predictions, indices, min_var=0.01):
    """
    Compute variance regularization loss for specified parameter indices.
    
    Encourages model to maintain variance in predictions to prevent mode collapse.
    
    Args:
        predictions: (B, num_params) tensor of predictions
        indices: list of parameter indices to regularize
        min_var: minimum target variance (default 0.01 for [0,1] normalized data)
    
    Returns:
        var_loss: scalar loss (lower when variance is higher)
    """
    selected = predictions[:, indices]  # (B, len(indices))
    variances = torch.var(selected, dim=0)  # variance per parameter
    
    # Penalize when variance falls below min_var
    # Using smooth hinge: max(0, min_var - var)^2
    penalties = F.relu(min_var - variances) ** 2
    
    return penalties.mean()


def train_v9(file_path, num_samples=200001, num_epochs=100,
             batch_size=256, lr=1e-4, device='cuda', 
             d_model=256, num_heads=8, num_layers=6, d_ff=1024, dropout=0.2,
             var_reg_weight=0.05, var_reg_min=0.02):
    """
    Train the v9 model with variance regularization.
    
    Args:
        file_path: path to HDF5 data file
        num_samples: number of samples to use
        num_epochs: training epochs
        batch_size: batch size
        lr: learning rate
        device: 'cuda' or 'cpu'
        d_model: transformer dimension
        num_heads: attention heads
        num_layers: transformer layers
        d_ff: feedforward dimension
        dropout: dropout rate
        var_reg_weight: weight for variance regularization loss (default 0.8)
        var_reg_min: minimum target variance for regularization (default 0.03)
    """
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data = MultiASDDataLoaderV9(file_path, num_samples)
    
    model = MultiASDEncoderV9(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout,
        num_freq_bins=1024
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Variance regularization: weight={var_reg_weight}, min_var={var_reg_min}")
    print(f"Mode collapse indices: {data.mode_collapse_indices}")
    
    # Loss functions
    mse = nn.MSELoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    num_batches = len(data.qasd) // batch_size
    best_val_loss = float('inf')
    
    # Training history
    history = {
        'train_direct': [], 'train_angles': [], 'train_var_reg': [],
        'val_direct': [], 'val_angles': [],
        'var_per_param': []  # Track variance of each mode-collapse param
    }
    
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_losses = {'direct': 0, 'angles': 0, 'var_reg': 0}
        
        for i in range(num_batches):
            start = i * batch_size
            stop = start + batch_size
            
            asd, direct_target, angles_target = data.batch(device, start, stop)
            
            # Forward
            direct_pred, angles_pred = model(asd)
            
            # MSE Losses
            loss_direct = mse(direct_pred, direct_target)
            loss_angles = mse(angles_pred, angles_target)
            
            # Variance regularization for mode-collapse parameters
            loss_var_reg = variance_regularization_loss(
                direct_pred, 
                data.mode_collapse_indices,
                min_var=var_reg_min
            )
            
            # Total loss
            loss = loss_direct + loss_angles + var_reg_weight * loss_var_reg
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses['direct'] += loss_direct.item()
            train_losses['angles'] += loss_angles.item()
            train_losses['var_reg'] += loss_var_reg.item()
        
        scheduler.step()
        for k in train_losses:
            train_losses[k] /= num_batches
        
        # Validation
        model.eval()
        with torch.no_grad():
            direct_pred, angles_pred = model(data.qasd_val.to(device))
            
            direct_target = data.params_val.to(device)
            angles_target = data.sqz_val.to(device)
            
            val_direct = mse(direct_pred, direct_target).item()
            val_angles = mse(angles_pred, angles_target).item()
            val_loss = val_direct + val_angles
            
            # Track variance of mode-collapse parameters
            var_per_param = torch.var(direct_pred[:, data.mode_collapse_indices], dim=0)
            var_per_param = var_per_param.cpu().numpy()
        
        # Record history
        history['train_direct'].append(train_losses['direct'])
        history['train_angles'].append(train_losses['angles'])
        history['train_var_reg'].append(train_losses['var_reg'])
        history['val_direct'].append(val_direct)
        history['val_angles'].append(val_angles)
        history['var_per_param'].append(var_per_param)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model_v9.pt')
        
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            var_str = ', '.join([f'{v:.4f}' for v in var_per_param])
            print(f"Epoch {epoch:3d}: "
                  f"direct={train_losses['direct']:.5f}/{val_direct:.5f}, "
                  f"angles={train_losses['angles']:.5f}/{val_angles:.5f}, "
                  f"var_reg={train_losses['var_reg']:.5f}, "
                  f"var=[{var_str}]")
    
    # Save training history
    np.savez('training_history_v9.npz', **{k: np.array(v) for k, v in history.items()})
    
    return model, data, history


def pearson_corrcoef(x, y):
    """
    Compute Pearson correlation coefficient using PyTorch.
    Memory-efficient alternative to np.corrcoef.
    
    Args:
        x: (N,) tensor
        y: (N,) tensor
    Returns:
        r: scalar correlation coefficient
    """
    x_mean = x.mean()
    y_mean = y.mean()
    
    x_centered = x - x_mean
    y_centered = y - y_mean
    
    numerator = (x_centered * y_centered).sum()
    denominator = torch.sqrt((x_centered ** 2).sum() * (y_centered ** 2).sum())
    
    # Avoid division by zero
    if denominator < 1e-8:
        return 0.0
    
    return (numerator / denominator).item()


def circular_correlation(pred_angles, target_angles):
    """
    Compute circular correlation for π-periodic angles.
    
    Uses cos(2*(pred - target)) which equals 1 when pred ≈ target or pred ≈ target + π
    
    Args:
        pred_angles: (N,) tensor of predicted angles in [0, π]
        target_angles: (N,) tensor of target angles in [0, π]
    Returns:
        r: mean circular correlation in [0, 1] (1 = perfect)
    """
    # For π-periodic angles, use 2x the angle difference
    diff = 2 * (pred_angles - target_angles)
    return torch.cos(diff).mean().item()


def evaluate_v9(model, data, device='cuda'):
    """Evaluate and visualize results with sin/cos angle decoding."""
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    with torch.no_grad():
        direct_pred, angles_sincos_pred = model(data.qasd_val.to(device))
    
    # Denormalize direct params (keep as tensors for correlation computation)
    direct_pred_phys = data.denormalize_direct(direct_pred)
    direct_target_phys = data.denormalize_direct(data.params_val)
    
    # Decode sin/cos angles back to [0, π]
    angles_pred_phys = data.decode_angles(angles_sincos_pred)
    angles_target_phys = data.sqz_raw_val  # Raw angles stored during data loading
    
    # === COMPUTE CORRELATIONS ===
    direct_names = [
        'fc_detune', 'inj_sqz', 'inj_lss', 'arm_power', 'sec_detune',
        'ifo_omc_mm', 'sqz_omc_mm', 'fc_mm', 'lo_angle', 'phase_noise'
    ]
    angle_names = [f'sqz_angle_{i}' for i in range(5)]
    
    print("\n=== Correlation Coefficients ===")
    print("Direct Parameters:")
    correlations = {}
    for j, name in enumerate(direct_names):
        x = direct_target_phys[:, j]
        y = direct_pred_phys[:, j]
        r = pearson_corrcoef(x, y)
        correlations[name] = r
        marker = "**" if j in data.mode_collapse_indices else ""
        print(f"  {name:20s}: r = {r:.4f} {marker}")
    
    print("\nSqueezing Angles (circular correlation):")
    for j, name in enumerate(angle_names):
        x = angles_target_phys[:, j]
        y = angles_pred_phys[:, j]
        # Use both Pearson and circular correlation
        r_pearson = pearson_corrcoef(x, y)
        r_circular = circular_correlation(y, x)
        correlations[name] = r_pearson
        correlations[name + '_circ'] = r_circular
        print(f"  {name:20s}: r = {r_pearson:.4f}, circ = {r_circular:.4f}")
    
    # === PLOTTING ===
    # Convert to numpy for matplotlib
    direct_pred_np = direct_pred_phys.numpy()
    direct_target_np = direct_target_phys.numpy()
    angles_pred_np = angles_pred_phys.numpy()
    angles_target_np = angles_target_phys.numpy()
    
    all_names = direct_names + angle_names
    n_total = len(all_names)
    
    fig, axes = plt.subplots(3, 5, figsize=(25, 15))
    axes = axes.flatten()
    
    # Direct params
    for j, name in enumerate(direct_names):
        ax = axes[j]
        x = direct_target_np[:, j]
        y = direct_pred_np[:, j]
        r = correlations[name]
        
        ax.scatter(x, y, s=3, alpha=0.3)
        mn, mx = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([mn, mx], [mn, mx], 'r--', lw=1)
        
        # Highlight mode-collapse parameters
        color = 'orange' if j in data.mode_collapse_indices else 'black'
        ax.set_title(f"{name}\nr = {r:.3f}", color=color, fontweight='bold' if j in data.mode_collapse_indices else 'normal')
        ax.set_xlabel("target")
        ax.set_ylabel("prediction")
        ax.grid(True, alpha=0.3)
    
    # Squeezing angles
    for j in range(5):
        ax = axes[10 + j]
        name = angle_names[j]
        x = angles_target_np[:, j]
        y = angles_pred_np[:, j]
        r = correlations[name]
        r_circ = correlations[name + '_circ']
        
        ax.scatter(x, y, s=3, alpha=0.3)
        # Plot y=x line
        ax.plot([0, np.pi], [0, np.pi], 'r--', lw=1)
        ax.set_xlim(0, np.pi)
        ax.set_ylim(0, np.pi)
        ax.set_title(f"{name}\nr = {r:.3f}, circ = {r_circ:.3f}")
        ax.set_xlabel("target (rad)")
        ax.set_ylabel("prediction (rad)")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('v9_results.png', dpi=150)
    plt.show()
    
    return correlations


def plot_training_history(history):
    """Plot training history including variance tracking."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = np.arange(len(history['train_direct']))
    
    # Loss curves
    ax = axes[0, 0]
    ax.semilogy(epochs, history['train_direct'], label='Train Direct')
    ax.semilogy(epochs, history['val_direct'], label='Val Direct')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Direct Parameter Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.semilogy(epochs, history['train_angles'], label='Train Angles')
    ax.semilogy(epochs, history['val_angles'], label='Val Angles')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Squeezing Angle Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Variance regularization
    ax = axes[1, 0]
    ax.plot(epochs, history['train_var_reg'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Variance Reg Loss')
    ax.set_title('Variance Regularization Loss')
    ax.grid(True, alpha=0.3)
    
    # Variance per mode-collapse parameter
    ax = axes[1, 1]
    var_history = np.array(history['var_per_param'])
    param_names = ['ifo_omc_mm', 'sqz_omc_mm', 'fc_mm']
    for i, name in enumerate(param_names):
        ax.plot(epochs, var_history[:, i], label=name)
    ax.axhline(y=0.02, color='r', linestyle='--', label='Min target var')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Prediction Variance')
    ax.set_title('Variance of Mode-Collapse Parameters')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('v9_training_history.png', dpi=150)
    plt.show()


def predict_physical_params(model, data, asd_input, device='cuda'):
    """
    Convenience function to get physical parameters from model output.
    
    Returns dict with all learnable parameters in physical units.
    Note: Phase parameters are NOT included as they are unlearnable from PSD.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    with torch.no_grad():
        direct_pred, angles_sincos_pred = model(asd_input.to(device))
    
    # Denormalize direct params
    direct_phys = data.denormalize_direct(direct_pred).numpy()[0]
    
    # Decode sin/cos angles to [0, π]
    angles_phys = data.decode_angles(angles_sincos_pred).numpy()[0]
    
    return {
        'fc_detune': direct_phys[0],
        'inj_sqz': direct_phys[1],
        'inj_lss': direct_phys[2],
        'arm_power': direct_phys[3],
        'sec_detune': direct_phys[4],
        'ifo_omc_mm': direct_phys[5],
        'sqz_omc_mm': direct_phys[6],
        'fc_mm': direct_phys[7],
        'lo_angle': direct_phys[8],
        'phase_noise': direct_phys[9],
        'sqz_angles': angles_phys,
    }
    
# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    
    FILE_PATH = r'Generate_Data/Samples_Train.hdf5'
    
    model, data, history = train_v9(
        FILE_PATH,
        num_samples=200001,
        num_epochs=101,
        batch_size=256,
        lr=3e-4,
        d_model=256,
        num_heads=16,
        num_layers=8,
        d_ff=1024,
        dropout=0.22,
        var_reg_weight=0.8,  # Variance regularization weight
        var_reg_min=0.03      # Minimum target variance
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate
    correlations = evaluate_v9(model, data)
