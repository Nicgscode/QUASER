"""
add_noise.py - Add realistic noise and spikes to simulated QASDs

Based on real LIGO data characteristics

Usage:
    python add_noise.py --input Samples_Train.hdf5 --output Samples_Train_noisy.hdf5
"""

import numpy as np
import h5py
import argparse
from tqdm import tqdm


def add_gaussian_noise(qasd, freq, noise_std=(2.0, 5.0)):
    """
    Add Gaussian noise to QASDs.
    
    Args:
        qasd: ndarray (10, 1024) - single sample of 10 ASD curves
        freq: ndarray (1024,) - frequency array
        noise_std: tuple (min, max) - range of noise standard deviations in dB
        
    Returns:
        noisy qasd
    """
    qasd = np.array(qasd, dtype=np.float32)
    
    # Random noise level for this sample
    base_std = np.random.uniform(noise_std[0], noise_std[1])
    
    # Frequency-dependent noise profile (more noise at edges)
    freq_norm = np.log10(freq / freq[0]) / np.log10(freq[-1] / freq[0])  # 0 to 1
    freq_profile = 1.0 + 0.5 * (4 * (freq_norm - 0.5)**2)  # U-shaped: more at edges
    
    for i in range(len(qasd)):
        noise = np.random.normal(0, base_std, qasd.shape[1]) * freq_profile
        qasd[i] += noise
    
    return qasd


def add_spikes(qasd, freq, num_spikes_range=(5, 25), spike_amplitude_range=(10, 35)):
    """
    Add random spikes to QASDs.
    
    Args:
        qasd: ndarray (10, 1024)
        freq: ndarray (1024,)
        num_spikes_range: (min, max) number of spikes to add
        spike_amplitude_range: (min, max) amplitude in dB
        
    Returns:
        qasd with spikes
    """
    qasd = np.array(qasd, dtype=np.float32)
    num_spikes = np.random.randint(num_spikes_range[0], num_spikes_range[1])
    
    for _ in range(num_spikes):
        # Random frequency index
        idx = np.random.randint(0, len(freq))
        
        # Random amplitude (positive or negative)
        amplitude = np.random.uniform(spike_amplitude_range[0], spike_amplitude_range[1])
        amplitude *= np.random.choice([-1, 1])
        
        # Spike width (narrow: 1-3 bins)
        width = np.random.randint(1, 4)
        
        # Decide which curves get this spike
        if np.random.random() < 0.3:
            # Common mode: all curves
            curves = range(len(qasd))
        elif np.random.random() < 0.5:
            # Single curve
            curves = [np.random.randint(0, len(qasd))]
        else:
            # Random subset
            num_curves = np.random.randint(2, len(qasd))
            curves = np.random.choice(len(qasd), num_curves, replace=False)
        
        # Apply spike
        start = max(0, idx - width // 2)
        end = min(len(freq), idx + width // 2 + 1)
        
        for c in curves:
            qasd[c, start:end] += amplitude
    
    return qasd


def add_line_features(qasd, freq):
    """
    Add power line harmonics (60 Hz and harmonics) and other persistent lines.
    """
    qasd = np.array(qasd, dtype=np.float32)
    
    # Power line harmonics
    power_lines = [60, 120, 180, 240, 300, 360]
    
    for line_freq in power_lines:
        if line_freq < freq[0] or line_freq > freq[-1]:
            continue
            
        # Find closest frequency bin
        idx = np.argmin(np.abs(freq - line_freq))
        
        # Random amplitude for this line
        if np.random.random() < 0.5:  # 50% chance to have this line
            amplitude = np.random.uniform(5, 20) * np.random.choice([-1, 1])
            
            # Apply to all curves (common mode)
            for i in range(len(qasd)):
                qasd[i, idx] += amplitude
    
    return qasd


def process_sample(qasd, freq, noise_std=(2.0, 5.0), num_spikes_range=(5, 25), 
                   spike_amplitude_range=(10, 35), add_power_lines=True):
    """
    Process a single sample: add noise, spikes, and optional power lines.
    """
    # Add Gaussian noise
    qasd = add_gaussian_noise(qasd, freq, noise_std)
    
    # Add random spikes
    qasd = add_spikes(qasd, freq, num_spikes_range, spike_amplitude_range)
    
    # Optionally add power line harmonics
    if add_power_lines and np.random.random() < 0.3:
        qasd = add_line_features(qasd, freq)
    
    return qasd


def process_hdf5(input_path, output_path, noise_std=(2.0, 5.0), 
                 num_spikes_range=(5, 25), spike_amplitude_range=(10, 35),
                 add_power_lines=True):
    """
    Process entire HDF5 file.
    """
    print(f"Loading {input_path}...")
    
    with h5py.File(input_path, 'r') as hf:
        qasd = hf['Simulated_ASD']['QASD'][:]
        freq = hf['Simulated_Params']['Frequency'][:]
        
        # Load all parameter datasets
        params_group = hf['Simulated_Params']
        param_datasets = {}
        for key in params_group.keys():
            if key != 'Frequency':
                param_datasets[key] = params_group[key][:]
        
        # Get attributes
        param_attrs = dict(params_group.attrs)
    
    num_samples = len(qasd)
    print(f"Processing {num_samples} samples...")
    print(f"  Noise std range: {noise_std[0]:.1f} - {noise_std[1]:.1f} dB")
    print(f"  Spikes per sample: {num_spikes_range[0]} - {num_spikes_range[1]}")
    print(f"  Spike amplitude: {spike_amplitude_range[0]:.0f} - {spike_amplitude_range[1]:.0f} dB")
    
    noisy_qasd = np.zeros_like(qasd)
    
    for i in tqdm(range(num_samples)):
        noisy_qasd[i] = process_sample(
            qasd[i], freq, 
            noise_std=noise_std,
            num_spikes_range=num_spikes_range,
            spike_amplitude_range=spike_amplitude_range,
            add_power_lines=add_power_lines
        )
    
    # Save
    print(f"Saving to {output_path}...")
    
    with h5py.File(output_path, 'w') as hf:
        g1 = hf.create_group('Simulated_ASD')
        g2 = hf.create_group('Simulated_Params')
        
        # Save noisy and clean versions
        g1.create_dataset('QASD', data=noisy_qasd)
        g1.create_dataset('QASD_clean', data=qasd)
        
        # Save all parameters
        g2.create_dataset('Frequency', data=freq)
        for key, data in param_datasets.items():
            g2.create_dataset(key, data=data)
        
        # Copy attributes
        for key, val in param_attrs.items():
            g2.attrs[key] = val
        
        # Add noise metadata
        g2.attrs['noise_std_range'] = noise_std
        g2.attrs['num_spikes_range'] = num_spikes_range
        g2.attrs['spike_amplitude_range'] = spike_amplitude_range
    
    print("Done!")
    print(f"\nStatistics:")
    print(f"  Clean range: [{qasd.min():.2f}, {qasd.max():.2f}]")
    print(f"  Noisy range: [{noisy_qasd.min():.2f}, {noisy_qasd.max():.2f}]")


def visualize_comparison(input_path, output_path, sample_idx=0, save_path='noise_comparison.png'):
    """
    Visualize clean vs noisy for a sample.
    """
    import matplotlib.pyplot as plt
    
    with h5py.File(output_path, 'r') as hf:
        noisy = hf['Simulated_ASD']['QASD'][sample_idx]
        clean = hf['Simulated_ASD']['QASD_clean'][sample_idx]
        freq = hf['Simulated_Params']['Frequency'][:]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Clean
    ax1 = axes[0]
    for i in range(10):
        ax1.semilogx(freq, clean[i], label=f'qasd# {i}', alpha=0.8)
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('dB (Sqz/noSqz)')
    ax1.set_title('Clean Simulated Data')
    ax1.legend(loc='upper right', ncol=2, fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([freq[0], freq[-1]])
    
    # Noisy
    ax2 = axes[1]
    for i in range(10):
        ax2.semilogx(freq, noisy[i], label=f'qasd# {i}', alpha=0.8)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('dB (Sqz/noSqz)')
    ax2.set_title('Noisy Simulated Data (Realistic)')
    ax2.legend(loc='upper right', ncol=2, fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([freq[0], freq[-1]])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add realistic noise to simulated QASDs')
    parser.add_argument('--input', type=str, required=True, help='Input HDF5 path')
    parser.add_argument('--output', type=str, default=None, help='Output HDF5 path')
    parser.add_argument('--noise_min', type=float, default=0.05, help='Min noise std (dB)')
    parser.add_argument('--noise_max', type=float, default=0.2, help='Max noise std (dB)')
    parser.add_argument('--spikes_min', type=int, default=8, help='Min spikes per sample')
    parser.add_argument('--spikes_max', type=int, default=25, help='Max spikes per sample')
    parser.add_argument('--spike_amp_min', type=float, default=4, help='Min spike amplitude (dB)')
    parser.add_argument('--spike_amp_max', type=float, default=25, help='Max spike amplitude (dB)')
    parser.add_argument('--no_power_lines', action='store_true', help='Disable power line harmonics')
    parser.add_argument('--visualize', action='store_true', help='Show comparison plot')
    parser.add_argument('--sample_idx', type=int, default=0, help='Sample to visualize')
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = args.input.replace('.hdf5', '_noisy.hdf5')
    
    process_hdf5(
        args.input,
        args.output,
        noise_std=(args.noise_min, args.noise_max),
        num_spikes_range=(args.spikes_min, args.spikes_max),
        spike_amplitude_range=(args.spike_amp_min, args.spike_amp_max),
        add_power_lines=not args.no_power_lines
    )
    
    if args.visualize:
        visualize_comparison(args.input, args.output, sample_idx=args.sample_idx)
