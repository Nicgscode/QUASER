"""
Generate_ASDsv8.py - Production Quantum ASD Data Generation

Updated to match LIGO commissioning parameters from alog (begum.kabagoz@LIGO.ORG, March 2025):
https://alog.ligo-la.caltech.edu/aLOG/

Parameters from real MCMC/hand-fitting workflow:
    - Arm Power: 280 kW
    - SEC detuning: -0.216°
    - IFO readout efficiency: 0.925
    - IFO-OMC mismatch: 2.8%, 3.31 rad
    - Injected Squeezing: 17.27 dB
    - Injection loss: 6.4 dB
    - SQZ-OMC mismatch: 3.9%, 2.5 rad
    - SQZ-FC mismatch: 1.8%, 3.95 rad
    - FC detuning: -27.7 Hz
    - Phase noise: 18 mrad

Output Parameters (14 physical + 6 encoded = 20 total for model):
    Physical (directly predicted):
        0: fc_detune      - Filter cavity detuning [Hz]
        1: inj_sqz        - Injected squeezing [dB]  
        2: inj_lss        - Injection loss [fraction]
        3: arm_power      - Arm cavity power [W]
        4: sec_detune     - SEC detuning [rad]
        5: ifo_omc_mm     - IFO-OMC mode mismatch [fraction]
        6: sqz_omc_mm     - SQZ-OMC mode mismatch [fraction]
        7: fc_mm          - Filter cavity mismatch [fraction]
        8: lo_angle       - Local oscillator / homodyne angle [rad]
        9: phase_noise    - Phase noise RMS [rad]
    
    Sin/Cos Encoded (for 2π-periodic phases):
        10: sin(ifo_omc_mm_phase)
        11: cos(ifo_omc_mm_phase)
        12: sin(sqz_omc_mm_phase)
        13: cos(sqz_omc_mm_phase)
        14: sin(fc_mm_phase)
        15: cos(fc_mm_phase)
    
    + sqz_angles: (N, 5) in [0, π] (π-periodic, predicted directly)
"""

import numpy as np
import h5py
import gwinc
import tqdm

np.random.seed(2025)  # New seed for production dataset


def generate_params(N):
    """
    Generate N sets of quantum noise parameters matching LIGO commissioning ranges.
    
    Returns:
        params: ndarray (N, 16) - 10 physical + 6 sin/cos encoded
        sqz_angles: ndarray (N, 5) - squeezing angles
        params_raw: ndarray (N, 13) - raw physical values for GWINC
    """
    
    # === TIER 1: CRITICAL (Filter Cavity) ===
    # FC detuning: -27.7 Hz nominal, range based on commissioning experience
    fc_detune = np.random.uniform(-51, 51, N).astype(np.float32)
    
    # === TIER 2: HIGH IMPACT (Squeezing) ===
    # Injected squeezing: 17.27 dB nominal
    inj_sqz = np.random.uniform(0, 21, N).astype(np.float32)
    
    # Injection loss: 6.4 dB nominal (~77% efficiency)
    # As fraction: 0 = no loss, 1 = total loss
    inj_lss = np.random.uniform(0, 1, N).astype(np.float32)
    
    # Squeezing angles: π-periodic (Eq. 4.65 sin²/cos² dependence)
    sqz_angles = np.random.uniform(0, np.pi, (N, 5)).astype(np.float32)
    
    # === TIER 2: HIGH IMPACT (IFO) ===
    # Arm power: 280 kW nominal
    arm_power = np.random.uniform(200e3, 400e3, N).astype(np.float32)
    
    # SEC detuning: -0.216° nominal
    sec_detune = np.random.uniform(-2, 2, N).astype(np.float32) * (np.pi / 180)
    
    # === TIER 3: MODE MISMATCHES (magnitude + phase) ===
    # IFO-OMC: 2.8%, 3.31 rad nominal
    ifo_omc_mm = np.random.uniform(0, 0.2, N).astype(np.float32)
    ifo_omc_mm_phase = np.random.uniform(0, 2*np.pi, N).astype(np.float32)
    
    # SQZ-OMC: 3.9%, 2.5 rad nominal
    sqz_omc_mm = np.random.uniform(0, 0.2, N).astype(np.float32)
    sqz_omc_mm_phase = np.random.uniform(0, 2*np.pi, N).astype(np.float32)
    
    # SQZ-FC (filter cavity): 1.8%, 3.95 rad nominal
    fc_mm = np.random.uniform(0, 0.1, N).astype(np.float32)
    fc_mm_phase = np.random.uniform(0, 2*np.pi, N).astype(np.float32)
    
    # === TIER 3: READOUT ===
    # LO angle (homodyne angle): centered around π/2 with ±21° variation
    # This is Quadrature.dc in GWINC
    lo_angle = np.random.uniform(-19, 19, N).astype(np.float32) * (np.pi/180) + np.pi/2
    
    # === DEGENERATE BUT MEASURED ===
    # Phase noise: 18 mrad nominal
    phase_noise = np.random.uniform(0, 0.5, N).astype(np.float32)
    
    # === ENCODE PHASE PARAMETERS ===
    # Sin/cos encoding handles 2π periodicity naturally
    sin_ifo_phase = np.sin(ifo_omc_mm_phase).astype(np.float32)
    cos_ifo_phase = np.cos(ifo_omc_mm_phase).astype(np.float32)
    sin_sqz_phase = np.sin(sqz_omc_mm_phase).astype(np.float32)
    cos_sqz_phase = np.cos(sqz_omc_mm_phase).astype(np.float32)
    sin_fc_phase = np.sin(fc_mm_phase).astype(np.float32)
    cos_fc_phase = np.cos(fc_mm_phase).astype(np.float32)
    
    # Stack for model training (16 values)
    params_encoded = np.column_stack([
        # Direct physical parameters (10)
        fc_detune,        # 0
        inj_sqz,          # 1
        inj_lss,          # 2
        arm_power,        # 3
        sec_detune,       # 4
        ifo_omc_mm,       # 5
        sqz_omc_mm,       # 6
        fc_mm,            # 7
        lo_angle,         # 8
        phase_noise,      # 9
        # Sin/cos encoded phases (6)
        sin_ifo_phase,    # 10
        cos_ifo_phase,    # 11
        sin_sqz_phase,    # 12
        cos_sqz_phase,    # 13
        sin_fc_phase,     # 14
        cos_fc_phase,     # 15
    ])
    
    # Raw parameters for GWINC (13 values, phases in radians)
    params_raw = np.column_stack([
        fc_detune,
        inj_sqz,
        inj_lss,
        arm_power,
        sec_detune,
        ifo_omc_mm,
        ifo_omc_mm_phase,
        sqz_omc_mm,
        sqz_omc_mm_phase,
        fc_mm,
        fc_mm_phase,
        lo_angle,
        phase_noise,
    ])
    
    return params_encoded, sqz_angles, params_raw


def change_model(budget, params_raw, sqz_angles):
    """
    Apply parameters to GWINC budget and generate ASDs.
    
    Args:
        budget: GWINC budget object
        params_raw: Raw parameter array (13 values with phases in radians)
        sqz_angles: Array of 5 squeezing angles
        
    Returns:
        List of budget runs (5 FDS + 5 FIS + 1 unsqueezed = 11 total)
    """
    budgets = []
    ifo = budget.ifo
    ifo.Squeezer.Type = 'Freq Dependent'
    
    # === Apply parameters ===
    # Filter Cavity
    ifo.Squeezer.FilterCavity.fdetune = params_raw[0]
    
    # Squeezing
    ifo.Squeezer.AmplitudedB = params_raw[1]
    ifo.Squeezer.InjectionLoss = params_raw[2]
    
    # IFO
    ifo.Laser.ArmPower = params_raw[3]
    ifo.Optics.SRM.Tunephase = params_raw[4]
    
    # Mode Mismatches with phases
    ifo.Optics.MM_IFO_OMC = params_raw[5]
    ifo.Optics.MM_IFO_OMCphi = params_raw[6]
    
    ifo.Squeezer.MM_SQZ_OMC = params_raw[7]
    ifo.Squeezer.MM_SQZ_OMCphi = params_raw[8]
    
    ifo.Squeezer.FilterCavity.L_mm = params_raw[9]
    ifo.Squeezer.FilterCavity.psi_mm = params_raw[10]
    
    # Readout - LO angle (homodyne angle)
    ifo.Optics.Quadrature.dc = params_raw[11]
    
    # Phase noise
    ifo.Squeezer.SQZAngleRMS = params_raw[12]
    
    # === Generate ASDs at different squeezing angles ===
    # 5 Frequency-Dependent Squeezing measurements
    for angle in sqz_angles:
        ifo.Squeezer.SQZAngle = angle
        budgets.append(budget.run())
    
    # 5 Frequency-Independent Squeezing measurements
    del ifo.Squeezer.FilterCavity
    ifo.Squeezer.Type = 'Freq Independent'
    for angle in sqz_angles:
        ifo.Squeezer.SQZAngle = angle
        budgets.append(budget.run())
    
    # 1 Unsqueezed reference
    del ifo.Squeezer
    budgets.append(budget.run())
    
    return budgets


def get_qmodel(params_raw, sqz_angles, freq, yaml='april9.yaml'):
    """
    Generate quantum ASD ratios for given parameters.
    """
    qbudget = gwinc.load_budget(yaml, freq, bname='Quantum')
    qtraces = change_model(qbudget, params_raw, sqz_angles)
    
    # Extract ASDs and normalize
    qasd = [qtraces[i].asd / 3995 for i in range(len(qtraces))]
    
    # Ratios relative to unsqueezed
    rqasd = [qasd[i] / qasd[-1] for i in range(len(qasd) - 1)]

    #noise_level = np.random.uniform(0.05, 0.08)  # 5%-10% relative noise
    #noise = [noise_level * rqasd[ind] * np.random.randn(*rqasd[ind].shape) for ind in range(len(rqasd))]    
    
    # Convert to dB
    rqasd = 10 * np.log10(rqasd)
    
    return np.array(rqasd, dtype=np.float32)


def save_data(qasd, params_encoded, sqz_angles, params_raw, freq, 
              filename='Samples_TrainV9.hdf5'):
    """Save generated data to HDF5."""
    with h5py.File(filename, 'w') as hf:
        g1 = hf.create_group('Simulated_ASD')
        g2 = hf.create_group('Simulated_Params')
        
        g1.create_dataset('QASD', data=qasd)
        
        # Encoded parameters for model training
        g2.create_dataset('Parameters_Encoded', data=params_encoded)
        # Raw parameters (for reference/debugging)
        g2.create_dataset('Parameters_Raw', data=params_raw)
        g2.create_dataset('SQZ_Angles', data=sqz_angles)
        g2.create_dataset('Frequency', data=freq)
        
        # Metadata
        g2.attrs['encoded_param_names'] = [
            'fc_detune', 'inj_sqz', 'inj_lss', 'arm_power', 'sec_detune',
            'ifo_omc_mm', 'sqz_omc_mm', 'fc_mm', 'lo_angle', 'phase_noise',
            'sin_ifo_phase', 'cos_ifo_phase', 
            'sin_sqz_phase', 'cos_sqz_phase',
            'sin_fc_phase', 'cos_fc_phase'
        ]
        g2.attrs['raw_param_names'] = [
            'fc_detune', 'inj_sqz', 'inj_lss', 'arm_power', 'sec_detune',
            'ifo_omc_mm', 'ifo_omc_mm_phase', 
            'sqz_omc_mm', 'sqz_omc_mm_phase',
            'fc_mm', 'fc_mm_phase',
            'lo_angle', 'phase_noise'
        ]


def main():
    # Configuration
    num_samples = 200001
    batch_size = 10000
    freq = np.geomspace(10, 8000, 1024)
    
    # IFO configurations
    yaml_configs = np.array(['march19.yaml', 'april9.yaml', 'october23.yaml'])
    
    print("Generating parameters...")
    params_encoded, sqz_angles, params_raw = generate_params(num_samples)
    
    # Randomly assign IFO configurations
    config_indices = np.random.randint(0, len(yaml_configs), num_samples)
    np.save('yaml_orderV9.npy', yaml_configs[config_indices])
    
    # Initialize output
    qASD = np.zeros((num_samples, 10, 1024), dtype=np.float32)
    
    print(f"Generating {num_samples} ASD samples...")
    for i in tqdm.tqdm(range(num_samples)):
        qASD[i] = get_qmodel(
            params_raw[i], 
            sqz_angles[i], 
            freq, 
            yaml_configs[config_indices[i]]
        )
        
        if (i % batch_size == 0) or (i == num_samples - 1):
            save_data(qASD[:i+1], params_encoded[:i+1], sqz_angles[:i+1], 
                     params_raw[:i+1], freq)
    
    print("\nDone!")
    print(f"QASD shape: {qASD.shape}")
    print(f"Encoded params: {params_encoded.shape} (16 values)")
    print(f"Raw params: {params_raw.shape} (13 values)")
    print(f"SQZ angles: {sqz_angles.shape}")


if __name__ == '__main__':
    main()
