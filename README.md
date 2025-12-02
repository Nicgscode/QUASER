# LIGO Quantum Noise Parameter Estimation with Transformers ~ QUASER (QUAntum Squeezing EstimatoR)

A deep learning approach to real-time quantum noise characterization for gravitational wave detectors, achieving >1000× speedup over traditional hand-fitting methods.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Table of Contents

1. [Problem Statement & Overview](#1-problem-statement--overview)
2. [Methodology](#2-methodology)
3. [Implementation & Demo](#3-implementation--demo)
4. [Assessment & Evaluation](#4-assessment--evaluation)
5. [Model & Data Cards](#5-model--data-cards)
6. [Critical Analysis](#6-critical-analysis)
7. [References & Resources](#7-references--resources)
8. [Formal Algorithms](#8-formal-algorithms)

---

## 1. Problem Statement & Overview

### The Challenge

LIGO (Laser Interferometer Gravitational-Wave Observatory) detectors are limited by **quantum noise** at most frequencies. To optimize detector sensitivity, operators must characterize 15+ quantum noise parameters from measured Amplitude Spectral Density (ASD) curves. The current approach uses **Hand Fitting** methods, which require **days of work** per measurement—far too slow for real-time detector commissioning.

![Handfitting Example](Images/hand_fit_example.png)

### Our Solution

We developed a **transformer-based neural network** that predicts quantum noise parameters directly from ASD measurements in **milliseconds**, enabling real-time detector optimization during commissioning activities.

### Key Innovation

This work reveals a fundamental physics insight: certain parameters are unlearnable from power spectral density measurements alone. Specifically, mode mismatch phase parameters cannot be recovered because the scattered field amplitude scales as √Υ (small for typical 2-5% mismatches), making the phase contribution negligible compared to order-unity effects from other parameters. This is consistent with the parametric study in Section 6.3.1 of [Jia 2024], which shows that varying mismatch phase parameters (e.g., MM_IFO_OMCphi, MM ARM SRCphi) from -90° to +90° produces minimal change in the sensing function (see Figure 6-3 [Jia 2024]).

The speed and consistent output is another key innovation, with hand fitting two individuals may conclude on two valid parameter values. Having a model decide the value lowers the ambiguity for quantum parameters. 

### Pipeline Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Input: ASD    │     │   Transformer   │     │    Output:      │
│   (10 × 1024)   │ ──► │    Encoder      │ ──► │  15 Parameters  │
│   10 configs    │     │   8 layers      │     │   + 5 angles    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
     ~1 ms                  ~10 ms                   ~1 ms
                    Total: ~12 ms vs ~2 hours (MCMC)
```

---

## 2. Methodology

### 2.1 Theoretical Background

#### Quantum Noise in Gravitational Wave Detectors
![LIGO Design](Images/LIGO_design.jpeg)

LIGO's sensitivity is fundamentally limited by quantum fluctuations of light:

![LIGO's noise curve](Images/LIGO_Noise_Curve.png)
- **Shot Noise**: Dominates at high frequencies (>100 Hz), arises from photon counting statistics
- **Quantum Radiation Pressure Noise**: Dominates at low frequencies (<60 Hz), momentum transfer from photons (Photons hitting the mirrors are a noise source!)

The quantum noise power spectral density depends on multiple parameters including arm cavity power, squeezing injection, filter cavity detuning, and various optical losses and mode mismatches.

#### Frequency-Dependent Squeezing

![SQZ_Example](Images/Squeezing_Example.png)
Modern LIGO uses **frequency-dependent squeezing (FDS)** to reduce quantum noise across all frequencies. A 300m filter cavity rotates the squeezing ellipse as a function of frequency, requiring precise characterization of:

- Squeezing level and injection losses
- Filter cavity detuning and bandwidth  
- Mode mismatch between optical cavities
- Squeezing angle for each measurement configuration

#### Data Prep
 We generate the frequency bins as 
 ``` python
     frequencies = np.geomspace(10, 6000, 1024)
 ```


### 2.2 Transformer Architecture

We employ a **Vision Transformer (ViT)-inspired** architecture adapted for 1D spectral data:

#### Architecture Details

| Component | Specification |
|-----------|---------------|
| Input | 10 ASD curves × 1024 frequency bins |
| Embedding Dimension | 256 |
| Attention Heads | 16 |
| Transformer Layers | 8 |
| Feed-Forward Dimension | 2048 |
| Dropout | 0.22 |
| Total Parameters | ~12.4M |

#### Key Design Choices

1. **Multi-ASD Processing**: Processes 10 ASDs simultaneously (5 FDS + 5 FIS configurations) to maximize information extraction

2. **Learnable Frequency Embedding**: A learnable positional embedding over frequency bins helps the model understand the physics at different frequency ranges:
   ```python
   self.freq_pos_embed = nn.Parameter(torch.randn(1, 1, 1024) * 0.02)
   ```
   
3. **Structured Positional Embeddings**:
   - Position embedding: Which of 10 ASDs
   - Type embedding: FDS (0-4) vs FIS (5-9)
   - Angle embedding: Which squeezing angle configuration
   - Frequency embedding: Which frequency bin

4. **Dual Output Heads**:
   - CLS token → Direct parameters (shared across configurations)
   - Paired ASD features → Squeezing angles (configuration-specific)

#### Attention Mechanism

The self-attention mechanism from "Attention Is All You Need" (Vaswani et al., 2017):

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

This allows the model to learn correlations between different frequency regions and ASD configurations.

### 2.3 Handling Angular Periodicity

#### The Problem

Squeezing angles exhibit π-periodicity due to the sin²/cos² dependence in the quantum noise power spectrum (Eq. 4.63 in [Jia 2024]). Specifically, the squeezed quantum noise scales as:

$$S_P(\Omega) \propto e^{2r}\sin^2(\phi - \theta) + e^{-2r}\cos^2(\phi - \theta)$$

where angles $\phi$ and $\phi + \pi$ produce identical measurements.

#### The Solution: Sin/Cos Encoding

For π-periodic angles, we use:
```python
# Encoding
sin(2θ), cos(2θ)  # Maps θ=0 and θ=π to same point (0, 1)

# Decoding  
θ = arctan2(sin, cos) / 2 mod π
```

### 2.4 Loss Functions & Regularization

#### Multi-Task Loss

```python
loss = loss_direct + loss_angles + λ * loss_variance_reg
```

Where:
- `loss_direct`: MSE on normalized direct parameters
- `loss_angles`: MSE on sin/cos encoded angles
- `loss_variance_reg`: Prevents mode collapse on degenerate parameters

#### Variance Regularization

Some parameters (mode mismatch magnitudes) are highly degenerate with other parameters. We add variance regularization to prevent collapse to mean predictions:

```python
def variance_regularization_loss(predictions, indices, min_var=0.02):
    selected = predictions[:, indices]
    variances = torch.var(selected, dim=0)
    penalties = F.relu(min_var - variances) ** 2
    return penalties.mean()
```

### 2.5 Training Strategy

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning Rate | 3×10⁻⁴ |
| Weight Decay | 0.01 |
| Scheduler | Cosine Annealing |
| Batch Size | 256 |
| Epochs | 101 |
| Training Samples | 160,000 |
| Validation Samples | 20,000 |

---

## 3. Implementation & Demo

### 3.1 Installation

```bash
# Clone repository
git clone https://github.com/Nicgscode/ligo-quantum-transformer.git
cd ligo-quantum-transformer

# Create environment
conda create -n ligo-transformer python=3.10
conda activate ligo-transformer

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy h5py matplotlib tqdm
```

### 3.3 Data Generation

Generate synthetic training data using quantum noise physics model:

```python
python Generate_Data/Generate_ASDs.py
```

Add realistic noise to simulations:

```python
python add_noise.py --input Generate_Data/Samples_Train.hdf5 --output Generate_Data/Samples_Train_noisy.hdf5
```

### 3.4 Training

```python
from train_v9 import train_v9

model, data, history = train_v9(
    FILE_PATH,
    num_samples=200001,
    num_epochs=101,
    batch_size=512,
    lr=2e-4,
    d_model=256,
    num_heads=16,
    num_layers=8,
    d_ff=2048,
    dropout=0.22,
    var_reg_weight=0.9,  # Variance regularization weight
    var_reg_min=0.04      # Minimum target variance
)
```

### 3.5 Inference on Real Data

```bash
python inference_real_data_v9.py \
    --model best_model_v9.pt \
    --data Generate_Data/Real_Data_08_08_25.hdf5 \
    --training_data Generate_Data/Samples_Train.hdf5 \
    --save_fig results.png
```

Example output (LIGO alog format):

```
+----------------------------------------+
| Arm Power                   | 280 kW   |
| SEC detuning                | -0.216°  |
| IFO-OMC mismatch            | 2.8 %    |
| Injected Squeezing          | 17.27 dB |
| Injection loss              | 23.1 %   |
| SQZ-OMC mismatch            | 3.9 %    |
| SQZ-FC mismatch             | 1.8 %    |
| FC detuning                 | -27.7 Hz |
| Phase noise                 | 18 mrad  |
+----------------------------------------+
| Squeezing Angles                       |
+----------------------------------------+
|   θ_0                       | 45.23°   |
|   θ_1                       | 32.11°   |
|   θ_2                       | 67.89°   |
|   θ_3                       | 12.45°   |
|   θ_4                       | 88.76°   |
+----------------------------------------+
```

## 4. Assessment & Evaluation

### 4.1 Model Performance
![Model Performance](Images/model_ouput_clean.png)

### 4.3 Intended Uses

**Primary Use Cases:**
- Real-time quantum noise monitoring during LIGO commissioning
- Rapid parameter estimation for detector optimization
- Initial parameter guesses for MCMC refinement

**Out of Scope:**
- Final parameter values for publications (use MCMC with uncertainty)
- Phase parameter estimation (fundamentally unlearnable from PSD)
- Operation on detectors with significantly different configurations

### 4.4 Limitations & Known Issues

1. **Mode Mismatch Degeneracy**: Parameters `sqz_omc_mm` and `fc_mm` show weak correlations due to degeneracy with arm power and losses

2. **Phase Parameters Unlearnable**: Mismatch phase parameters cannot be recovered from PSD measurements (fundamental physics limitation)

3. **Domain Shift**: Model trained on simulated data; real data performance depends on simulation fidelity

### 4.5 Ethical Considerations & Bias

**Data Bias:**
- Training data generated from physics models may not capture all real-world effects
- Parameter ranges chosen based on expected LIGO operating conditions

**Potential Misuse:**
- Model should not be used as sole source of truth for detector parameters
- Uncertainty quantification not provided (future work)

**Fairness:**
- N/A (scientific instrument, no human subjects)

### 4.6 License

This project is released under the **MIT License**.

```
MIT License

Copyright (c) 2024 LIGO Laboratory / Caltech

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software...
```

---

## 5. Model & Data Cards

### 5.1 Model Card

| Field | Description |
|-------|-------------|
| **Model Name** | MultiASDEncoderV9 |
| **Model Type** | Transformer Encoder |
| **Task** | Multi-output Regression |
| **Input** | 10 ASD curves × 1024 frequency bins |
| **Output** | 15 physical parameters + 5 squeezing angles |
| **Parameters** | ~12.4 million |
| **Training Data** | 160,000 synthetic samples |
| **Validation Data** | 20,000 synthetic samples |
| **Framework** | PyTorch 2.0+ |
| **Hardware** | NVIDIA GeForce RTX 4060  |
| **Training Time** | ~2 hours (101 epochs on 4060) |

### 5.2 Data Card

| Field | Description |
|-------|-------------|
| **Dataset Name** | Samples_Train_noisy.hdf5 |
| **Size** | ~3 GB (200,001 samples) |
| **Format** | HDF5 |
| **Generation** | Physics-based quantum noise model |
| **Frequency Range** | 10 - 8000 Hz |
| **Frequency Bins** | 1024 (log-spaced) |
| **Noise Model** | Gaussian + uncertainty spikes |

#### Parameter Ranges

| Parameter | Min | Max | Units |
|-----------|-----|-----|-------|
| arm_power | 200 | 400 | kW |
| fc_detune | -50 | 50 | Hz |
| inj_sqz | 0 | 21 | dB |
| inj_lss | 0.0 | 1 | fraction |
| sec_detune | -0.03 | 0.03 | rad |
| ifo_omc_mm | 0.01 | 0.20 | fraction |
| sqz_omc_mm | 0.01 | 0.20 | fraction |
| fc_mm | 0.01 | 0.10 | fraction |
| lo_angle | 1.3 | 1.9 | rad |
| phase_noise | 0.01 | 0.50 | rad |
| sqz_angles | 0 | π | rad |

---

## 6. Critical Analysis

### 6.1 Impact of This Project

This work addresses a critical bottleneck in gravitational wave detector operations:

1. **Operational Impact**: Enables real-time quantum noise characterization during LIGO commissioning, reducing parameter estimation from days to milliseconds.

2. **Scientific Impact**: Demonstrates which quantum noise parameters are fundamentally identifiable from power spectral measurements.

3. **Methodological Impact**: Shows that transformer architectures can effectively learn complex physics relationships.

### 6.2 Key Findings & Revelations

#### Finding 1: Phase Information Loss

**Mismatch phase parameters are fundamentally unlearnable from ASD/PSD measurements.**

This occurs because:
- ASD = |FFT(signal)|² discards phase information
- Scattered field amplitude ∝ √Υ (small for 2-5% mismatches)
- Phase contribution is negligible compared to amplitude effects

This is a **physics limitation**, not a model limitation—validated by thesis Section 6.3.1.

#### Finding 2: Parameter Degeneracy

Mode mismatch magnitudes show weak correlations due to degeneracy with arm power and optical losses. The effect of 5% mode mismatch can be nearly indistinguishable from a small change in arm power.

#### Finding 3: Transformer Effectiveness

The transformer learns meaningful correlations for 12/15 parameters—demonstrating that attention mechanisms can capture complex frequency-dependent physics.

### 6.3 Next Steps

1. **Uncertainty Quantification**: Implement ensemble predictions or MC dropout for confidence intervals

2. **Multi-Modal Input**: Incorporate transfer function measurements to recover phase parameters

3. **Domain Adaptation**: Fine-tune on real LIGO data to bridge simulation-reality gap

4. **Deployment**: Integrate with LIGO control systems for live monitoring

5. **Extended Physics**: Include additional noise sources (seismic, thermal, technical)

---

## 7. References & Resources

### 7.1 Key Papers

1. **Vaswani, A., et al.** (2017). "Attention Is All You Need." *NeurIPS*.
   - Foundation of transformer architecture
   - [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

2. **Jia, W.** (2024). "Quantum Noise Parameter Estimation for LIGO." *Caltech Thesis*.
   - Theoretical foundation for quantum noise model
   - MCMC baseline comparisons

3. **LIGO Scientific Collaboration** (2015). "Advanced LIGO." *Classical and Quantum Gravity*.
   - Detector configuration and noise sources
   - [DOI:10.1088/0264-9381/32/7/074001](https://doi.org/10.1088/0264-9381/32/7/074001)

4. **Dosovitskiy, A., et al.** (2020). "An Image is Worth 16x16 Words." *ICLR*.
   - Vision Transformer architecture inspiration
   - [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

### 7.2 Code Repositories

- **This Repository**: [GitHub Link]
- **LIGO Algorithm Library (LAL)**: https://git.ligo.org/lscsoft/lalsuite
- **GWpy**: https://gwpy.github.io/ (Gravitational wave data analysis)

### 7.3 Additional Resources

- LIGO Open Science Center: https://losc.ligo.org/
- Gravitational Wave Open Data Workshop: https://gw-odw.thinkific.com/
- PyTorch Documentation: https://pytorch.org/docs/

---

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/Nicgscode/ligo-quantum-transformer.git
cd ligo-quantum-transformer
pip install -r requirements.txt

# 2. Download pre-trained model
wget https://github.com/Nicgscode/QUASER/releases/download/Pre/best_model_v9_nonoise.pt

# 3. Run inference
python inference_real_data_v9.py \
    --model best_model_v9.pt \
    --data your_data.hdf5 \
    --training_data Samples_TrainV9_noisy.hdf5
```

---

## Citation

If you use this work, please cite:

```bibtex
@misc{ligo-quantum-transformer,
  author = {Howard, Nic},
  title = {LIGO Quantum Noise Parameter Estimation with Transformers},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Nicgscode/ligo-quantum-transformer}
}
```

---

## Contact

- **Author**: Nicholas-Tyler Howard (Nic)
- **Affiliation**: LIGO Laboratory / Caltech / Fisk University
- **Email**: nicholastyler.howard@ligo.org


## 8. Formal Algorithms
---
## Algorithm 1: Multi-ASD Transformer Forward Pass

**Algorithm 1:** $\hat{\mathbf{y}} \leftarrow \text{MultiASDEncoder}(\mathbf{A} | \boldsymbol{\theta})$

```
/* Multi-ASD Transformer encoder forward pass for quantum parameter estimation */

Input: A ∈ ℝ^(N_asd × N_freq), matrix of N_asd = 10 ASD spectra with N_freq = 1024 frequency bins
Output: ŷ_direct ∈ ℝ^10, ŷ_angles ∈ ℝ^10, predicted direct parameters and sin/cos encoded angles

Hyperparameters: d_model, N_heads, N_layers, d_ff, p_drop ∈ ℕ × ℕ × ℕ × ℕ × ℝ

Parameters: θ includes all of the following parameters:
    W_freq ∈ ℝ^(1 × N_freq), learnable frequency positional embedding,
    W_proj ∈ ℝ^(N_freq × d_model), b_proj ∈ ℝ^d_model, ASD projection parameters,
    W_pos ∈ ℝ^(N_asd × d_model), positional embeddings for each ASD,
    W_type ∈ ℝ^(2 × d_model), type embeddings (FDS vs FIS),
    W_angle ∈ ℝ^(5 × d_model), angle configuration embeddings,
    e_cls ∈ ℝ^d_model, learnable CLS token,
    For l ∈ [N_layers]:
        𝒲_l^enc, multi-head self-attention parameters for layer l,
        γ_l^1, β_l^1, γ_l^2, β_l^2 ∈ ℝ^d_model, two sets of layer-norm parameters,
        W_mlp1^l ∈ ℝ^(d_model × d_ff), b_mlp1^l ∈ ℝ^d_ff, W_mlp2^l ∈ ℝ^(d_ff × d_model), b_mlp2^l ∈ ℝ^d_model,
    W_direct ∈ ℝ^(d_model × d_ff), b_direct ∈ ℝ^d_ff, W_direct2 ∈ ℝ^(d_ff × 10), b_direct2 ∈ ℝ^10,
    W_angle_head ∈ ℝ^(2·d_model × d_ff), b_angle_head ∈ ℝ^d_ff, W_angle2 ∈ ℝ^(d_ff × 2), b_angle2 ∈ ℝ^2.

/* Add frequency positional embedding to input ASDs: */
 1  for i ∈ [N_asd]: A[i, :] ← A[i, :] + W_freq

/* Project ASDs to model dimension: */
 2  for i ∈ [N_asd]: z_i ← A[i, :] · W_proj + b_proj
 3  Z ← [z_1, z_2, ..., z_{N_asd}]^T                          /* Z ∈ ℝ^(N_asd × d_model) */

/* Add structured positional embeddings: */
 4  for i ∈ [N_asd]: Z[i, :] ← Z[i, :] + W_pos[i, :]
 5  for i ∈ [5]: Z[i, :] ← Z[i, :] + W_type[0, :]            /* FDS type embedding */
 6  for i ∈ [5, 10): Z[i, :] ← Z[i, :] + W_type[1, :]        /* FIS type embedding */
 7  for i ∈ [5]: Z[i, :] ← Z[i, :] + W_angle[i, :]           /* FDS angle embedding */
 8  for i ∈ [5, 10): Z[i, :] ← Z[i, :] + W_angle[i-5, :]     /* FIS angle embedding */

/* Prepend CLS token: */
 9  Z ← [e_cls; Z]                                            /* Z ∈ ℝ^(11 × d_model) */

/* Apply transformer encoder layers: */
10  for l = 1, 2, ..., N_layers do
11      Z ← Z + MHAttention(LayerNorm(Z | γ_l^1, β_l^1) | 𝒲_l^enc)
12      Z ← Z + W_mlp2^l · GELU(W_mlp1^l · LayerNorm(Z | γ_l^2, β_l^2) + b_mlp1^l) + b_mlp2^l
13  end
14  Z ← LayerNorm(Z | γ_final, β_final)

/* Extract CLS token and ASD features: */
15  h_cls ← Z[0, :]                                           /* h_cls ∈ ℝ^d_model */
16  H_asd ← Z[1:, :]                                          /* H_asd ∈ ℝ^(10 × d_model) */

/* Compute direct parameters via CLS token: */
17  ŷ_direct ← σ(W_direct2 · GELU(W_direct · h_cls + b_direct) + b_direct2)

/* Compute squeezing angles via paired FDS/FIS features: */
18  for i ∈ [5]:
19      h_combined ← [H_asd[i, :]; H_asd[i+5, :]]             /* Concatenate FDS_i and FIS_i */
20      [ŝ_i, ĉ_i] ← tanh(W_angle2 · GELU(W_angle_head · h_combined + b_angle_head) + b_angle2)
21  end
22  ŷ_angles ← [ŝ_0, ĉ_0, ŝ_1, ĉ_1, ..., ŝ_4, ĉ_4]          /* Interleaved sin/cos */

23  return ŷ_direct, ŷ_angles
```

**Where:**
- $\sigma(\cdot)$ denotes the sigmoid activation function
- $\text{GELU}(\cdot)$ denotes the Gaussian Error Linear Unit activation
- $\tanh(\cdot)$ denotes the hyperbolic tangent activation
- $\text{MHAttention}(\cdot)$ denotes multi-head self-attention (see Algorithm 3)
- $\text{LayerNorm}(\cdot | \gamma, \beta)$ denotes layer normalization with scale $\gamma$ and shift $\beta$

---

## Algorithm 2: Training Procedure with Variance Regularization

**Algorithm 2:** $\boldsymbol{\theta}^* \leftarrow \text{Train}(\mathcal{D} | \boldsymbol{\theta}_0, \eta, \lambda_{var})$

```
/* Training procedure for Multi-ASD Transformer with variance regularization */

Input: 𝒟 = {(A^(n), y_direct^(n), θ_raw^(n))}_{n=1}^N, training dataset of ASD-parameter pairs
Output: θ*, optimized model parameters

Hyperparameters: 
    N_epochs ∈ ℕ, number of training epochs,
    B ∈ ℕ, batch size,
    η ∈ ℝ+, initial learning rate,
    λ_var ∈ ℝ+, variance regularization weight (default 0.05),
    σ_min ∈ ℝ+, minimum target variance (default 0.02),
    ℐ_collapse = {5, 6, 7}, indices of mode-collapse prone parameters.

/* Precompute target encodings: */
 1  for n ∈ [N]:
 2      y_direct_norm^(n) ← normalize(y_direct^(n))           /* Min-max to [0, 1] */
 3      for i ∈ [5]:
 4          s_i^(n) ← sin(2 · θ_raw^(n)[i])                   /* π-periodic encoding */
 5          c_i^(n) ← cos(2 · θ_raw^(n)[i])
 6      end
 7      y_angles^(n) ← [s_0^(n), c_0^(n), ..., s_4^(n), c_4^(n)]
 8  end

/* Initialize optimizer and scheduler: */
 9  optimizer ← AdamW(θ, lr=η, weight_decay=0.01)
10  scheduler ← CosineAnnealingLR(optimizer, T_max=N_epochs)

/* Training loop: */
11  for epoch = 1, 2, ..., N_epochs do
12      for each minibatch ℬ ⊂ 𝒟 of size B do
13          /* Forward pass: */
14          {ŷ_direct^(n), ŷ_angles^(n)}_{n∈ℬ} ← MultiASDEncoder({A^(n)}_{n∈ℬ} | θ)
          
15          /* Compute MSE losses: */
16          ℒ_direct ← (1/B) Σ_{n∈ℬ} ||ŷ_direct^(n) - y_direct_norm^(n)||_2^2
17          ℒ_angles ← (1/B) Σ_{n∈ℬ} ||ŷ_angles^(n) - y_angles^(n)||_2^2
          
18          /* Compute variance regularization for mode-collapse parameters: */
19          for j ∈ ℐ_collapse:
20              σ_j^2 ← Var({ŷ_direct^(n)[j]}_{n∈ℬ})
21              penalty_j ← max(0, σ_min - σ_j^2)^2
22          end
23          ℒ_var ← (1/|ℐ_collapse|) Σ_{j∈ℐ_collapse} penalty_j
          
24          /* Total loss: */
25          ℒ ← ℒ_direct + ℒ_angles + λ_var · ℒ_var
          
26          /* Backward pass and update: */
27          θ ← θ - optimizer_step(∇_θ ℒ, clip_norm=1.0)
28      end
29      scheduler.step()
30  end

31  return θ* ← θ
```

**Where:**
- $\text{Var}(\cdot)$ denotes sample variance over the batch dimension
- $\text{normalize}(\cdot)$ denotes min-max normalization to $[0, 1]$
- The variance regularization (lines 18-23) prevents mode collapse on degenerate parameters

---

## Algorithm 3: Multi-Head Self-Attention

**Algorithm 3:** $\mathbf{Y} \leftarrow \text{MHAttention}(\mathbf{X} | \mathcal{W})$

```
/* Multi-head self-attention mechanism */

Input: X ∈ ℝ^(L × d_model), sequence of L token embeddings
Output: Y ∈ ℝ^(L × d_model), attended representations

Parameters: 𝒲 = {W_Q^h, W_K^h, W_V^h ∈ ℝ^(d_model × d_k), W_O ∈ ℝ^(H·d_k × d_model)} for h ∈ [H]

 1  for h = 1, 2, ..., H do                                   /* H attention heads */
 2      Q^h ← X · W_Q^h                                       /* Queries: ℝ^(L × d_k) */
 3      K^h ← X · W_K^h                                       /* Keys: ℝ^(L × d_k) */
 4      V^h ← X · W_V^h                                       /* Values: ℝ^(L × d_k) */
 5      A^h ← softmax((Q^h · (K^h)^T) / √d_k)                 /* Attention weights */
 6      head_h ← A^h · V^h
 7  end
 8  Y ← [head_1; head_2; ...; head_H] · W_O                   /* Concatenate and project */

 9  return Y
```

---

## Algorithm 4: Angle Decoding (Inference)

**Algorithm 4:** $\hat{\boldsymbol{\theta}} \leftarrow \text{DecodeAngles}(\hat{\mathbf{y}}_{angles})$

```
/* Decode sin/cos predictions back to angles in [0, π] */

Input: ŷ_angles ∈ ℝ^10, predicted [sin(2θ_0), cos(2θ_0), ..., sin(2θ_4), cos(2θ_4)]
Output: θ̂ ∈ ℝ^5, decoded squeezing angles in [0, π]

 1  for i = 0, 1, ..., 4 do
 2      ŝ_i ← ŷ_angles[2i]                                    /* Predicted sin(2θ_i) */
 3      ĉ_i ← ŷ_angles[2i + 1]                                /* Predicted cos(2θ_i) */
 4      φ_i ← atan2(ŝ_i, ĉ_i)                                 /* Recover 2θ ∈ [-π, π] */
 5      θ̂_i ← (φ_i / 2) mod π                                 /* Map to [0, π] */
 6  end

 7  return θ̂ = [θ̂_0, θ̂_1, θ̂_2, θ̂_3, θ̂_4]
```

**Note:** The $\text{mod } \pi$ operation ensures the output respects the π-periodicity of squeezing angles, where $\theta$ and $\theta + \pi$ produce physically identical quantum noise spectra.

---

## Algorithm 5: Circular Correlation Metric

**Algorithm 5:** $r_{circ} \leftarrow \text{CircularCorrelation}(\hat{\boldsymbol{\theta}}, \boldsymbol{\theta})$

```
/* Compute circular correlation for π-periodic angles */

Input: θ̂ ∈ ℝ^N, predicted angles; θ ∈ ℝ^N, target angles
Output: r_circ ∈ [-1, 1], circular correlation coefficient

 1  r_circ ← (1/N) Σ_{i=1}^N cos(2 · (θ̂_i - θ_i))

 2  return r_circ
```

**Interpretation:**
- $r_{circ} = 1$: Perfect prediction ($\hat{\theta} = \theta$ or $\hat{\theta} = \theta + \pi$)
- $r_{circ} = 0$: Random predictions
- $r_{circ} = -1$: Maximally wrong (off by $\pi/2$)

---

## Summary of Notation

| Symbol | Description | Dimensions |
|--------|-------------|------------|
| $\mathbf{A}$ | Input ASD matrix | $10 \times 1024$ |
| $d_{model}$ | Model embedding dimension | 256 |
| $N_{heads}$ | Number of attention heads | 16 |
| $N_{layers}$ | Number of transformer layers | 7 |
| $d_{ff}$ | Feed-forward hidden dimension | 1024 |
| $\mathbf{Z}$ | Sequence of token embeddings | $(11) \times d_{model}$ |
| $\hat{\mathbf{y}}_{direct}$ | Predicted direct parameters | 10 |
| $\hat{\mathbf{y}}_{angles}$ | Predicted angle sin/cos values | 10 |
| $\mathcal{I}_{collapse}$ | Mode-collapse parameter indices | $\{5, 6, 7\}$ |
| $\lambda_{var}$ | Variance regularization weight | 0.05 |

---

## Parameter Definitions

**Direct Parameters** (indices 0-9):
| Index | Symbol | Description | Units |
|-------|--------|-------------|-------|
| 0 | $f_{FC}$ | Filter cavity detuning | Hz |
| 1 | $r_{inj}$ | Injected squeezing level | dB |
| 2 | $\eta_{inj}$ | Injection loss | fraction |
| 3 | $P_{arm}$ | Arm cavity power | W |
| 4 | $\phi_{SEC}$ | SEC detuning | rad |
| 5 | $\Upsilon_{IFO}$ | IFO-OMC mode mismatch | fraction |
| 6 | $\Upsilon_{SQZ}$ | SQZ-OMC mode mismatch | fraction |
| 7 | $\Upsilon_{FC}$ | Filter cavity mismatch | fraction |
| 8 | $\zeta$ | Local oscillator angle | rad |
| 9 | $\phi_{noise}$ | Phase noise RMS | rad |

**Squeezing Angles** (5 values):
| Index | Symbol | Description | Range |
|-------|--------|-------------|-------|
| 0-4 | $\theta_0, ..., \theta_4$ | Squeezing angle per config | $[0, \pi]$ |

---

*This project was developed as part of research at the LIGO Laboratory, California Institute of Technology.*
