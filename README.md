# PLHead-TTA: Test-Time Adaptation for Robust Image Classification

<p align="center">
  <img src="https://github.com/user-attachments/assets/26d52ade-dddd-469c-97ab-028c2f370e5a" width="1000">
</p>

## Overview

This project investigates **Test-Time Adaptation (TTA)** for improving the robustness of deep learning models under distribution shifts. The work reproduces **TENT (Test-Time Entropy Minimization)** and introduces advanced extensions including **MATTA** (Multi-View Augmentation with EMA Teacher) and **PLHead-TTA** (Pseudo-Label Head Adaptation).

Experiments are conducted on the **CIFAR-10-C** benchmark using a pre-trained **WideResNet-28-10** backbone. The objective is to adapt the model during inference without requiring access to source training data or target labels.

---

## Key Features

- Reproduction of TENT on CIFAR-10-C
- Five TENT variant experiments
- BatchNorm layer ablation studies
- Learning rate hyperparameter tuning
- MATTA (EMA Teacher + Multi-View Augmentation)
- PLHead-TTA (Pseudo-Label Guided Head Adaptation)
- Evaluation across 15 corruption types
- Analysis of robustness under severe distribution shifts

---

## Dataset

### CIFAR-10-C

CIFAR-10-C is a benchmark dataset designed to evaluate model robustness against common image corruptions.

#### Corruption Categories

**Noise**
- Gaussian Noise
- Shot Noise
- Impulse Noise

**Blur**
- Defocus Blur
- Glass Blur
- Motion Blur
- Zoom Blur

**Weather**
- Snow
- Frost
- Fog
- Brightness

**Digital**
- Contrast
- Elastic Transform
- Pixelate
- JPEG Compression

**Severity Level Used:** 5

---

## Methods Implemented

### Phase 1 – Baseline

- Source Model (No Adaptation)
- TENT Baseline

### Phase 2 – Reproduction Experiments

- Continual TENT
- Episodic TENT
- Shift-Only Adaptation
- Confidence-Filtered Adaptation
- Squared Entropy Loss

### Phase 3 – Improvements

- Deep BN Layer Adaptation
- Shallow BN Layer Adaptation
- Learning Rate Sweep
- MATTA (Multi-View Augmentation + EMA Teacher)

### Phase 4 – Proposed Method

- PLHead-TTA (Pseudo-Label Head Adaptation)

---

## Results

### Mean Error Across 15 Corruptions

| Method | Mean Error (%) |
|----------|----------:|
| Source | 43.47 |
| Continual TENT | 19.68 |
| Episodic TENT | 20.25 |
| Shift-Only | 20.38 |
| Confidence Filtered | 20.16 |
| Squared Entropy | 20.41 |
| PLHead-TTA | 20.27 |

### Mean Accuracy

| Method | Mean Accuracy (%) |
|----------|----------:|
| Source | 56.53 |
| TENT | 79.75 |
| PLHead-TTA | 83.60 |

---

## Folder Structure

```text
Test_Time_Adaption/
│
├── CODES/
│   ├── phase1_baseline_setup.ipynb
│   ├── phase2_reproduction_experiments.ipynb
│   ├── phase3_model_improvement.ipynb
│   └── phase4_enhancement.ipynb
│
├── report/
│   └── Deep_Learning_Project_Report.pdf
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Requirements

Create a file named:

```text
requirements.txt
```

with the following dependencies:

```text
torch
torchvision
numpy
pandas
matplotlib
seaborn
robustbench
timm
einops
tqdm
jupyter
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Reproducing Results

### 1. Clone the Repository

```bash
git clone https://github.com/RAKSHITART/Test_Time_Adaption.git

cd Test_Time_Adaption
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 4. Run Experiments

#### Baseline Evaluation

```text
notebooks/phase1_baseline_setup.ipynb
```

#### Reproduction Experiments

```text
notebooks/phase2_reproduction_experiments.ipynb
```

#### Ablation Studies and MATTA

```text
notebooks/phase3_model_improvement.ipynb
```

#### PLHead-TTA

```text
notebooks/phase4_enhancement.ipynb
```

Run the notebooks sequentially to reproduce all experiments and reported results.

---

## Experimental Setup

| Parameter | Value |
|------------|---------|
| Backbone | WideResNet-28-10 |
| Dataset | CIFAR-10-C |
| Batch Size | 64 |
| Optimizer | Adam |
| BN Learning Rate | 1e-3 |
| Head Learning Rate | 5e-4 |
| Confidence Threshold | 0.75 |
| EMA Alpha | 0.99 |
| Augmentation Views | 4 |
| Severity Level | 5 |

---

## Technologies Used

- Python
- PyTorch
- RobustBench
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook

---

## References

1. Wang, D., Shelhamer, E., Liu, S., Olshausen, B., & Darrell, T. (2021). *Tent: Fully Test-Time Adaptation by Entropy Minimization*. ICLR 2021.

2. Hendrycks, D., & Dietterich, T. (2019). *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations*. ICLR 2019.

3. RobustBench: A Standardized Adversarial Robustness Benchmark.

---

## Author

**Rakshita R Talegaon**

Deep Learning Course Project

KLE Technological University

---
