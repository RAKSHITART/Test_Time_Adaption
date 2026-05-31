

# PLHead-TTA: Test-Time Adaptation for Robust Image Classification

## Overview

This project explores **Test-Time Adaptation (TTA)** for improving the robustness of deep learning models under distribution shifts. The work reproduces the **TENT (Test-Time Entropy Minimization)** method and proposes two extensions:

* **MATTA** (Multi-View Augmentation with EMA Teacher)
* **PLHead-TTA** (Pseudo-Label Head Adaptation)

Experiments are conducted on the **CIFAR-10-C** benchmark using a pre-trained **WideResNet-28-10** model. The objective is to adapt the model during inference without access to source training data or target labels.

---

## Key Features

✅ Reproduction of TENT on CIFAR-10-C

✅ Five TENT variant experiments

✅ BatchNorm layer ablation studies

✅ Learning rate hyperparameter tuning

✅ MATTA (EMA Teacher + Multi-View Augmentation)

✅ PLHead-TTA (Pseudo-Label Guided Head Adaptation)

✅ Comprehensive evaluation across 15 corruption types

---

## Project Architecture

```text
CIFAR-10-C Images
        │
        ▼
 Preprocessing
        │
        ▼
WideResNet-28-10
        │
        ▼
 BatchNorm Adaptation
        │
        ▼
   Prediction
```

### PLHead-TTA Extension

```text
BN Adapted Predictions
           │
           ▼
Pseudo Label Generation
           │
           ▼
Confidence Filtering
           │
           ▼
Classifier Head Update
           │
           ▼
Final Prediction
```

---

## Dataset

### CIFAR-10-C

The dataset contains 15 corruption types grouped into four categories:

* Noise

  * Gaussian Noise
  * Shot Noise
  * Impulse Noise

* Blur

  * Defocus Blur
  * Glass Blur
  * Motion Blur
  * Zoom Blur

* Weather

  * Snow
  * Frost
  * Fog
  * Brightness

* Digital

  * Contrast
  * Elastic Transform
  * Pixelate
  * JPEG Compression

Severity Level Used:

```text
Severity = 5
```

---

## Methods Implemented

### Phase 1: Baseline

* Source Model (No Adaptation)
* TENT Baseline

### Phase 2: Reproduction Experiments

* Continual TENT
* Episodic TENT
* Shift-Only Adaptation
* Confidence-Filtered Adaptation
* Squared Entropy Loss

### Phase 3: Improvements

* Deep BN Layer Adaptation
* Shallow BN Layer Adaptation
* Learning Rate Sweep
* MATTA

### Phase 4: Proposed Method

* PLHead-TTA

---

## Results

| Method              | Mean Error (%) |
| ------------------- | -------------: |
| Source              |          43.47 |
| TENT                |          20.25 |
| Continual TENT      |          19.68 |
| Episodic TENT       |          20.25 |
| Shift-Only          |          20.38 |
| Confidence Filtered |          20.16 |
| Squared Entropy     |          20.41 |
| PLHead-TTA          |          20.27 |

### Accuracy Comparison

| Method     | Mean Accuracy (%) |
| ---------- | ----------------: |
| Source     |             56.53 |
| TENT       |             79.75 |
| PLHead-TTA |             83.60 |

---

## Folder Structure

```text
PLHead-TTA/
│
├── notebooks/
│   ├── phase1_baseline_setup.ipynb
│   ├── phase2_reproduction_experiments.ipynb
│   ├── phase3_model_improvement.ipynb
│   └── phase4_enhancement.ipynb
│
│
├── report/
│   └── Deep_Learning_Project_Report.pdf
│
├── requirements.txt
│
├── README.md
│
└── LICENSE
```

---

## Requirements

Create a file named:

```text
requirements.txt
```

Contents:

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

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Reproducing Results

### Step 1: Clone Repository

```bash
git clone https://github.com/your-username/PLHead-TTA.git

cd PLHead-TTA
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Launch Jupyter

```bash
jupyter notebook
```

### Step 4: Run Experiments

#### Baseline

```text
notebooks/phase1_baseline_setup.ipynb
```

#### Reproduction Experiments

```text
notebooks/phase2_reproduction_experiments.ipynb
```

#### Ablation Studies & MATTA

```text
notebooks/phase3_model_improvement.ipynb
```

#### PLHead-TTA

```text
notebooks/phase4_enhancement.ipynb
```

Run notebooks sequentially to reproduce all reported results.

---

## Experimental Setup

| Parameter            | Value            |
| -------------------- | ---------------- |
| Backbone             | WideResNet-28-10 |
| Dataset              | CIFAR-10-C       |
| Batch Size           | 64               |
| Optimizer            | Adam             |
| BN Learning Rate     | 1e-3             |
| Head Learning Rate   | 5e-4             |
| Confidence Threshold | 0.75             |
| EMA Alpha            | 0.99             |
| Augmentation Views   | 4                |
| Severity Level       | 5                |

---

## Technologies Used

* Python
* PyTorch
* RobustBench
* NumPy
* Pandas
* Matplotlib
* Jupyter Notebook

---

## References

1. Wang et al., **Tent: Fully Test-Time Adaptation by Entropy Minimization**, ICLR 2021.
2. Hendrycks & Dietterich, **Benchmarking Neural Network Robustness to Common Corruptions and Perturbations**, ICLR 2019.
3. RobustBench Benchmark Framework.

---

## Author

**Rakshita R Talegaon**

Deep Learning Course Project

KLE Technological University

---


