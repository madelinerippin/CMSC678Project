# EEG Motor Imagery Classification — CMSC 678 Project

Benchmarking traditional and deep learning approaches for EEG-based motor imagery classification on the BCI Competition IV Dataset 2a. Models are evaluated with and without self-supervised augmentation (BYOL).

## Models

| Model | Type |
|---|---|
| FBCSP | Traditional (Filter Bank CSP) |
| EEGNet | Compact CNN for EEG |
| ATCNet | Attention Temporal Convolutional Network |
| Mamba | State Space Model |
| SSL-BYOL | Self-Supervised Learning (Bootstrap Your Own Latent) |

## Requirements

- GPU access required (tested on Google Colab A100)
- Python 3.x

## Setup

### 1. Get the data

Download the dataset from Google Drive and place it in `/content/`:

[BCI Competition IV Dataset 2a](https://drive.google.com/drive/folders/15tBQjr5Rcl8dZMTU2jONHxQoII2xrbPx?usp=sharing)

### 2. Clone the repo in Google Colab

```python
!git clone <your-repo-url>
%cd CMSC678Project
```

### 3. Install dependencies

```python
!pip install -r requirements.txt
!pip install mamba-ssm --no-build-isolation
```

> `mamba-ssm` must be installed separately due to custom CUDA build requirements.

## Running

```python
!python MAIN_RUN.py
```

This will:
- Train and evaluate all models (with and without augmentation)
- Run the ablation study
- Generate confusion matrices, ERD plots, and augmentation visualizations
- Save all results to `/content/drive/MyDrive/BCI/results/`

## Output Files

| File | Description |
|---|---|
| `ALL_MODEL_RESULTS.txt` | Per-subject and mean accuracy for each model |
| `ABLATION_RESULTS.txt` | Ablation study results |
| `ERD_RESULTS.txt` | Event-related desynchronization analysis |
| `confusion_matrices.png` | Confusion matrices (no augmentation) |
| `confusion_matrices_aug.png` | Confusion matrices (with augmentation) |
| `erd_comparison.png` | ERD comparison plot |
| `byol_ablation.png` | BYOL ablation study plot |
| `signal_dif_sr_augmentation/` | Augmentation visualization samples |
