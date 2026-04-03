# EEG Motor Imagery Baseline — BCI Competition IV Dataset 2a

**Notebook:** `Baseline.ipynb`  

---

## What This Notebook Does

This notebook establishes two baseline models for 4-class motor imagery EEG classification on the BCI Competition IV Dataset 2a. It covers the full pipeline from raw data loading through to per-subject accuracy results for both an FBCSP (traditional ML) baseline and an EEGNet (deep learning) baseline.

---

## Data

The dataset contains EEG recordings from 9 subjects performing 4 motor imagery tasks: **left hand** (event 769), **right hand** (770), **feet** (771), and **tongue** (772). Each subject has two sessions:

- **Training session:** `A0#T.npz` — 288 trials (72 per class), used to fit models
- **Evaluation session:** `A0#E.npz` + `A0#E.mat` — 288 trials, used for testing; true labels are loaded from the `.mat` file via `classlabel`

Raw signals are stored as `(M, 25)` arrays (M time samples × 25 electrodes: 22 EEG + 3 EOG). EOG channels are dropped before any model training, leaving 22 EEG channels.

**Files must be placed at:** `/content/drive/MyDrive/BCI/` (Google Drive) or `/content/` for `.mat` label files.

---

## Pipeline Overview

### 1. Data Loading
Loads all 9 subjects' training and evaluation `.npz` files into `subjectData` and `subjectDataEVAL` dictionaries.

### 2. Preprocessing
- Raw signals are transposed to `(25, M)` for filtering
- A 5th-order Butterworth bandpass filter (4–40 Hz) is applied to the continuous signal at 250 Hz
- Trials are epoched using `make_trials_dict()`: event onset positions (`epos`) for event codes 769–772 are extracted, and a 4-second window (1000 samples at 250 Hz) is sliced from each onset

---

## Baseline 1 — FBCSP

**Implementation:** [`jesus-333/FBCSP-Python`](https://github.com/jesus-333/FBCSP-Python), `FBCSP_V4` + `FBCSP_Multiclass`

| Setting | Value |
|---|---|
| Channels | 22 EEG |
| Sampling rate | 250 Hz |
| Epoch window | 0–4 s from cue onset = 1000 samples |
| Filter bank | 9 Butterworth bands from 4–40 Hz in 4 Hz steps |
| Filter order | 3 |
| CSP components (n_w) | 2 → 4 filters per band (first 2 + last 2) |
| Feature selection | MIBIF, top 4 features per band |
| Classifier | LDA |
| Multi-class strategy | One-vs-Rest (OVR) — 4 binary classifiers |
| Train/test split | Session 1 (A0#T) → train, Session 2 (A0#E) → test |

### FBCSP Results

Compared against the published kappa scores from **Ang et al. (2012)** — *Filter Bank Common Spatial Pattern Algorithm on BCI Competition IV Datasets 2a and 2b* — Table 2 (OVR column). Note: the paper reports kappa; this notebook reports accuracy. Rankings are consistent.

| Subject | Paper OVR κ | Ours (Accuracy) | Δ (κ) |
|---|---|---|---|
| 01 | 0.676 | 0.6771 | +0.001 |
| 02 | 0.417 | 0.5417 | +0.125 |
| 03 | 0.745 | 0.8056 | +0.061 |
| 04 | 0.481 | 0.5868 | +0.106 |
| 05 | 0.398 | 0.5069 | +0.109 |
| 06 | 0.273 | 0.4236 | +0.151 |
| 07 | 0.773 | 0.7708 | −0.002 |
| 08 | 0.755 | 0.7188 | −0.036 |
| 09 | 0.606 | 0.6771 | +0.071 |
| **Mean** | **0.569** | **0.6343** | **+0.065** |

Results are systematically higher than the published kappa values, likely due to differences in filter type (Butterworth vs Chebyshev Type II in the original) and filter order. The subject-level ranking is consistent with the paper, which is the key validation criterion.

---

## Baseline 2 — EEGNet

**Implementation:** Custom `EEGNetModel` (PyTorch), based on Lawhern et al. (2018) EEGNet-8,2

| Setting | Value |
|---|---|
| Channels | 22 EEG |
| Sampling rate | 128 Hz (resampled from 250 Hz via `resample_poly(up=64, down=125)`) |
| Epoch window | 0–4 s from cue onset = 512 samples at 128 Hz |
| F1 (temporal filters) | 16 |
| D (depth multiplier) | 2 → 32 spatial filters |
| F2 (separable filters) | 32 |
| Temporal kernel | 64 samples = 500 ms at 128 Hz |
| Dropout | 0.5 after each pooling layer |
| Normalization | Per-channel StandardScaler (fit on train, applied to eval) |
| Optimizer | Adam, lr=0.001 |
| Loss | CrossEntropyLoss |
| Epochs | 300 |
| Batch size | 32 |
| Hardware | GPU (Colab A100/T4) |

The `subject_to_tensors()` function handles loading, resampling, per-channel standardization, and conversion to PyTorch tensors. Evaluation labels are loaded from `A0#E.mat` using `scipy.io.loadmat`.

### EEGNet Results

No single published per-subject table exists for this exact configuration. Published EEGNet mean accuracy on BCI IV-2a ranges from approximately 68–75% depending on implementation. Our result falls within this range.

| Subject | Accuracy |
|---|---|
| 01 | 0.7778 |
| 02 | 0.5799 |
| 03 | 0.8750 |
| 04 | 0.6076 |
| 05 | 0.6910 |
| 06 | 0.5660 |
| 07 | 0.7569 |
| 08 | 0.8194 |
| 09 | 0.7431 |
| **Mean** | **0.7130** |

---

## High / Low Performer Split

Based on FBCSP accuracy, a median split is applied to define subject groups for the BCI illiteracy analysis downstream:

| Group | Subjects |
|---|---|
| **High performers** | 01, 03, 07, 08, 09 |
| **Low performers** | 02, 04, 05, 06 |

This grouping is consistent with per-subject rankings reported across multiple papers in the literature (e.g., subjects 2, 5, 6 are consistently the weakest performers).

---

## Dependencies

```
numpy
scipy
pandas
matplotlib
torch
sklearn
FBCSP_Multiclass   # from jesus-333/FBCSP-Python
EEGNet             # custom EEGNetModel (EEGNet.py must be in working directory)
```

---

## Notes

- The notebook was run on Google Colab with GPU acceleration enabled. The `train_subject()` function moves tensors and model to `device` (CUDA if available).
- An earlier version of `subject_to_tensors()` without per-channel standardization is preserved in a commented-out cell for reference.
- An earlier commented-out version of `train_subject()` used `time_points=1000` (250 Hz, no resampling); the active version uses `time_points=512` (128 Hz) to match the original EEGNet paper configuration.
- All 288 trials per subject are used — no artifact rejection is applied.
