# CMSC 678 Final Project — BCI Motor Imagery Classification

4-class EEG motor imagery classification on BCI Competition IV Dataset 2a. We compare FBCSP, EEGNet, ATCNet, MI-Mamba, and BYOL-pretrained EEGNet with and without S&R augmentation, with a focus on whether different approaches help BCI-inefficient users (low performers).

---

## Files needed

All of these need to be in the same folder as `driver.ipynb` (or on `sys.path`):

| File | What it does |
|------|-------------|
| `utils.py` | Data loading and preprocessing |
| `normal_results.py` | Runs all model training conditions |
| `EEGNet.py` | EEGNet model |
| `ATCNet.py` | ATCNet model |
| `Mamba.py` | MI-Mamba model |
| `FBCSP_Multiclass.py` | FBCSP pipeline |
| `EEGNetBYOL.py` | BYOL pretraining and finetuning |
| `ablation.py` | Leave-one-out BYOL ablation |
| `sr_augmentation.py` | S&R data augmentation |
| `plot.py` | All plotting functions |
| `requirements.txt` | Python dependencies |

---

## Data files

You need 27 files in a single folder (`DATA_DIR`):

- **18 signal files** — `A01T.npz` through `A09T.npz` (training) and `A01E.npz` through `A09E.npz` (eval)
- **9 label files** — `A01E.mat` through `A09E.mat` (true eval labels)

**Shortcut (no download needed):** The data folder is shared at [temporary link]. Open it, click **Add shortcut to My Drive**, then point `DATA_DIR` in cell 2 to wherever you placed the shortcut. You can skip the download cell entirely.

If you don't have access to the shared folder, cell 3 of the notebook (Download data) will download everything automatically from the public sources.

---

## How to run

1. Open `driver.ipynb` in Google Colab
2. In **cell 2**, set `HOME_FOLDER` to your project directory on Drive (where all the `.py` files live) and set `DATA_DIR` to the folder containing the 27 data files
3. Run all cells top to bottom
4. Each experiment saves its results to a pickle file in `HOME_FOLDER/results/` — if you stop and come back, re-running a cell will load from cache instead of retraining

GPU is required for experiments 2–10. A100 recommended, T4 works but is slower.

**Rough runtimes on A100:**
- Experiments 1–7 (FBCSP + supervised models): ~2–3 hours total
- Experiments 8–9 (BYOL pretrain + finetune): ~30–60 min each
- Experiment 10 (ablation, 32 runs): several hours — checkpointed, safe to interrupt and resume

---

## Dependencies

```
torch
numpy
scipy
scikit-learn
matplotlib
mne
mamba-ssm
```

Install with:
```bash
pip install -r requirements.txt
pip install mamba-ssm --no-build-isolation
```
