import os
import shutil
import numpy as np
import torch

from normal_results import (
    FBCSP_results, EEG_results, Mamba_results, ATCNet_results, SSL_results,
    subjectData, subjectDataEVAL
)
from ablation import run_ablation, baseline_ssl_aug
from plot import (
    run_erd_analysis, run_ablation_plot, run_augmentation_visualization,
    plot_all_confusion_matrices
)

DATA_DIR = '/content'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#get accuracy and confusion matrices for each model both augmented and not
fbcsp_acc, fbcsp_cm = FBCSP_results()
eeg_acc, eeg_cm = EEG_results(aug_bool=False)
eeg_aug_acc, eeg_aug_cm = EEG_results(aug_bool=True)
mamba_acc, mamba_cm = Mamba_results(aug_bool=False)
mamba_aug_acc, mamba_aug_cm = Mamba_results(aug_bool=True)
atcnet_acc, atcnet_cm = ATCNet_results(aug_bool=False)
atcnet_aug_acc, atcnet_aug_cm = ATCNet_results(aug_bool=True)
ssl_acc, ssl_cm = SSL_results(aug_bool=False)
ssl_aug_acc, ssl_aug_cm = SSL_results(aug_bool=True)

results = {
    'FBCSP': fbcsp_acc,
    'EEGNet': eeg_acc,
    'EEGNet+aug': eeg_aug_acc,
    'Mamba': mamba_acc,
    'Mamba+aug': mamba_aug_acc,
    'ATCNet': atcnet_acc,
    'ATCNet+aug': atcnet_aug_acc,
    'SSL-BYOL': ssl_acc,
    'SSL-BYOL+aug': ssl_aug_acc,
}

with open('/content/ALL_MODEL_RESULTS.txt', 'w') as f:
    for name, accs in results.items():
        per_subj = '  '.join(f'S{i+1:02d}:{a:.4f}' for i, a in enumerate(accs))
        line = f'{name:<14} Mean:{np.mean(accs):.4f}  [{per_subj}]\n'
        f.write(line)
        print(line, end='')

#run ablation study for all the low performers
ablation_results = run_ablation(subjectData, subjectDataEVAL, DATA_DIR)

#create confusion matrices for each model
plot_all_confusion_matrices(
    {'FBCSP': fbcsp_cm, 'EEGNet': eeg_cm, 'Mamba': mamba_cm, 'ATCNet': atcnet_cm},
    aug_bool=False
)
plot_all_confusion_matrices(
    {'EEGNet': eeg_aug_cm, 'Mamba': mamba_aug_cm, 'ATCNet': atcnet_aug_cm},
    aug_bool=True
)

#create ablation and augmentation plots
run_erd_analysis(subjectData)
run_augmentation_visualization(subjectData, subjectDataEVAL, DATA_DIR)
run_ablation_plot(ablation_results, baseline_ssl_aug)

#save results
DRIVE_RESULTS = '/content/drive/MyDrive/BCI/results'
os.makedirs(DRIVE_RESULTS, exist_ok=True)

for fname in [
    'ALL_MODEL_RESULTS.txt',
    'ABLATION_RESULTS.txt',
    'ERD_RESULTS.txt',
    'confusion_matrices.png',
    'confusion_matrices_aug.png',
    'erd_comparison.png',
    'erd_topomap.png',
    'byol_ablation.png',
]:
    src = f'/content/{fname}'
    if os.path.exists(src):
        shutil.copy(src, DRIVE_RESULTS)

aug_src = '/content/signal_dif_sr_augmentation'
if os.path.exists(aug_src):
    shutil.copytree(aug_src, os.path.join(DRIVE_RESULTS, 'signal_dif_sr_augmentation'),
                    dirs_exist_ok=True)

print(f'All results copied to {DRIVE_RESULTS}')
