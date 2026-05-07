import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.signal import welch
import mne
from sr_augmentation import augment_trials_SR
from utils import subject_to_arrays

CLASS_NAMES = ['Left', 'Right', 'Feet', 'Tongue']
high_subjects = [1, 3, 7, 8, 9]
low_subjects  = [2, 4, 5, 6]

#creates the 4x4 confusion matrix per model
def plot_confusion_matrix(cm, title, save_path=None):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)

    thresh = cm.max() / 2
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

#puts the single confusion matrices together side by side
def plot_all_confusion_matrices(cms_dict, aug_bool=False):
    n = len(cms_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    suffix = ' (+aug)' if aug_bool else ''
    fig.suptitle(f'Aggregate Confusion Matrices{suffix}', fontsize=13)

    for ax, (name, cm) in zip(axes, cms_dict.items()):
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(name)

        thresh = cm.max() / 2
        for i in range(4):
            for j in range(4):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    suffix_str = '_aug' if aug_bool else ''
    plt.savefig(f'/content/confusion_matrices{suffix_str}.png', dpi=150, bbox_inches='tight')
    plt.show()

#visually verify the difference between augmented and non-augmented signals
#specifically using channel 8 (C3/motor cortex) which shows ERD patterns
def plot_augmentation_comparison(X_original, X_augmented, y, subject_id,
                                  n_classes=4, channel=8):

    class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
    fig, axes = plt.subplots(n_classes, 2, figsize=(14, 10))
    fig.suptitle(f'Subject {subject_id:02d} — Original vs S&R Augmented Trial\n'
                 f'Channel C3 (ch {channel})', fontsize=13)

    t = np.linspace(0, 4, 512) #4 seconds at 128 Hz

    for cls in range(n_classes):
        orig_idx = np.where(y[:len(X_original)] == cls)[0][0]
        aug_idx  = np.where(y[len(X_original):len(X_original)*2] == cls)[0][0]
        orig_signal = X_original[orig_idx, channel, :]
        aug_signal  = X_augmented[len(X_original) + aug_idx, channel, :]
        axes[cls, 0].plot(t, orig_signal, color='steelblue', linewidth=0.8)
        axes[cls, 0].set_title(f'{class_names[cls]} — Original')
        axes[cls, 0].set_ylabel('Amplitude (z-scored)')
        axes[cls, 1].plot(t, aug_signal, color='darkorange', linewidth=0.8)
        axes[cls, 1].set_title(f'{class_names[cls]} — S&R Augmented')

        #marking the segment boundary
        for ax in axes[cls]:
            ax.axvline(x=2.0, color='red', linestyle='--',
                      alpha=0.5, label='Segment boundary')
            ax.set_xlim([0, 4])

    for ax in axes[-1]:
        ax.set_xlabel('Time (s)')

    plt.tight_layout()
    """plt.savefig(f'augmentation_comparison_subject{subject_id:02d}.png',
                dpi=150, bbox_inches='tight')"""
    plt.savefig(f'/content/signal_dif_sr_augmentation/augmentation_subject{subject_id:02d}.png',
            dpi=150, bbox_inches='tight')
    plt.show()

###########################################################

#get ERD per channel per class per subject
def compute_erd(npz_data, fs=250, rest_duration=2, imagery_duration=4):
    s = npz_data['s'][:, :22].astype(np.float64)
    epos = npz_data['epos'].flatten().astype(int)
    etyp = npz_data['etyp'].flatten().astype(int)
    rest_samples = int(rest_duration * fs)   
    imagery_samples = int(imagery_duration * fs)
    results = {}
    code_to_class = {769: 'left', 770: 'right', 771: 'feet', 772: 'tongue'}

    for code, cls_name in code_to_class.items():
        cue_positions = epos[etyp == code]
        rest_psds    = []
        imagery_psds = []

        for cue_pos in cue_positions:
            rest_start = cue_pos - rest_samples
            imagery_start = cue_pos
            if rest_start < 0:
                continue

            rest_epoch = s[rest_start:rest_start + rest_samples, :].T 
            imagery_epoch = s[imagery_start:imagery_start + imagery_samples, :].T

            #calculate power in mu and beta bands
            f_rest, p_rest = welch(rest_epoch, fs=fs, nperseg=fs, axis=-1)
            f_img, p_img = welch(imagery_epoch, fs=fs, nperseg=fs, axis=-1)
            rest_psds.append(p_rest)
            imagery_psds.append(p_img)

        rest_mean = np.mean(rest_psds, axis=0) 
        imagery_mean = np.mean(imagery_psds, axis=0)
        results[cls_name] = {
            'freqs': f_rest,
            'rest_psd': rest_mean,
            'imagery_psd': imagery_mean,
            'erd': (imagery_mean - rest_mean) / rest_mean * 100
        }

    return results

#calculate mean ERD% within frequency band
def compute_band_erd(erd_result, band=(8, 12)):
    freqs = erd_result['freqs']
    mask  = (freqs >= band[0]) & (freqs <= band[1])
    return erd_result['erd'][:, mask].mean(axis=1)  # (22,)

###################################################################################################

#calculates erd by class
def get_group_erd_topo(subject_list, subjectData, band=(8, 12)):
    group_erd = []
    for idx in subject_list:
        erd = compute_erd(subjectData[f'subject{idx:02d}'])
        subj_erd = np.mean([compute_band_erd(erd[cls], band)
                            for cls in ['left','right','feet','tongue']], axis=0)
        group_erd.append(subj_erd)
    return np.mean(group_erd, axis=0)  # (22,)

#plots the erd difference between high and low performers
def plot_erd_comparison(high_erd_mu, low_erd_mu, high_erd_beta, low_erd_beta):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('ERD% at C3: High vs Low Performers\n'
                 '(Negative = desynchronization = stronger motor imagery response)',
                 fontsize=12)

    x = np.arange(2)
    width = 0.35

    for ax, high_vals, low_vals, band_name in zip(
        axes,
        [high_erd_mu, high_erd_beta],
        [low_erd_mu,  low_erd_beta],
        ['Mu Band (8–12 Hz)', 'Beta Band (13–30 Hz)']
    ):
        ax.bar(x[0], np.mean(high_vals), width,
               yerr=np.std(high_vals), color='steelblue',
               label='High performers', capsize=5)
        ax.bar(x[1], np.mean(low_vals), width,
               yerr=np.std(low_vals), color='tomato',
               label='Low performers', capsize=5)
        ax.scatter(np.full(len(high_vals), x[0]), high_vals,
                   color='steelblue', zorder=5, s=40, alpha=0.7)
        ax.scatter(np.full(len(low_vals), x[1]), low_vals,
                   color='tomato', zorder=5, s=40, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        ax.set_title(band_name)
        ax.set_ylabel('ERD (%)')
        ax.set_xticks([])
        ax.legend()

    plt.tight_layout()
    plt.savefig('/content/erd_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

#shows the topomap for high and low performers and the difference between them
def plot_topomap_erd(high_erd_topo, low_erd_topo):
    ch_names = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
                'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
                'P1', 'Pz', 'P2', 'POz']

    info = mne.create_info(ch_names=ch_names, sfreq=128, ch_types='eeg')
    info.set_montage('standard_1020')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Mu+Beta Band ERD% Topography (8–30 Hz)\n'
                 'During Motor Imagery (Negative = Desynchronization)',
                 fontsize=13)

    vmin = min(high_erd_topo.min(), low_erd_topo.min())
    vmax = max(high_erd_topo.max(), low_erd_topo.max())
    mne.viz.plot_topomap(high_erd_topo, info, axes=axes[0],
                         show=False, vlim=(vmin, vmax), cmap='RdBu_r')
    axes[0].set_title('High Performers')
    mne.viz.plot_topomap(low_erd_topo, info, axes=axes[1],
                         show=False, vlim=(vmin, vmax), cmap='RdBu_r')
    axes[1].set_title('Low Performers')
    mne.viz.plot_topomap(high_erd_topo - low_erd_topo, info,
                         axes=axes[2], show=False, cmap='RdBu_r')
    axes[2].set_title('Difference (High − Low)')

    plt.tight_layout()
    plt.savefig('/content/erd_topomap.png', dpi=150, bbox_inches='tight')
    plt.show()

#returns quantitative stats for mu and beta for high vs low performers
def run_erd_analysis(subjectData):
    high_erd_mu,  high_erd_beta = [], []
    low_erd_mu,   low_erd_beta  = [], []
    for idx in high_subjects:
        erd = compute_erd(subjectData[f'subject{idx:02d}'])
        high_erd_mu.append(np.mean([compute_band_erd(erd[cls], (8, 12))[7]
                                    for cls in ['left','right','feet','tongue']]))
        high_erd_beta.append(np.mean([compute_band_erd(erd[cls], (13, 30))[7]
                                      for cls in ['left','right','feet','tongue']]))

    for idx in low_subjects:
        erd = compute_erd(subjectData[f'subject{idx:02d}'])
        low_erd_mu.append(np.mean([compute_band_erd(erd[cls], (8, 12))[7]
                                   for cls in ['left','right','feet','tongue']]))
        low_erd_beta.append(np.mean([compute_band_erd(erd[cls], (13, 30))[7]
                                     for cls in ['left','right','feet','tongue']]))

    lines = [
        'ERD% at C3 (channel 7, 0-indexed)\n',
        f'{"":30} {"Mean":>8}  {"Per subject"}\n',
        f'{"Mu (8-12 Hz)  High performers":<30} {np.mean(high_erd_mu):>8.1f}%  '
        f'{" ".join(f"S{i:02d}:{v:.1f}" for i, v in zip(high_subjects, high_erd_mu))}\n',
        f'{"Mu (8-12 Hz)  Low performers":<30} {np.mean(low_erd_mu):>8.1f}%  '
        f'{" ".join(f"S{i:02d}:{v:.1f}" for i, v in zip(low_subjects, low_erd_mu))}\n',
        f'{"Beta (13-30 Hz) High performers":<30} {np.mean(high_erd_beta):>8.1f}%  '
        f'{" ".join(f"S{i:02d}:{v:.1f}" for i, v in zip(high_subjects, high_erd_beta))}\n',
        f'{"Beta (13-30 Hz) Low performers":<30} {np.mean(low_erd_beta):>8.1f}%  '
        f'{" ".join(f"S{i:02d}:{v:.1f}" for i, v in zip(low_subjects, low_erd_beta))}\n',
    ]
    for line in lines:
        print(line, end='')
    with open('/content/ERD_RESULTS.txt', 'w') as f:
        f.writelines(lines)

    plot_erd_comparison(high_erd_mu, low_erd_mu, high_erd_beta, low_erd_beta)
    high_erd_topo = get_group_erd_topo(high_subjects, subjectData, band=(8, 30))
    low_erd_topo = get_group_erd_topo(low_subjects,  subjectData, band=(8, 30))
    plot_topomap_erd(high_erd_topo, low_erd_topo)

#creates visual of the accuracy drop/increase per ablation per target sub
def run_ablation_plot(ablation_results, baseline_ssl_aug):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('BYOL Ablation: Accuracy Drop When Each Subject Removed\n'
                 'Positive drop = that subject contributed to pretraining',
                 fontsize=13)

    for ax, target_idx in zip(axes.flatten(), low_subjects):
        other_subjects = sorted(ablation_results[target_idx].keys())
        drops  = [baseline_ssl_aug[target_idx] - ablation_results[target_idx][s]
                  for s in other_subjects]
        colors = ['steelblue' if s in high_subjects else 'tomato'
                  for s in other_subjects]
        labels = [f'S{s:02d}\n({"H" if s in high_subjects else "L"})'
                  for s in other_subjects]
        bars = ax.bar(labels, drops, color=colors, edgecolor='black', linewidth=0.5)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        ax.set_title(f'Target: Subject {target_idx:02d} — '
                     f'Baseline: {baseline_ssl_aug[target_idx]:.4f}')
        ax.set_ylabel('Accuracy Drop (baseline − ablated)\n'
                      'Positive = contributor, Negative = interference')
        ax.set_xlabel('Excluded Subject')

        for bar, drop in zip(bars, drops):
            ypos = bar.get_height() + 0.002 if drop >= 0 else bar.get_height() - 0.008
            ax.text(bar.get_x() + bar.get_width()/2, ypos,
                    f'{drop:+.3f}', ha='center', va='bottom', fontsize=8)

    legend_elements = [Patch(facecolor='steelblue', label='High performer'),
                       Patch(facecolor='tomato',    label='Low performer')]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=2, fontsize=11, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('/content/byol_ablation.png', dpi=150, bbox_inches='tight')
    plt.show()

#full run for comparing the augmented vs not augmented signals
def run_augmentation_visualization(subjectData, subjectDataEVAL, DATA_DIR):
    os.makedirs('/content/signal_dif_sr_augmentation', exist_ok=True)

    for idx in range(1, 10):
        X_tr, y_tr, _, _ = subject_to_arrays(
            subjectData[f'subject{idx:02d}'],
            subjectDataEVAL[f'subject{idx:02d}'],
            f'{DATA_DIR}/A{idx:02d}E.mat'
        )
        X_aug, y_aug = augment_trials_SR(X_tr, y_tr, n_segments=2, n_augmented=2)
        print(f'Subject {idx:02d} — Original: {X_tr.shape}, Augmented: {X_aug.shape}')
        plot_augmentation_comparison(X_tr, X_aug, y_aug, subject_id=idx, channel=8)