import pickle
import os
import copy
import numpy as np
import torch
from EEGNetBYOL import EEGNetBYOL, byol_loss, eeg_augment, finetune_byol_subject
from utils import subject_to_arrays, arrays_to_tensors

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pretrain_byol_loso_ablation(target_idx, exclude_idx, subjectData, subjectDataEVAL, DATA_DIR,
                                  epochs=300, batch_size=64, lr=0.001):
    """
    Pretrain BYOL excluding both target_idx and one ablation subject (exclude_idx).
    target_idx: subject we're evaluating on
    exclude_idx: subject we're removing from pretraining to test their contribution
    """
    model = EEGNetBYOL().to(device)
    optimizer = torch.optim.Adam(
        list(model.online_backbone.parameters()) +
        list(model.online_projector.parameters()) +
        list(model.online_predictor.parameters()),
        lr=lr, weight_decay=1e-4
    )

    all_trials = []
    for idx in range(1, 10):
        if idx == target_idx or idx == exclude_idx:
            continue
        X_tr, _, _, _ = subject_to_arrays(
            subjectData[f'subject{idx:02d}'],
            subjectDataEVAL[f'subject{idx:02d}'],
            f'{DATA_DIR}/A{idx:02d}E.mat'
        )
        all_trials.append(X_tr)

    all_trials = np.concatenate(all_trials, axis=0)

    for epoch in range(epochs):
        idx_perm = np.random.permutation(len(all_trials))
        epoch_loss = 0
        n_batches = 0

        for start in range(0, len(all_trials), batch_size):
            batch = all_trials[idx_perm[start:start + batch_size]]
            v1 = np.stack([eeg_augment(t)[0] for t in batch])
            v2 = np.stack([eeg_augment(t)[1] for t in batch])
            v1 = torch.tensor(v1[:, np.newaxis],
                               dtype=torch.float32).to(device)
            v2 = torch.tensor(v2[:, np.newaxis],
                               dtype=torch.float32).to(device)

            optimizer.zero_grad()
            loss = (byol_loss(model.online_forward(v1),
                               model.target_forward(v2)) +
                    byol_loss(model.online_forward(v2),
                               model.target_forward(v1))) / 2
            loss.backward()
            optimizer.step()
            model.update_target(momentum=0.996)
            epoch_loss += loss.item()
            n_batches += 1

    return model

###############################################################

low_subjects  = [2, 4, 5, 6]
all_subjects  = list(range(1, 10))
high_subjects = {1, 3, 7, 8, 9}

baseline_ssl_aug = {2: 0.5868, 4: 0.6076, 5: 0.6806, 6: 0.5139}
fbcsp            = {2: 0.542,  4: 0.587,  5: 0.507,  6: 0.424}


def run_ablation(subjectData, subjectDataEVAL, DATA_DIR,
                 baselines=None, checkpoint_dir=None, results_path=None):
    if baselines is None:
        baselines = baseline_ssl_aug
    if checkpoint_dir is None:
        checkpoint_dir = os.environ.get('ABLATION_CHECKPOINT_DIR',
                                        '/content/drive/MyDrive/BCI/ablation_checkpoints')
    if results_path is None:
        results_path = os.environ.get('ABLATION_RESULTS_PATH',
                                      '/content/ABLATION_RESULTS.txt')
    os.makedirs(checkpoint_dir, exist_ok=True)

    ablation_results = {t: {} for t in low_subjects}

    print('Checking for existing checkpoints...')
    for target_idx in low_subjects:
        checkpoint_path = os.path.join(checkpoint_dir, f'ablation_subject{target_idx:02d}.pkl')
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'rb') as f:
                ablation_results[target_idx] = pickle.load(f)
            n_done  = len(ablation_results[target_idx])
            n_total = len([s for s in all_subjects if s != target_idx])
            print(f'  Subject {target_idx:02d}: {n_done}/{n_total} runs loaded from checkpoint')
        else:
            print(f'  Subject {target_idx:02d}: no checkpoint found, starting fresh')

    for target_idx in low_subjects:
        other_subjects = [s for s in all_subjects if s != target_idx]
        n_total = len(other_subjects)
        n_done  = len(ablation_results[target_idx])

        if n_done == n_total:
            print(f'\nSubject {target_idx:02d} already complete — skipping')
            continue

        print(f'\n{"="*60}')
        print(f'Target subject: {target_idx:02d} '
              f'(baseline SSL+aug: {baselines[target_idx]:.4f})')
        print(f'{n_done}/{n_total} runs already completed')
        print(f'{"="*60}')

        X_tr, y_tr, X_ev, y_ev = subject_to_arrays(
            subjectData[f'subject{target_idx:02d}'],
            subjectDataEVAL[f'subject{target_idx:02d}'],
            f'{DATA_DIR}/A{target_idx:02d}E.mat'
        )
        _, _, X_eval, y_eval = arrays_to_tensors(X_tr, y_tr, X_ev, y_ev)

        for exclude_idx in other_subjects:
            if exclude_idx in ablation_results[target_idx]:
                acc  = ablation_results[target_idx][exclude_idx]
                drop = baselines[target_idx] - acc
                contributor = 'High' if exclude_idx in high_subjects else 'Low'
                print(f'  Exclude {exclude_idx:02d} ({contributor}): '
                      f'acc={acc:.4f}  drop={drop:+.4f}  [cached]')
                continue

            byol_model = pretrain_byol_loso_ablation(
                target_idx, exclude_idx, subjectData, subjectDataEVAL, DATA_DIR,
                epochs=150, batch_size=64
            )

            model_copy = copy.deepcopy(byol_model)
            acc = finetune_byol_subject(
                model_copy, X_tr, y_tr, X_eval, y_eval, aug_bool=True
            )

            ablation_results[target_idx][exclude_idx] = acc
            drop = baselines[target_idx] - acc
            contributor = 'High' if exclude_idx in high_subjects else 'Low'
            print(f'  Exclude {exclude_idx:02d} ({contributor}): '
                  f'acc={acc:.4f}  drop={drop:+.4f}')

            checkpoint_path = os.path.join(checkpoint_dir, f'ablation_subject{target_idx:02d}.pkl')
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(ablation_results[target_idx], f)
            print(f'    Checkpoint saved ({len(ablation_results[target_idx])}/{n_total} runs complete)')

    header = (
        f'\n{"="*60}\n'
        f'ABLATION COMPLETE — Summary\n'
        f'{"="*60}\n'
        f'\n{"Target":<10} {"Excluded":<10} {"Type":<8} {"Acc":<8} {"Drop":>8}\n'
        f'{"-" * 48}\n'
    )
    rows = []
    for target_idx in low_subjects:
        for exclude_idx in sorted(ablation_results[target_idx].keys()):
            acc  = ablation_results[target_idx][exclude_idx]
            drop = baselines[target_idx] - acc
            t    = 'High' if exclude_idx in high_subjects else 'Low'
            rows.append(f'{target_idx:<10} {exclude_idx:<10} {t:<8} {acc:<8.4f} {drop:>+8.4f}\n')
        rows.append('\n')

    print(header, end='')
    for row in rows:
        print(row, end='')

    with open(results_path, 'w') as f:
        f.write(header)
        f.writelines(rows)

    return ablation_results