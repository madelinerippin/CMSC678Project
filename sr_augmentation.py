import numpy as np


def augment_trials_SR(X_train, y_train, n_segments=2, n_augmented=2):
    """
    Segmentation and Recombination (S&R) augmentation.
    Lotte et al. / Roy et al. standard approach for EEG.

    X_train: numpy (N, 22, 512)
    y_train: numpy (N,)
    n_segments: number of segments to split each trial into (2 = halves)
    n_augmented: number of synthetic copies to generate per original trial

    Returns: (N * (1 + n_augmented), 22, 512) — original + synthetic
    """
    all_X = [X_train]
    all_y = [y_train]

    seg_len = X_train.shape[2] // n_segments  # 512 // 2 = 256 samples per segment

    for _ in range(n_augmented):
        synthetic_X = np.zeros_like(X_train)

        for i in range(len(X_train)):
            label = y_train[i]
            same_class = X_train[y_train == label]

            synthetic_trial = np.zeros_like(X_train[i])
            for seg_idx in range(n_segments):
                start = seg_idx * seg_len
                end = start + seg_len
                donor_idx = np.random.randint(0, len(same_class))
                synthetic_trial[:, start:end] = same_class[donor_idx, :, start:end]

            synthetic_X[i] = synthetic_trial

        all_X.append(synthetic_X)
        all_y.append(y_train)

    return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0)
    # Result: 288 * 3 = 864 trials
