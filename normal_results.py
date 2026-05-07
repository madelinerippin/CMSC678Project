import os
import numpy as np
import torch
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from FBCSP_Multiclass import FBCSP_Multiclass
from EEGNet import EEGNetModel
from Mamba import MI_Mamba
from ATCNet import ATCNetWrapper
from sr_augmentation import augment_trials_SR
from EEGNetBYOL import pretrain_byol_loso, finetune_byol_subject
from utils import load_data, make_trials_dict, subject_to_tensors, train_subject, subject_to_arrays, arrays_to_tensors

DATA_DIR = os.environ.get('BCI_DATA_DIR', '/content/drive/MyDrive/BCI')
subjectData, subjectDataEVAL = load_data(data_dir=DATA_DIR)

def get_aug():
    X_train_augs = []
    y_train_augs = []
    X_evals = []
    y_evals = []
    for idx in range(1, 10):
        X_tr, y_tr, X_ev, y_ev = subject_to_arrays(
            subjectData[f'subject{idx:02d}'],
            subjectDataEVAL[f'subject{idx:02d}'],
            f'{DATA_DIR}/A{idx:02d}E.mat'
        )
        X_tr_aug, y_tr_aug = augment_trials_SR(X_tr, y_tr)
        X_train_aug, y_train_aug, X_eval, y_eval = arrays_to_tensors(X_tr_aug, y_tr_aug, X_ev, y_ev)
        X_train_augs.append(X_train_aug)
        y_train_augs.append(y_train_aug)
        X_evals.append(X_eval)
        y_evals.append(y_eval)

    return X_train_augs, y_train_augs, X_evals, y_evals

def FBCSP_results():
    accuracies = []
    all_preds = []
    all_true = []

    for idx in range(1, 10):
        sub = f'subject{idx:02d}'
        train_dict = make_trials_dict(subjectData[sub])

        #get subject data
        s = subjectDataEVAL[sub]['s'][:, :22].astype(np.float64)
        epos = subjectDataEVAL[sub]['epos'].flatten().astype(int)
        etyp = subjectDataEVAL[sub]['etyp'].flatten().astype(int)
        win = np.arange(0, 4 * 250)

        #perform FBCSP and get prediction
        cue_pos = epos[etyp == 783]
        X_test = np.stack([s[pos + win, :].T for pos in cue_pos])
        mat = loadmat(f'{DATA_DIR}/A{idx:02d}E.mat')
        y_test = mat['classlabel'].flatten()
        clf = FBCSP_Multiclass(train_dict, 250, print_var=False)
        y_pred = clf.evaluateTrial(X_test)

        #compute accuracy and change classes from 1-4 to 0-3 like others
        acc = np.mean(y_pred == y_test)
        accuracies.append(acc)
        all_preds.extend(y_pred - 1)
        all_true.extend(y_test - 1)

    aggregate_cm = sk_confusion_matrix(all_true, all_preds, labels=[0, 1, 2, 3])
    return accuracies, aggregate_cm


##############################################################################################

def EEG_results(aug_bool):
    accuracies = []
    all_preds = []
    all_true = []

    #if augmentation is true use augmented data
    if aug_bool:
        X_train_augs, y_train_augs, X_evals, y_evals = get_aug()

    for idx in range(1, 10):
        model = EEGNetModel(chans=22, classes=4, time_points=512, temp_kernel=64).to(device)
        if aug_bool:
            acc, y_pred, y_true = train_subject(
                X_train_augs[idx-1], y_train_augs[idx-1],
                X_evals[idx-1], y_evals[idx-1], model, return_preds=True)
        else:
            X_train, y_train, X_eval, y_eval = subject_to_tensors(
                subjectData[f'subject{idx:02d}'],
                subjectDataEVAL[f'subject{idx:02d}'],
                f'{DATA_DIR}/A{idx:02d}E.mat'
            )
            acc, y_pred, y_true = train_subject(X_train, y_train, X_eval, y_eval, model, return_preds=True)
        accuracies.append(acc)
        all_preds.extend(y_pred)
        all_true.extend(y_true)

    aggregate_cm = sk_confusion_matrix(all_true, all_preds, labels=[0, 1, 2, 3])
    return accuracies, aggregate_cm

##############################################################################################

def Mamba_results(aug_bool):
    accuracies = []
    all_preds = []
    all_true = []

    #if augmentation is true use augmented data
    if aug_bool:
        X_train_augs, y_train_augs, X_evals, y_evals = get_aug()

    for idx in range(1, 10):
        model = MI_Mamba().to(device)
        if aug_bool:
            acc, y_pred, y_true = train_subject(
                X_train_augs[idx-1], y_train_augs[idx-1],
                X_evals[idx-1], y_evals[idx-1], model, epochs=500, return_preds=True)
        else:
            X_train, y_train, X_eval, y_eval = subject_to_tensors(
                subjectData[f'subject{idx:02d}'],
                subjectDataEVAL[f'subject{idx:02d}'],
                f'{DATA_DIR}/A{idx:02d}E.mat'
            )
            acc, y_pred, y_true = train_subject(X_train, y_train, X_eval, y_eval, model, epochs=500, return_preds=True)
        accuracies.append(acc)
        all_preds.extend(y_pred)
        all_true.extend(y_true)

    aggregate_cm = sk_confusion_matrix(all_true, all_preds, labels=[0, 1, 2, 3])
    return accuracies, aggregate_cm

##############################################################################################

def ATCNet_results(aug_bool):
    accuracies = []
    all_preds = []
    all_true = []

    #if augmentation is true use augmented data
    if aug_bool:
        X_train_augs, y_train_augs, X_evals, y_evals = get_aug()

    for idx in range(1, 10):
        model = ATCNetWrapper().to(device)
        if aug_bool:
            acc, y_pred, y_true = train_subject(
                X_train_augs[idx-1], y_train_augs[idx-1],
                X_evals[idx-1], y_evals[idx-1], model, return_preds=True)
        else:
            X_train, y_train, X_eval, y_eval = subject_to_tensors(
                subjectData[f'subject{idx:02d}'],
                subjectDataEVAL[f'subject{idx:02d}'],
                f'{DATA_DIR}/A{idx:02d}E.mat'
            )
            acc, y_pred, y_true = train_subject(X_train, y_train, X_eval, y_eval, model, return_preds=True)
        accuracies.append(acc)
        all_preds.extend(y_pred)
        all_true.extend(y_true)

    aggregate_cm = sk_confusion_matrix(all_true, all_preds, labels=[0, 1, 2, 3])
    return accuracies, aggregate_cm

##############################################################################################

def SSL_results(aug_bool):
    ssl_results = []
    all_preds = []
    all_true = []

    for idx in range(1, 10):
        byol_model = pretrain_byol_loso(idx, subjectData, subjectDataEVAL, DATA_DIR, epochs=300, batch_size=64)

        X_tr, y_tr, X_ev, y_ev = subject_to_arrays(
            subjectData[f'subject{idx:02d}'],
            subjectDataEVAL[f'subject{idx:02d}'],
            f'{DATA_DIR}/A{idx:02d}E.mat'
        )
        _, _, X_eval, y_eval = arrays_to_tensors(X_tr, y_tr, X_ev, y_ev)

        #get metrics after finetuning on target subject
        acc, y_pred, y_true = finetune_byol_subject(
            byol_model, X_tr, y_tr, X_eval, y_eval, aug_bool, return_preds=True)
        ssl_results.append(acc)
        all_preds.extend(y_pred)
        all_true.extend(y_true)

    aggregate_cm = sk_confusion_matrix(all_true, all_preds, labels=[0, 1, 2, 3])
    return ssl_results, aggregate_cm
