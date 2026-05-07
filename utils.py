import numpy as np
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from scipy.signal import resample_poly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(data_dir=None):
    if data_dir is None:
        data_dir = os.environ.get('BCI_DATA_DIR', '/content/drive/MyDrive/BCI')

    subjectData = {}
    subjectDataEVAL = {}

    if not os.path.exists('/content/drive/MyDrive'):
        from google.colab import drive
        drive.mount('/content/drive')

    #filling the subject data and evaluation data
    for i in range(1,10):
        subject = 'subject{:02d}'.format(i)
        subjectData[subject]     = np.load(os.path.join(data_dir, 'A{:02d}T.npz'.format(i)))
        subjectDataEVAL[subject] = np.load(os.path.join(data_dir, 'A{:02d}E.npz'.format(i)))

    return subjectData, subjectDataEVAL


def make_trials_dict(subject):
    s = subject['s'][:, :22].astype(np.float64)
    epos = subject['epos'].flatten().astype(int)
    etyp = subject['etyp'].flatten().astype(int)
    win = np.arange(0, 4 * 250)
    buckets = {'left': [], 'right': [], 'feet': [], 'tongue': []}
    code_to_name = {769: 'left', 770: 'right', 771: 'feet', 772: 'tongue'}

    #filtering the classes accordingly
    for pos, code in zip(epos, etyp):
        if code in code_to_name:
            buckets[code_to_name[code]].append(s[pos + win, :].T)

    return {k: np.stack(v) for k, v in buckets.items()}

#preprocess one subject's data and return numpy arrays
def subject_to_arrays(npz_train, npz_eval, mat_path):
    train_dict = make_trials_dict(npz_train)
    X_train = np.concatenate([train_dict[cls] for cls in ['left','right','feet','tongue']])
    y_train = np.concatenate([
        np.full(train_dict[cls].shape[0], lbl)
        for cls, lbl in zip(['left','right','feet','tongue'], [0,1,2,3])
    ])

    s = npz_eval['s'][:, :22].astype(np.float64)
    epos = npz_eval['epos'].flatten().astype(int)
    etyp = npz_eval['etyp'].flatten().astype(int)
    win = np.arange(0, 4 * 250)
    X_eval = np.stack([s[pos + win, :].T for pos in epos[etyp == 783]])
    y_eval = loadmat(mat_path)['classlabel'].flatten() - 1
    X_train = resample_poly(X_train, up=64, down=125, axis=2)
    X_eval = resample_poly(X_eval,  up=64, down=125, axis=2)

    for ch in range(X_train.shape[1]):
      scaler = StandardScaler()
      X_train[:, ch, :] = scaler.fit_transform(X_train[:, ch, :])
      X_eval[:, ch, :]  = scaler.transform(X_eval[:, ch, :])

    return X_train, y_train, X_eval, y_eval

#converting arrays to tensors for SSL
def arrays_to_tensors(X_train, y_train, X_eval, y_eval):
    X_train = torch.tensor(X_train[:, np.newaxis], dtype=torch.float32)
    X_eval = torch.tensor(X_eval[:, np.newaxis], dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_eval = torch.tensor(y_eval, dtype=torch.long)
    return X_train, y_train, X_eval, y_eval

#converting subject data into the tensors
def subject_to_tensors(npz_train, npz_eval, mat_path):
    return arrays_to_tensors(*subject_to_arrays(npz_train, npz_eval, mat_path))

#function for every model to train
def train_subject(X_train, y_train, X_eval, y_eval, model, epochs=300, lr=0.001,
                  batch_size=32, return_preds=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss_fn(model(X_batch), y_batch).backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_eval.to(device)).argmax(dim=1).cpu()
    acc = (preds == y_eval).float().mean().item()
    if return_preds:
        return acc, preds.numpy(), y_eval.numpy()
    return acc