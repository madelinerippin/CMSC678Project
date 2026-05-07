import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sr_augmentation import augment_trials_SR
from utils import subject_to_arrays

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#adds noise, temporal masking, and amplitude scaling for augmentation
def eeg_augment(x, fs=128):
    def augment_single(x):
        aug = x.copy()
        aug = aug + np.random.randn(*aug.shape) * aug.std() * 0.03
        mask_len = np.random.randint(25, 75)  #0.2–0.6s at 128Hz
        mask_start = np.random.randint(0, 512 - mask_len)
        aug[:, mask_start:mask_start + mask_len] = 0
        scale = np.random.uniform(0.8, 1.2)
        aug = aug * scale
        return aug

    return augment_single(x), augment_single(x)

#doing BYOL for EEG which doesn't need any negative pairs
#online network tries to predict target network's representations
class EEGNetBYOL(nn.Module):
    def __init__(self, chans=22, time_points=512, temp_kernel=64,
                 proj_dim=256, pred_dim=128):
        super().__init__()
        from EEGNet import EEGNetModel

        #building the online network
        self.online_backbone = EEGNetModel(chans=chans, classes=4,time_points=time_points, temp_kernel=temp_kernel)
        self.online_backbone.fc = nn.Identity()
        feature_dim = 128

        self.online_projector = nn.Sequential(
            nn.Linear(feature_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        self.online_predictor = nn.Sequential(
            nn.Linear(proj_dim, pred_dim),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(),
            nn.Linear(pred_dim, proj_dim)
        )

        #building the target network (no grads for params)
        self.target_backbone = copy.deepcopy(self.online_backbone)
        self.target_projector = copy.deepcopy(self.online_projector)

        for p in self.target_backbone.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

    def target_forward(self, x):
        with torch.no_grad():
            feat = self.target_backbone(x)
            proj = self.target_projector(feat)
            return F.normalize(proj, dim=1)

    def online_forward(self, x):
        feat = self.online_backbone(x)
        proj = self.online_projector(feat)
        pred = self.online_predictor(proj)
        return F.normalize(pred, dim=1)

    #exponential moving avg update of target network
    @torch.no_grad()
    def update_target(self, momentum=0.996):
        for online_p, target_p in zip(
            list(self.online_backbone.parameters()) +
            list(self.online_projector.parameters()),
            list(self.target_backbone.parameters()) +
            list(self.target_projector.parameters())
        ):
            target_p.data = momentum * target_p.data + (1 - momentum) * online_p.data

#calculating the negative cosine similarity for the loss
def byol_loss(pred, target):
    return -F.cosine_similarity(pred, target, dim=1).mean()

#pretraining on all the subjects except for the target one
def pretrain_byol_loso(target_idx, subjectData, subjectDataEVAL, DATA_DIR,
                        epochs=300, lr=0.001, batch_size=64):
    model = EEGNetBYOL().to(device)
    
    #optimizing the online network
    optimizer = torch.optim.Adam(
        list(model.online_backbone.parameters()) +
        list(model.online_projector.parameters()) +
        list(model.online_predictor.parameters()),
        lr=lr, weight_decay=1e-4
    )

    all_trials = []
    for idx in range(1, 10):
        if idx == target_idx:
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

            #applying augmentation
            v1 = np.stack([eeg_augment(t)[0] for t in batch])
            v2 = np.stack([eeg_augment(t)[1] for t in batch])
            v1 = torch.tensor(v1[:, np.newaxis], dtype=torch.float32).to(device)
            v2 = torch.tensor(v2[:, np.newaxis], dtype=torch.float32).to(device)
            optimizer.zero_grad()

            #use the online to predict the target in both directions 
            loss = (byol_loss(model.online_forward(v1),
                               model.target_forward(v2)) +
                    byol_loss(model.online_forward(v2),
                               model.target_forward(v1))) / 2

            loss.backward()
            optimizer.step()
            model.update_target(momentum=0.996)
            epoch_loss += loss.item()
            n_batches += 1

        if epoch % 50 == 0:
            print(f'Epoch {epoch}: loss={epoch_loss/n_batches}')

    return model

#fine tune online network on labeled target's data
def finetune_byol_subject(byol_model, X_tr_np, y_tr_np, X_eval, y_eval, aug_bool=False,
                           epochs=300, lr=0.001, batch_size=32, return_preds=False):
    if aug_bool:
        X_aug, y_aug = augment_trials_SR(X_tr_np, y_tr_np, n_segments=2, n_augmented=2)
        X_train = torch.tensor(X_aug[:, np.newaxis], dtype=torch.float32)
        y_train = torch.tensor(y_aug, dtype=torch.long)
    else:
        X_train = torch.tensor(X_tr_np[:, np.newaxis], dtype=torch.float32)
        y_train = torch.tensor(y_tr_np, dtype=torch.long)

    classifier = nn.Linear(128, 4).to(device)
    byol_model.online_backbone.fc = classifier
    optimizer = torch.optim.Adam(
        byol_model.online_backbone.parameters(),
        lr=lr, weight_decay=1e-4
    )
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(X_train, y_train),
                        batch_size=batch_size, shuffle=True)

    byol_model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = byol_model.online_backbone(X_batch)
            loss_fn(preds, y_batch).backward()
            optimizer.step()

    byol_model.eval()
    with torch.no_grad():
        preds = byol_model.online_backbone(
            X_eval.to(device)).argmax(dim=1).cpu()
    acc = (preds == y_eval).float().mean().item()
    if return_preds:
        return acc, preds.numpy(), y_eval.numpy()
    return acc