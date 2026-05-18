from braindecode.models import ATCNet
import torch.nn as nn

class ATCNetWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ATCNet(n_chans=22, n_outputs=4, n_times=512, sfreq=128)

    def forward(self, x):
        # x arrives as (batch, 1, 22, 512) from subject_to_tensors
        x = x.squeeze(1)  # (batch, 22, 512)
        return self.model(x)