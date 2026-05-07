from braindecode.models import ATCNet
import torch.nn as nn

class ATCNetWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ATCNet(n_chans=22, n_outputs=4, n_times=512, sfreq=128)

    def forward(self, x):
        x = x.squeeze(1)  #convert (batch, 1, 22, 512) to (batch, 22, 512)
        return self.model(x)