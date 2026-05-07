import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

class MI_Mamba(nn.Module):
    def __init__(self, n_channels=22, n_classes=4, n_times=512, d_model=32):
        super().__init__()
        self.spatial_conv = nn.Conv2d(1, d_model, kernel_size=(n_channels, 1))
        self.bn = nn.BatchNorm2d(d_model)
        self.dropout = nn.Dropout(0.5)
        self.mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x):
        x = F.relu(self.bn(self.spatial_conv(x))) #start with (batch, 32, 1, 512)
        x = x.squeeze(2).permute(0, 2, 1)
        x = self.mamba(x)                
        x = self.dropout(x.mean(dim=1))        
        return self.fc(x) #final return is (batch, 4)