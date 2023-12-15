import torch 
import torch.nn as nn
import torch.nn.functional as F


class FMS(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_channels, in_channels)
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        out = F.adaptive_avg_pool1d(x, 1).reshape(x.shape[0], -1)
        out = self.activation(self.linear(out)).reshape(x.shape[0], x.shape[1], 1)
        return out * x + out
        

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_first=False, **kwargs):
        super().__init__()

        pre_layers = []
        if not is_first:
            pre_layers.append(nn.BatchNorm1d(in_channels))
            pre_layers.append(nn.ReLU())
        pre_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, padding="same"))
        pre_layers.append(nn.BatchNorm1d(out_channels))
        pre_layers.append(nn.LeakyReLU(0.3))
        pre_layers.append(nn.Conv1d(out_channels, out_channels, kernel_size=3, padding="same"))
        self.pre = nn.Sequential(*pre_layers)

        self.chan_resizer = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

        self.post = nn.Sequential(
            nn.MaxPool1d(3),
            FMS(out_channels)
        )
    
    def forward(self, x):
        out = self.pre(x)
        if self.chan_resizer:
            x = self.chan_resizer(x)
        out = out + x
        out = self.post(out)
        return out


