import torch
import torch.nn as nn

from hw_as.model.sincconv import SincConv_fast
from hw_as.model.resblock import ResBlock


class RawNet2(nn.Module):
    def __init__(self, sinc_channels, sinc_size, gru_hidden_size, gru_num_layers, embedding_size, **kwargs):
        super().__init__()
        self.sincconv = SincConv_fast(sinc_channels[0][0], sinc_size)

        self.seq = nn.Sequential(
            nn.MaxPool1d(3),
            nn.BatchNorm1d(sinc_channels[0][0]),
            nn.LeakyReLU(0.3),
            ResBlock(sinc_channels[0][0], sinc_channels[0][1], is_first=True),
            ResBlock(sinc_channels[0][0], sinc_channels[0][1]),
            ResBlock(sinc_channels[0][1], sinc_channels[1][0]),
            *[ResBlock(sinc_channels[1][0], sinc_channels[1][1]) for _ in range(3)],
            nn.BatchNorm1d(sinc_channels[1][1]),
            nn.LeakyReLU(0.3),
        )

        self.gru = nn.GRU(sinc_channels[1][1], gru_hidden_size, gru_num_layers, batch_first=True)
        self.linear1 = nn.Linear(gru_hidden_size, embedding_size)
        self.linear2 = nn.Linear(embedding_size, 2)
    
    def forward(self, audio, **kwargs):
        x = audio.unsqueeze(1)
        out = self.sincconv(x)
        out = self.seq(torch.abs(out))
        out, _ = self.gru(out.transpose(1, 2))
        out = out[:, -1, :]
        out = self.linear1(out)
        out = self.linear2(out)
        return {"preds": out}


