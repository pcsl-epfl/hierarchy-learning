import torch
from torch import nn


class DenseBlock(nn.Module):
    def __init__(self, input_dim, h, last=False, dropout=False, batch_norm=False):
        super().__init__()
        self.w = nn.Parameter(torch.randn(input_dim, h))
        self.b = nn.Parameter(torch.randn(h))
        self.last = last
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(h)
        else:
            self.batch_norm = None

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        input_dim = x.size(1)
        if self.last:
            y = x @ self.w / input_dim
        else:
            y = x @ self.w / input_dim ** 0.5 + self.b
        if self.dropout is not None:
            y = self.dropout(y)
        y = y.squeeze() if self.last else y.relu()
        if self.batch_norm is not None:
            y = self.batch_norm(y)
        return y


class DenseNet(nn.Module):
    def __init__(self, n_layers, input_dim, h, out_dim, dropout=None, batch_norm=None):
        super().__init__()
        self.fcn = nn.Sequential(
            DenseBlock(input_dim, h, dropout=dropout, batch_norm=batch_norm),
            *[DenseBlock(h, h, dropout=dropout, batch_norm=batch_norm) for _ in range(n_layers - 2)],
            DenseBlock(h, out_dim, last=True, dropout=dropout)
        )

    def forward(self, x):
        return self.fcn(x)