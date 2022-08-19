import torch
from torch import nn
from .fcn import DenseBlock


class LocallyConnected1d(nn.Module):
    def __init__(
        self, input_ch, out_channels, out_dim, kernel_size, stride, bias=False
    ):
        super(LocallyConnected1d, self).__init__()
        self.weight = nn.Parameter(
            torch.randn(
                1,
                out_channels,
                input_ch,
                out_dim,
                kernel_size,
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(1, out_channels, out_dim))
        else:
            self.register_parameter("bias", None)
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        k = self.kernel_size
        s = self.stride
        x = x.unfold(2, k, s)
        x = x.contiguous()
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out


class LocallyHierarchicalNet(nn.Module):
    def __init__(self, input_ch, h, out_dim, filter_size, num_layers, bias=False):
        super(LocallyHierarchicalNet, self).__init__()

        d = filter_size ** num_layers

        self.net = nn.Sequential(
            LocallyConnected1d(
                input_ch, h, d // filter_size, filter_size, filter_size, bias
            ),
            nn.ReLU(),
            *[
                nn.Sequential(
                    LocallyConnected1d(
                        h,
                        h,
                        d // filter_size ** (l + 2),
                        filter_size,
                        filter_size,
                        bias,
                    ),
                    nn.ReLU(),
                )
                for l in range(num_layers - 1)
            ],
            DenseBlock(h, out_dim, last=True)
        )

    def forward(self, x):
        return self.net(x)
