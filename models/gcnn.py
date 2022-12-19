import torch
from torch import nn


def tensor_roll(x, shifts=None):
    d = x.shape[-1]
    device = x.device
    if shifts is None:
        shifts = -torch.arange(0, d, 2, device=device)
    a = torch.arange(d, device=device)[None].repeat(x.shape[-2], 1)
    index = (a + shifts[:, None]) % d
    return torch.gather(x, -1, index.expand_as(x))

def global_unfold(x):
    d = x.shape[-1]
    x = x[..., None, :]
    x = x.expand(*x.shape[:-2], d // 2, d)
    return tensor_roll(x)


class GlobalConv1d(nn.Module):
    def __init__(
        self, input_channels, out_channels, out_dim, bias=False
    ):
        super(GlobalConv1d, self).__init__()
        self.weight = nn.Parameter( # input [bs, cin, space], weight [cout, cin, 1, d]
            torch.randn(
                out_channels,
                input_channels,
                1,
                out_dim * 2,
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(1, out_channels, out_dim))
        else:
            self.register_parameter("bias", None)

        self.input_channels = input_channels

    def forward(self, x):

        # [bs, cin, d] -> [bs, 1, cin, d // 2, d]
        x = global_unfold(x)
        x = x[:, None] * self.weight # [bs, cout, cin, d // 2, d]
        x = x.sum(dim=[-1, -3]) # [bs, cout, d // 2]
        x /= self.input_channels ** .5
        if self.bias is not None:
            x += self.bias * 0.1
        return x


class GCNN(nn.Module):
    """
        Global convolutional neural network crafted to have an effective size equal to the corresponding HLCN.
    """
    def __init__(self, input_channels, h, out_dim, num_layers, bias=False):
        super(GCNN, self).__init__()

        d = 2 ** num_layers

        self.hier = nn.Sequential(
            GlobalConv1d(
                input_channels, h, d // 2, bias
            ),
            nn.ReLU(),
            *[nn.Sequential(
                    GlobalConv1d(
                        h, h, d // 2 ** (l + 1), bias
                    ),
                    nn.ReLU(),
                )
                for l in range(1, num_layers)
            ],
        )
        self.beta = nn.Parameter(torch.randn(h, out_dim))

    def forward(self, x):
        y = self.hier(x)
        y = y.mean(dim=[-1])
        y = y @ self.beta / self.beta.size(0)
        return y
