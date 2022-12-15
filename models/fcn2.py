import torch
from torch import nn

class Linear1d(nn.Module):
    def __init__(
        self, input_channels, out_channels, out_dim, bias=False
    ):
        super(Linear1d, self).__init__()
        self.weight = nn.Parameter( # input [bs, cin * space], weight [cout * space // 2, cin * space]
            torch.randn(
                out_channels * out_dim,
                input_channels * out_dim * 2,
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(1, out_channels * out_dim))
        else:
            self.register_parameter("bias", None)

        self.input_channels = input_channels
        self.out_dim = out_dim

    def forward(self, x):
        x = x[:, None] * self.weight # [bs, cout * space // 2, cin * space]
        x = x.sum(dim=-1) # [bs, cout * space // 2]
        x /= self.input_channels ** .5
        if self.bias is not None:
            x += self.bias * 0.1
        return x


class FCN2(nn.Module):
    """
        FCN crafted to have an effective size equal to the corresponding HLCN.
    """
    def __init__(self, input_channels, h, out_dim, num_layers, bias=False):
        super(FCN2, self).__init__()

        d = 2 ** num_layers

        self.hier = nn.Sequential(
            Linear1d(
                input_channels, h, d // 2, bias
            ),
            nn.ReLU(),
            *[nn.Sequential(
                    Linear1d(
                        h, h, d // 2 ** (l + 1), bias
                    ),
                    nn.ReLU(),
                )
                for l in range(1, num_layers)
            ],
        )
        self.beta = nn.Parameter(torch.randn(h * d // 2 ** num_layers, out_dim))

    def forward(self, x):
        y = x.flatten(1) # [bs, cin, space] -> [bs, cin * space]
        y = self.hier(y)
        y = y @ self.beta / self.beta.size(0)
        return y
