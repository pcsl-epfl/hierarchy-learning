import torch
from torch import nn

class Linear1d(nn.Module):
    def __init__(
        self, input_channels, out_channels, bias=False
    ):
        super(Linear1d, self).__init__()
        self.weight = nn.Parameter( # input [bs, cin], weight [cout, cin]
            torch.randn(
                out_channels,
                input_channels,
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(1, out_channels))
        else:
            self.register_parameter("bias", None)

        self.input_channels = input_channels

    def forward(self, x):
        x = x[:, None] * self.weight # [bs, cout, cin]
        x = x.sum(dim=-1) # [bs, cout]
        x /= self.input_channels ** .5
        if self.bias is not None:
            x += self.bias * 0.1
        return x


class FCN(nn.Module):
    def __init__(self, input_channels, h, out_dim, num_layers, bias=False):
        super(FCN, self).__init__()

        self.hier = nn.Sequential(
            Linear1d(
                input_channels, h, bias
            ),
            nn.ReLU(),
            *[nn.Sequential(
                    Linear1d(
                        h, h, bias
                    ),
                    nn.ReLU(),
                )
                for _ in range(1, num_layers)
            ],
        )
        self.beta = nn.Parameter(torch.randn(h, out_dim))

    def forward(self, x):
        y = x.flatten(1) # [bs, cin, space] -> [bs, cin * space]
        y = self.hier(y)
        y = y @ self.beta / self.beta.size(0)
        return y
