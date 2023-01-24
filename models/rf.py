# random features (layer-wise)

import torch
from torch import nn

class RandomFeaturesLayer(nn.Module):
    def __init__(
            self, input_size, out_channels, bias=False
    ):
        super(RandomFeaturesLayer, self).__init__()

        rp = torch.randn(out_channels, input_size)
        rp /= rp.pow(2).sum(dim=-1, keepdim=True).sqrt()
        self.register_buffer("random_projection", rp)

        self.weight = nn.Parameter(torch.zeros(out_channels, out_channels)
                                   )
        if bias:
            self.bias = nn.Parameter(torch.randn(1, out_channels))
        else:
            self.register_buffer("bias", None)

    def forward(self, x):
        # x: [bs, cin]
        x = x[:, None] * self.random_projection  # [bs, cout, cin]
        x = x.sum(dim=-1)  # [bs, cout]

        x = x[:, None] * self.weight  # [bs, cout, cout]
        x = x.sum(dim=-1)  # [bs, cout]

        if self.bias is not None:
            x += self.bias * 0.1
        return x


class RFLayerwise(nn.Module):
    """
        Layerwise random features with zero initialization.
    """

    def __init__(self, input_channels, h, out_dim, num_layers, bias=False):
        super(RFLayerwise, self).__init__()

        d = 2 ** num_layers * input_channels

        layers = []
        for l in range(num_layers):
            layers.append(RandomFeaturesLayer(h if l else d, h, bias))
            layers.append(nn.ReLU())
        self.layers = layers

        self.hier = nn.Sequential(*layers)
        self.beta = nn.Parameter(torch.randn(h, out_dim))

        self.h = h
        self.out_dim = out_dim

    def init_layerwise_(self, l):
        device = self.beta.device
        self.hier = nn.Sequential(*self.layers[:2 * (l + 1)])
        self.beta = nn.Parameter(torch.randn(self.h, self.out_dim, device=device))

    def forward(self, x):
        y = x.flatten(1)  # [bs, cin, space] -> [bs, cin * space]
        if len(self.hier) == len(self.layers):
            y = self.hier(y)
        else:
            with torch.no_grad():
                y = self.hier[:-2](y)
            y = self.hier[-2:](y)
        y = y @ self.beta / self.beta.size(0)
        return y