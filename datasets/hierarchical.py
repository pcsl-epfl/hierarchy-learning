import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from itertools import *
import random
import numpy as np

def hierarchical_features(num_features, num_layers, m, num_classes, seed=0):
    random.seed(seed)
    features = [torch.arange(num_classes)]
    for i in range(num_layers):
        previous_features = features[-1].flatten()
        features_set = list(set([i.item() for i in previous_features]))
        num_layer_features = len(features_set)
        new_features = list(combinations(range(num_features), 2))
        assert len(new_features) >= m * num_layer_features, "Not enough features to choose from!!"
        random.shuffle(new_features)
        new_features = new_features[:m * num_layer_features]
        new_features = list(sum(new_features, ()))  # tuples to list

        new_features = torch.tensor(new_features)
        new_features = new_features.reshape(-1, m, 2)  # [n_features h-1, m, 2]

        # map features to indices
        feature_to_index = dict([(e, i) for i, e in enumerate(features_set)])

        indices = [feature_to_index[f.item()] for f in previous_features]

        new_features = new_features[indices]
        features.append(new_features)
    return features


def features_to_data(features, m, num_classes, num_features, num_layers, samples_per_class, seed=0):
    np.random.seed(seed)
    x = features[-1].reshape(num_classes, *sum([(m, 2) for _ in range(num_layers)], ()))
    y = torch.arange(num_classes)[None].repeat(samples_per_class, 1).t().flatten()

    indices = []
    for l in range(num_layers):

        if l != 0:
            left_right = torch.arange(2)[None].repeat(2 ** (num_layers - 2), 1).reshape(2 ** (num_layers - l - 1),
                                                                                        -1).t().flatten()
            left_right = left_right[None].repeat(samples_per_class * num_classes, 1)
            indices.append(left_right)

        random_features = np.random.choice(range(m), size=(samples_per_class * num_classes, 2 ** l)).repeat(
            2 ** (num_layers - l - 1), 1)
        indices.append(torch.tensor(random_features))

    yi = y[:, None].repeat(1, 2 ** (num_layers - 1))

    x = x[tuple([yi, *indices])].flatten(1)

    return x, y


def dec2bin(x, bits=None):
    if bits is None:
        bits = (x.max() + 1).log2().ceil().item()
    x = x.int()
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()


# def features_to_data(features, m, num_classes, num_features, num_layers, samples_per_class):
#     x = features[-1].reshape(num_classes, *sum([(m, 2) for _ in range(num_layers)], ()))
#     y = torch.arange(num_classes)[None].repeat(samples_per_class, 1).t().flatten()
#     ii = np.random.choice(range(m), size=(num_layers, samples_per_class * num_classes))
#     if num_layers == 1:
#         x = x[y, ii[0]]
#     elif num_layers == 2:
#         x = x[y, ii[0], :, ii[1], :]
#     elif num_layers == 3:
#         x = x[y, ii[0], :, ii[1], :, ii[2], :]
#     elif num_layers == 4:
#         x = x[y, ii[0], :, ii[1], :, ii[2], :, ii[3], :]
#     elif num_layers == 5:
#         x = x[y, ii[0], :, ii[1], :, ii[2], :, ii[3], :, ii[4], :]
#
#     x = x.flatten(1)
#     x = (x + 1) / num_features
#
#     return x, y


# def features_to_data(features, m, num_classes, num_features):
#     x = features[-1]
#     x = x.reshape(num_classes, -1, m, 2)
#     N = x.shape[1]
#     combs = list(product(*[range(m) for _ in range(N)]))
#     x_by_class = []
#     for xi in x:
#         combs_class = deepcopy(combs)
#         shuffle(combs_class)
#         combs_class = combs_class[:50000]
#         x_by_class.append(torch.stack([xi[range(N), c] for c in combs_class]))
#     x = torch.stack(x_by_class).reshape(-1, N * 2)
#     x = (x + 1) / num_features
#
#     samples_per_class = x.shape[0] // num_classes
#
#     y = torch.arange(num_classes)
#     y = y[None].repeat(samples_per_class, 1).T.reshape(-1)
#
#     return x, y


class HierarchicalDataset(Dataset):
    """
        Hierarchical dataset.
    """

    def __init__(
            self,
            num_features=8,
            m=2,  # features multiplicity
            num_layers=2,
            num_classes=2,
            seed=0,
            train=True,
            input_format=0,
            transform=None,
            testsize=-1
    ):
        torch.manual_seed(seed)
        self.num_features = num_features
        self.m = m  # features multiplicity
        self.num_layers = num_layers
        self.num_classes = num_classes

        features = hierarchical_features(num_features, num_layers, m, num_classes, seed=seed)
        self.x, self.targets = features_to_data(features, m, num_classes, num_features, num_layers, samples_per_class=10000, seed=seed)

        if input_format == 'binary':
            self.x = dec2bin(self.x)
            self.x = self.x.permute(0, 2, 1)
        elif input_format == 'decimal':
            self.x = (self.x[:, None] + 1) / num_features
        elif input_format == 'onehot':
            self.x = F.one_hot(self.x.long()).float()
            self.x = self.x.permute(0, 2, 1)
        else:
            raise ValueError

        if testsize == -1:
            testsize = len(self.x) // 5

        P = torch.randperm(len(self.targets))
        if train:
            P = P[:-testsize]
        else:
            P = P[-testsize:]

        self.x, self.targets = self.x[P], self.targets[P]

        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        """
        :param idx: sample index
        :return (torch.tensor, torch.tensor): (sample, label)
        """

        x, y = self.x[idx], self.targets[idx]

        if self.transform:
            x = self.transform(x)

        # if self.background_noise:
        #     g = torch.Generator()
        #     g.manual_seed(idx)
        #     x += torch.randn(x.shape, generator=g) * self.background_noise

        return x, y