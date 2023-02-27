import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from itertools import *
import random
import numpy as np

from .utils import unique, dec2bin


def hierarchical_features(num_features, num_layers, m, num_classes, seed=0):
    """
    Build hierarchy of features.

    :param num_features: number of features to choose from at each layer (short: `n`).
    :param num_layers: number of layers in the hierarchy (short: `l`)
    :param m: features multiplicity (number of ways in which a feature can be made from sub-feat.)
    :param num_classes: number of different classes
    :param seed: sampling sub-features seed
    :return: features hierarchy as a list of length num_layers
    """
    random.seed(seed)
    features = [torch.arange(num_classes)]
    for l in range(num_layers):
        previous_features = features[-1].flatten()
        features_set = list(set([i.item() for i in previous_features]))
        num_layer_features = len(features_set)
        # new_features = list(combinations(range(num_features), 2))
        new_features = list(product(range(num_features), range(num_features)))
        assert (
            len(new_features) >= m * num_layer_features
        ), "Not enough features to choose from!!"
        random.shuffle(new_features)
        new_features = new_features[: m * num_layer_features]
        new_features = list(sum(new_features, ()))  # tuples to list

        new_features = torch.tensor(new_features)
        new_features = new_features.reshape(-1, m, 2)  # [n_features h-1, m, 2]

        # here new_features are ordered as what makes a 2, what makes a 1 etc...]

        # map features to indices
        feature_to_index = dict([(e, i) for i, e in enumerate(features_set)])

        indices = [feature_to_index[f.item()] for f in previous_features]

        new_features = new_features[indices]
        features.append(new_features)
    return features


def features_to_data(features, m, num_classes, num_layers, samples_per_class, seed=0, seed_reset_layer=42):
    """
    Build hierarchical dataset from features hierarchy.

    :param features: hierarchy of features
    :param m: features multiplicity (number of ways in which a feature can be made from sub-feat.)
    :param num_classes: number of different classes
    :param num_layers: number of layers in the hierarchy (short: `l`)
    :param samples_per_class: self-expl.
    :param seed: controls randomness in sampling
    :return: dataset {x, y}
    """

    np.random.seed(seed)
    x = features[-1].reshape(num_classes, *sum([(m, 2) for _ in range(num_layers)], ())) # [nc, m, 2, m, 2, ...]
    y = torch.arange(num_classes)[None].repeat(samples_per_class, 1).t().flatten()

    indices = []
    for l in range(num_layers):

        if l != 0:
            # indexing the left AND right sub-features (i.e. dimensions of size 2 in x)
            # Repeat is there such that higher level features are chosen consistently for a give data-point
            left_right = (
                torch.arange(2)[None]
                .repeat(2 ** (num_layers - 2), 1)
                .reshape(2 ** (num_layers - l - 1), -1)
                .t()
                .flatten()
            )
            left_right = left_right[None].repeat(samples_per_class * num_classes, 1)
            indices.append(left_right)

        # randomly choose sub-features
        # TODO: to avoid resampling, enumerate all sub-features and only later randomize. Too large tensor for memory though.
        # (for the moment, this is solved by resampling + filtering unique samples.)
        if l >= seed_reset_layer:
            np.random.seed(seed + 42 + l)
        random_features = np.random.choice(
            range(m), size=(samples_per_class * num_classes, 2 ** l)
        ).repeat(2 ** (num_layers - l - 1), 1)
        indices.append(torch.tensor(random_features))

    yi = y[:, None].repeat(1, 2 ** (num_layers - 1))

    x = x[tuple([yi, *indices])].flatten(1)

    return x, y


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
        seed_traintest_split=0,
        train=True,
        input_format='onehot',
        whitening=0,
        transform=None,
        testsize=-1,
        memory_constraint=5e5,
        seed_reset_layer=42,
        unique_datapoints=1
    ):
        assert testsize or train, "testsize must be larger than zero when generating a test set!"
        torch.manual_seed(seed)
        self.num_features = num_features
        self.m = m  # features multiplicity
        self.num_layers = num_layers
        self.num_classes = num_classes
        Pmax = m ** (2 ** num_layers - 1) * num_classes


        samples_per_class = min(10 * Pmax, int(memory_constraint)) # constrain dataset size for memory budget

        features = hierarchical_features(
            num_features, num_layers, m, num_classes, seed=seed
        )
        self.x, self.targets = features_to_data(
            features, m, num_classes, num_layers, samples_per_class=samples_per_class, seed=seed, seed_reset_layer=seed_reset_layer
        )

        if unique_datapoints:
            self.x, unique_indices = unique(self.x, dim=0)
            self.targets = self.targets[unique_indices]

        # encode input pairs instead of features
        if "pairs" in input_format:
            self.x = pairing_features(self.x, num_features)

        if 'onehot' not in input_format:
            assert not whitening, "Whitening only implemented for one-hot encoding"

        if "binary" in input_format:
            self.x = dec2bin(self.x)
            self.x = self.x.permute(0, 2, 1)
        elif "long" in input_format:
            self.x = self.x.long() + 1
        elif "decimal" in input_format:
            self.x = ((self.x[:, None] + 1) / num_features - 1) * 2
        elif "onehot" in input_format:
            self.x = F.one_hot(self.x.long()).float()
            self.x = self.x.permute(0, 2, 1)

            if whitening:
                inv_sqrt_n = (num_features - 1) ** -.5
                self.x = self.x * (1 + inv_sqrt_n) - inv_sqrt_n
            else:
                exp = int("pairs" in input_format) + 1
                self.x *= num_features ** exp
        else:
            raise ValueError

        if testsize == -1:
            testsize = min(len(self.x) // 5, 20000)

        g = torch.Generator()
        g.manual_seed(seed_traintest_split)
        P = torch.randperm(len(self.targets), generator=g)
        if train and testsize:
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
            x, y = self.transform(x, y)

        # if self.background_noise:
        #     g = torch.Generator()
        #     g.manual_seed(idx)
        #     x += torch.randn(x.shape, generator=g) * self.background_noise

        return x, y

def pairs_to_num(xi, n):
    """
        Convert one long input with n-features encoding to n^2 pairs encoding.
    """
    ln = len(xi)
    xin = torch.zeros(ln // 2)
    for ii, xii in enumerate(xi):
        xin[ii // 2] += xii * n ** (1 - ii % 2)
    return xin

def pairing_features(x, n):
    """
        Batch of inputs from n to n^2 encoding.
    """
    xn = torch.zeros(x.shape[0], x.shape[-1] // 2)
    for i, xi in enumerate(x.squeeze()):
        xn[i] = pairs_to_num(xi, n)
    return xn