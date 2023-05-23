import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from itertools import *
import random
import numpy as np

from .utils import dec2bin


def sample_hierarchical_rules(num_features, num_layers, m, num_classes, s, seed=0):
    """
    Build hierarchy of features.
    :param num_features: number of features to choose from at each layer (short: `n`).
    :param num_layers: number of layers in the hierarchy (short: `l`)
    :param m: features multiplicity (number of ways in which a feature can be made from sub-feat.)
    :param num_classes: number of different classes
    :param s: sub-features tuple size
    :param seed: sampling sub-features seed
    :return: features hierarchy as a list of length num_layers.
             Each layer contains all paths going from label to layer.
    """
    random.seed(seed)
    all_levels_paths = [torch.arange(num_classes)]
    all_levels_tuples = []
    for l in range(num_layers):
        old_paths = all_levels_paths[-1].flatten()
        # unique features in the previous level
        old_features = list(set([i.item() for i in old_paths]))
        num_old_features = len(old_features)
        # new_features = list(combinations(range(num_features), 2))
        # generate all possible new features at this level
        new_tuples = list(product(*[range(num_features) for _ in range(s)]))
        assert (
                len(new_tuples) >= m * num_old_features
        ), "Not enough features to choose from!!"
        random.shuffle(new_tuples)
        # samples as much as needed
        new_tuples = new_tuples[: m * num_old_features]
        new_tuples = list(sum(new_tuples, ()))  # tuples to list

        new_tuples = torch.tensor(new_tuples)

        new_tuples = new_tuples.reshape(-1, m, s)  # [n_features h-1, m, 2]

        old_feature_to_index = dict([(e, i) for i, e in enumerate(old_features)])

        indices = [old_feature_to_index[f.item()] for f in old_paths]

        new_paths = new_tuples[indices]

        all_levels_tuples.append(new_tuples)
        all_levels_paths.append(new_paths)
    return all_levels_paths, all_levels_tuples


def sample_data_from_paths(samples_indices, paths, m, num_classes, num_layers, s, seed=0, seed_reset_layer=42):
    """
    Build hierarchical dataset from features hierarchy.
    :param samples_indices: torch tensor containing indices in [0, 1, ..., Pmax - 1] of datapoints to sample
    :param paths: hierarchy of feature paths
    :param m: features multiplicity (number of ways in which a feature can be made from sub-feat.)
    :param num_classes: number of different classes
    :param num_layers: number of layers in the hierarchy (short: `l`)
    :param s: sub-features tuple size
    :param seed: controls randomness in sampling for stability measurements
    :param seed_reset_layer: layer from which to randomize the choice of semantically equivalent subfeatures (for stability measurements)
    :return: dataset {x, y}
    """

    Pmax = m ** ((s ** num_layers - 1) // (s - 1)) * num_classes

    x = paths[-1].reshape(num_classes, *sum([(m, s) for _ in range(num_layers)], ()))  # [nc, m, s, m, s, ...]

    groups_size = Pmax // num_classes
    y = samples_indices.div(groups_size, rounding_mode='floor')
    samples_indices = samples_indices % groups_size

    indices = []
    for l in range(num_layers):

        if l != 0:
            # indexing the left AND right sub-features (i.e. dimensions of size 2 in x)
            # Repeat is there such that higher level features are chosen consistently for a give data-point
            left_right = (
                torch.arange(s)[None]
                    .repeat(s ** (num_layers - 2), 1)
                    .reshape(s ** (num_layers - l - 1), -1)
                    .t()
                    .flatten()
            )
            left_right = left_right[None].repeat(len(samples_indices), 1)

            indices.append(left_right)

        if l >= seed_reset_layer:
            np.random.seed(seed + 42 + l)
            perm = torch.randperm(len(samples_indices))
            samples_indices = samples_indices[perm]

        groups_size //= m ** (s ** l)
        layer_indices = samples_indices.div(groups_size, rounding_mode='floor')

        rules = number2base(layer_indices, m, string_length=s ** l)
        rules = (
            rules[:, None]
                .repeat(1, s ** (num_layers - l - 1), 1)
                .permute(0, 2, 1)
                .flatten(1)
        )

        indices.append(rules)

        samples_indices = samples_indices % groups_size

    yi = y[:, None].repeat(1, s ** (num_layers - 1))
    
    x = x[tuple([yi, *indices])].flatten(1)

    return x, y


class RandomHierarchyModel(Dataset):
    """
    Implement the Random Hierarchy Model (RHM) as a PyTorch dataset.
    """

    def __init__(
            self,
            num_features=8,
            m=2,  # features multiplicity
            num_layers=2,
            num_classes=2,
            s=2,
            seed=0,
            max_dataset_size=None,
            seed_traintest_split=0,
            train=True,
            input_format='onehot',
            whitening=0,
            transform=None,
            testsize=-1,
            seed_reset_layer=42,
    ):
        assert testsize or train, "testsize must be larger than zero when generating a test set!"
        torch.manual_seed(seed)
        self.num_features = num_features
        self.m = m  # features multiplicity
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.s = s

        paths, _ = sample_hierarchical_rules(
            num_features, num_layers, m, num_classes, s, seed=seed
        )

        Pmax = m ** ((s ** num_layers - 1) // (s - 1)) * num_classes
        assert Pmax < 1e19, "Pmax cannot be represented with int64!! Parameters too large! Please open a github issue if you need a solution."
        if max_dataset_size is None or max_dataset_size > Pmax:
            max_dataset_size = Pmax
        if testsize == -1:
            testsize = min(max_dataset_size // 5, 20000)

        g = torch.Generator()
        g.manual_seed(seed_traintest_split)

        if Pmax < 5e6:  # there is a crossover in computational time of the two sampling methods around this value of Pmax
            samples_indices = torch.randperm(Pmax, generator=g)[:max_dataset_size]
        else:
            samples_indices = torch.randint(Pmax, (2 * max_dataset_size,), generator=g)
            samples_indices = torch.unique(samples_indices)
            perm = torch.randperm(len(samples_indices), generator=g)[:max_dataset_size]
            samples_indices = samples_indices[perm]

        if train and testsize:
            samples_indices = samples_indices[:-testsize]
        else:
            samples_indices = samples_indices[-testsize:]

        self.x, self.targets = sample_data_from_paths(
            samples_indices, paths, m, num_classes, num_layers, s, seed=seed, seed_reset_layer=seed_reset_layer
        )

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
            self.x = F.one_hot(
                self.x.long(),
                num_classes=num_features if 'pairs' not in input_format else num_features ** 2
            ).float()
            self.x = self.x.permute(0, 2, 1)

            if whitening:
                inv_sqrt_n = (num_features - 1) ** -.5
                self.x = self.x * (1 + inv_sqrt_n) - inv_sqrt_n

        else:
            raise ValueError

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


def number2base(numbers, base, string_length=None):
    digits = []
    while numbers.sum():
        digits.append(numbers % base)
        numbers = numbers.div(base, rounding_mode='floor')
    if string_length:
        assert len(digits) <= string_length, "String length required is too small to represent numbers!"
        digits += [torch.zeros(len(numbers), dtype=int)] * (string_length - len(digits))
    return torch.stack(digits[::-1]).t()


def compute_true_occurrences(v, L, m, nc, model_seed):
    """
        Compute N_j(\alpha, \mu) at Pmax.
    """
    paths, _ = sample_hierarchical_rules(
        v, L, m, nc, 2, seed=model_seed
    )[-1]
    # unroll paths tensor
    x = paths.reshape(nc, *sum([(m, 2) for _ in range(L)], ()))
    # multiplicity factor when going from paths to datapoints
    mul = m ** (2 ** L - 1 - L)
    # convert to base v (-> encoding of pairs)
    x = x[..., 0] * v + x[..., 1]
    # convert to one-hot-encoding so that N (occurrences ) can be obtained by summing
    x = torch.nn.functional.one_hot(x, num_classes=v**2)
    # sum over paths
    x = x.sum(dim=[2 * k + 1 for k in range(L)]) # what's left is number of classes and space dimensions: [nc, {s,} x (L - 1)]
    # if I don't care about space, can just flatten space dimensions (pay attention that they may not be ordered)!
    x = x.flatten(start_dim=1, end_dim=-2) # [nc, 2 ** (L-1), v ** 2]
    return x.permute(0, 2, 1) * mul # N(\alpha, \mu, j)
