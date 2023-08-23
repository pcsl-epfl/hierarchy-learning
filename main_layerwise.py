import os
import argparse
import pickle

from utils import args2train_test_sizes
from datasets import dataset_initialization

import torch
from torch import nn

from torch.nn import functional as F
from sklearn.cluster import KMeans

from datasets.hierarchical import pairing_features


def two_layers(w1, seed, x, y):
    h, v2 = w1.size()
    assert v2 == x.shape[-1], "Input dim. not matching!"
    v = int(v2 ** .5)

    g = torch.Generator()
    g.manual_seed(seed)
    w2 = torch.randn(v, h, generator=g)

    o = (w2 @ (w1 @ x.t()).div(v2 ** .5).relu() / h).t()

    loss = torch.nn.functional.cross_entropy(o, y, reduction="mean")

    return loss, o

def run(args):

    m, v, nc, L, seed, fmt = args.m, args.num_features, args.num_classes, args.num_layers, args.seed_trainset, args.input_format
    h = args.width

    trainset, testset, _, _ = dataset_initialization(args)

    ws = [nn.Parameter(torch.ones(h, v ** 2)) for _ in range(L)]
    clus = []

    ##### TRAIN #####

    trainset = trainset.dataset
    x = trainset.x - 1
    y = trainset.targets

    for l in range(L):

        x = pairing_features(x, v)
        x = F.one_hot(x.long(), num_classes=v ** 2).float()
        x = x.permute(0, 2, 1)

        loss, _ = two_layers(ws[l], l+args.seed_net, x[..., 0], y)
        loss.backward()
        with torch.no_grad():
            ws[l] = ws[l] - 1000 * h * ws[l].grad

        if l == L - 1:
            with torch.no_grad():
                _, o = two_layers(ws[l], l, x[..., 0], y)

        else:

            kms = KMeans(n_clusters=v, n_init=10)
            kms.fit(ws[l].t())
            clus.append(kms.labels_)

            x = trainset.x - 1
            for c in clus:
                x = c[pairing_features(x, v).int()]

    train_error = (o.max(dim=1).indices == y).float().mean()

    ##### TEST #####

    x = testset.x - 1
    y = testset.targets

    for l in range(L):

        if l == L - 1:

            x = pairing_features(x, v)
            x = F.one_hot(x[:, 0].long(), num_classes=v ** 2).float()
            # x = x.permute(0, 2, 1)
            with torch.no_grad():
                _, o = two_layers(ws[l], l+args.seed_net, x, y)

        else:
            x = testset.x - 1
            for c in clus:
                x = c[pairing_features(x, v).int()]

    test_error = (o.max(dim=1).indices == y).float().mean()

    print(
        f"[tr.acc: {train_error * 100:.02f}]"
        f"[te.acc: {test_error  * 100:.02f}]"
    )

    out = {
        "args": args,
        "train_error": train_error.item(),
        "test_error": test_error.item(),
    }

    yield out


def main():

    parser = argparse.ArgumentParser()

    ### Tensors type ###
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32")

    ### Seeds ###
    parser.add_argument("--seed_init", type=int, default=0)
    parser.add_argument("--seed_net", type=int, default=-1)
    parser.add_argument("--seed_trainset", type=int, default=-1)

    ### DATASET ARGS ###
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ptr", type=float, default=0.8,
        help="Number of training point. If in [0, 1], fraction of training points w.r.t. total.",
    )
    parser.add_argument("--pte", type=float, default=.2)

    # Hierarchical dataset #
    parser.add_argument("--num_features", type=int, default=8)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--s", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_classes", type=int, default=-1)
    parser.add_argument("--input_format", type=str, default="long")
    parser.add_argument("--whitening", type=int, default=0)
    parser.add_argument("--auto_regression", type=int, default=0)
    parser.add_argument("--loss", type=str, default="cross_entropy")

    parser.add_argument("--width", type=int, default=256)


    ## saving path ##
    parser.add_argument("--pickle", type=str, required=False, default="None")
    parser.add_argument("--output", type=str, required=False, default="None")
    args = parser.parse_args()

    if args.pickle == "None":
        assert (
            args.output != "None"
        ), "either `pickle` or `output` must be given to the parser!!"
        args.pickle = args.output

    # special value -1 to set some equal arguments
    if args.seed_trainset == -1:
        args.seed_trainset = args.seed_init
    if args.seed_net == -1:
        args.seed_net = args.seed_init
    if args.num_classes == -1:
        args.num_classes = args.num_features
    if args.m == -1:
        args.m = args.num_features

    # define train and test sets sizes

    args.ptr, args.pte = args2train_test_sizes(args)

    with open(args.output, "wb") as handle:
        pickle.dump(args, handle)
    try:
        for data in run(args):
            with open(args.output, "wb") as handle:
                pickle.dump(args, handle)
                pickle.dump(data, handle)
    except:
        os.remove(args.output)
        raise


if __name__ == "__main__":
    main()
