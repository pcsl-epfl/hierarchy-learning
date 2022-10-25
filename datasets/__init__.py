import torch
from .hierarchical import HierarchicalDataset
from .parity import ParityDataset


def dataset_initialization(args):
    """
    Initialize train and test loaders for chosen dataset and transforms.
    :param args: parser arguments (see main.py)
    :return: trainloader, testloader, image size, number of classes.
    """

    nc = args.num_classes

    if args.dataset == 'hier1':
        trainset = HierarchicalDataset(
            num_features=args.num_features,
            m=args.m,  # features multiplicity
            num_layers=args.num_layers,
            num_classes=nc,
            input_format=args.input_format,
            whitening=args.whitening,
            seed=args.seed_init,
            train=True,
            transform=None,
            testsize=args.pte
        )

        testset = HierarchicalDataset(
            num_features=args.num_features,
            m=args.m,  # features multiplicity
            num_layers=args.num_layers,
            num_classes=nc,
            input_format=args.input_format,
            whitening=args.whitening,
            seed=args.seed_init,
            train=False,
            transform=None,
            testsize=args.pte
        )

    elif args.dataset == 'parity':

        assert args.num_classes == 2, "Simple parity can only have two classes!!"

        trainset = ParityDataset(
            num_layers=args.num_layers,
            seed=args.seed_init,
            train=True,
            transform=None,
            testsize=args.pte
        )

        testset = ParityDataset(
            num_layers=args.num_layers,
            seed=args.seed_init,
            train=False,
            transform=None,
            testsize=args.pte
        )


    else:
        raise ValueError('`dataset` argument is invalid!')

    input_dim = trainset[0][0].shape[-1]
    ch = trainset[0][0].shape[-2]

    if args.loss == 'hinge':
        # change to binary labels
        trainset.targets = 2 * (torch.as_tensor(trainset.targets) >= nc // 2) - 1
        testset.targets = 2 * (torch.as_tensor(testset.targets) >= nc // 2) - 1

    P = len(trainset)
    assert args.ptr <= 32 + P, "ptr is too large!!"

    # take random subset of training set
    torch.manual_seed(args.seed_trainset)
    perm = torch.randperm(P)
    trainset = torch.utils.data.Subset(trainset, perm[:args.ptr])

    return trainset, testset, input_dim, ch
