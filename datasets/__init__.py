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

    transform = None
    if args.auto_regression:
        # def transform(x, y):
        #     return x[:-1], x[1:]
        print('BERT mode')
        def transform(x, _):
            """ BERT-like masking. """
            idx = torch.randint(2 ** args.num_layers, (1,))[0]
            p = torch.rand(1)[0]
            y = torch.tensor([idx, x[idx]])
            x = x.clone()
            if p > .2:
                x[idx] = 0
            elif .1 < p < .2:
                x[idx] = torch.randint(args.num_features, (1,))[0]
            return x.contiguous(), y
            # mask = torch.rand((len(x),)) > args.pmask
            # return x * mask, x

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
            transform=transform,
            testsize=args.pte,
            memory_constraint= 10 * (args.ptr + args.pte)
        )

        if args.pte:
            testset = HierarchicalDataset(
                num_features=args.num_features,
                m=args.m,  # features multiplicity
                num_layers=args.num_layers,
                num_classes=nc,
                input_format=args.input_format,
                whitening=args.whitening,
                seed=args.seed_init,
                train=False,
                transform=transform,
                testsize=args.pte,
                memory_constraint= 10 * (args.ptr + args.pte)
            )
        else:
            testset = None

    elif args.dataset == 'parity':

        assert args.num_classes == 2, "Simple parity can only have two classes!!"

        trainset = ParityDataset(
            num_layers=args.num_layers,
            seed=args.seed_init,
            train=True,
            transform=transform,
            testsize=args.pte
        )

        if args.pte:
            testset = ParityDataset(
                num_layers=args.num_layers,
                seed=args.seed_init,
                train=False,
                transform=transform,
                testsize=args.pte
            )
        else:
            testset = None


    else:
        raise ValueError('`dataset` argument is invalid!')

    input_dim = trainset[0][0].shape[-1]
    ch = trainset[0][0].shape[-2] if args.input_format != 'long' else 0

    if args.loss == 'hinge':
        # change to binary labels
        trainset.targets = 2 * (torch.as_tensor(trainset.targets) >= nc // 2) - 1
        if testset:
            testset.targets = 2 * (torch.as_tensor(testset.targets) >= nc // 2) - 1

    P = len(trainset)
    assert args.ptr <= 32 + P, "ptr is too large!!"

    # take random subset of training set
    torch.manual_seed(args.seed_trainset)
    perm = torch.randperm(P)
    trainset = torch.utils.data.Subset(trainset, perm[:args.ptr])

    return trainset, testset, input_dim, ch
