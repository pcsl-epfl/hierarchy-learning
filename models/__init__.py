import torch
import torch.backends.cudnn as cudnn

from .fcn import FCN
from .lcn import LocallyHierarchicalNet
from .cnn import CNN, CNNLayerWise
from .fcn2 import FCN2
from .gcnn import GCNN


def model_initialization(args, input_dim, ch):
    """
    Neural netowrk initialization.
    :param args: parser arguments
    :return: neural network as torch.nn.Module
    """

    num_outputs = 1 if args.loss == "hinge" else args.num_classes

    ### Define network architecture ###
    torch.manual_seed(args.seed_net)

    net = None

    if args.net == "fcn":
        net = FCN(
            num_layers=args.net_layers,
            input_channels=ch * input_dim,
            h=args.width,
            out_dim=num_outputs,
            bias=args.bias,
        )
    elif args.net == "fcn2": # (h * space_dimension) neurons per layer instead of h
        net = FCN2(
            num_layers=args.net_layers,
            input_channels=ch,
            h=args.width,
            out_dim=num_outputs,
            bias=args.bias,
        )
    elif args.net == "hlcn":
        net = LocallyHierarchicalNet(
            num_layers=args.net_layers,
            input_channels=ch,
            h=args.width,
            # filter_size=args.filter_size,
            out_dim=num_outputs,
            bias=args.bias,
        )
    elif args.net == "cnn":
        net = CNN(
            num_layers=args.net_layers,
            input_channels=ch,
            h=args.width,
            patch_size=args.filter_size,
            out_dim=num_outputs,
            bias=args.bias,
        )
    elif args.net == "gcnn":
        net = GCNN(
            num_layers=args.net_layers,
            input_channels=ch,
            h=args.width,
            out_dim=num_outputs,
            bias=args.bias,
        )
    elif args.net == "cnn_layerwise":
        net = CNNLayerWise(
            num_layers=args.net_layers,
            input_channels=ch,
            h=args.width,
            # filter_size=args.filter_size,
            out_dim=num_outputs,
            bias=args.bias,
        )

    assert net is not None, "Network architecture not in the list!"

    if args.random_features:
        for param in [p for p in net.parameters()][:-2]:
            param.requires_grad = False

    net = net.to(args.device)

    # if args.device == "cuda":
    #     net = torch.nn.DataParallel(net)
    #     cudnn.benchmark = True

    return net
