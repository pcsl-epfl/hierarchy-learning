import torch
import torch.backends.cudnn as cudnn

from .fcn import DenseNet
from .cnn import ConvNetGAPMF # ConvNet2L
from .lcn import LocallyHierarchicalNet
from .cnn2 import CNN2
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
        net = DenseNet(
            n_layers=args.net_layers,
            input_dim=input_dim * ch,
            h=args.width,
            out_dim=num_outputs,
            batch_norm=args.batch_norm,
            # bias=args.bias,
        )
    elif args.net == "cnn":
        net = ConvNetGAPMF(
            n_blocks=args.net_layers,
            input_ch=ch,
            h=args.width,
            filter_size=args.filter_size,
            stride=args.stride,
            pbc=args.pbc,
            out_dim=num_outputs,
            batch_norm=args.batch_norm,
        )

    ### The next 4 architectures are built to have the same *effective* number of parameters ###
    elif args.net == "hlcn":
        net = LocallyHierarchicalNet(
            num_layers=args.net_layers,
            input_channels=ch,
            h=args.width,
            # filter_size=args.filter_size,
            out_dim=num_outputs,
            bias=args.bias,
        )
    elif args.net == "cnn2":
        net = CNN2(
            num_layers=args.net_layers,
            input_channels=ch,
            h=args.width,
            # filter_size=args.filter_size,
            out_dim=num_outputs,
            bias=args.bias,
        )
    elif args.net == "fcn2":
        net = FCN2(
            num_layers=args.net_layers,
            input_channels=ch,
            h=args.width,
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
    # elif args.net == "cnn2L":
    #     net = ConvNet2L(
    #         n=ch,
    #         h=args.width,
    #         out_dim=num_outputs
    #     )

    assert net is not None, "Network architecture not in the list!"

    if args.random_features:
        for param in [p for p in net.parameters()][:-2]:
            param.requires_grad = False

    net = net.to(args.device)

    if args.device == "cuda":
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    return net
