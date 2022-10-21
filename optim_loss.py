from torch import nn
import torch.optim as optim


def loss_func(args, o, y):
    """
    Compute the loss.
    :param args: parser args
    :param o: network prediction
    :param y: true labels
    :return: value of the loss
    """

    if args.loss == "cross_entropy":
        loss = nn.functional.cross_entropy(args.alpha * o, y, reduction="mean")
    elif args.loss == "hinge":
        loss = ((1 - args.alpha * o * y).relu()).mean()
    else:
        raise NameError("Specify a valid loss function in [cross_entropy, hinge]")

    return loss

def regularize(loss, f, l, reg_type):
    """
    add L1/L2 regularization to the loss.
    :param loss: current loss
    :param f: network function
    :param args: parser arguments
    """
    for p in f.parameters():
        if reg_type == 'l1':
            loss += l * p.abs().mean()
        elif reg_type == 'l2':
            loss += l * p.pow(2).mean()

def measure_accuracy(args, out, targets, correct, total):
    """
    Compute out accuracy on targets. Returns running number of correct and total predictions.
    """
    if args.loss != "hinge":
        _, predicted = out.max(1)
        correct += predicted.eq(targets).sum().item()
    else:
        correct += (out * targets > 0).sum().item()

    total += targets.size(0)

    return correct, total


def opt_algo(args, net):
    """
    Define training optimizer and scheduler.
    :param args: parser args
    :param net: network function
    :return: torch scheduler and optimizer.
    """

    # rescale loss by alpha or alpha**2 if doing feature-lazy
    args.lr = args.lr / args.alpha

    if args.optim == "sgd":
        # if args.net == 'hlcn':
        #     optimizer = optim.SGD([
        #         {
        #             "params": (p for n, p in net.named_parameters() if 'b' not in n and 'weight' not in n),
        #             "lr": args.lr * args.width ** .5,
        #         },
        #         {
        #             "params": (p for n, p in net.named_parameters() if 'b' in n or 'weight' in n),
        #             "lr": args.lr * args.width,
        #         },
        #     ], lr=args.lr * args.width, momentum=0.9
        # )
        # else:
        optimizer = optim.SGD(
            net.parameters(), lr=args.lr * args.width, momentum=args.momentum
        )  ## 5e-4
    elif args.optim == "adam":
        optimizer = optim.Adam(
            net.parameters(), lr=args.lr * args.width
        )  ## 1e-5
    else:
        raise NameError("Specify a valid optimizer [Adam, (S)GD]")

    if args.scheduler == "cosineannealing":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs * 0.8
        )
    elif args.scheduler == "none":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.epochs // 3, gamma=1.0
        )
    elif args.scheduler == "exponential":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.975)
    else:
        raise NameError(
            "Specify a valid scheduler [cosineannealing, exponential, none]"
        )

    return optimizer, scheduler
