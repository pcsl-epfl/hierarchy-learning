"""
    Train networks on 1d hierarchical models of data.
"""

import os
import argparse
import time
import math
import pickle
from models import *
import copy
from functools import partial

from init import init_fun
from optim_loss import loss_func, regularize, opt_algo, measure_accuracy
from utils import cpu_state_dict
from observables import locality_measure

def run(args):

    best_acc = 0  # best test accuracy
    criterion = partial(loss_func, args)

    trainloader, testloader, net0 = init_fun(args)

    # scale batch size when smaller than train-set size
    if (args.batch_size <= args.ptr) and args.scale_batch_size:
        args.batch_size = args.ptr // 2

    if args.save_dynamics:
        dynamics = [{"acc": 0.0, "epoch": 0., "net": cpu_state_dict(net0)}]
    else:
        dynamics = None

    loss = []
    terr = []
    locality = []
    epochs_list = []

    best = dict()
    trloss_flag = 0

    for net, epoch, losstr, avg_epoch_time in train(args, trainloader, net0, criterion):

        assert str(losstr) != "nan", "Loss is nan value!!"
        loss.append(losstr)
        epochs_list.append(epoch)

        # measuring locality for fcn nets
        if args.net == 'fcn':
            state = net.state_dict()
            hidden_layers = [state[k] for k in state if 'w' in k][:-2]
            with torch.no_grad():
                locality.append(locality_measure(hidden_layers, args)[0])

        # avoid computing accuracy each and every epoch if dataset is small and epochs are rescaled
        # if epoch > 250:
        #     if epoch % (args.epochs // 250) != 0:
        #         continue

        if epoch % 10 != 0 and not args.save_dynamics: continue

        if testloader:
            acc = test(args, testloader, net, criterion, print_flag=epoch % 5 == 0)
        else:
            acc = torch.nan
        terr.append(100 - acc)

        if args.save_dynamics:
        #     and (
        #     epoch
        #     in (10 ** torch.linspace(-1, math.log10(args.epochs), 30)).int().unique()
        # ):
            # save dynamics at 30 log-spaced points in time
            dynamics.append(
                {"acc": acc, "epoch": epoch, "net": cpu_state_dict(net)}
            )
        if acc > best_acc:
            best["acc"] = acc
            best["epoch"] = epoch
            if args.save_best_net:
                best["net"] = cpu_state_dict(net)
            # if args.save_dynamics:
            #     dynamics.append(best)
            best_acc = acc
            print(f"BEST ACCURACY ({acc:.02f}) at epoch {epoch:.02f} !!", flush=True)

        out = {
            "args": args,
            "epoch": epochs_list,
            "train loss": loss,
            "terr": terr,
            "locality": locality,
            "dynamics": dynamics,
            "best": best,
        }

        yield out

        if (losstr == 0 and args.loss == 'hinge') or (losstr < args.zero_loss_threshold and args.loss == 'cross_entropy'):
            trloss_flag += 1
            if trloss_flag >= args.zero_loss_epochs:
                break

    try:
        wo = weights_evolution(net0, net)
    except:
        print("Weights evolution failed!")
        wo = None

    out = {
        "args": args,
        "epoch": epochs_list,
        "train loss": loss,
        "terr": terr,
        "locality": locality,
        "dynamics": dynamics,
        "init": cpu_state_dict(net0) if args.save_init_net else None,
        "best": best,
        "last": cpu_state_dict(net) if args.save_last_net else None,
        "weight_evo": wo,
        'avg_epoch_time': avg_epoch_time,
    }
    yield out


def train(args, trainloader, net0, criterion):

    net = copy.deepcopy(net0)

    optimizer, scheduler = opt_algo(args, net)
    print(f"Training for {args.epochs} epochs...")

    start_time = time.time()

    num_batches = math.ceil(args.ptr / args.batch_size)
    checkpoint_batches = torch.linspace(0, num_batches, 10, dtype=int)

    for epoch in range(args.epochs):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, targets)
            train_loss += loss.detach().item()
            regularize(loss, net, args.weight_decay, reg_type=args.reg_type)
            loss.backward()
            optimizer.step()

            correct, total = measure_accuracy(args, outputs, targets, correct, total)

            # during first epoch, save some sgd steps instead of after whole epoch
            if epoch < 10 and batch_idx in checkpoint_batches and batch_idx != (num_batches - 1):
                yield net, epoch + (batch_idx + 1) / num_batches, train_loss / (batch_idx + 1), None

        avg_epoch_time = (time.time() - start_time) / (epoch + 1)

        if epoch % 5 == 0:
            print(
                f"[Train epoch {epoch+1} / {args.epochs}, {print_time(avg_epoch_time)}/epoch, ETA: {print_time(avg_epoch_time * (args.epochs - epoch - 1))}]"
                f"[tr.Loss: {train_loss * args.alpha / (batch_idx + 1):.03f}]"
                f"[tr.Acc: {100.*correct/total:.03f}, {correct} / {total}]",
                flush=True
            )

        scheduler.step()

        yield net, epoch + 1, train_loss / (batch_idx + 1), avg_epoch_time


def test(args, testloader, net, criterion, print_flag=True):

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = net(inputs)

            loss = criterion(outputs, targets)

            test_loss += loss.item()

            correct, total = measure_accuracy(args, outputs, targets, correct, total)

        if print_flag:
            print(
                f"[TEST][te.Loss: {test_loss * args.alpha / (batch_idx + 1):.03f}]"
                f"[te.Acc: {100. * correct / total:.03f}, {correct} / {total}]",
                flush=True
            )

    return 100.0 * correct / total


# timing function
def print_time(elapsed_time):

    # if less than a second, print milliseconds
    if elapsed_time < 1:
        return f"{elapsed_time * 1000:.00f}ms"

    elapsed_seconds = round(elapsed_time)

    m, s = divmod(elapsed_seconds, 60)
    h, m = divmod(m, 60)

    elapsed_time = []
    if h > 0:
        elapsed_time.append(f"{h}h")
    if not (h == 0 and m == 0):
        elapsed_time.append(f"{m:02}m")
    elapsed_time.append(f"{s:02}s")

    return "".join(elapsed_time)


def weights_evolution(f0, f):
    s0 = f0.state_dict()
    s = f.state_dict()
    nd = 0
    for k in s:
        nd += (s0[k] - s[k]).norm() / s0[k].norm()
    nd /= len(s)
    return nd


def main():

    parser = argparse.ArgumentParser()

    ### Tensors type ###
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float64")

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
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--scale_batch_size", type=int, default=0)

    parser.add_argument("--background_noise", type=float, default=0)

    # Hierarchical dataset #
    parser.add_argument("--num_features", type=int, default=8)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_classes", type=int, default=-1)
    parser.add_argument("--input_format", type=str, default="onehot")
    parser.add_argument("--whitening", type=int, default=0)
    parser.add_argument("--auto_regression", type=int, default=0)

    ### ARCHITECTURES ARGS ###
    parser.add_argument("--net", type=str, required=True)
    parser.add_argument("--random_features", type=int, default=0)

    ## Nets params ##
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--net_layers", type=int, default=3)
    parser.add_argument("--filter_size", type=int, default=2)
    parser.add_argument("--pooling_size", type=int, default=2)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--batch_norm", type=int, default=0)
    parser.add_argument("--bias", type=int, default=1, help="for some archs, controls bias presence")
    parser.add_argument("--pbc", type=int, default=0, help="periodic boundaries cnn")

    ## Auto-regression with Transformers ##
    parser.add_argument("--pmask", type=float, default=.2)


    ### ALGORITHM ARGS ###
    parser.add_argument("--loss", type=str, default="cross_entropy")
    parser.add_argument("--optim", type=str, default="sgd")
    parser.add_argument("--scheduler", type=str, default="cosineannealing")
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--weight_decay", default=5e-4, type=float)
    parser.add_argument("--reg_type", default='l2', type=str)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--zero_loss_epochs", type=int, default=0)
    parser.add_argument("--zero_loss_threshold", type=float, default=0.01)
    parser.add_argument("--rescale_epochs", type=int, default=0)

    parser.add_argument(
        "--alpha", default=1.0, type=float, help="alpha-trick parameter"
    )

    ### SAVING ARGS ###
    parser.add_argument("--save_init_net", type=int, default=1)
    parser.add_argument("--save_best_net", type=int, default=1)
    parser.add_argument("--save_last_net", type=int, default=1)
    parser.add_argument("--save_dynamics", type=int, default=0)

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
    if args.net_layers == -1:
        args.net_layers = args.num_layers
    if args.m == -1:
        args.m = args.num_features

    # define train and test sets sizes
    Pmax = args.m ** (2 ** args.num_layers - 1) * args.num_classes

    if 0 < args.pte <= 1:
        args.pte = int(args.pte * Pmax)
    elif args.ptr == -1:
        args.pte = min(Pmax // 5, 20000)
    else:
        args.pte = int(args.pte)

    if args.ptr >= 0:
        if args.ptr <= 1:
            args.ptr = int(args.ptr * Pmax)
        else:
            args.ptr = int(args.ptr)
        assert args.ptr > 0, "relative dataset size (P/Pmax) too small for such dataset!"
    else:
        args.ptr = int(- args.ptr * args.m ** (args.num_layers) * args.num_features)

    args.pte = min(Pmax - args.ptr, args.pte)

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
