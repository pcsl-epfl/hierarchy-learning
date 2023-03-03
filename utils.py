import math
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

def plot_std(x, std, c='C0', alpha=.3):
    log10e = math.log10(math.exp(1))
    P = x.keys()
    rerr = log10e *  std / x
    plt.fill_between(P, 10 ** (np.log10(x) - rerr), 10 ** (np.log10(x) + rerr), color=c, alpha=alpha)

# timing function
def format_time(elapsed_time):
    """
    format time into hours, minutes, seconds.
    :param float elapsed_time: elapsed time in seconds
    :return str: time formatted as `{hrs}h{mins}m{secs}s`
    """

    elapsed_seconds = round(elapsed_time)

    m, s = divmod(elapsed_seconds, 60)
    h, m = divmod(m, 60)

    elapsed_time = []
    if h > 0:
        elapsed_time.append(f'{h}h')
    if not (h == 0 and m == 0):
        elapsed_time.append(f'{m:02}m')
    elapsed_time.append(f'{s:02}s')

    return ''.join(elapsed_time)


def cpu_state_dict(f):
    return {k: deepcopy(f.state_dict()[k].cpu()) for k in f.state_dict()}

def args2train_test_sizes(args, max_pte=20000):
    Pmax = args.m ** (2 ** args.num_layers - 1) * args.num_classes

    if 0 < args.pte <= 1:
        args.pte = int(args.pte * Pmax)
    elif args.pte == -1:
        args.pte = max_pte
    else:
        args.pte = int(args.pte)

    if args.ptr >= 0:
        if args.ptr <= 1:
            args.ptr = int(args.ptr * Pmax)
        else:
            args.ptr = int(args.ptr)
        assert args.ptr > 1, "relative dataset size (P/Pmax) too small for such dataset!"
    else:
        args.ptr = int(- args.ptr * args.m ** args.num_layers * args.num_features)

    args.pte = min(Pmax - args.ptr, args.pte)

    return args.ptr, args.pte
