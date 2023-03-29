import os
import argparse
import time
import pickle
import torch
from utils import format_time, args2train_test_sizes
from datasets import dataset_initialization
from kernels import select_kernel
from sklearn.svm import SVC, LinearSVC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def kernel_regression(ktrtr, ktetr, ytr, yte, ridge):
    """
    Perform kernel ridge regression
    :param ktrtr: train-train gram matrix
    :param ktrtr: test-train gram matrix
    :param ytr: training labels
    :param yte: testing labels
    :param ridge: ridge value
    :return: mean square error.
    """
    yte, ytr = yte.float(), ytr.float()

    alpha = torch.linalg.solve(
        ktrtr + ridge * torch.eye(ytr.size(0), device=ktrtr.device),
        ytr
    )

    f = ktetr @ alpha
    mse = (f - yte).pow(2).mean()
    return mse

def svc(ktrtr, ktetr, ytr, yte, l, kernel='precomputed'):
    """
    Train a Support Vector Classifier
    :param ktrtr: train-train gram matrix
    :param ktrtr: test-train gram matrix
    :param ytr: training labels
    :param yte: testing labels
    :param l: l2 penalty
    :return: classification error.
    """
    clf = SVC(C=1/l, kernel=kernel, max_iter=-1)
    clf.fit(ktrtr, ytr)

    y_hat = torch.tensor(clf.predict(ktetr))
    terr = 1 - y_hat.eq(yte).float().mean()

    return terr

def linear_svc(xtr, xte, ytr, yte, l):
    """
    Train a Support Vector Classifier
    :param xtr: train samples
    :param xte: test samples
    :param ytr: training labels
    :param yte: testing labels
    :param l: l2 penalty
    :return: classification error.
    """
    clf = LinearSVC(C=1/l, max_iter=-1)
    clf.fit(xtr, ytr)

    y_hat = torch.tensor(clf.predict(xte))
    terr = 1 - y_hat.eq(yte).float().mean()

    return terr


def run_krr(args):

    ptr, pte = args.ptr, args.pte

    t1 = time.time()
    def timing_fun(t1):
        t2 = time.time()
        print(format_time(t2 - t1), flush=True)
        t1 = t2
        return t1

    args.device = device

    # initialize dataset
    print('Init dataset...', flush=True)
    trainset, testset, _, _ = dataset_initialization(args)
    xtr, ytr = trainset.dataset.x[:ptr].permute(0, 2, 1).flatten(1), trainset.dataset.targets[:ptr]
    xte, yte = testset.x[:pte].permute(0, 2, 1).flatten(1), testset.targets[:pte]

    t1 = timing_fun(t1)

    if args.algo == 'linear_svc':
        assert args.kernel == 'linear', "Kernel must be linear for linearSVC algo."
        print('Linear SVC...', flush=True)
        err = linear_svc(xtr, xte, ytr, yte, args.l)
        timing_fun(t1)
    else:
        gram = select_kernel(args)
        print('Compute gram matrix (train)...', flush=True)
        ktrtr = gram(xtr, xtr)
        t1 = timing_fun(t1)

        print('Compute gram matrix (test)...', flush=True)
        ktetr = gram(xte, xtr)
        t1 = timing_fun(t1)

        if args.algo == 'krr':
            print('KRR...', flush=True)
            err = kernel_regression(ktrtr, ktetr, ytr, yte, args.l)
            timing_fun(t1)
        elif args.algo == 'svc':
            print('SVC...', flush=True)
            err = svc(ktrtr, ktetr, ytr, yte, args.l)
            timing_fun(t1)
        else:
            raise ValueError('`algo` argument is invalid, must be either svc or krr!')

    res = {
        'args': args,
        'err': err.item(),
    }

    yield res


def main():
    parser = argparse.ArgumentParser(
        description="Perform a kernel method on hierarchical dataset."
    )

    """
    	   DATASET ARGS
    """

    parser.add_argument("--dataset", type=str, default='hier1')
    parser.add_argument("--num_features", type=int, default=8)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_classes", type=int, default=-1)
    parser.add_argument("--input_format", type=str, default="onehot")
    parser.add_argument("--seed_init", type=int, default=-1)
    parser.add_argument("--seed_trainset", type=int, default=-1)
    parser.add_argument("--whitening", type=int, default=0)


    """
           TRAINING ARGS
    """
    parser.add_argument("--algo", type=str, required=True)
    parser.add_argument("--kernel", type=str, required=True)

    parser.add_argument("--ptr", metavar="P", type=float, help="size of the training set")
    parser.add_argument("--pte", type=float, help="size of the validation set", default=512)

    ### ridge parameter ###
    parser.add_argument("--l", metavar="lambda", type=float, help="regularisation parameter")

    parser.add_argument("--output", type=str, required=False, default="None")

    args = parser.parse_args()
    args.loss = 'none'
    args.auto_regression = 0
    if args.seed_trainset == -1:
        args.seed_trainset = args.seed_init
    if args.num_classes == -1:
        args.num_classes = args.num_features
    if args.m == -1:
        args.m = args.num_features
    args.ptr, args.pte = args2train_test_sizes(args, max_pte=1000)

    ## PAY ATTENTION TO THIS!!! ##
    # condition to run only if m^L/n < 1. #
    upper_bound = args.m ** args.num_layers / args.num_features ** (2 * (1 - 2 ** -args.num_layers)) < 0.8
    lower_bound = args.m ** args.num_layers / args.num_features ** (1 - 2 ** -args.num_layers) > 1.2
    if upper_bound and lower_bound:
        raise ValueError('Parameters outsize range for using couples to classify!')
    if args.ptr > 30000:
        raise ValueError("ptr too large!! (>30k)")

    with open(args.output, "wb") as handle:
        pickle.dump(args, handle)
    try:
        for data in run_krr(args):
            with open(args.output, "wb") as handle:
                pickle.dump(args, handle)
                pickle.dump(data, handle)
    except:
        os.remove(args.output)
        raise

if __name__ == "__main__":
    main()
