import os
import argparse
import time
import pickle
import torch
from utils import format_time
from datasets import dataset_initialization
from kernels import gram_ntk
from sklearn.svm import SVC

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

def svc(ktrtr, ktetr, ytr, yte, l):
    """
    Train a Support Vector Classifier
    :param ktrtr: train-train gram matrix
    :param ktrtr: test-train gram matrix
    :param ytr: training labels
    :param yte: testing labels
    :param l: l2 penalty
    :return: classification error.
    """
    clf = SVC(C=1/l, kernel="precomputed", max_iter=-1)
    clf.fit(ktrtr, ytr)

    y_hat = torch.tensor(clf.predict(ktetr))
    terr = 1 - y_hat.eq(yte).float().mean()

    return terr


def run_krr(args):

    ptr, pte = args.ptr, args.pte

    t1 = time.time()
    def timing_fun(t1):
        t2 = time.time()
        print(format_time(t2 - t1))
        t1 = t2
        return t1

    args.device = device

    # initialize dataset
    print('Init dataset...')
    trainset, testset, _, _ = dataset_initialization(args)
    xtr, ytr = trainset.dataset.x[:ptr].flatten(1), 2 * trainset.dataset.targets[:ptr] - 1
    xte, yte = testset.x[:pte].flatten(1), 2 * testset.targets[:pte] - 1

    t1 = timing_fun(t1)

    print('Compute NTK gram matrix (train)...')
    ktrtr = gram_ntk(xtr, xtr)
    t1 = timing_fun(t1)

    print('Compute NTK gram matrix (test)...')
    ktetr = gram_ntk(xte, xtr)
    t1 = timing_fun(t1)

    if args.algo == 'krr':
        print('KRR...')
        err = kernel_regression(ktrtr, ktetr, ytr, yte, args.l)
        timing_fun(t1)
    elif args.algo == 'svc':
        print('SVC...')
        err = svc(ktrtr, ktetr, ytr, yte, args.l)
        timing_fun(t1)
    else:
        raise ValueError('`algo` argument is invalid, must be either svc or krr!')

    res = {
        'args': args,
        'err': err,
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
    parser.add_argument("--seed_data", type=int, default=-1)


    """
           TRAINING ARGS
    """
    parser.add_argument("--algo", type=str, required=True)

    parser.add_argument("--ptr", metavar="P", type=int, help="size of the training set")
    parser.add_argument("--pte", type=int, help="size of the validation set", default=8192)

    ### ridge parameter ###
    parser.add_argument("--l", metavar="lambda", type=float, help="regularisation parameter")

    # parser.add_argument("--pickle", type=str, required=False)
    parser.add_argument("--output", type=str, required=False, default="None")


    args = parser.parse_args()

    args.loss = 'none'

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
