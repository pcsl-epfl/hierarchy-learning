import torch

from datasets import dataset_initialization
from models import model_initialization

import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def init_fun(args):
    """
        Initialize dataset and architecture.
    """
    torch.manual_seed(args.seed_init)

    trainset, testset, input_dim, ch = dataset_initialization(args)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=0)

    net = model_initialization(args, input_dim=input_dim, ch=ch)

    return trainloader, testloader, net



