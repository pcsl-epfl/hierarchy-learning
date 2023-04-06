import torch

from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

from datasets.hierarchical import RandomHierarchyModel
from models import model_initialization

def state2permutation_stability(state, args):

    x = build_permuted_datasets(args)
    _, ch, input_dim = x.shape
    l = state2feature_extractor(state, args, input_dim, ch)

    with torch.no_grad():
        o = l(x.to(args.device))

    return measure_stability(o, args)

def build_permuted_datasets(args):
    x = []
    for seed_reset_layer in range(args.num_layers, -1, -1):
        dataset = RandomHierarchyModel(
            num_features=args.num_features,
            m=args.m,  # features multiplicity
            num_layers=args.num_layers,
            num_classes=args.num_features,
            s=args.s,
            input_format=args.input_format,
            whitening=args.whitening,
            seed=args.seed_init,
            train=True,
            transform=None,
            testsize=0,
            seed_reset_layer=seed_reset_layer,
        )
        x.append(dataset.x)
    return torch.cat(x)

def state2feature_extractor(state, args, input_dim, ch):
    net = model_initialization(args, input_dim=input_dim, ch=ch)
    net.load_state_dict(state)
    net.to(args.device)
    nodes, _ = get_graph_node_names(net)
    nodes = ['hier.1', 'truediv'] + [f'hier.{i}.1' for i in range(2, args.num_layers + 1)]
    nodes.sort()
    return create_feature_extractor(net, return_nodes=nodes)


def measure_stability(o, args):
    stability = {}
    for node in o.keys():
        on = o[node].detach()
        on = on.flatten(1)
        on = on.reshape(args.num_layers + 1, -1, on.shape[1])
        normalization = (on[0][None] - on[0][:, None]).pow(2).sum(dim=-1)
        stability[node] = ((on[0] - on[1:]).pow(2).sum(dim=-1).mean(1) / normalization.mean()).cpu()

    return stability
