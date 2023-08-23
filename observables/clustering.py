import torch
from torch.nn import functional as F

from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

from datasets.hierarchical import sample_hierarchical_rules
from datasets.utils import unique

from models import model_initialization
from sklearn.cluster import KMeans


def state2clustering_error(state, args):

    v, L, m, nc, seed = args.num_features, args.num_layers, args.m, args.num_classes, args.seed_init
    idxs = order_features(v, L, m, nc, seed)
    ordered_inputs = torch.cat(
        [torch.stack([torch.div(idxs, args.num_features, rounding_mode='trunc'), idxs % args.num_features]).t(),
         torch.zeros(len(idxs), 2 ** L - 2)], dim=1)
    x = F.one_hot(ordered_inputs.long(), num_classes=v).float().permute(0, 2, 1)

    l = state2feature_extractor(state, args, 2 ** L, v)

    with torch.no_grad():
        o = l(x.to(args.device))

    o = o[list(o.keys())[0]][..., 0]

    kmeans = KMeans(n_clusters=len(idxs) // m, n_init=10).fit(o.cpu())

    torch.tensor(kmeans.labels_).view(v, -1)

    lbs = torch.tensor(kmeans.labels_).view(v, -1)
    return 1 - lbs.t().eq(lbs.mode(dim=-1).values).float().mean()


def state2feature_extractor(state, args, input_dim, ch):
    net = model_initialization(args, input_dim=input_dim, ch=ch)
    net.load_state_dict(state)
    net.to(args.device)
    nodes, _ = get_graph_node_names(net)
    nodes = ['hier.1', 'truediv'] + [f'hier.{i}.1' for i in range(2, args.num_layers + 1)]
    nodes.sort()
    nodes = [nodes[1]]
    return create_feature_extractor(net, return_nodes=nodes)


def pairs_to_int(x, n):
    return n * x[..., 0] + x[..., 1]

def order_features(v, L, m, nc, seed):
    # build hierarchy
    features, _ = sample_hierarchical_rules(
        v, L, m, nc, s=2, seed=seed
    )

    # ordering of n^2 features into n groups of size n
    spelled_out_features = pairs_to_int(features[-1], v)
    ue, ui = unique(spelled_out_features, dim=0)
    sorted_ui = features[-2].flatten(-3)[ui].sort().indices
    return ue[sorted_ui].flatten()