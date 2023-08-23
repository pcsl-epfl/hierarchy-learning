import torch

def locality_measure(states, args, idxs=None, norm_weighted=False):
    """
        Measure locality of a set of weights.
        Takes as input a list of weights matrices for layers l=0,1,2... .
    """

    assert args.s == 2, "Locality measure only implemented for s=2!"

    locality = []
    idxs_none = idxs is None
    if idxs_none:
        idxs = []

    for l, w in enumerate(states):

        if l == 0:
            # reconstruct input structure + pairs: [ch, 2**(L-1), 2 (pair), width]
            w = w.view(args.num_features, 2 ** (args.num_layers - 1), 2, -1)
            # measure the norm over channels and pairs dimensions
            patch_norm = w.pow(2).sum([0, 2]).sqrt()
            # wn will be a matrix that has [space_resolution, width]
        else:
            # compute patch-norms by grouping together input weights that look at the same spatial locations
            patch_norm = torch.stack([w[idxs[l - 1] == i].pow(2).sum(0) for i in range(2 ** (args.num_layers - l))])
            patch_norm = patch_norm.view(2 ** (args.num_layers - l - 1), 2, -1).sum(1)
        # save info about spatial structure
        if idxs_none:
            idxs.append(patch_norm.max(0).indices)

        # sort the vectors norm in the space dimension to later compute relative norms
        sorted_patch_norm = patch_norm.sort(dim=0, descending=True).values
        # compute relative norms as locality measures
        norm_ratios = sorted_patch_norm[0] / sorted_patch_norm.mean(dim=0) # informative norm divided by total norm

        # use weights norm to do a weighted average
        if norm_weighted:
            norm = patch_norm.sum(0)
            norm /= norm.sum()
        else:
            norm = 1 / patch_norm.shape[1]

        locality.append((norm_ratios * norm).sum().item())

    return locality, idxs