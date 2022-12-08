import torch
from datasets import HierarchicalDataset
from itertools import product
import pandas as pd

def bincount(x, max_val=None):
    """Bincount on the last dimension of x"""
    if max_val is None:
        max_val = x.max().item() + 1
    assert x.dtype is torch.int64, "only integral (int64) tensor is supported"
    cnt = torch.zeros((*x.shape[:-1], max_val), dtype=x.dtype, device=x.device)
    return cnt.scatter_add_(dim=2, index=x, src=torch.ones_like(x))


df = []
device = 'cpu'
for L, n, seed in product([4], [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 22, 24, 26, 28, 30, 32], [0, 1, 2, 3]):

    print(f'{L=}, {n=}, {seed=}', flush=True)
    #     try:
    dataset = HierarchicalDataset(
        num_features=n,
        m=n,  # features multiplicity
        num_layers=L,
        num_classes=n,
        input_format='long',
        seed=seed,
        train=True,
        transform=None,
        testsize=0,
        memory_constraint=1e10
    )

    x, y = dataset.x.to(device) - 1, dataset.targets.to(device)

    y, idxs = y.sort()
    if len(y) % n:
        y, idxs = y[:-(len(y) % n)], idxs[:-(len(y) % n)]
    x = x[idxs].reshape(len(x) // n, n, -1)

    #     except:
    #         continue

    bnc = bincount(x.permute(1, 2, 0)).float()  # [alpha, j, beta]
    bnc /= bnc.sum(dim=-1, keepdim=True).float()
    res1 = bnc.std(dim=-1).mean()
    res2 = bnc.std(dim=[-1, 0]).mean()

    df.append({
        'n': n,
        'L': L,
        'seed': seed,
        'res1': res1,
        'res2': res2,
    })

# df = pd.DataFrame.from_records(df)

    torch.save(df, './dataframes/correlations_measurement_fidis2.torch')