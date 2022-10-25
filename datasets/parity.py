import torch
from torch.utils.data import Dataset
from .utils import unique

class ParityDataset(Dataset):
    """
        Parity dataset.
    """

    def __init__(
        self,
        num_layers=2,
        seed=0,
        train=True,
        transform=None,
        testsize=-1,
    ):
        torch.manual_seed(seed)

        self.input_size = 2 ** num_layers

        Pmax = 2 ** self.input_size

        samples_per_class = min(10 * Pmax, int(5e5)) # constrain dataset size for memory budget

        self.x = torch.randint(2, size=(int(2 * samples_per_class), self.input_size)) * 2. - 1
        self.targets = self.x.prod(dim=1).eq(1).long()

        self.x, unique_indices = unique(self.x, dim=0)
        self.targets = self.targets[unique_indices]

        self.x = self.x[:, None]

        print(f"Data set size: {self.x.shape[0]}")

        if testsize == -1:
            testsize = min(len(self.x) // 5, 100000)

        P = torch.randperm(len(self.targets))
        if train:
            P = P[:-testsize]
        else:
            P = P[-testsize:]

        self.x, self.targets = self.x[P], self.targets[P]

        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        """
        :param idx: sample index
        :return (torch.tensor, torch.tensor): (sample, label)
        """

        x, y = self.x[idx], self.targets[idx]

        if self.transform:
            x = self.transform(x)

        return x, y