from torch.utils.data import Dataset
from itertools import accumulate


class BatchIdxDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in datasets]
        self.offsets = [0] + list(accumulate(self.lengths))

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        for i, offset in enumerate(self.offsets):
            if idx < offset:
                dataset_idx = i - 1
                element_idx = idx - self.offsets[dataset_idx]
                break
        data, label = self.datasets[dataset_idx][element_idx]
        return data, label, idx