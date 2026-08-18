from torch.utils.data import Dataset, IterableDataset


class ValidationWrapper(Dataset):
    """Expose one advancing sample from an iterable validation dataset."""

    def __init__(self, dataset, length=1):
        self.dataset = dataset
        self.length = length
        self.dataset_iterator = None

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if isinstance(self.dataset, IterableDataset):
            if self.dataset_iterator is None:
                self.dataset_iterator = iter(self.dataset)
            return next(self.dataset_iterator)
        return self.dataset[index]
