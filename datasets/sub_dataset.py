import torch.utils.data as data

class SubDataset(data.Dataset):
    def __init__(self, dataset, indecas):
        self._base = dataset
        self._indecas = indecas

    def __len__(self):
        return len(self._indecas)

    def __getitem__(self, index):
        return self._base[self._indecas[index]]
