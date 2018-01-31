from torch.utils.data import Dataset
from skimage import io

class PreprocessedDataset(Dataset):
    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image = io.imread(self.pairs[idx][0])
        label = self.pairs[idx][1]
        if self.transform:
            image = self.transform(image)
        return image, label
