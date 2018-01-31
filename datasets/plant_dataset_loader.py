from tqdm import tqdm
import numpy as np

from .datasets_path import DatasetsPath
from .module import create_module

class PlantDatasetLoader():
    def __init__(self, module, is_flip=True, is_multilabel=True, is_data_augmentation=True, root=None):
        self.module = create_module(module)
        self.is_multilabel = is_multilabel
        self.n_fold = 4

        if root is None:
            self.datasets_path = DatasetsPath(0, module, is_flip)
        else:
            self.datasets_path = DatasetsPath(0, module, is_flip, root=root)

        if not is_data_augmentation:
            self.datasets_path.not_data_augmentation()

    def four_fold_cross_val(self):
        for n in range(self.n_fold):
            print("{} / {}-fold cross validation".format(n+1, self.n_fold))
            train_data, val_data = self.plant_pairs(n)

            train = self.plant_dataset(train_data)
            val = self.plant_dataset(val_data)

            yield (train, val)

    def plant_pairs(self, val_n):
        print("Loading datasets...")

        self.datasets_path.val_n = val_n

        train_paths = self.datasets_path.four_fold_cross_validation_train_paths
        val_path = self.datasets_path.four_fold_cross_validation_val_path

        train_data = []
        for train_path in train_paths:
            train_data.extend(self.data_pairs(train_path))
        val_data = self.data_pairs(val_path)

        return (train_data, val_data)

    def plant_dataset(self, pairs):
        dataset_type = self.module.dataset_type
        return dataset_type(pairs)

    def data_pairs(self, path):
        if not path.exists():
            raise NameError("{} does not exist".format(str(path)))

        suffix = "*{}".format(self.module.suffix)

        if self.is_multilabel:
            # TODO: クラス数がハードコードになってる
            return [(str(y), [1 if i == int(str(y.parts[-1])[0]) else 0 for i in range(8)]) for y in tqdm(path.glob(suffix))]
        else:
            return [(str(y), int(str(y.parts[-1])[0])) for y in tqdm(path.glob(suffix))]
