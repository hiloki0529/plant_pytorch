# -*- coding: utf-8 -*-

from pathlib import Path
from .module import create_module

class DatasetsPath():

    four_fold_cross_validation = "four_fold_cross_validation"
    route_flip = "rotate_flip"
    route = "rotate"
    train = "train"
    val = "val"
    vgg = "vgg"
    bad = "bad"

    def __init__(self, val_n, module, is_flip, root="/home/agri/datasets_sai/leaves_datasets"):
    
        """
        Args:
            val_n: int. 4-fold cross validation で何番目か.
            module: 文字列またはModule. 今のところ 'pil' か 'cv' か 'vgg'. 'cv' :推奨
            is_flip: bool. 回転処理を入れているか, 入れてないか.
            root: ルート
        """
    
        self.root = Path(root)

        self.val_n = val_n

        if type(module) is str:
            self.module = create_module(module).path
        else:
            self.module = module.path

        if is_flip:
            self.process = DatasetsPath.route_flip
        else:
            self.process = DatasetsPath.route
        self.is_flip = is_flip

    @property
    def sub_root(self):
        return (self.root / DatasetsPath.four_fold_cross_validation)

    @property
    def all_train_paths(self):
        return [ (self.sub_root / DatasetsPath.train / self.process / self.module / str(i)) for i in range(4) ]

    @property
    def all_val_paths(self):
        return [(self.sub_root / DatasetsPath.val / self.module / str(i)) for i in range(4)]

    @property
    def four_fold_cross_validation_train_paths(self):
        train_paths = self.all_train_paths[:]
        train_paths.pop(self.val_n)
        return train_paths

    @property
    def four_fold_cross_validation_val_path(self):
        return (self.sub_root / DatasetsPath.val / self.module / str(self.val_n))

    def vgg_train_and_val_paths(self, vgg_per):
        return (self.vgg_train_path(vgg_per), self.vgg_val_path(vgg_per))

    def vgg_train_path(self, vgg_per):
        return self.sub_root / DatasetsPath.train / self.process / DatasetsPath.vgg / vgg_per / str(self.val_n)

    def vgg_val_path(self, vgg_per):
        return self.sub_root / DatasetsPath.val / DatasetsPath.vgg / vgg_per / str(self.val_n)

    def not_data_augmentation(self):
        DatasetsPath.train = DatasetsPath.val
        self.process = ""


if __name__ == '__main__':
    print(DatasetsPath.four_fold_cross_fold_validation)
