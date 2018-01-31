from .module import Module
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parents[1].resolve()))

from preprocessed_dataset import PreprocessedDataset

class BadCondition(Module):
    def __init__(self):
        pass
    
    @property
    def dataset_type(self):
        return PreprocessedDataset

    @property
    def path(self):
        return Path("bad")
