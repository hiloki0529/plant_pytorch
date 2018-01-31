from abc import ABCMeta, abstractmethod

class Module(metaclass=ABCMeta):
    @property
    def suffix(self):
        return ".jpg"

    @abstractmethod
    def dataset_type(self):
        pass

    @abstractmethod
    def path(self):
        pass
