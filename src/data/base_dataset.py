from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class BaseVisionDataset(ABC, Dataset):

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass
