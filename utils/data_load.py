import numpy as np
import torch
from datasets.dataset import CLASS_MAP
from torch.utils.data import Dataset


class BrainDataset(Dataset):
    def __init__(self, voxels, labels, transform=None):
        self.voxels = voxels
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.voxels)
    def __getitem__(self, index):
        voxel = self.voxels[index]
        label = self.labels[index]
        if self.transform:
            voxel = self.transform(voxel, self.phase)
        voxel = self._preprocess(voxel)
        return voxel, label
    def _preprocess(self, voxel):
        cut_range = 4
        voxel = np.clip(voxel, 0, cut_range * np.std(voxel))
        voxel = normalize(voxel, np.min(voxel), np.max(voxel))
        voxel = voxel[np.newaxis, ]
        return voxel.astype('f')
    def __call__(self, index):
        return self.__getitem__(index)


def normalize(voxel: np.ndarray, floor: int, ceil: int) -> np.ndarray:
    return (voxel - floor) / (ceil - floor)
