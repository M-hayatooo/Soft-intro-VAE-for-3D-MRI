import torch
from torch.utils.data import Dataset
import numpy as np
from datasets.dataset import CLASS_MAP


class BrainDataset(Dataset):

    def __init__(self, data_dict, transform=None, phase="train", class_map=CLASS_MAP):
        self.data = data_dict
        self.voxels = [self._preprocess(data["voxel"]) for data in self.data]
        self.phase = phase
        self.class_map = class_map
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        voxel = self.voxels[index]
        label = self.class_map[self.data[index]["label"]]
        if self.transform:
            voxel = self.transform(voxel, self.phase)
        return voxel, label

    def _preprocess(self, voxel):
        cut_range = 4
        # voxel = voxel[:, 8:88, :]
        voxel = np.clip(voxel, 0, cut_range * np.std(voxel))
        voxel = normalize(voxel, np.min(voxel), np.max(voxel))
        voxel = voxel[np.newaxis, ]
        return voxel.astype('f')

    def __call__(self, index):
        return self.__getitem__(index)


def normalize(voxel: np.ndarray, floor: int, ceil: int) -> np.ndarray:
    return (voxel - floor) / (ceil - floor)


class BrainData(Dataset):
    def __init__(self, data, transform=None, class_map=CLASS_MAP):
        """
        data: dataset.py の load_data() で手に入るもの
        transform: あとでやる(画像処理する関数?)
        class_map: labelを数字に変換
        """
        self.data = data
        self.class_map = class_map
        self.transform = transform

        if self.transform:
            for sample in data:
                sample = transform(sample)
        # voxelを4次元にする
        for sample in data:
            sample["voxel"] = torch.unsqueeze(torch.tensor(sample["voxel"]), 0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        voxel = self.data[idx]["voxel"]
        label = self.class_map[self.data[idx]["label"]]
        sample = {"voxel": voxel, "label": label}

        return sample