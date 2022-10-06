import argparse
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
# pytorch
import torch.nn as nn
import torch.optim as optim
# import os.path as osp
import torchio as tio
import torchvision.utils as vutils
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader, Dataset
from torchio.transforms.augmentation.intensity.random_bias_field import \
    RandomBiasField
from torchio.transforms.augmentation.intensity.random_noise import RandomNoise
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

import models.models as models
import utils.confusion as confusion
import utils.my_trainer as trainer
import utils.train_result as train_result
from datasets.dataset import load_data
from utils.data_class import BrainDataset

CLASS_MAP = {"CN": 0, "AD": 1}
SEED_VALUE = 82


def parser():
    parser = argparse.ArgumentParser(description="example")
    parser.add_argument("--model", type=str, default="VAE")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--log", type=str, default="output")
    parser.add_argument("--n_train", type=float, default=0.8)
    parser.add_argument("--train_or_loadnet", type=str, default="train")# train or loadnet
    args = parser.parse_args()
    return args


# TorchIO
class ImageTransformio():
    def __init__(self):
        self.transform = {
            "train": tio.Compose([
                tio.transforms.RandomAffine(scales=(0.9, 1.2), degrees=10, isotropic=True,
                                 center="image", default_pad_value="mean", image_interpolation='linear'),
                # tio.transforms.RandomNoise(),
                tio.transforms.RandomBiasField(),
                # tio.ZNormalization(),
                tio.transforms.RescaleIntensity((0, 1))
            ]),
            "val": tio.Compose([
                # tio.ZNormalization(),
                # tio.RescaleIntensity((0, 1))  # , in_min_max=(0.1, 255)),
            ])
        }

    def __call__(self, img, phase="train"):
        img_t = torch.tensor(img)
        return self.transform[phase](img_t)


def load_dataloader(n_train_rate, batch_size):
    data = load_data(kinds=["ADNI2", "ADNI2-2"], classes=["CN", "AD"], unique=False, blacklist=True)
  # data = load_data(kinds=["ADNI2","ADNI2-2"], classes=["CN", "AD"], unique=True, blacklist=True)
  # data = dataset.load_data(kinds=kinds,classes=classes,unique=False)
    pids = []
    for i in range(len(data)):
        pids.append(data[i]["pid"])
    gss = GroupShuffleSplit(test_size=1 - n_train_rate, random_state=SEED_VALUE)
    train_idx, val_idx = list(gss.split(data, groups=pids))[0]
    train_data = data[train_idx]
    val_data = data[val_idx]

    #train_datadict, val_datadict = train_test_split(dataset, test_size=1-n_train_rate, shuffle=True, random_state=SEED_VALUE)
    transform = ImageTransformio()
    # transform = None
    train_dataset = BrainDataset(data_dict=train_data, transform=transform, phase="train")
    val_dataset = BrainDataset(data_dict=val_data, transform=transform, phase="val")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True, shuffle=False)

    return train_dataloader, val_dataloader


def main():
    # randam.seed(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    torch.manual_seed(SEED_VALUE)

    args = parser()

    if args.model == "ResNetCAE":
        net = models.ResNetCAE(12, [[12,1,2],[24,1,2],[32,2,2],[48,2,2]]) # ここでmodelの block 内容指定
        log_path = "./logs/" + args.log + "_ResNetCAE/"
        print("net: ResNetCAE") # ------------------------------------- #
    elif args.model == "ResNetVAE":
        net = models.ResNetVAE(12, [[12,1,2],[24,1,2],[32,2,2],[48,2,2]])
        log_path = "./logs/" + args.log + "_ResNetVAE/"
        print("net: ResNetVAE") # ------------------------------------- #
    elif args.model == "softintroVAE":
        net = models.SoftIntroVAE(12, [[12,1,2],[24,1,2],[32,2,2],[48,2,2]])
        log_path = "./logs/" + args.log + "_SoftIntroVAE/"
        print("net: SoftIntroVAE") # ------------------------------------- #


    os.makedirs(log_path, exist_ok=True)
    os.makedirs(log_path + "csv/", exist_ok=True)
    # save args
    with open(log_path + "my_args.txt", "w") as f:
        f.write("{}".format(args))



#   os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6, 7",   #  os.environ["CUDA_VISIBLE_DEVICES"]="6"
    device = torch.device("cuda" if torch.cuda.is_available() and True else "cpu")
    print("device:", device)

    train_loader, val_loader = load_dataloader(args.n_train, args.batch_size)
    # loadnet or train
    if args.train_or_loadnet == "loadnet":
        net.load_state_dict(torch.load(log_path + 'weight.pth'))
        # とりあえずvalidationで確認 テストデータあとで作る
        confusion.make_confusion_matrix(
            net, val_loader, CLASS_MAP, device, log_path)

    elif args.train_or_loadnet == "train":

        if args.model == "ResNetCAE":
            train_loss, val_loss = trainer.train_ResNetCAE(net, train_loader, val_loader, args.epoch, args.lr, device, log_path)
            torch.save(net.state_dict(), log_path + "resnetcae_weight.pth")
            print("saved net weight!")
            train_result.result_ae(train_loss, val_loss, log_path)

        elif args.model == "ResNetVAE":
            train_loss, val_loss = trainer.train_ResNetVAE(net, train_loader, val_loader, args.epoch, args.lr, device, log_path)
            torch.save(net.state_dict(), log_path + "resnetvae_weight.pth")
            print("saved net weight!")
            train_result.result_ae(train_loss, val_loss, log_path)
            #ここの result_ae は result_AutoEncoder
        elif args.model == "softintroVAE":
            train_loss, val_loss = trainer.train_soft_intro_vae(net, train_loader, val_loader, args.epoch, args.lr, device, log_path)
            torch.save(net.state_dict(), log_path + "soft_intro_vae_weight.pth")
            print("saved net weight!")
            train_result.result_ae(train_loss, val_loss, log_path)



if __name__ == "__main__":
    main()
