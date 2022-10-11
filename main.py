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
# import torchvision.utils as vutils
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
from utils.data_load import BrainDataset

CLASS_MAP = {"CN": 0, "AD": 1}
SEED_VALUE = 82

def parser():
    parser = argparse.ArgumentParser(description="example")
    parser.add_argument("--model", type=str, default="SoftIntroVAE")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--log", type=str, default="output")
    parser.add_argument("--n_train", type=float, default=0.8)
    parser.add_argument("--train_or_loadnet", type=str, default="train")# train or loadnet
    args = parser.parse_args()
    return args


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True #この行をFalseにすると再現性はとれるが、速度が落ちる
    torch.backends.cudnn.deterministic = True
    return
fix_seed(0)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_dataloader(n_train_rate, batch_size):
    data = load_data(kinds=["ADNI2", "ADNI2-2"], classes=["CN", "AD"], unique=False, blacklist=True)

    pids = []
    voxels = np.zeros((len(data), 80, 96, 80))
    labels = np.zeros(len(data))
    for i in tqdm(range(len(data))):
        pids.append(data[i]["pid"])
        voxels[i] = data[i]["voxel"]
        labels[i] = CLASS_MAP[data[i]["label"]]
    pids = np.array(pids)

    gss = GroupShuffleSplit(test_size=0.2, random_state=42)
    tid, vid = list(gss.split(voxels, groups=pids))[0]
    train_voxels = voxels[tid]
    val_voxels = voxels[vid]
    train_labels = labels[tid]
    val_labels = labels[vid]

    train_dataset = BrainDataset(train_voxels, train_labels)
    val_dataset = BrainDataset(val_voxels, val_labels)

    g = torch.Generator()
    g.manual_seed(0)
    batch_size = 16
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=os.cpu_count(),
                                  pin_memory=True, shuffle=True, worker_init_fn=seed_worker, generator=g)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=os.cpu_count(),
                                pin_memory=True, shuffle=False, worker_init_fn=seed_worker, generator=g)

    #train_datadict, val_datadict = train_test_split(dataset, test_size=1-n_train_rate, shuffle=True, random_state=SEED_VALUE)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True, shuffle=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True, shuffle=False)

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
    elif args.model == "SoftIntroVAE":
        net = models.SoftIntroVAE(12, [[12,1,2],[24,1,2],[32,2,2],[48,2,2]])
        log_path = "./logs/" + args.log + "_SoftIntroVAE/"
        print("net: SoftIntroVAE") # ------------------------------------- #


    os.makedirs(log_path, exist_ok=True)
    os.makedirs(log_path + "csv/", exist_ok=True)
    # save args
    with open(log_path + "my_args.txt", "w") as f:
        f.write("{}".format(args))


    os.environ["CUDA_VISIBLE_DEVICES"] = "5"   #  os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"
    device = torch.device("cuda" if torch.cuda.is_available() and True else "cpu")
    print("device:", device)


#   ここで データをロードする .
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
        elif args.model == "SoftIntroVAE":
            train_lossE, train_lossD, val_lossE, val_lossD = trainer.train_soft_intro_vae(net, train_loader, val_loader, args.epoch, args.lr, device, log_path)
            torch.save(net.state_dict(), log_path + "soft_intro_vae_weight.pth")
            print("saved net weight!")
            train_result.result_ae(train_lossE, train_lossD, val_lossE, val_lossD, log_path)


if __name__ == "__main__":
    main()
