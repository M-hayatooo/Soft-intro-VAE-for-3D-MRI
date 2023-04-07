import argparse
import csv
import os
import random
import time

import matplotlib.pyplot as plt
import models.vaemodel as models
import numpy as np
import torch
# pytorch
import torch.nn as nn
import torch.optim as optim
# import os.path as osp
import torchio as tio
import utils.confusion as confusion
import utils.my_trainer as trainer
import utils.train_result as train_result
from datasets.dataset import load_data
# import torchvision.utils as vutils
from sklearn.model_selection import (GroupShuffleSplit, StratifiedGroupKFold,
                                     train_test_split)
from torch.utils.data import DataLoader, Dataset
from torchio.transforms.augmentation.intensity.random_bias_field import \
    RandomBiasField
from torchio.transforms.augmentation.intensity.random_noise import RandomNoise
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from utils.data_load import BrainDataset

#"CN", "AD", "EMCI", "LMCI", "SMC", "MCI"
CLASS_MAP = {"CN": 0, "AD": 1, "EMCI":2, "LMCI":3, "SMC":4, "MCI":5}
SEED_VALUE = 82


def parser():
    parser = argparse.ArgumentParser(description="example")
    parser.add_argument("--model", type=str, default="ResNetVAE")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=400)
    # parser.add_argument("--Softepoch", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--log", type=str, default="output")
    parser.add_argument("--n_train", type=float, default=0.8)
    parser.add_argument("--train_or_loadnet", type=str, default="train") # train or loadnet
    parser.add_argument("--conv_model", type=str, default="(12, [[12,1,2],[24,1,2],[32,2,2],[48,2,2]])")
    # parser.add_argument("--conv_model", type=str, default="(32, [[32,1,2],[64,1,2],[128,2,2]])")
    # parser.add_argument("--conv_to_latent", type=str, default="Fully Connectedlayer")
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--upsample_mode", type=str, default="Nearest")
    # parser.add_argument("--rec:kl_weight", type=str, default="rec:kl=1:1")
    parser.add_argument("--mse_weight", type=float, default=1.0)
    parser.add_argument("--kl_weight", type=int, default=1)
    parser.add_argument("--activation_func", type=str, default="ReLU")
    parser.add_argument("--noise_mean", type=float, default=0.03)
    # parser.add_argument("--noise_std", type=float, default=0.08)
    parser.add_argument("--noise_std", type=str, default="(0.03, 0.03)")
    # parser.add_argument("--last_path", type=str, default="nonaug_003_003003_kl1/")
    parser.add_argument("--last_path", type=str, default="nonaug_rec1_kl1/")

    args = parser.parse_args()
    return args


seed_ti = 103

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True #この行をFalseにすると再現性はとれるが、速度が落ちる
    torch.backends.cudnn.deterministic = True
    return


fix_seed(seed_ti)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# TorchIO
class ImageTransformio():
    def __init__(self):
        self.spatial_transforms = {
            tio.transforms.RandomNoise(mean=0.0, std=(0, 0.1)): 0.5,
            tio.transforms.RandomNoise(mean=0.0, std=(0, 0.2)): 0.5,
        }
        self.transform = {
            "train": tio.Compose([
                tio.OneOf(self.spatial_transforms, p=0.5),
            ]),
            "val": tio.Compose([
                # ========================================
            ])
        }

    def __call__(self, img, phase="train"):
        img_t = torch.tensor(img)
        return self.transform[phase](img_t)
    
# load_dataloader(train:test, batch size, noise_mean_value, noise_std_value)
def load_dataloader(n_train_rate, batch_size, noise_mean, noise_std):
    data = load_data(kinds=["ADNI2", "ADNI2-2"], classes=["CN", "AD", "EMCI", "LMCI", "SMC", "MCI"], unique=False, blacklist=True)

    pids = []
    voxels = np.zeros((len(data), 80, 96, 80))
    labels = np.zeros(len(data))
    for i in tqdm(range(len(data))):
        pids.append(data[i]["pid"])
        voxels[i] = data[i]["voxel"]
        labels[i] = CLASS_MAP[data[i]["label"]]
    pids = np.array(pids)

    split_index = 4   #   split index  を指定
    sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed_ti)
    tid, vid = list(sgk.split(voxels, y=labels, groups=pids))[split_index]
    # tid, vid = list(gss.split(voxels, groups=pids))[0]

    train_voxels = voxels[tid]
    val_voxels = voxels[vid]
    train_labels = labels[tid]
    val_labels = labels[vid]

    # transform = ImageTransformio()
    spatial_transforms = { # stdは0.1くらいでやってみる？それと0.3
        # tio.transforms.RandomNoise(mean=noise_mean, std=noise_std): 1.0,
        tio.transforms.RandomNoise(mean=0.03, std=(0.03, 0.03) ): 1.0,
        # tio.transforms.RandomAffine(degrees=35): 0.35,
        # tio.transforms.RandomAffine(degrees=25): 0.4,
        # tio.transforms.RandomAffine(degrees=10): 0.35,
    }
    transform = tio.Compose([
        tio.OneOf(spatial_transforms, p=0.5),
    ])
    
    train_dataset = BrainDataset(train_voxels, train_labels, transform=transform, phase="train")
    val_dataset = BrainDataset(val_voxels, val_labels, transform=None, phase="val")

    g = torch.Generator()
    g.manual_seed(seed_ti)

    print(f"batch size:{batch_size}")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4,
                                  pin_memory=True, shuffle=True, worker_init_fn=seed_worker, generator=g)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4,
                                pin_memory=True, shuffle=False, worker_init_fn=seed_worker, generator=g)

    # train_datadict, val_datadict = train_test_split(dataset, test_size=1-n_train_rate, shuffle=True, random_state=SEED_VALUE)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True, shuffle=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True, shuffle=False)

    return train_dataloader, val_dataloader


def write_csv(epoch, train_loss, val_loss, path):
    with open(path, "a") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_loss, val_loss])


def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "6"   #  os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"
    device = torch.device("cuda:0" if torch.cuda.is_available() and True else "cpu")
    print("device:", device)
    # randam.seed(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    torch.manual_seed(SEED_VALUE)

    args = parser()
    ##############################################
    print(f"destination file path = {args.last_path}")
    ##############################################

    if args.model == "ResNetVAE":
        net = models.ResNetVAE(12, [[12,1,2],[24,1,2],[32,2,2],[48,2,2]])
        # net = models.ResNetVAE(12, [[12,1,2],[24,1,2],[32,2,2]])
        log_path = "./logs/" + args.log + "_ResNetVAE/" + args.last_path
        # log_path = "./logs/" + args.log + "_ResNetVAE/data-augment-004-008/"
        print("net: ResNetVAE") # ------------------------------------- #


    os.makedirs(log_path, exist_ok=True)
    os.makedirs(log_path + "csv/", exist_ok=True)
    # save args
    with open(log_path + "my_args.txt", "w") as f:
        f.write("{}".format(args))


#   ここで データをロードする. load_dataloader( train:test, batch size,  noise_mean_value, noise_std_value)
    train_loader, val_loader = load_dataloader(args.n_train, args.batch_size, args.noise_mean, args.noise_std)
    # loadnet or train
    if args.train_or_loadnet == "loadnet":
        net.load_state_dict(torch.load(log_path + 'weight.pth'))
        # とりあえずvalidationで確認 テストデータあとで作る
        confusion.make_confusion_matrix(
            net, val_loader, CLASS_MAP, device, log_path)

    elif args.train_or_loadnet == "train":
        if args.model == "ResNetVAE":
            train_loss, val_loss = trainer.train_ResNetVAE(net, train_loader, val_loader, args.epoch, args.lr, args.mse_weight, args.kl_weight, device, log_path)
            torch.save(net.to('cpu').state_dict(), log_path + "resnetvae_weight.pth")
            print("saved ResNetVAE net weight! ")
            train_result.result_ae(train_loss, val_loss, log_path)
            #write_csv(args.epoch, train_loss, val_loss, log_path)
            #ここの result_ae は result_AutoEncoder


if __name__ == "__main__":
    main()
