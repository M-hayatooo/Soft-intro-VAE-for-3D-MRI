import argparse
import csv
import os
import random
import time

import matplotlib.pyplot as plt
import models.mymodel as models
import numpy as np
import torch
# pytorch
import torch.nn as nn
import torch.optim as optim
# import os.path as osp
import torchio as tio
import utils.confusion as confusion
import utils.train_result as train_result
import utils.trainer_fc as trainer
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
    parser.add_argument("--model", type=str, default="SoftIntroVAE")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=1500)
    parser.add_argument("--Softepoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--log", type=str, default="latent150")
    parser.add_argument("--n_train", type=float, default=0.8)
    parser.add_argument("--train_or_loadnet", type=str, default="train")# train or loadnet
    parser.add_argument("--beta_rec", type=float, default=1.0)
    parser.add_argument("--beta_neg", type=float, default=1024.0)
    parser.add_argument("--beta_kl", type=float, default=0.3)
    parser.add_argument("--gamma_r", type=float, default=1e-8)
    parser.add_argument("--normalizing_value", type=str, default="8/80*96*80")
    # parser.add_argument("--conv", type=str, default="16-32-64-128")
    parser.add_argument("--conv", type=str, default="8-16-32-64")
    parser.add_argument("--optimizer", type=str, default="Adam")
    # parser.add_argument("--optimizer", type=str, default="RAdam")
    # parser.add_argument("--conv", type=str, default="32-64-128-256")
    parser.add_argument("--upsample_mode", type=str, default="Nearest")
    # parser.add_argument("--upsample_mode", type=str, default="trilinear_align-True")
    
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


def load_dataloader(n_train_rate, batch_size):
    data = load_data(kinds=["ADNI2", "ADNI2-2"], classes=["CN", "AD", "EMCI", "LMCI", "SMC", "MCI"], unique=False, blacklist=False)

    pids = []
    voxels = np.zeros((len(data), 80, 96, 80))
    labels = np.zeros(len(data))
    for i in tqdm(range(len(data))):
        pids.append(data[i]["pid"])
        voxels[i] = data[i]["voxel"]
        labels[i] = CLASS_MAP[data[i]["label"]]
    pids = np.array(pids)

#   split index  を指定
    split_index = 4
    sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed_ti)
    tid, vid = list(sgk.split(voxels, y=labels, groups=pids))[split_index]
#    tid, vid = list(gss.split(voxels, groups=pids))[0]

    train_voxels = voxels[tid]
    val_voxels = voxels[vid]
    train_labels = labels[tid]
    val_labels = labels[vid]

    train_dataset = BrainDataset(train_voxels, train_labels)
    val_dataset = BrainDataset(val_voxels, val_labels)

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
    #   os.environ["CUDA_VISIBLE_DEVICES"] = "6"   #  os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"
    device = torch.device("cuda:6" if torch.cuda.is_available() and True else "cpu")
    print("device:", device)

    # randam.seed(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    torch.manual_seed(SEED_VALUE)

    args = parser()
    # conv 12 → 24 → 32   4回poolingで 80*96*80 → 5*6*5に
    # conv 24 → 48 → 64  ##  32 → 64 → 128   ##  64 → 128 → 256
    if args.model == "SoftIntroVAE": #                  # チャネル変化
        net = models.SoftIntroVAE(8, 16, 32, 64, 150) # 16 → 32 → 64 → 128;  z=600
        #net = models.SoftIntroVAE(32, 64, 128, 256, 600) # 32 → 64 → 128 → 256;  z=600
        log_path = "./logs/" + args.log + "_SoftIntroVAE/8per_rec1_ng1024_kl03_ch64_el05/"
        print("net: SoftIntroVAE")
        # ------------------------------------- #

    elif args.model == "ResNetCAE":
        net = models.ResNetCAE(12, [[12,1,2],[24,1,2],[32,2,2]]) # ここでmodelの block 構造指定
        log_path = "./logs/" + args.log + "_ResNetCAE/"
        print("net: ResNetCAE") # ------------------------------------- #
    elif args.model == "ResNetVAE":
        # net = models.ResNetVAE(12, [[12,1,2],[24,1,2],[32,2,2],[48,2,2]])
        net = models.ResNetVAE(32, [[32,1,2],[64,1,2],[128,2,2]])
        # net = models.ResNetVAE(64, [[64,1,2],[128,1,2],[256,2,2]])
        log_path = "./logs/" + args.log + "_ResNetVAE/ch32_64_128_kl30_b32/"
        print("net: ResNetVAE") # ------------------------------------- #
    elif args.model == "VAEtoSoftVAE":
        resnet = models.ResNetVAE(12, [[12,1,2],[24,1,2],[32,2,2]])
        net = models.SoftIntroVAE(12, [[12,1,2],[24,1,2],[32,2,2]])
        log_path = "./logs/" + args.log + "_VAEtoSoftVAE_fixed_seed/only_VAE1/"
        print("net: VAE to SoftVAE") # ------------------------------------- #


    os.makedirs(log_path, exist_ok=True)
    os.makedirs(log_path + "csv/", exist_ok=True)
    # save args
    with open(log_path + "my_args.txt", "w") as f:
        f.write("{}".format(args))


#   ここでデータをロードする.
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
            print("saved ResNetCAE net weight!")
            train_result.result_ae(train_loss, val_loss, log_path)
        elif args.model == "ResNetVAE":
            train_loss, val_loss = trainer.train_ResNetVAE(net, train_loader, val_loader, args.epoch, args.lr, device, log_path)
            torch.save(net.state_dict(), log_path + "resnetvae_weight.pth")
            print("saved ResNetVAE net weight! ")
            train_result.result_ae(train_loss, val_loss, log_path)
            #write_csv(args.epoch, train_loss, val_loss, log_path)
            #ここの result_ae は result_AutoEncoder
        elif args.model == "SoftIntroVAE":
            # pretrained_path = log_path + "S-IntroVAE_4184_epoch399.pth"
            train_lossE, train_lossD, val_lossE, val_lossD = trainer.train_soft_intro_vae(
                net, train_loader, val_loader, args.epoch, args.lr, device, log_path, args.beta_rec, args.beta_neg, args.beta_kl,
            )
            torch.save(net.state_dict(), log_path + "last_s_introVAE_weight.pth")
            print("saved S-IntroVAE net weight!")
            train_result.result_S_IntroVAE(train_lossE, train_lossD, val_lossE, val_lossD, log_path)

        elif args.model == "VAEtoSoftVAE":
            train_loss, val_loss = trainer.train_ResNetVAE(resnet, train_loader, val_loader, args.epoch, args.lr, device, log_path)
            pretrained_path = log_path + "resnetvae_weight.pth"
            train_lossE, train_lossD, val_lossE, val_lossD = trainer.train_soft_intro_vae(net, train_loader, val_loader, args.Softepoch, args.lr, device, log_path, pretrained_path)
            torch.save(net.state_dict(), log_path + "soft_intro_vae_weight.pth")
            print("saved S-IntroVAE net weight!")
            train_result.result_S_IntroVAE(train_lossE, train_lossD, val_lossE, val_lossD, log_path)
#            train_result.result_ae(train_lossE, train_lossD, val_lossE, val_lossD, log_path)


if __name__ == "__main__":
    main()
