import argparse
import os

import numpy as np
import torch
import torchio as tio
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader
from torchio.transforms.augmentation.intensity.random_bias_field import \
    RandomBiasField
from torchio.transforms.augmentation.intensity.random_noise import RandomNoise
from torchvision import transforms

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
    # CNN or CAE or VAE
    parser.add_argument("--model", type=str, default="ResNetVAE")
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

    if args.model == "CNN":
        net = models.FujiNet1()
        log_path = "./logs/" + args.log + "_cnn-IIP1-drop+DA3/"
        print("net: CNN") # ------------------------------------- #
    elif args.model == "CAE":
        net = models.Cae()
        #net = models.CAE3()
        log_path = "./logs/" + args.log + "_cae/"
        print("net: CAE") # ------------------------------------- #
    elif args.model == "Caee":
        net = models.Caee() ### """ loadnet """
        log_path = "./logs/" + args.log + "_caee/"
        print("net: Caee")
    elif args.model == "Cae_avgpool":
        net = models.Cae() #### ------------------------------------- #
        log_path = "./logs/" + args.log + "_cae_avgpool/"
        print("net: Cae_avgpool")
    elif args.model == "Cae_nearest":
        net = models.Cae() #### ------------------------------------- #
        log_path = "./logs/" + args.log + "_cae_nearest/"
        print("net: Cae_nearest")
    elif args.model == "VAE":
        net = models.Vae()
        log_path = "./logs/" + args.log + "_vae/"
        print("net: VAE") # ------------------------------------- #
    elif args.model == "Vaee":
        net = models.Vaee()
        log_path = "./logs/" + args.log + "_vaee/"
        print("net: Vaee") # ------------------------------------- #
    elif args.model == "ResNetCAE":
        net = models.ResNetCAE(12, [[12,1,2],[24,1,2],[32,2,2],[48,2,2]]) # ここでmodelの block 内容指定
        log_path = "./logs/" + args.log + "_ResNetCAE/"
        print("net: ResNetCAE") # ------------------------------------- #
    elif args.model == "ResNetVAE":
        net = models.ResNetVAE(12, [[12,1,2],[24,1,2],[32,2,2],[48,2,2]])
        log_path = "./logs/" + args.log + "_ResNetVAE/"
        print("net: ResNetVAE") # ------------------------------------- #


    os.makedirs(log_path, exist_ok=True)
    os.makedirs(log_path + "csv/", exist_ok=True)
    # save args
    with open(log_path + "my_args.txt", "w") as f:
        f.write("{}".format(args))


#   os.environ["CUDA_VISIBLE_DEVICES"]="6"
#   os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6, 7"
    device = torch.device("cuda" if torch.cuda.is_available() and True else "cpu")
#    device = torch.device("cpu")
    print("device:", device)

    train_loader, val_loader = load_dataloader(args.n_train, args.batch_size)
    # loadnet or train
    if args.train_or_loadnet == "loadnet":
        net.load_state_dict(torch.load(log_path + 'weight.pth'))
        # とりあえずvalidationで確認 テストデータあとで作る
        confusion.make_confusion_matrix(
            net, val_loader, CLASS_MAP, device, log_path)

    elif args.train_or_loadnet == "train":
        # CNN or CAE or VAE
        if args.model == "CNN":
            train_loss, train_acc, val_loss, val_acc = trainer.train(net, train_loader, val_loader, args.epoch, args.lr, device, log_path)
            # torch.save(net.state_dict(), log_path + "weight.pth")
            train_result.result(train_acc, train_loss, val_acc, val_loss, log_path)
            # とりあえずvalidationで確認 テストデータあとで作る
            confusion.make_confusion_matrix(net, val_loader, CLASS_MAP, device, log_path)

        elif args.model == "CAE":
            train_loss, val_loss = trainer.train_cae(net, train_loader, val_loader, args.epoch, args.lr, device, log_path)
            torch.save(net.state_dict(), log_path + "cae_weight.pth")
            print("saved net weight!")
            train_result.result_ae(train_loss, val_loss, log_path)

        elif args.model == "Caee":
            train_loss, val_loss = trainer.train_cae(net, train_loader, val_loader, args.epoch, args.lr, device, log_path)
            torch.save(net.state_dict(), log_path + "caee_weight.pth")
            print("saved net weight!")
            train_result.result_ae(train_loss, val_loss, log_path)

        elif args.model == "Cae_avgpool":
            train_loss, val_loss = trainer.train_cae(net, train_loader, val_loader, args.epoch, args.lr, device, log_path)
            torch.save(net.state_dict(), log_path + "cae_avgpool_weight.pth")
            print("saved net weight!")
            train_result.result_ae(train_loss, val_loss, log_path)

        elif args.model == "Cae_nearest":
            train_loss, val_loss = trainer.train_cae(net, train_loader, val_loader, args.epoch, args.lr, device, log_path)
            torch.save(net.state_dict(), log_path + "cae_nearest_weight.pth")
            print("saved net weight!")
            train_result.result_ae(train_loss, val_loss, log_path)

        elif args.model == "VAE":
            train_loss, val_loss = trainer.train_vae(net, train_loader, val_loader, args.epoch, args.lr, device, log_path)
            torch.save(net.state_dict(), log_path + "vae_weight.pth")
            print("saved net weight!")
            train_result.result_ae(train_loss, val_loss, log_path)

        elif args.model == "Vaee":
            train_loss, val_loss = trainer.train_vae(net, train_loader, val_loader, args.epoch, args.lr, device, log_path)
            torch.save(net.state_dict(), log_path + "vaee_weight.pth")
            print("saved net weight!")
            train_result.result_ae(train_loss, val_loss, log_path)

        elif args.model == "ResNetCAE":
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


if __name__ == "__main__":
    main()
