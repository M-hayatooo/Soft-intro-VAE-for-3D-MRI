import argparse
import os
import random
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from datasets.dataset import CLASS_MAP, load_data
from utils.data_load import BrainDataset


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True #この行をFalseにすると再現性はとれるが、速度が落ちる
    torch.backends.cudnn.deterministic = True
    return


fix_seed(0)

CLASS_MAP = {"CN": 0, "AD": 1, "EMCI":2, "LMCI":3, "SMC":4, "MCI":5}
SEED_VALUE = 0
#data = load_data(kinds=["ADNI2", "ADNI2-2"], classes=["CN", "AD"], unique=False, blacklist=True)
data = load_data(kinds=["ADNI2", "ADNI2-2"], classes=["CN", "AD", "EMCI", "LMCI", "SMC", "MCI"], unique=False, blacklist=True)

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

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)


# ===  batch size definetion  ===
batch_size = 32
# === === === === === === === ===

train_dataset = BrainDataset(train_voxels, train_labels)
val_dataset = BrainDataset(val_voxels, val_labels)
train_dataloader = DataLoader(train_dataset,batch_size=batch_size,num_workers=os.cpu_count(), pin_memory=True,shuffle=True,worker_init_fn=seed_worker,generator=g)
val_dataloader = DataLoader(val_dataset,batch_size=batch_size,num_workers=os.cpu_count(), pin_memory=True,shuffle=False,worker_init_fn=seed_worker,generator=g)


def calc_kl(logvar, mu, mu_o=0.0, logvar_o=0.0, reduce='sum'):
    if not isinstance(mu_o, torch.Tensor):
        mu_o = torch.tensor(mu_o).to(mu.device)
    if not isinstance(logvar_o, torch.Tensor):
        logvar_o = torch.tensor(logvar_o).to(mu.device)
    kl = -0.5 * (1 + logvar - logvar_o - logvar.exp() / torch.exp(logvar_o) - (mu - mu_o).pow(2) / torch.exp(
        logvar_o)).sum(1)
    if reduce == 'sum':
        kl = torch.sum(kl)
    elif reduce == 'mean':
        kl = torch.mean(kl)
    return kl


def reparameterize(mu, logvar):
    device = mu.device
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std).to(device)
    return mu + eps * std


def calc_reconstruction_loss(x, recon_x, loss_type='mse', reduction='mean'):
    x = x.view(x.size(0), -1)
    recon_x = recon_x.view(recon_x.size(0), -1)

    recon_error = F.mse_loss(recon_x, x, reduction='none')
    recon_error = recon_error.sum(1)
    recon_error = recon_error.mean()
    return recon_error


def load_model(model, pretrained, device):
    weights = torch.load(pretrained, map_location=device)
    model.load_state_dict(weights['model'], strict=False)


def save_checkpoint(model, epoch, iteration, prefix=""):
    model_out_path = "./saves/" + prefix + "model_epoch_{}_iter_{}.pth".format(epoch, iteration)
    state = {"epoch": epoch, "model": model.state_dict()}
    if not os.path.exists("./saves/"):
        os.makedirs("./saves/")

    torch.save(state, model_out_path)

    print("model checkpoint saved @ {}".format(model_out_path))


class BuildingBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, bias=False):
        super(BuildingBlock, self).__init__()
        self.res = stride == 1
        self.shortcut = self._shortcut()
        self.relu = nn.ReLU(inplace=True)
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.AvgPool3d(kernel_size=stride),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm3d(out_ch),
        )

    def _shortcut(self):
        return lambda x: x

    def forward(self, x):
        if self.res:
            shortcut = self.shortcut(x)
            return self.relu(self.block(x) + shortcut)
        else:
            return self.relu(self.block(x))

class UpsampleBuildingkBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, bias=False):
        super(UpsampleBuildingkBlock, self).__init__()
        self.res = stride == 1
        self.shortcut = self._shortcut()
        self.relu = nn.ReLU(inplace=True)
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm3d(in_ch),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=stride),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm3d(out_ch),
        )

    def _shortcut(self):
        return lambda x: x

    def forward(self, x):
        if self.res:
            shortcut = self.shortcut(x)
            return self.relu(self.block(x) + shortcut)
        else:
            return self.relu(self.block(x))


class ResNetEncoder(nn.Module):
    def __init__(self, in_ch, block_setting):
        super(ResNetEncoder, self).__init__()
        self.block_setting = block_setting
        self.in_ch = in_ch
        last = 1
        blocks = [nn.Sequential(
            nn.Conv3d(1, in_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(in_ch),
            nn.ReLU(inplace=True),
        )]
        for line in self.block_setting:
            c, n, s = line[0], line[1], line[2]
            for i in range(n):
                stride = s if i == 0 else 1
                blocks.append(nn.Sequential(BuildingBlock(in_ch, c, stride)))
                in_ch = c
        self.inner_ch = in_ch
        self.blocks = nn.Sequential(*blocks)
        self.conv = nn.Conv3d(in_ch, last, kernel_size=1, stride=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(x)
        return self.conv(h)

class ResNetDecoder(nn.Module):
    def __init__(self, encoder: ResNetEncoder, blocks=None):
        super(ResNetDecoder, self).__init__()
        last = encoder.block_setting[-1][0]
        if blocks is None:
            blocks = [nn.Sequential(
                nn.Conv3d(1, last, 1, 1, bias=True),
                nn.BatchNorm3d(last),
                nn.ReLU(inplace=True),
            )]
        in_ch = last
        for i in range(len(encoder.block_setting)):
            if i == len(encoder.block_setting) - 1:
                nc = encoder.in_ch
            else:
                nc = encoder.block_setting[::-1][i + 1][0]
            c, n, s = encoder.block_setting[::-1][i]
            for j in range(n):
                stride = s if j == n - 1 else 1
                c = nc if j == n - 1 else c
                blocks.append(nn.Sequential(UpsampleBuildingkBlock(in_ch, c, stride)))
                in_ch = c
        blocks.append(nn.Sequential(
            nn.Conv3d(in_ch, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
        ))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class BaseEncoder(nn.Module):
    def __init__(self) -> None:
        super(BaseEncoder, self).__init__()
class BaseDecoder(nn.Module):
    def __init__(self) -> None:
        super(BaseDecoder, self).__init__()

class BaseCAE(nn.Module):
    def __init__(self) -> None:
        super(BaseCAE, self).__init__()
        self.encoder = BaseEncoder()
        self.decoder = BaseDecoder()
    def encode(self, x):
        z = self.encoder(x)
        return z
    def decode(self, z):
        out = self.decoder(z)
        return out
    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z)
        return out, z

class BaseVAE(nn.Module):
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()
        self.encoder = BaseEncoder()
        self.decoder = BaseDecoder()
    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar
    def decode(self, vec):
        out = self.decoder(vec)
        return out
    def reparameterize(self, mu, logvar) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, x):
        mu, logvar = self.encode(x)
        vec = self.reparameterize(mu, logvar)
        x_hat = self.decode(vec)
        return x_hat, vec, mu, logvar


class ResNetCAE(BaseCAE):
    def __init__(self, in_ch, block_setting) -> None:
        super(ResNetCAE, self).__init__()
        self.encoder = ResNetEncoder(in_ch=in_ch, block_setting=block_setting)
        self.decoder = ResNetDecoder(self.encoder)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def __call__(self, x):
        x = self.forward(x)
        return x


class VAEResNetEncoder(ResNetEncoder):
    def __init__(self, in_ch, block_setting) -> None:
        super(VAEResNetEncoder, self).__init__(in_ch, block_setting)
        self.mu = nn.Conv3d(self.inner_ch, 1, kernel_size=1, stride=1, bias=True)
        self.var = nn.Conv3d(self.inner_ch, 1, kernel_size=1, stride=1, bias=True)

    def forward(self, x: torch.Tensor):
        h = self.blocks(x)
        mu = self.mu(h)
        var = self.var(h)
        return mu, var


class ResNetVAE(BaseVAE):
    def __init__(self, in_ch, block_setting) -> None:
        super(ResNetVAE, self).__init__()
        self.encoder = VAEResNetEncoder(in_ch=in_ch, block_setting=block_setting)
        self.decoder = ResNetDecoder(self.encoder)

    def reparamenterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparamenterize(mu, logvar)
        x_re = self.decoder(z)
        return x_re, mu, logvar

    def loss(self, x_re, x, mu, logvar):
        re_err = torch.sqrt(torch.mean((x_re - x)**2)) # ==  self.Rmse(x_re, x)
        kld = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
        return re_err + kld


class SoftIntroVAE(nn.Module):
    def __init__(self, in_ch, block_setting, zdim=150, conditional=False):
        super(SoftIntroVAE, self).__init__()
        self.zdim = zdim
        self.conditional = conditional
        self.encoder = VAEResNetEncoder(
            in_ch=in_ch,
            block_setting=block_setting,
        )
        self.decoder = ResNetDecoder(self.encoder)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_re = self.decoder(z)
        return mu, logvar, z, x_re

    def loss(self, x_re, x, mu, logvar):
        re_err = torch.sqrt(torch.mean((x_re - x)**2)) # ==  self.Rmse(x_re, x)
        kld = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
        return re_err + kld

    def sample(self, z, y_cond=None):
        # x.view(-1, 2)
        y = self.decode(z, y_cond=y_cond)
        return y

    def sample_with_noise(self, num_samples=1, device=torch.device("cpu"), y_cond=None):
        z = torch.randn(num_samples, self.z_dim).to(device)
        return self.decode(z, y_cond=y_cond)

    def encode(self, x, o_cond=None):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def decode(self, z, y_cond=None):
        y = self.decoder(z)
        return y


def train_soft_intro_vae(lr_e=2e-4, lr_d=2e-4, batch_size=batch_size, start_epoch=0,
                         num_epochs=500, num_vae=0, save_interval=5000, recon_loss_type="mse",
                         beta_kl=1.0, beta_rec=1.0, beta_neg=1.0, test_iter=1000, seed=-1, pretrained=None,
                         device=torch.device("cpu"), num_row=8, gamma_r=1e-8):
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        print("random seed: ", seed)


    # model = SoftIntroVAE(12, [[12,1,2],[24,1,2],[32,2,2],[48,2,2]], conditional=False)

#    model.to(device)
    # もしpretrainedが存在しているのならば model param load
    if pretrained is not None:
        load_model(model, pretrained, device)
    #print(model)

    optimizer_e = optim.Adam(model.module.encoder.parameters(), lr=lr_e)
    optimizer_d = optim.Adam(model.module.decoder.parameters(), lr=lr_d)

    e_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_e, milestones=(350,), gamma=0.1)
    d_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=(350,), gamma=0.1)

    scale = 1 / (80 * 96 * 80)  # normalizing constant, 's' in the paper  desu

#    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_dataset,batch_size=batch_size,num_workers=os.cpu_count(), pin_memory=True,shuffle=True,worker_init_fn=seed_worker,generator=g)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,num_workers=os.cpu_count(), pin_memory=True,shuffle=False,worker_init_fn=seed_worker,generator=g)

#   train_data_loader = load_data(kinds=["ADNI2","ADNI2-2"], classes=["CN", "AD"], unique=False, blacklist=True)

    start_time = time.time()

#    cur_iter = 0
    kls_real = []
    kls_fake = []
    kls_rec = []
    rec_errs = []

    train_lossE_list, train_lossD_list, val_lossE_list, val_lossD_list = [], [], [], []
    train_lossE, train_lossD, val_lossE, val_lossD = 0.0, 0.0, 0.0, 0.0

# ================= following part is for using DataParallel ==================
    for epoch in range(start_epoch, num_epochs):
        loop_start_time = time.time()
        diff_kls = []

        model.train()
        batch_kls_real = []
        batch_kls_fake = []
        batch_kls_rec = []
        batch_rec_errs = []
        counter = 0
        for batch, labels in train_loader:
            # ======================== train ==========================
            b_size = batch.size(0)

            noise_batch = torch.randn(size=(b_size, 1, 5, 6, 5)).to(device)
            real_batch = batch.to(device)

            # ====================== Update E =========================
            fake = model.module.decode(noise_batch)

            real_mu, real_logvar = model.module.encode(real_batch)
            z = model.module.reparameterize(real_mu, real_logvar)
            rec = model.module.decode(z)

            loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type=recon_loss_type, reduction="mean")
            lossE_real_kl = calc_kl(real_logvar, real_mu, reduce="mean")
            # {{ mu,    logvar,    z   ,    y }}を返す
            rec_mu, rec_logvar, z_rec, rec_rec = model(rec.detach())
            fake_mu, fake_logvar, z_fake, rec_fake = model(fake.detach())

            fake_kl_e = calc_kl(fake_logvar, fake_mu, reduce="none")
            rec_kl_e = calc_kl(rec_logvar, rec_mu, reduce="none")

            loss_fake_rec = calc_reconstruction_loss(fake, rec_fake, loss_type=recon_loss_type, reduction="none")
            loss_rec_rec = calc_reconstruction_loss(rec, rec_rec, loss_type=recon_loss_type, reduction="none")

            exp_elbo_fake = (-2 * scale * (beta_rec * loss_fake_rec + beta_neg * fake_kl_e)).exp().mean()
            exp_elbo_rec = (-2 * scale * (beta_rec * loss_rec_rec + beta_neg * rec_kl_e)).exp().mean()
            # total loss
            lossE = scale * (beta_rec * loss_rec + beta_kl * lossE_real_kl) + 0.25 * (exp_elbo_fake + exp_elbo_rec)
            # backprop
            optimizer_e.zero_grad()
            lossE.backward()
            optimizer_e.step()

            # ========= Update D ==================

            fake = model.module.decode(noise_batch)#
            rec = model.module.decode(z.detach())

            loss_rec = calc_reconstruction_loss(real_batch, rec.detach(),loss_type=recon_loss_type, reduction="mean")

            rec_mu, rec_logvar = model.module.encode(rec)
            z_rec = reparameterize(rec_mu, rec_logvar)

            fake_mu, fake_logvar = model.module.encode(fake)
            z_fake = reparameterize(fake_mu, fake_logvar)

            rec_rec = model.module.decode(z_rec)
            rec_fake = model.module.decode(z_fake)

            loss_rec_rec = calc_reconstruction_loss(rec.detach(), rec_rec, loss_type=recon_loss_type, reduction="mean")
            loss_fake_rec = calc_reconstruction_loss(fake.detach(), rec_fake, loss_type=recon_loss_type, reduction="mean")

            rec_kl = calc_kl(rec_logvar, rec_mu, reduce="mean")
            fake_kl = calc_kl(fake_logvar, fake_mu, reduce="mean")

            lossD = scale * (loss_rec * beta_rec + (rec_kl + fake_kl) * 0.5 * beta_kl
                             + gamma_r * 0.5 * beta_rec * (loss_rec_rec + loss_fake_rec))

            optimizer_d.zero_grad()
            lossD.backward()
            optimizer_d.step()

            if torch.isnan(lossD) or torch.isnan(lossE):
                raise SystemError

            # statistics for plotting later
            diff_kls.append(-lossE_real_kl.data.cpu().item() + fake_kl.data.cpu().item())
            batch_kls_real.append(lossE_real_kl.data.cpu().item())
            batch_kls_fake.append(fake_kl.cpu().item())
            batch_kls_rec.append(rec_kl.data.cpu().item())
            batch_rec_errs.append(loss_rec.data.cpu().item())

            counter += 1


        model.eval()
        with torch.no_grad():
            #_, _, _, rec_det = model(real_batch)
            for batch, labels in val_loader:
                b_size = batch.size(0)
                noise_batch = torch.randn(size=(b_size,1,5,6,5)).to(device)
                real_batch = batch.to(device)
                fake = model.module.decode(noise_batch)

                real_mu, real_logvar = model.module.encode(real_batch)
                z = model.module.reparameterize(real_mu, real_logvar)
                rec = model.module.decode(z)

                loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type=recon_loss_type, reduction="mean")
                lossE_real_kl = calc_kl(real_logvar, real_mu, reduce="mean")

                rec_mu, rec_logvar, z_rec, rec_rec = model(rec.detach())
                fake_mu, fake_logvar, z_fake, rec_fake = model(fake.detach())

                fake_kl_e = calc_kl(fake_logvar, fake_mu, reduce="none")
                rec_kl_e = calc_kl(rec_logvar, rec_mu, reduce="none")

                loss_fake_rec = calc_reconstruction_loss(fake, rec_fake, loss_type=recon_loss_type, reduction="none")
                loss_rec_rec = calc_reconstruction_loss(rec, rec_rec, loss_type=recon_loss_type, reduction="none")

                exp_elbo_fake = (-2 * scale * (beta_rec * loss_fake_rec + beta_neg * fake_kl_e)).exp().mean()
                exp_elbo_rec = (-2 * scale * (beta_rec * loss_rec_rec + beta_neg * rec_kl_e)).exp().mean()

                lossE = scale * (beta_rec * loss_rec + beta_kl * lossE_real_kl) + 0.25 * (exp_elbo_fake + exp_elbo_rec)

                val_lossE += lossE.item()

                #======================== Decoder Part ===========================

                loss_rec = calc_reconstruction_loss(real_batch, rec.detach(),loss_type=recon_loss_type, reduction="mean")

                rec_mu, rec_logvar, z_rec, rec_rec = model(rec.detach())
                fake_mu, fake_logvar, z_fake, rec_fake = model(fake.detach())

                rec_rec = model.module.decode(z_rec)#.detach())
                rec_fake = model.module.decode(z_fake)#.detach())

                loss_rec_rec = calc_reconstruction_loss(rec.detach(), rec_rec,loss_type=recon_loss_type, reduction="mean")
                loss_fake_rec = calc_reconstruction_loss(fake.detach(), rec_fake,loss_type=recon_loss_type, reduction="mean")

                rec_kl = calc_kl(rec_logvar, rec_mu, reduce="mean")
                fake_kl = calc_kl(fake_logvar, fake_mu, reduce="mean")

                lossD = scale * (loss_rec * beta_rec + (rec_kl + fake_kl) * 0.5 * beta_kl + gamma_r * 0.5 * beta_rec * (loss_rec_rec + loss_fake_rec))

                val_lossD += lossD.item()

        train_lossE /= len(train_loader)
        train_lossD /= len(train_loader)
        train_lossE_list.append(train_lossE)
        train_lossD_list.append(train_lossD)
        val_lossE /= len(val_loader)
        val_lossD /= len(val_loader)
        val_lossE_list.append(val_lossE)
        val_lossD_list.append(val_lossD)

        savename = f"Parallel/S-IntroVAE_weight_epoch{epoch}.pth"
    #   torch.save(model.state_dict(), file_path)
        torch.save(model.state_dict(), log_path + savename)
#           torch.save(model.state_dict(), log_path + f"softintroVAE_weight_epoch{str(epoch)}.pth")

        now_time = time.time()
        print(f"Epoch [{epoch+1}/{num_epochs}]  train_lossE:{train_lossE:.3f}  train_lossD:{train_lossD:.3f}  val_lossE:{val_lossE:.3f}  "
              f"val_lossD:{val_lossD:.3f}, 1epoch:{(now_time - loop_start_time):.1f}秒  total time:{(now_time - start_time):.1f}秒")

        _, _, _, rec_det = model(real_batch)

    e_scheduler.step()
    d_scheduler.step()

    if epoch > num_vae - 1:
        kls_real.append(np.mean(batch_kls_real))
        kls_fake.append(np.mean(batch_kls_fake))
        kls_rec.append(np.mean(batch_kls_rec))
        rec_errs.append(np.mean(batch_rec_errs))

    return train_lossE, train_lossD, val_lossE, val_lossD


# hyperparameters
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu") #ここでもしcuda番号指定したら、Data Parallel のcuda開始番号と一致させないとダメ
print("device:", device)
model = SoftIntroVAE(12, [[12,1,2],[24,1,2],[32,2,2],[48,2,2]], conditional=False).to(device)
###
#  ==============   Data Parallel 指定  ================
###
model = torch.nn.DataParallel(model, device_ids=[2, 3])
###
print("Use DataParallel device_ids=2,3,") # ここ手動...
num_epochs = 501
lr = 2e-4
#batch_size = 16  batch size is definited above
beta_kl = 1.0
beta_rec = 1.0
beta_neg = 256
log_path = "logs/output_SoftIntroVAE/"
train_lossE, train_lossD, val_lossE, val_lossD = train_soft_intro_vae(lr_e=2e-4, lr_d=2e-4, batch_size=batch_size, start_epoch=0,
                                                                      num_epochs=num_epochs, num_vae=0, save_interval=5000, recon_loss_type="mse",
                                                                      beta_kl=beta_kl, beta_rec=beta_rec, beta_neg=beta_neg, test_iter=1000, seed=-1, pretrained=None,
                                                                      device=device)
# train soft intro vae の引数の中にpretrainedがあるが、指定すれば呼べる？？？？

torch.save(model.state_dict(), log_path + "Parallel/S-IntroVAE_weight.pth")
print("saved net weight!")
train_result.result_ae(train_lossE, train_lossD, val_lossE, val_lossD, log_path)
