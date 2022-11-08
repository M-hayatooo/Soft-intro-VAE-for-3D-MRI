import csv
import os
import random
import time
from asyncore import loop

import models.lossf as lossf
import models.models as models
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils.train_result as train_result
from sklearn.datasets import make_classification
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from skorch import NeuralNetClassifier
from skorch.callbacks import Callback, Checkpoint, EarlyStopping
from skorch.dataset import CVSplit
# from tune_sklearn import TuneGridSearchCV
from tune_sklearn import TuneSearchCV


def write_csv(epoch, train_loss, val_loss, path):
    with open(path, "a") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_loss, val_loss])


def calc_kl(logvar, mu):
    bsize = mu.size(0)
    mu = mu.view(bsize, -1)
    logvar = logvar.view(bsize, -1)
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)


def reparameterize(mu, logvar):
    device = mu.device
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std).to(device)
    return mu + eps * std


def calc_reconstruction_loss(x, recon_x, reduction='None'):
    bsize = x.size(0)
    x = x.view(bsize, -1)
    out = recon_x.view(bsize, -1)
    if reduction == 'mean':
        # 平均とるよ
        mse = torch.mean(torch.sum(F.mse_loss(x, out, reduction='none'), dim=1), dim=0)
    else:
        # 平均取らないよ
        mse = torch.sum(F.mse_loss(x, out, reduction='none'), dim=1)

    return mse


'''
    # recon_x = recon_x.view(recon_x.size(0), -1)

    # recon_error = F.mse_loss(recon_x, x, reduction='none')
    # recon_error = recon_error.sum(1)
    # recon_error = recon_error.mean()
#    return recon_error
'''


def load_model(model, pretrained, device):
    # weights = torch.load(pretrained, map_location=device)
    model.load_state_dict(torch.load(pretrained, map_location=device), strict=False)


def save_checkpoint(model, epoch, iteration, prefix=""):
    model_out_path = "./saves/" + prefix + f"model_epoch_{epoch}_iter_{iteration}.pth"
    state = {"epoch": epoch, "model": model.state_dict()}
    if not os.path.exists("./saves/"):
        os.makedirs("./saves/")

    torch.save(state, model_out_path)

    print(f"model checkpoint saved @ {model_out_path}")


#  ===============  Soft Intro VAE  function  =================
def train_soft_intro_vae(
    model,
    train_loader,
    val_loader,
    epochs,
    lr=0.001,
    device=torch.device("cpu"),
    path="./output_SoftIntroVAE/",
    pretrained_path=None,
):
    seed = 77

    path2 = path + "train_result.csv"
    with open(path2, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_lossE", "train_lossD", "val_lossE", "val_lossD"])


    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        print("random seed: ", seed)

    #model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model.to(device)
    # もしpretrainedが存在しているのならば model param load
    if pretrained_path is not None:
        load_model(model, pretrained_path, device)
    # print(model)
    # lr_e = lr
    # lr_d = lr
    optimizer_e = optim.Adam(model.encoder.parameters(), lr=2e-4)
    optimizer_d = optim.Adam(model.decoder.parameters(), lr=2e-4)
    e_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_e, milestones=(350,), gamma=0.1)
    d_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=(350,), gamma=0.1)

# ===============================================================================================
#   パラメータ定義  いじるならここかなぁ
    recon_loss_type = "mse"
    beta_rec = 100.0
    beta_neg = 150.0
    beta_kl = 1.0
    gamma_r = 1.0

    scale = 1 / (80 * 96 * 80)  # normalizing constant, 's' in the paper desu

    start_time = time.time()

#    cur_iter = 0
    train_lossE_list, train_lossD_list, val_lossE_list, val_lossD_list = [], [], [], []
    train_lossE, train_lossD, val_lossE, val_lossD = 0.0, 0.0, 0.0, 0.0
    kls_real, kls_fake, kls_rec, rec_errs = [], [], [], []
    print(f"beta kl = {beta_kl},  beta neg = {beta_neg},  beta rec = {beta_rec}")
    start_epoch = 0
    print(f"training epoch:{epochs}")
    for epoch in range(start_epoch, epochs):
        loop_start_time = time.time()
        diff_kls = []
        # save models
        # if epoch % save_interval == 0 and epoch > 0:
        #     save_epoch = (epoch // save_interval) * save_interval
        #     prefix = dataset + "_soft_intro_vae" + "_betas_" + str(beta_kl) + "_" + str(beta_neg) + "_" + str(beta_rec) + "_"
        #     save_checkpoint(model, save_epoch, cur_iter, prefix)

        model.train()
        batch_kls_real = []
        batch_kls_fake = []
        batch_kls_rec = []
        batch_rec_errs = []

        for batch, labels in train_loader:
            #---------------- train ------------------
            b_size = batch.size(0)
            noise_batch = torch.randn(size=(b_size, 1, 5, 6, 5)).to(device)
            real_batch = batch.to(device)
            # ============== Update E ================
            fake = model.decode(noise_batch)

            real_mu, real_logvar = model.encode(real_batch)
            z = model.reparameterize(real_mu, real_logvar)
            rec = model.decode(z)

            loss_rec = calc_reconstruction_loss(real_batch, rec, reduction="mean")
            lossE_real_kl = calc_kl(real_logvar, real_mu)
            # {{ mu,    logvar,   z  ,    y }}を返す
            rec_mu, rec_logvar, z_rec, rec_rec = model(rec.detach())
            fake_mu, fake_logvar, z_fake, rec_fake = model(fake.detach())

            fake_kl_e = calc_kl(fake_logvar, fake_mu)
            rec_kl_e  = calc_kl(rec_logvar, rec_mu)

            loss_fake_rec = calc_reconstruction_loss(fake, rec_fake, reduction="none")
            loss_rec_rec  = calc_reconstruction_loss(rec,  rec_rec,  reduction="none")

            exp_elbo_fake = (-2 * scale * (beta_rec*loss_fake_rec + beta_neg*fake_kl_e)).exp().mean()
            exp_elbo_rec  = (-2 * scale * (beta_rec*loss_rec_rec  + beta_neg*rec_kl_e)).exp().mean()
            # ===== encoder total loss =====
            lossE = scale * (beta_rec*loss_rec + beta_kl*lossE_real_kl) + 0.25*(exp_elbo_fake+exp_elbo_rec)
            # backprop part of encoder
            optimizer_e.zero_grad()
            lossE.backward()
            optimizer_e.step()
            train_lossE += lossE.item()

            # ================ Update D ==================
            for param in model.encoder.parameters():
                param.requires_grad = False
            for param in model.decoder.parameters():
                param.requires_grad = True

            fake = model.decode(noise_batch)
            rec  = model.decode(z.detach())

            loss_rec = calc_reconstruction_loss(real_batch, rec.detach(), loss_type=recon_loss_type, reduction="mean")

            rec_mu, rec_logvar = model.encode(rec)
            z_rec = reparameterize(rec_mu, rec_logvar)

            fake_mu, fake_logvar = model.encode(fake)
            z_fake = reparameterize(fake_mu, fake_logvar)

            rec_rec  = model.decode(z_rec)
            rec_fake = model.decode(z_fake)

            loss_rec_rec  = calc_reconstruction_loss(rec.detach(), rec_rec, loss_type=recon_loss_type, reduction="mean")
            loss_fake_rec = calc_reconstruction_loss(fake.detach(), rec_fake, loss_type=recon_loss_type, reduction="mean")

            rec_kl  = calc_kl(rec_logvar, rec_mu)
            fake_kl = calc_kl(fake_logvar, fake_mu)

            lossD = scale * (loss_rec*beta_rec + (rec_kl+fake_kl)*0.5*beta_kl + gamma_r*0.5*beta_rec*(loss_rec_rec+loss_fake_rec))

            optimizer_d.zero_grad()
            lossD.backward()
            optimizer_d.step()
            train_lossD += lossD.item()

            if torch.isnan(lossD) or torch.isnan(lossE):
                raise SystemError

            # statistics for plotting later
            diff_kls.append(-lossE_real_kl.data.cpu().item() + fake_kl.data.cpu().item())
            batch_kls_real.append(lossE_real_kl.data.cpu().item())
            batch_kls_fake.append(fake_kl.cpu().item())
            batch_kls_rec.append(rec_kl.data.cpu().item())
            batch_rec_errs.append(loss_rec.data.cpu().item())
        # print(f"value of train_loader.dataset:{len(train_loader.dataset)}")
        # print(f"value of train_loader:{len(train_loader)}")

        train_lossE /= len(train_loader)  # train_lossE /= len(train_loader.dataset)
        train_lossD /= len(train_loader)
        train_lossE_list.append(train_lossE)
        train_lossD_list.append(train_lossD)


        info  = f"Epoch[{epoch}]({epochs}/{len(train_loader)})  "
        info += f'Rec:{loss_rec.data.cpu():.4f}, '
        info += f'Kl_E:{lossE_real_kl.data.cpu():.4f}, expELBO_R:{exp_elbo_rec.data.cpu():.4e}, expELBO_F:{exp_elbo_fake.cpu():.4e}, '
        info += f'Kl_F:{rec_kl.data.cpu():.4f}, KL_R:{ fake_kl.data.cpu():.4f}, '
        info += f'DIFF_Kl_F:{(-lossE_real_kl.data.cpu() + fake_kl.data.cpu()):.4f}'
        print(info)

        #_, _, _, rec_det = model(real_batch)

        model.eval()
        with torch.no_grad():
            for batch, labels in train_loader:
                #####
                b_size = batch.size(0)
                noise_batch = torch.randn(size=(b_size, 1, 5, 6, 5)).to(device)
                real_batch = batch.to(device)
                # ============================ Encoder ===============================
                fake = model.decode(noise_batch)
                real_mu, real_logvar = model.encode(real_batch)
                z = model.reparameterize(real_mu, real_logvar)
                rec = model.decode(z)

                loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type=recon_loss_type, reduction="mean")
                lossE_real_kl = calc_kl(real_logvar, real_mu)

                rec_mu,   rec_logvar,  z_rec, rec_rec  = model(rec.detach())
                fake_mu, fake_logvar, z_fake, rec_fake = model(fake.detach())

                fake_kl_e = calc_kl(fake_logvar, fake_mu)
                rec_kl_e  = calc_kl(rec_logvar, rec_mu)

                loss_fake_rec = calc_reconstruction_loss(fake, rec_fake, loss_type=recon_loss_type, reduction="none")
                loss_rec_rec  = calc_reconstruction_loss(rec, rec_rec, loss_type=recon_loss_type, reduction="none")

                exp_elbo_fake = (-2 * scale * (beta_rec * loss_fake_rec + beta_neg * fake_kl_e)).exp().mean()
                exp_elbo_rec  = (-2 * scale * (beta_rec * loss_rec_rec + beta_neg * rec_kl_e)).exp().mean()
                # total loss
                lossE = scale * (beta_rec*loss_rec + beta_kl*lossE_real_kl) + 0.25*(exp_elbo_fake+exp_elbo_rec)
                # backprop
                # ============================ Decoder ==============================
                # fake = model.decode(noise_batch)
                # rec = model.decode(z.detach())
                loss_rec = calc_reconstruction_loss(real_batch, rec.detach(),loss_type=recon_loss_type, reduction="mean")

                rec_mu, rec_logvar = model.encode(rec)
                z_rec = reparameterize(rec_mu, rec_logvar)

                fake_mu, fake_logvar = model.encode(fake)
                z_fake = reparameterize(fake_mu, fake_logvar)

                rec_rec  = model.decode(z_rec)
                rec_fake = model.decode(z_fake)

                loss_rec_rec  = calc_reconstruction_loss(rec.detach(),   rec_rec, loss_type=recon_loss_type, reduction="mean")
                loss_fake_rec = calc_reconstruction_loss(fake.detach(), rec_fake, loss_type=recon_loss_type, reduction="mean")

                rec_kl  = calc_kl(rec_logvar, rec_mu)
                fake_kl = calc_kl(fake_logvar, fake_mu)

                lossD = scale * (loss_rec*beta_rec + (rec_kl+fake_kl)*0.5*beta_kl + gamma_r*0.5*beta_rec*(loss_rec_rec+loss_fake_rec))
                val_lossD += lossD.item()


            val_lossE /= len(val_loader)
            val_lossD /= len(val_loader)
            val_lossE_list.append(val_lossE)
            val_lossD_list.append(val_lossD)

        #max_imgs = min(batch.size(0), 16)
        # vutils.save_image(
        #         torch.cat([real_batch[:max_imgs], rec_det[:max_imgs], fake[:max_imgs]], dim=0).data.cpu(),
        #         '{}/image_{}.jpg'.format("./", cur_iter), nrow=num_row)
        #cur_iter += 1
        kls_real.append(np.mean(batch_kls_real))
        kls_fake.append(np.mean(batch_kls_fake))
        kls_rec.append(np.mean(batch_kls_rec))
        rec_errs.append(np.mean(batch_rec_errs))
        # beta_rec = 10.0
        # beta_neg = 1
        # beta_kl = 0.05
        # gamma_r = 1
        savename = f"VAEtoS-IntroVAE_4184_epoch{epoch}.pth"
    #   torch.save(model.state_dict(), file_path)
        torch.save(model.state_dict(), path + savename)
        # path == ... _VAEtoSoftVAE_fixed_seed/rec1_ng300_kl1_gamma1/
#       torch.save(model.state_dict(), log_path + f"softintroVAE_weight_epoch{str(epoch)}.pth")


        now_time = time.time()
        print(f"Epoch [{epoch+1}/{epochs}]  train_lossE:{train_lossE:.3f}  train_lossD:{train_lossD:.3f}  val_lossE:{val_lossE:.3f}  "
              f"val_lossD:{val_lossD:.3f}, 1epoch:{(now_time - loop_start_time):.1f}秒  total time:{(now_time - start_time):.1f}秒")

        train_lossE_list.append(train_lossE)
        train_lossD_list.append(train_lossD)
        val_lossE_list.append(val_lossE)
        val_lossD_list.append(val_lossD)

        write_fig(path + "/loss.txt", train_lossE_list, val_lossE_list, train_lossD_list, val_lossD_list)
        write_kl_losses(path + "/kl_losses.txt", kls_real, kls_fake, kls_rec, rec_errs)

    e_scheduler.step()
    d_scheduler.step()

    train_result.result_rec_kls_loss(kls_real, kls_fake, kls_rec, rec_errs, path)
    print("Finished S-IntroVAE Traininig !!")
    return train_lossE_list, train_lossD_list, val_lossE_list, val_lossD_list


def init_weights_he(m):
    if type(m) == nn.Conv3d or type(m) == nn.ConvTranspose3d:
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
    return


def write_fig(path, trainE, valE, trainD, valD):
    with open(path, "w") as f:
        for t,v,td,vd in zip(trainE, valE, trainD, valD):
            f.write("trainE=%s\n" % str(t))
            f.write("valE===%s\n" % str(v))
            f.write("trainD=%s\n" % str(td))
            f.write("valD===%s\n" % str(vd))
    return

def write_kl_losses(path, kls_real, kls_fake, kls_rec, rec_errs):
    with open(path, "w") as f:
        for t,v,td,vd in zip(kls_real, kls_fake, kls_rec, rec_errs):
            f.write("kls_real==%s\n" % str(t))
            f.write("kls_fake==%s\n" % str(v))
            f.write("kls_rec===%s\n" % str(td))
            f.write("rec_errs==%s\n" % str(vd))
    return


def write_figres(path, train, val):
    with open(path, "w") as f:
        for t,v in zip(train, val):
            f.write("train=%s\n" % str(t))
            f.write("val===%s\n" % str(v))
    return

# trainer for ResNet mackysan VAE
def train_ResNetVAE(
    net,
    train_loader,
    val_loader,
    epochs=1,
    lr=0.001,
    device=torch.device("cpu"),
    path="./output_ResNetVAE/",
):
    path2 = path + "train_result.csv"
    with open(path2, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])

    optimizer = optim.Adam(net.parameters(), lr)
    print(optimizer)
    #net = torch.nn.DataParallel(net, device_ids=[0, 1, 2, 3])

    net.to(device)
    net.apply(init_weights_he)
    train_losses, val_losses = [], []
    start_time = time.time()
    for epoch in range(epochs):
        loop_start_time = time.time()
        train_run_mse = 0.0
        train_run_kl = 0.0
        train_run_loss = 0.0
        net.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            x_re, mu, logvar = net.forward(inputs) #  forward:  return x_re, mu, logvar
            loss, mse, kl = lossf.normal_loss(x_re, mu, logvar, inputs)
            loss.backward()
            optimizer.step()
            train_run_loss += loss.item()
            train_run_mse += mse.item()
            train_run_kl += kl.item()
#____________________________________________________#
        train_run_loss /= len(train_loader)
        train_run_mse /= len(train_loader)
        train_run_kl /= len(train_loader)
        train_losses.append(train_run_loss)
# __________________________________________________________________________________ #

        net.eval()
        val_run_loss = 0.0
        val_run_mse = 0.0
        val_run_kl = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                x_re, mu, logvar = net.forward(inputs)
                loss, mse, kl = lossf.normal_loss(x_re, mu, logvar, inputs)
                val_run_loss += loss.item()
                val_run_mse += mse.item()
                val_run_kl += kl.item()
            val_run_loss /= len(val_loader)
            val_run_mse /= len(val_loader)
            val_run_kl /= len(val_loader)

        val_run_loss /= len(val_loader)
        val_losses.append(val_run_loss)

        if epoch % 10 == 0:
            savename = f"ResNetVAE_4184epoch{epoch}.pth"
        #   torch.save(model.state_dict(), file_path)
            torch.save(net.state_dict(), path + savename)
    #       torch.save(model.state_dict(), log_path + f"softintroVAE_weight_epoch{str(epoch)}.pth")

        now_time = time.time()
        print(f"Epoch [{epoch+1}/{epochs}] train_loss:{train_run_loss:.3f}  val_loss:{val_run_loss:.3f} "
              f" 1epoch:{(now_time - loop_start_time):.1f}秒  total time:{(now_time - start_time):.1f}秒")

        write_figres(path + "/loss.txt", train_losses, val_losses)
        # write_fig(path + "/mse.txt",train_losses_mse,val_losses_mse)
        # write_fig(path + "/kl.txt",train_losses_kl,val_losses_kl)

    if epochs != 0:
        torch.save(net.state_dict(), path + "resnetvae_weight.pth")
        print("saved ResNetVAE param and ", end="") # これで改行しない

    print("finished ResNetVAE traininig.")
    return train_losses, val_losses
# --------------------------------------------------------------------------------#


# trainer for ResCAE
def train_ResNetCAE(
    net,
    train_loader,
    val_loader,
    epochs=1,
    lr=0.001,
    device=torch.device("cpu"),
    path="./output_ResNetCAE/",
):
    path = path + "train_result.csv"
    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr)
    print(optimizer)
    net = torch.nn.DataParallel(net, device_ids=[0, 1, 2, 3])
    net.to(device)
    train_loss_list, val_loss_list = [], []
    start_time = time.time()
    for epoch in range(epochs):
        loop_start_time = time.time()
        net.train()
        train_loss, val_loss = 0.0, 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            #labels = labels.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        net.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                #labels = labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, inputs)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        now_time = time.time()
        print(f"Epoch [{epoch+1}/{epochs}] train_loss:{train_loss:.3f}  val_loss:{val_loss:.3f} "
              f" 1epoch:{(now_time - loop_start_time):.1f}秒  total time:{(now_time - start_time):.1f}秒")

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        #write_csv(epoch, train_loss, val_loss, path)

    print("Finished Traininig")
    return train_loss_list, val_loss_list

    # ----------------------------------------------------------------------------------------------------------- #


# trainer for CNN
def train(
    net,
    train_loader,
    val_loader,
    epochs=1,
    lr=0.001,
    device=torch.device("cpu"),
    path="./output/",
):
    path = path + "train_result.csv"
    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    #criterion = nn.MSELoss()
    #criterion = nn.BCELoss()
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(params=net.parameters(), lr=0.0005, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr)
    print(optimizer)

    net.to(device)
    train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []
    start_time = time.time()
    for epoch in range(epochs):
        loop_start_time = time.time()
        net.train()
        train_loss, train_acc, val_loss, val_acc = 0.0, 0.0, 0.0, 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            #labels = nn.functional.one_hot(labels, num_classes=2).to(torch.float32)         # sigmoid用
            loss = criterion(outputs, labels)
            #loss = criterion(outputs, labels.unsqueeze(1).float())          # bce用
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            #predicted = (outputs > 0.5).float()                    # bce用
            #predicted = predicted.squeeze(1)                       # bce用
            #labels = torch.argmax(labels, dim=1)                   # sigmoid用
            train_acc += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader.dataset)


        net.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = net(inputs)
                #labels = nn.functional.one_hot(labels, num_classes=2).to(torch.float32)         # sigmoid用
                loss = criterion(outputs, labels)
                #loss = criterion(outputs, labels.unsqueeze(1).float())          # bce用
                val_loss += loss.item()
                predicted = torch.argmax(outputs, dim=1)
                #predicted = (outputs > 0.5).float()                  # bce用
                #predicted = predicted.squeeze(1)
                #labels = torch.argmax(labels, dim=1)                # sigmoid用
                val_acc += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc /= len(val_loader.dataset)

        now_time = time.time()
        print(f"Epoch [{epoch+1}/{epochs}] train_loss:{train_loss:.3f}  acc:{train_acc * 100:.3f}  val_loss:{val_loss:.3f} "
              f"val acc:{val_acc * 100:.3f}  1epoch:{(now_time - loop_start_time):.1f}秒  total time:{(now_time - start_time):.1f}秒")

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        write_csv(epoch, train_loss, train_acc, val_loss, val_acc, path)

    print("Finished Traininig")
    return train_loss_list, train_acc_list, val_loss_list, val_acc_list

# ----------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------- #

# trainer for VAE
def train_vae(
    net,
    train_loader,
    val_loader,
    epochs=1,
    lr=0.001,
    device=torch.device("cpu"),
    path="./output_vae/",
):
    path = path + "train_result.csv"
    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])

    optimizer = optim.Adam(net.parameters(), lr)
    print(optimizer)

    net.to(device)
    train_loss_list, val_loss_list = [], []
    start_time = time.time()
    for epoch in range(epochs):
        loop_start_time = time.time()
        net.train()
        train_loss, val_loss = 0.0, 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            optimizer.zero_grad()

            x_re, mu, logvar = net.forward(inputs)
            loss = net.loss(x_re, inputs, mu, logvar)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)


        net.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                #labels = labels.to(device)
                x_re, mu, logvar = net.forward(inputs)
                loss = net.loss(x_re, inputs, mu, logvar)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        now_time = time.time()
        print(f"Epoch [{epoch+1}/{epochs}] train_loss:{train_loss:.3f}  val_loss:{val_loss:.3f} "
              f" 1epoch:{(now_time - loop_start_time):.1f}秒  total time:{(now_time - start_time):.1f}秒")

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        #write_csv(epoch, train_loss, val_loss, path)

    print("Finished Traininig")
    return train_loss_list, val_loss_list

# --------------------------------------------------------------------------------#
