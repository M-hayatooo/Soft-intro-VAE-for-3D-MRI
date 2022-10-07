import csv
import os
import random
import time
from asyncore import loop

import models.models as models
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def write_csv(epoch, train_loss, train_acc, val_loss, val_acc, path):
    with open(path, "a") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])


def calc_kl(logvar, mu, mu_o=0.0, logvar_o=0.0, reduce='sum'):
    if not isinstance(mu_o, torch.Tensor):
        mu_o = torch.tensor(mu_o).to(mu.device)
    if not isinstance(logvar_o, torch.Tensor):
        logvar_o = torch.tensor(logvar_o).to(mu.device)
    kl = -0.5 * (1 + logvar - logvar_o - logvar.exp() / torch.exp(logvar_o) - (mu - mu_o).pow(2) / torch.exp(logvar_o)).sum(1)
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

#  ============= first =================
# #  ==================================================================
# def train_soft_intro_vae(z_dim=150, lr_e=2e-4, lr_d=2e-4, batch_size=16, num_workers=os.cpu_count(), start_epoch=0,
#                            num_epochs=250, num_vae=0, save_interval=5000, recon_loss_type="mse",
#                            beta_kl=1.0, beta_rec=1.0, beta_neg=1.0, test_iter=1000, seed=-1, pretrained=None,
#                            device=torch.device("cpu"), num_row=8, gamma_r=1e-8):
#     if seed != -1:
#         random.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         torch.cuda.manual_seed(seed)
#         torch.backends.cudnn.deterministic = True
#         print("random seed: ", seed)

#     model = models.SoftIntroVAE(12, [[12,1,2],[24,1,2],[32,2,2],[48,2,2]], conditional=False)
#     #model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
#     model.to(device)
#     # もしpretrainedが存在しているのならば model param load
#     if pretrained is not None:
#         load_model(model, pretrained, device)
#     print(model)

#     optimizer_e = optim.Adam(model.encoder.parameters(), lr=lr_e)
#     optimizer_d = optim.Adam(model.decoder.parameters(), lr=lr_d)

#     e_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_e, milestones=(350,), gamma=0.1)
#     d_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=(350,), gamma=0.1)

#     scale = 1 / (80 * 96 * 80)  # normalizing constant, 's' in the paper  desu

#     train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
# #   train_data_loader = load_data(kinds=["ADNI2","ADNI2-2"], classes=["CN", "AD"], unique=False, blacklist=True)

#     start_time = time.time()

#     cur_iter = 0
#     kls_real = []
#     kls_fake = []
#     kls_rec = []
#     rec_errs = []

#     for epoch in range(start_epoch, num_epochs):
#         diff_kls = []
#         # save models
#         if epoch % save_interval == 0 and epoch > 0:
#             save_epoch = (epoch // save_interval) * save_interval
#             prefix = dataset + "_soft_intro_vae" + "_betas_" + str(beta_kl) + "_" + str(beta_neg) + "_" + str(beta_rec) + "_"
#             save_checkpoint(model, save_epoch, cur_iter, prefix)

#         model.train()
#         batch_kls_real = []
#         batch_kls_fake = []
#         batch_kls_rec = []
#         batch_rec_errs = []


#         for iteration, (batch, labels) in enumerate(train_data_loader, 0):# iterationには 自動で割り振られたindex番号が適用される
#             b_size = batch.size(0)

#             noise_batch = torch.randn(size=(b_size, 1, 5, 6, 5)).to(device)
#             real_batch = batch.to(device)

#             # =========== Update E ================
#             fake = model.decode(noise_batch) #  fake = model.sample(noise_batch)

#             real_mu, real_logvar = model.encode(real_batch)
#             z = model.reparameterize(real_mu, real_logvar)
#             rec = model.decode(z)

#             loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type=recon_loss_type, reduction="mean")
#             lossE_real_kl = calc_kl(real_logvar, real_mu, reduce="mean")
#             # {{ mu,    logvar,    z   ,    y }}を返す
#             rec_mu, rec_logvar, z_rec, rec_rec     = model( rec.detach())
#             fake_mu, fake_logvar, z_fake, rec_fake = model(fake.detach())

#             fake_kl_e = calc_kl(fake_logvar, fake_mu, reduce="none")
#             rec_kl_e = calc_kl(rec_logvar, rec_mu, reduce="none")


#             loss_fake_rec = calc_reconstruction_loss(fake, rec_fake, loss_type=recon_loss_type, reduction="none")
#             loss_rec_rec = calc_reconstruction_loss(rec, rec_rec, loss_type=recon_loss_type, reduction="none")

#             exp_elbo_fake = (-2 * scale * (beta_rec * loss_fake_rec + beta_neg * fake_kl_e)).exp().mean()
#             exp_elbo_rec = (-2 * scale * (beta_rec * loss_rec_rec + beta_neg * rec_kl_e)).exp().mean()
#             # total loss
#             lossE = scale * (beta_rec * loss_rec + beta_kl * lossE_real_kl) + 0.25 * (exp_elbo_fake + exp_elbo_rec)
#             # backprop
#             optimizer_e.zero_grad()
#             lossE.backward()
#             optimizer_e.step()

#             # ========= Update D ==================
#             for param in model.encoder.parameters():
#                 param.requires_grad = False
#             for param in model.decoder.parameters():
#                 param.requires_grad = True

#             fake = model.decode(noise_batch)
#             rec = model.decode(z.detach())

#             loss_rec = calc_reconstruction_loss(real_batch, rec.detach(),loss_type=recon_loss_type, reduction="mean")

#             rec_mu, rec_logvar = model.encode(rec)
#             z_rec = reparameterize(rec_mu, rec_logvar)

#             fake_mu, fake_logvar = model.encode(fake)
#             z_fake = reparameterize(fake_mu, fake_logvar)

#             rec_rec = model.decode(z_rec) # rec_rec = model.decode(z_rec.detach())
#             rec_fake = model.decode(z_fake) # rec_fake = model.decode(z_fake.detach())

#             loss_rec_rec = calc_reconstruction_loss(rec.detach(), rec_rec, loss_type=recon_loss_type, reduction="mean")
#             loss_fake_rec = calc_reconstruction_loss(fake.detach(), rec_fake, loss_type=recon_loss_type, reduction="mean")

#             rec_kl = calc_kl(rec_logvar, rec_mu, reduce="mean")
#             fake_kl = calc_kl(fake_logvar, fake_mu, reduce="mean")

#             lossD = scale * (loss_rec * beta_rec + (rec_kl + fake_kl) * 0.5 * beta_kl + gamma_r * 0.5 * beta_rec * (loss_rec_rec + loss_fake_rec))

#             optimizer_d.zero_grad()
#             lossD.backward()
#             optimizer_d.step()
#     #        print("finish updateD")

#             if torch.isnan(lossD) or torch.isnan(lossE):
#                 raise SystemError

#             # statistics for plotting later
#             diff_kls.append(-lossE_real_kl.data.cpu().item() + fake_kl.data.cpu().item())
#             batch_kls_real.append(lossE_real_kl.data.cpu().item())
#             batch_kls_fake.append(fake_kl.cpu().item())
#             batch_kls_rec.append(rec_kl.data.cpu().item())
#             batch_rec_errs.append(loss_rec.data.cpu().item())


#         info = "\nEpoch[{}]({}/{}): time: {:4.4f}: ".format(epoch, iteration, len(train_data_loader), time.time() - start_time)
#         info += 'Rec: {:.4f}, '.format(loss_rec.data.cpu())
#         info += 'Kl_E: {:.4f}, expELBO_R: {:.4e}, expELBO_F: {:.4e}, '.format(lossE_real_kl.data.cpu(), exp_elbo_rec.data.cpu(), exp_elbo_fake.cpu())
#         info += 'Kl_F: {:.4f}, KL_R: {:.4f}'.format(rec_kl.data.cpu(), fake_kl.data.cpu())
#         info += ' DIFF_Kl_F: {:.4f}'.format(-lossE_real_kl.data.cpu() + fake_kl.data.cpu())
#         print(info)

#         _, _, _, rec_det = model(real_batch)
#         max_imgs = min(batch.size(0), 16)

#     e_scheduler.step()
#     d_scheduler.step()

#     if epoch > num_vae - 1:
#         kls_real.append(np.mean(batch_kls_real))
#         kls_fake.append(np.mean(batch_kls_fake))
#         kls_rec.append(np.mean(batch_kls_rec))
#         rec_errs.append(np.mean(batch_rec_errs))

#     return model
# # =====================================================================

#torch.distributed.init_process_group(backend='nccl',rank=4, world_size=2)


#  ============= second =================
def train_soft_intro_vae(
    model,  #  net
    train_loader,
    val_loader,
    epochs,
    lr=0.001,
    device=torch.device("cpu"),
    path="./output_SoftIntroVAE/",
):
    seed = 77

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
    # if pretrained is not None:
    #     load_model(model, pretrained, device)
    # print(model)
    lr_e = lr
    lr_d = lr
    optimizer_e = optim.Adam(model.encoder.parameters(), lr=lr_e)
    optimizer_d = optim.Adam(model.decoder.parameters(), lr=lr_d)
    e_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_e, milestones=(350,), gamma=0.1)
    d_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=(350,), gamma=0.1)

#   パラメータ定義
    recon_loss_type = "mse"
    beta_rec = 1.0
    beta_neg = 1.0
    beta_kl = 1.0
    gamma_r = 1.0


    scale = 1 / (80 * 96 * 80)  # normalizing constant, 's' in the paper  desu

    start_time = time.time()

    cur_iter = 0
    kls_real = []
    kls_fake = []
    kls_rec = []
    rec_errs = []

    start_epoch, num_epochs = 0, 500
    for epoch in range(start_epoch, num_epochs):
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

#        for iteration, batch in enumerate(train_data_loader, 0):
        for batch, labels in train_loader:# iterationには 自動で割り振られたindex番号が適用される
            # --------------train------------
            b_size = batch.size(0)
            noise_batch = torch.randn(size=(b_size, 1, 5, 6, 5)).to(device)
            real_batch = batch.to(device)

            # =========== Update E ================
        #   fake = model.sample(noise_batch)
            fake = model.decode(noise_batch)

            real_mu, real_logvar = model.encode(real_batch)
            z = model.reparameterize(real_mu, real_logvar)
            rec = model.decode(z)

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
       #     print("finish updateE")
            # ========= Update D ==================
            for param in model.encoder.parameters():
                param.requires_grad = False
            for param in model.decoder.parameters():
                param.requires_grad = True

            fake = model.decode(noise_batch)
            rec = model.decode(z.detach())

            loss_rec = calc_reconstruction_loss(real_batch, rec.detach(),loss_type=recon_loss_type, reduction="mean")

            rec_mu, rec_logvar = model.encode(rec)
            z_rec = reparameterize(rec_mu, rec_logvar)

            fake_mu, fake_logvar = model.encode(fake)
            z_fake = reparameterize(fake_mu, fake_logvar)

            # rec_rec = model.decode(z_rec.detach())
            # rec_fake = model.decode(z_fake.detach())
            rec_rec = model.decode(z_rec)
            rec_fake = model.decode(z_fake)

            loss_rec_rec = calc_reconstruction_loss(rec.detach(), rec_rec, loss_type=recon_loss_type, reduction="mean")
            loss_fake_rec = calc_reconstruction_loss(fake.detach(), rec_fake, loss_type=recon_loss_type, reduction="mean")

            rec_kl = calc_kl(rec_logvar, rec_mu, reduce="mean")
            fake_kl = calc_kl(fake_logvar, fake_mu, reduce="mean")

            lossD = scale * (loss_rec * beta_rec + (rec_kl + fake_kl) * 0.5 * beta_kl + gamma_r * 0.5 * beta_rec * (loss_rec_rec + loss_fake_rec))

            optimizer_d.zero_grad()
            lossD.backward()
            optimizer_d.step()
            if epoch % 100 == 0:
                print("finish updateD")

            if torch.isnan(lossD) or torch.isnan(lossE):
                raise SystemError

            # statistics for plotting later
            diff_kls.append(-lossE_real_kl.data.cpu().item() + fake_kl.data.cpu().item())
            batch_kls_real.append(lossE_real_kl.data.cpu().item())
            batch_kls_fake.append(fake_kl.cpu().item())
            batch_kls_rec.append(rec_kl.data.cpu().item())
            batch_rec_errs.append(loss_rec.data.cpu().item())

            #if cur_iter % test_iter == 0:
        info = "\nEpoch[{}]({}/{}): time: {:4.4f}: ".format(epoch, iteration, len(train_data_loader), time.time() - start_time)
        info += 'Rec: {:.4f}, '.format(loss_rec.data.cpu())
        info += 'Kl_E: {:.4f}, expELBO_R: {:.4e}, expELBO_F: {:.4e}, '.format(lossE_real_kl.data.cpu(), exp_elbo_rec.data.cpu(), exp_elbo_fake.cpu())
        info += 'Kl_F: {:.4f}, KL_R: {:.4f}'.format(rec_kl.data.cpu(), fake_kl.data.cpu())
        info += ' DIFF_Kl_F: {:.4f}'.format(-lossE_real_kl.data.cpu() + fake_kl.data.cpu())
        print(info)

        _, _, _, rec_det = model(real_batch)
        max_imgs = min(batch.size(0), 16)
        # vutils.save_image(
        #         torch.cat([real_batch[:max_imgs], rec_det[:max_imgs], fake[:max_imgs]], dim=0).data.cpu(),
        #         '{}/image_{}.jpg'.format("./", cur_iter), nrow=num_row)                 
        #cur_iter += 1
    e_scheduler.step()
    d_scheduler.step()

    if epoch > num_vae - 1:
        kls_real.append(np.mean(batch_kls_real))
        kls_fake.append(np.mean(batch_kls_fake))
        kls_rec.append(np.mean(batch_kls_rec))
        rec_errs.append(np.mean(batch_rec_errs))


        now_time = time.time()
        print(
            "Epoch [%3d/%3d], train_lossE: %.3f, train_lossD: %.3f, val_lossE: %.3f, val_lossD: %.3f, %d秒/epoch, total time %d秒"
            % (
                epoch + 1,
                epochs,
                train_lossE,
                train_lossD,
                val_lossE,
                val_lossD,
                now_time - loop_start_time,
                now_time - start_time,
            )
        )
        train_lossE_list.append(train_lossE)
        train_lossD_list.append(train_lossD)
        val_lossE_list.append(val_lossE)
        val_lossD_list.append(val_lossD)

    print("Finished Traininig")
    return train_lossE_list, val_lossE_list


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
    path = path + "train_result.csv"
    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])

    optimizer = optim.Adam(net.parameters(), lr)
    print(optimizer)
    #net = torch.nn.DataParallel(net, device_ids=[0, 1, 2, 3])
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
                x_re, mu, logvar = net.forward(inputs)
                loss = net.loss(x_re, inputs, mu, logvar)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        now_time = time.time()
        print(
            "Epoch [%3d/%3d], loss: %.3f, val_loss: %.3f, %d秒/epoch, total time %d秒"
            % (
                epoch + 1,
                epochs,
                train_loss,
                val_loss,
                now_time - loop_start_time,
                now_time - start_time,
            )
        )
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

    print("Finished Traininig")
    return train_loss_list, val_loss_list
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
        print(
            "Epoch [%3d/%3d], loss: %.3f, val_loss: %.3f, %d秒/epoch, total time %d秒"
            % (
                epoch + 1,
                epochs,
                train_loss,
                val_loss,
                now_time - loop_start_time,
                now_time - start_time,
            )
        )
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
        elapsed_time = time.time()
        print(
            "Epoch [%3d/%3d], loss: %.3f acc: %.3f, val_loss: %.3f val_acc: %.3f total time %d秒"
            % (
                epoch + 1,
                epochs,
                train_loss,
                train_acc * 100,
                val_loss,
                val_acc * 100,
                elapsed_time - start_time,
            )
        )
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        write_csv(epoch, train_loss, train_acc, val_loss, val_acc, path)

    print("Finished Traininig")
    return train_loss_list, train_acc_list, val_loss_list, val_acc_list

# ----------------------------------------------------------------------------------------------------------- #


# trainer for CAE
def train_cae(
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
        writer.writerow(["epoch", "train_loss", "val_loss"])

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr)
    print(optimizer)
    net = torch.nn.DataParallel(net, device_ids=[0,1,2,3])
    net.to(device)
    train_loss_list, val_loss_list = [], []
    start_time = time.time()
    for epoch in range(epochs):
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

        elapsed_time = time.time()
        print(
            "Epoch [%3d/%3d], loss: %.3f, val_loss: %.3f, total time %d秒"
            % (
                epoch + 1,
                epochs,
                train_loss,
                val_loss,
                elapsed_time - start_time,
            )
        )
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        #write_csv(epoch, train_loss, val_loss, path)

    print("Finished Traininig")
    return train_loss_list, val_loss_list

    # ----------------------------------------------------------------------------------------------------------- #


# trainer for Caee
def train_caee(
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
        writer.writerow(["epoch", "train_loss", "val_loss"])

    criterion = nn.MSELoss()
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
        print(
            "Epoch [%3d/%3d], loss: %.3f, val_loss: %.3f, %d秒/epoch, total time %d秒"
            % (
                epoch + 1,
                epochs,
                train_loss,
                val_loss,
                loop_start_time - now_time,
                now_time - start_time,
            )
        )
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        #write_csv(epoch, train_loss, val_loss, path)

    print("Finished Traininig")
    return train_loss_list, val_loss_list

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

        elapsed_time = time.time()
        print(
            "Epoch [%3d/%3d], loss: %.3f, val_loss: %.3f, total time %d秒"
            % (
                epoch + 1,
                epochs,
                train_loss,
                val_loss,
                elapsed_time - start_time,
            )
        )
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        #write_csv(epoch, train_loss, val_loss, path)

    print("Finished Traininig")
    return train_loss_list, val_loss_list

# --------------------------------------------------------------------------------#
