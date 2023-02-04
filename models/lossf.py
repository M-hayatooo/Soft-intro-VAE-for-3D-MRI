import torch
import torch.nn.functional as F


def mse_loss(out, x):
    bsize = x.size(0)
    x = x.view(bsize, -1)
    out = out.view(bsize, -1)
    mse = torch.mean(torch.sum(F.mse_loss(x, out, reduction='none'), dim=1), dim=0)
    # mse = torch.mean(torch.sum(F.mse_loss(x, out, reduction='none'), dim=1), dim=0)
    # mse = torch.mean(torch.sum(F.mse_loss(x, out, reduction='none'), dim=1), dim=0)
    return mse

def kld_loss(mu, logvar):
    bsize = mu.size(0)
    mu = mu.view(bsize, -1)
    logvar = logvar.view(bsize, -1)
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)

def normal_loss(x_hat, mu, logvar, x, msew=1, kldw=10):
    mse = mse_loss(x_hat, x) * msew
    kld = kld_loss(mu, logvar) * kldw
    loss = mse + kld
    return loss, mse, kld

def localized_loss(x_hat, mu, logvar, localize_loss, x, msew=1, kldw=1, localizew=1):
    mse = mse_loss(x_hat, x) * msew
    kld = kld_loss(mu, logvar) * kldw
    localize_loss = torch.mean(torch.sum(localize_loss, dim=1), dim=0) * localizew
    loss = mse + kld + localize_loss
    return loss, mse, kld, localize_loss


# def loss(self, x_re, x, mu, logvar):
#     re_err = torch.sqrt(torch.mean((x_re - x)**2)) # ==  self.Rmse(x_re, x)
#     kld = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
#     return re_err + kld
