from typing import List

import torch
# from torch import Module
import torch.nn as nn
import torch.nn.functional as F


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

#       for c, n, s in self.block_setting:

        for line in self.block_setting:
           #c, n, s = *line
            c, n, s = line[0], line[1], line[2] # ex [12, 1, 2] ブロックの最終channel数の指定, 繰り返し回数, pooling_kernel_size
            for i in range(n):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                blocks.append(nn.Sequential(BuildingBlock(in_ch, c, stride)))
                in_ch = c
        self.inner_ch = in_ch
        self.blocks = nn.Sequential(*blocks)
        self.conv = nn.Sequential(nn.Conv3d(in_ch, last, kernel_size=1, stride=1, bias=True))

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
                if j == n - 1:
                    stride = s
                else:
                    stride = 1
                if j == n - 1:
                    c = nc
                else:
                    c = c
                blocks.append(nn.Sequential(UpsampleBuildingkBlock(in_ch, c, stride)))
                in_ch = c
        blocks.append(nn.Sequential(
            nn.Conv3d(in_ch, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
        ))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)





























class Encoder_lucky(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool3d(2, 2)
        self.avgpool = nn.AvgPool3d(2, 2)
        self.conv1 = nn.Conv3d(1, 3, 3, padding=1)
        self.conv2 = nn.Conv3d(3, 3, 3, padding=1)
        self.conv3 = nn.Conv3d(3, 32, 3, padding=1)
        self.conv4 = nn.Conv3d(32, 64, 3, padding=1)
        self.conv5 = nn.Conv3d(64, 64, 4, 2, padding=1)
        self.fc1 = nn.Linear(10 * 12 * 10 * 64, 512)

        self.sigmoid = nn.Sigmoid()
        self.batchnorm3d1 = nn.BatchNorm3d(3)
        self.batchnorm3d2 = nn.BatchNorm3d(3)
        self.batchnorm3d3 = nn.BatchNorm3d(32)
        self.batchnorm3d4 = nn.BatchNorm3d(64)

    #    self.batchnorm1 = nn.BatchNorm1d(512)

    def forward(self, x):
        x = F.relu(self.batchnorm3d1(self.conv1(x)))
        x = self.pool(x) # 40*48*80
        x = F.relu(self.batchnorm3d2(self.conv2(x)))
        x = self.pool(x) # 20
        x = F.relu(self.batchnorm3d3(self.conv3(x)))
        x = F.relu(self.batchnorm3d4(self.conv4(x)))
        x = self.pool(x) # 10
        x = x.view(-1, 10 * 12 * 10 * 64)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.batchnorm1(self.fc1(x))) ここの batchnorm は要らない
        return x


class Decoder_lucky(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsamp1 = nn.Upsample((20, 24, 20), mode='nearest')  # mode="trilinear", align_corners=True
        self.upsamp2 = nn.Upsample((40, 48, 40), mode='nearest')  # mode="nearest"  degault mode == nearest
        self.upsamp3 = nn.Upsample((80, 96, 80), mode='nearest')
      # self.upsamp3 = nn.Upsample((80, 96, 80), mode='nearest', align_corners=False)

        self.dfc1 = nn.Linear(512, 10 * 12 * 10 * 64)
        self.deconv1 = nn.ConvTranspose3d(64, 32, 3, padding=1)
        self.deconv2 = nn.ConvTranspose3d(32, 3, 3, padding=1)
        self.deconv3 = nn.ConvTranspose3d(3, 3, 3, padding=1)
        self.deconv4 = nn.ConvTranspose3d(3, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

        self.batchnorm_d1 = nn.BatchNorm1d(76800)
        self.batchnorm_d3d1 = nn.BatchNorm3d(32)
        self.batchnorm_d3d2 = nn.BatchNorm3d(3)
        self.batchnorm_d3d3 = nn.BatchNorm3d(3)

    def forward(self, x):
        x = F.relu(self.batchnorm_d1(self.dfc1(x)))
        x = x.view(-1, 64, 10, 12, 10)
        x = self.upsamp1(x)
        x = F.relu(self.batchnorm_d3d1(self.deconv1(x)))
        x = F.relu(self.batchnorm_d3d2(self.deconv2(x)))
        x = self.upsamp2(x)
        x = F.relu(self.batchnorm_d3d3(self.deconv3(x)))
        x = self.upsamp3(x)
        x = self.sigmoid(self.deconv4(x))
        return x
