import torch.nn.functional as F
import torch.nn as nn
import torch
from utils import *

# THIS MODEL IS v49
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, dropout_rate):
        super(ResidualBlock, self).__init__()

        self.res = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(in_channels, affine=True),
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(in_channels, affine=True),
            nn.Dropout2d(dropout_rate)
        )

    def forward(self, x):
        return x + self.res(x)


class MiniBlock(nn.Module):
    def __init__(self, in_channels, dropout_rate):
        super(MiniBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(in_channels, affine=True),
            nn.Dropout2d(dropout_rate)
        )

    def forward(self, x):
        return self.block(x)


class CustomUNet(nn.Module):
    def __init__(self, in_channels, dropout_rate, name="model_v54"):
        super(CustomUNet, self).__init__()
        self.dropout_rate = dropout_rate
        self.name = name

        self.lrelu = nn.LeakyReLU()

        # Initial block
        self.initial = self.down_block(in_channels, 64) # output 64 channels

        # Down blocks
        self.down1 = self.down_block(64, 128) # Input 64, output 128 channels
        self.down2 = self.down_block(128, 256) # Input 128, output 256 channels
        self.down3 = self.down_block(256, 512) # Input 256, output 512 channels
        self.down4 = self.down_block(512, 1024) # Input 512, output 1024 channels

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(1024, self.dropout_rate) for _ in range(3)]
        )

        self.attention1 = SelfAttention(1024)
        # self.attention2 = SelfAttention(1024)
        
        self.mini1 = MiniBlock(1024, dropout_rate)
        self.mini2 = MiniBlock(512, dropout_rate)
        self.mini3 = MiniBlock(256, dropout_rate)
        self.mini4 = MiniBlock(128, dropout_rate)
        self.mini5 = MiniBlock(64, dropout_rate)

        # Up blocks
        self.up1 = self.up_block(2048, 512)
        self.up2 = self.up_block(1024, 256)
        self.up3 = self.up_block(512, 128)
        self.up4 = self.up_block(256, 64)

        # Output block
        self.out = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=5, stride=1, padding=2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            # nn.Sigmoid()
        )

        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
        self.apply(weights_init)

    def down_block(self, in_channels, out_channels, kernel_sizes=[3, 5]):
        branches = []
        for kernel_size in kernel_sizes:
            padding = (kernel_size - 1) // 2
            # branch = nn.Sequential(
            #     nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            #     nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2),
            #     self.lrelu,
            #     nn.GroupNorm(8, out_channels),
            #     nn.MaxPool2d(2),
            #     nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            #     nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            #     self.lrelu,
            #     nn.GroupNorm(8, out_channels),
            #     nn.Dropout2d(self.dropout_rate)
            # )
            branch = nn.Sequential(
                nn.Conv2d(in_channels, out_channels//len(kernel_sizes), kernel_size=kernel_size, stride=1, padding=padding),
                nn.Conv2d(out_channels//len(kernel_sizes), out_channels//len(kernel_sizes), kernel_size=5, stride=1, padding=2),
                self.lrelu,
                nn.GroupNorm(8, out_channels//len(kernel_sizes)),
                nn.MaxPool2d(2),
                nn.Conv2d(out_channels//len(kernel_sizes), out_channels//len(kernel_sizes), kernel_size=kernel_size, stride=1, padding=padding),
                nn.Conv2d(out_channels//len(kernel_sizes), out_channels//len(kernel_sizes), kernel_size=3, stride=1, padding=1),
                self.lrelu,
                nn.GroupNorm(8, out_channels//len(kernel_sizes)),
                nn.Dropout2d(self.dropout_rate)
            )

            branches.append(branch)
        return nn.ModuleList(branches)
    
    def up_block(self, in_channels, out_channels, kernel_sizes=[3, 5]):
        branches = []
        for kernel_size in kernel_sizes:
            padding = (kernel_size - 1) // 2
            branch = nn.Sequential(
                nn.Conv2d(in_channels, out_channels//len(kernel_sizes), kernel_size=kernel_size, stride=1, padding=padding),
                nn.Conv2d(out_channels//len(kernel_sizes), out_channels//len(kernel_sizes), kernel_size=5, stride=1, padding=2),
                self.lrelu,
                nn.GroupNorm(8, out_channels//len(kernel_sizes)),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(out_channels//len(kernel_sizes), out_channels//len(kernel_sizes), kernel_size=kernel_size, stride=1, padding=padding),
                nn.Conv2d(out_channels//len(kernel_sizes), out_channels//len(kernel_sizes), kernel_size=3, stride=1, padding=1),
                self.lrelu,
                nn.GroupNorm(8, out_channels//len(kernel_sizes)),
                nn.Dropout2d(self.dropout_rate)
            )
            branches.append(branch)
        return nn.ModuleList(branches)

    def forward(self, arm):
        x = arm
        x0 = torch.cat([branch(x) for branch in self.initial], dim=1)
        x1 = torch.cat([branch(x0) for branch in self.down1], dim=1)
        x2 = torch.cat([branch(x1) for branch in self.down2], dim=1)
        x3 = torch.cat([branch(x2) for branch in self.down3], dim=1)
        x4 = torch.cat([branch(x3) for branch in self.down4], dim=1)
        x5 = self.res_blocks(x4)
        x5 = self.attention1(x5)
        x6 = torch.cat([branch(torch.cat((x5, self.mini1(x4)), dim=1)) for branch in self.up1], dim=1)
        x7 = torch.cat([branch(torch.cat((x6, self.mini2(x3)), dim=1)) for branch in self.up2], dim=1)
        x8 = torch.cat([branch(torch.cat((x7, self.mini3(x2)), dim=1)) for branch in self.up3], dim=1)
        x9 = torch.cat([branch(torch.cat((x8, self.mini4(x1)), dim=1)) for branch in self.up4], dim=1)
        output = self.out(torch.cat((x9, self.mini5(x0)), dim=1))
        return output
