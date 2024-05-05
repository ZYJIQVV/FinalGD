# -*- encoding: utf-8 -*-
"""
@Time: 2024-04-20 14:35
@Auth: xjjxhxgg
@File: pcwa.py
@IDE: PyCharm
@Motto: xhxgg
"""
import torch

from torch import nn
from gd.nn.modules import Conv


class PCWA(nn.Module):
    def __init__(self, channel=3, patch_size=2, h=224, w=224):
        super(PCWA, self).__init__()
        self.patch_size = patch_size
        self.max_pool = nn.MaxPool2d(patch_size, patch_size)
        self.avg_pool = nn.AvgPool2d(patch_size, patch_size)
        self.mlp = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(channel * 2, channel, 1)
        self.channel_max_pool = ChannelMaxPool()
        self.channel_avg_pool = ChannelAvgPool()
        self.mlp_out = nn.Linear(h * w, 1)
        self.conv_out = Conv(2, channel, 7)

    def _spatially_divide(self, x):
        B, C, H, W = x.size()
        ph, pw = self.patch_size, self.patch_size
        x = x.unfold(2, ph, ph).unfold(3, pw, pw)
        x = x.contiguous().view(B, C, -1, ph, pw)
        Fs = []
        for i in range(x.size(2)):
            Fs.append(x[:, :, i, :, :])
        return Fs

    def forward(self, x):
        Fs = self._spatially_divide(x)
        max_pool = [self.max_pool(F) for F in Fs]
        avg_pool = [self.avg_pool(F) for F in Fs]
        mlp = [self.mlp(m) + self.mlp(a) for m, a in zip(max_pool, avg_pool)]
        mlp = [self.sigmoid(m) for m in mlp]
        product = [Fs[i] * mlp[i] for i in range(len(Fs))]
        h_n, w_n = x.size(2) // self.patch_size, x.size(3) // self.patch_size
        # concat product into shape of x
        product = torch.cat([product[i] for i in range(len(product))], dim=2)
        product = product.view(x.size(0), x.size(1), h_n, w_n, self.patch_size, self.patch_size)
        product = product.permute(0, 1, 2, 4, 3, 5).contiguous().view(x.size(0), x.size(1), h_n * self.patch_size,
                                                                      w_n * self.patch_size)
        # cat product and x
        out = torch.cat([product, x], dim=1)
        out = self.conv(out)
        fs = out
        max_pool = self.channel_max_pool(out)
        avg_pool = self.channel_avg_pool(out)
        out = torch.cat([max_pool, avg_pool], dim=1)
        out = self.conv_out(out)
        out = self.sigmoid(out)
        out = out * fs
        return out


class ChannelMaxPool(nn.Module):
    def __init__(self):
        super(ChannelMaxPool, self).__init__()
    def forward(self, x):
        x = torch.max(x, dim=1, keepdim=True)[0]
        return x


class ChannelAvgPool(nn.Module):
    def __init__(self):
        super(ChannelAvgPool, self).__init__()
    def forward(self, x):
        x = torch.mean(x, dim=1, keepdim=True)
        return x


class AID(nn.Module):
    """Adaptive Interlace Downsample (AID) module."""
    def __init__(self,channel=3,stride=2, group=4,patch_size=2,h=224,w=224):
        super(AID,self).__init__()
        # stride表示h方向多长为一组，group表示一共分多少组，group应该是stride的整数倍，否则会报错，因为要保证每一组都有相同的元素
        # group_for_one_row表示在一行内，多长为一组
        # 例如stride=2,group=4,group_for_one_row=2
        # 表示一共分4组，步长为2，则每两行为一组，在单独一行内，会将像素分为group_for_one_row=2组，每组有group_for_one_row=2个像素，每组之间的像素间隔为group_for_one_row=2

        self.stride = stride
        self.group = group
        self.group_for_one_row = self.group // self.stride
        self.pcwa = PCWA(patch_size,channel=channel * group, h=h//stride, w=w//self.group_for_one_row)
        self.layer_norm = nn.LayerNorm([channel * group, h//stride, w//self.group_for_one_row])
        self.mlp = Conv(channel * group, channel * group // 2, 1)
    def forward(self, x):
        x = self._split(x)
        x = self.pcwa(x)
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x

    def _split(self, x):
        """Split input tensor into groups."""
        groups = []
        for i in range(self.group):
            groups.append(x[:,:,i // self.stride::self.stride,i // self.stride::self.group_for_one_row])
        groups = torch.cat(groups, dim=1)
        return groups
