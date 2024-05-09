import torch
from torch import nn
from einops import rearrange
class SE(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        # 读取批数据图片数量及通道数
        b, c, h, w = x.size()
        # Fsq操作：经池化后输出b*c的矩阵
        y = self.gap(x).view(b, c)
        # Fex操作：经全连接层输出（b，c，1，1）矩阵
        y = self.fc(y).view(b, c, 1, 1)
        # Fscale操作：将得到的权重乘以原来的特征图x
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    """
    CBAM混合注意力机制的通道注意力
    """

    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            # 全连接层
            # nn.Linear(in_planes, in_planes // ratio, bias=False),
            # nn.ReLU(),
            # nn.Linear(in_planes // ratio, in_planes, bias=False)

            # 利用1x1卷积代替全连接，避免输入必须尺度固定的问题，并减小计算量
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
       avg_out = self.fc(self.avg_pool(x))
       max_out = self.fc(self.max_pool(x))
       out = avg_out + max_out
       out = self.sigmoid(out)
       return out * x

class SpatialAttention(nn.Module):
    """
    CBAM混合注意力机制的空间注意力
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv1(out))
        return out * x

class CBAM(nn.Module):
    """
    CBAM混合注意力机制
    """

    def __init__(self, in_channels, ratio=16, kernel_size=3):
        super(CBAM, self).__init__()
        self.channelattention = ChannelAttention(in_channels, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = self.channelattention(x)
        x = self.spatialattention(x)
        return x


class RFCBAMConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1):
        super().__init__()
        if kernel_size % 2 == 0:
            assert ("the kernel_size must be  odd.")
        self.kernel_size = kernel_size
        self.generate = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size, padding=kernel_size // 2,
                      stride=stride, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU()
            )
        self.get_weight = nn.Sequential(nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False), nn.Sigmoid())
        self.se = SE(in_channel)

        self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size, stride=kernel_size),
                                  nn.BatchNorm2d(out_channel), nn.ReLu())

    def forward(self, x):
        b, c = x.shape[0:2]
        channel_attention = self.se(x)
        generate_feature = self.generate(x)

        h, w = generate_feature.shape[2:]
        generate_feature = generate_feature.view(b, c, self.kernel_size ** 2, h, w)

        # generate_feature = rearrange(generate_feature, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
        #                              n2=self.kernel_size)

        unfold_feature = generate_feature * channel_attention
        max_feature, _ = torch.max(generate_feature, dim=1, keepdim=True)
        mean_feature = torch.mean(generate_feature, dim=1, keepdim=True)
        receptive_field_attention = self.get_weight(torch.cat((max_feature, mean_feature), dim=1))
        conv_data = unfold_feature * receptive_field_attention
        return self.conv(conv_data)

class BottleneckRFCBAMConv(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = RFCBAMConv(c_, c2, k[1], 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f_RFCBAM(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(BottleneckRFCBAMConv(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class SPPFLSKA(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.lska = LSKA(c2, k=5)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(self.lska(torch.cat(y, 1)))




# -*- encoding: utf-8 -*-
"""
@Time: 2024-05-05 23:22
@Auth: xjjxhxgg
@File: lska.py
@IDE: PyCharm
@Motto: xhxgg
"""
import torch
from torch import nn
class LSKA(nn.Module):
    """Large Kernel Attention(LKA) of VAN.

    .. code:: text
            DW_conv (depth-wise convolution)
                            |
                            |
        DW_D_conv (depth-wise dilation convolution)
                            |
                            |
        Transition Convolution (1×1 convolution)

    Args:
        embed_dims (int): Number of input channels.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self, embed_dims, k):
        super(LSKA, self).__init__()

        self.k=k

        if k == 7:
            self.DW_conv_h = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=(1,3), padding=(0, (3-1)//2), groups=embed_dims)
            self.DW_conv_v = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=(3,1), padding=((3-1)//2, 0), groups=embed_dims)
            self.DW_D_conv_h = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=(1,3), stride=(1,1), padding=(0,2), groups=embed_dims, dilation=2)
            self.DW_D_conv_v = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=(3,1), stride=(1,1), padding=(2,0), groups=embed_dims, dilation=2)
        elif k == 11:
            self.DW_conv_h = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=(1,3), padding=(0, (3-1)//2), groups=embed_dims)
            self.DW_conv_v = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=(3,1), padding=((3-1)//2, 0), groups=embed_dims)
            self.DW_D_conv_h = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=(1,5), stride=(1,1), padding=(0,4), groups=embed_dims, dilation=2)
            self.DW_D_conv_v = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=(5,1), stride=(1,1), padding=(4,0), groups=embed_dims, dilation=2)
        elif k == 23:
            self.DW_conv_h = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=(1,5), padding=(0, (5-1)//2), groups=embed_dims)
            self.DW_conv_v = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=(5,1), padding=((5-1)//2, 0), groups=embed_dims)
            self.DW_D_conv_h = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=(1,7), stride=(1,1), padding=(0,9), groups=embed_dims, dilation=3)
            self.DW_D_conv_v = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=(7,1), stride=(1,1), padding=(9,0), groups=embed_dims, dilation=3)
        elif k == 35:
            self.DW_conv_h = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=(1,5), padding=(0, (5-1)//2), groups=embed_dims)
            self.DW_conv_v = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=(5,1), padding=((5-1)//2, 0), groups=embed_dims)
            self.DW_D_conv_h = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=(1,11), stride=(1,1), padding=(0,15), groups=embed_dims, dilation=3)
            self.DW_D_conv_v = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=(11,1), stride=(1,1), padding=(15,0), groups=embed_dims, dilation=3)
        elif k == 41:
            self.DW_conv_h = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=(1,5), padding=(0, (5-1)//2), groups=embed_dims)
            self.DW_conv_v = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=(5,1), padding=((5-1)//2, 0), groups=embed_dims)
            self.DW_D_conv_h = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=(1,13), stride=(1,1), padding=(0,18), groups=embed_dims, dilation=3)
            self.DW_D_conv_v = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=(13,1), stride=(1,1), padding=(18,0), groups=embed_dims, dilation=3)
        elif k == 53:
            self.DW_conv_h = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=(1,5), padding=(0, (5-1)//2), groups=embed_dims)
            self.DW_conv_v = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=(5,1), padding=((5-1)//2, 0), groups=embed_dims)
            self.DW_D_conv_h = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=(1,17), stride=(1,1), padding=(0,24), groups=embed_dims, dilation=3)
            self.DW_D_conv_v = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=(17,1), stride=(1,1), padding=(24,0), groups=embed_dims, dilation=3)

        self.conv1 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

    def forward(self, x):
        u = x.clone()
        attn = self.DW_conv_h(x)
        attn = self.DW_conv_v(attn)
        attn = self.DW_D_conv_h(attn)
        attn = self.DW_D_conv_v(attn)

        attn = self.conv1(attn)

        return u * attn



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
