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