# -*- encoding: utf-8 -*-
"""
@Time: 2024-04-18 18:14
@Auth: xjjxhxgg
@File: dtin.py
@IDE: PyCharm
@Motto: xhxgg
"""
from gd.nn.modules.conv import Conv
import torch
from torch import nn
import torch.nn.functional as F


class DTIN(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, router=False) -> None:
        super(DTIN, self).__init__()
        self.conv1 = Conv(c1, c2, 1)
        self.bn1 = nn.BatchNorm2d(c2)
        self.router = router
        self.only_wb = False  # if True then the dynamic convolution module only produces weight ans bias, or convolve after produces weight ans bias and only returns the product
        if self.router:
            self.dynamic_conv = ODConv2d(c2, c2, k, s, p, g, d, act, instance_norm=True)
        else:
            self.dynamic_conv = DyConv(c2, c2, k, s, p, g, d, only_wb=self.only_wb, instance_norm=True)
        if self.only_wb:
            self.instance_norm = nn.InstanceNorm2d(c2, affine=False)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.bn2 = nn.BatchNorm2d(c2)
        self.conv2 = Conv(c2, c2, 1)
        self.stride = s
        self.dilation = d
        self.groups = g
        self.out_planes = c2

    def _dyconv(self, x):
        batch_size = x.size(0)
        weight, bias = self.dynamic_conv(x)
        x = self.instance_norm(x)
        padding = autopad(weight.shape[2], None)
        output = F.conv2d(x, weight=weight, bias=bias, stride=self.stride, padding=padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output

    def _odconv(self, x):
        batch_size = x.size(0)
        weight, filter_attention = self.dynamic_conv(x)
        weight_size = weight.shape[2]
        self.padding = autopad(weight_size, None)
        output = F.conv2d(x, weight=weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = output * filter_attention
        return output

    def _dyconv_forward(self, x):
        return self.dynamic_conv(x)

    def _dtin(self, x):
        if not self.only_wb:
            return self._dyconv_forward(x)
        elif isinstance(self.dynamic_conv, ODConv2d):
            return self._odconv(x)
        elif isinstance(self.dynamic_conv, DyConv):
            return self._dyconv(x)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self._dtin(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.conv2(x)
        return x


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.avgpool(x)
        x = self.fc(x)
        if batch_size > 1:
            x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)


class ODConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 reduction=0.0625, kernel_num=4, only_wb=False, instance_norm=True):
        super(ODConv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = Attention(in_planes, out_planes, kernel_size, groups=groups,
                                   reduction=reduction, kernel_num=kernel_num)
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes // groups, kernel_size, kernel_size),
                                   requires_grad=True)
        self._initialize_weights()
        self.instance_norm = instance_norm
        if only_wb:
            self._forward = self._forward_wb
        else:
            self._forward = self._forward_conv
            if instance_norm:
                self.instance_norm = nn.InstanceNorm2d(in_planes, affine=False)

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    def _forward_wb(self, x):
        # Multiplying channel attention (or filter attention) to weights and feature maps are equivalent,
        # while we observe that when using the latter method the models will run faster with less gpu memory cost.
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x * channel_attention
        x = x.reshape(1, -1, height, width)
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
        return aggregate_weight, filter_attention

    def _forward_conv(self, x):
        # Multiplying channel attention (or filter attention) to weights and feature maps are equivalent,
        # while we observe that when using the latter method the models will run faster with less gpu memory cost.
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        if self.instance_norm:
            x = self.instance_norm(x)
        x = x * channel_attention
        x = x.reshape(1, -1, height, width)
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
        weight_size = aggregate_weight.shape[2]
        self.padding = autopad(weight_size, None)
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = output * filter_attention
        return output

    def forward(self, x):
        return self._forward(x)


class AttentionDyConv(nn.Module):
    def __init__(self, in_planes, K, ):
        super(AttentionDyConv, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, K, 1, )
        self.fc2 = nn.Conv2d(K, K, 1, )

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x, 1)


class DyConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4,
                 only_wb=False, instance_norm=True):
        super(DyConv, self).__init__()
        assert in_planes % groups == 0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = AttentionDyConv(in_planes, K)
        self.instance_norm = instance_norm
        self.weight = nn.Parameter(torch.Tensor(K, out_planes, in_planes // groups, kernel_size, kernel_size),
                                   requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_planes))
        else:
            self.bias = None

        if only_wb:
            self._forward = self._forward_wb
        else:
            self._forward = self._forward_conv
            if instance_norm:
                self.instance_norm = nn.InstanceNorm2d(in_planes, affine=False)

    def _forward_wb(self, x):  # 将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        softmax_attention = self.attention(x)
        weight = self.weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
        bias = self.bias
        self.padding = autopad(aggregate_weight.shape[2], None)
        if bias is not None:
            bias = torch.mm(softmax_attention, bias).view(-1)

        return aggregate_weight, bias

    def _forward_conv(self, x):  # 将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        softmax_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        if self.instance_norm:
            x = self.instance_norm(x)
        x = x.view(1, -1, height, width)  # 变化成一个维度进行组卷积
        weight = self.weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
        padding = autopad(aggregate_weight.shape[2], None)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=padding,
                              dilation=self.dilation, groups=self.groups * batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output

    def forward(self, x):
        return self._forward(x)
