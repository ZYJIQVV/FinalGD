# -*- encoding: utf-8 -*-
"""
@Time: 2024-04-18 18:15
@Auth: xjjxhxgg
@File: fft_norm.py
@IDE: PyCharm
@Motto: xhxgg
"""
import torch
from torch import nn
from gd.nn.modules.conv import Conv


class FFTNorm(nn.Module):
    def __init__(self, c2, lambda_=0.5) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(c2)
        self.lmd = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def fourier_transform(self, x: torch.Tensor) -> torch.Tensor:
        return torch.fft.fftn(x, dim=(2, 3))

    def inverse_fourier_transform(self, x: torch.Tensor) -> torch.Tensor:
        return torch.fft.ifftn(x, dim=(2, 3)).real

    def compose(self, alpha: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        real = alpha * torch.cos(phase)
        imag = alpha * torch.sin(phase)
        return torch.complex(real, imag)

    def decompose(self, x: torch.Tensor) -> torch.Tensor:
        real = x.real
        imag = x.imag
        alpha = torch.sqrt(real ** 2 + imag ** 2)
        phase = torch.atan2(imag, real)
        return alpha, phase


class PCNorm(FFTNorm):
    def __init__(self, c2) -> None:
        super().__init__(c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        F = self.fourier_transform(x)
        F_norm = self.fourier_transform(x_norm)
        alpha, phase = self.decompose(F)
        alpha_norm, phase_norm = self.decompose(F_norm)
        alpha_com = alpha * (1 - self.lmd) + alpha_norm * self.lmd
        phase_com = phase * (1 - self.lmd) + phase_norm * self.lmd
        composed = self.compose(alpha_com, phase_com)
        return self.inverse_fourier_transform(composed)


class SCNorm(FFTNorm):
    def __init__(self, c2) -> None:
        super().__init__(c2)
        self.norm = nn.InstanceNorm2d(c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        F = self.fourier_transform(x)
        F_norm = self.fourier_transform(x_norm)
        alpha, phase = self.decompose(F)
        alpha_norm, phase_norm = self.decompose(F_norm)
        alpha_com = alpha * (1 - self.lmd) + alpha_norm * self.lmd
        phase_com = phase * (1 - self.lmd) + phase_norm * self.lmd
        composed = self.compose(alpha_com, phase_com)
        return self.inverse_fourier_transform(composed)


class PCBlock(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True) -> None:
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, s, p, g, d, act)
        self.conv2 = Conv(c2, c2, k, s, p, g, d, act)
        self.conv3 = Conv(c2, c2, 1, s, p, g, d, act)
        self.conv4 = Conv(c1, c2, 1, s, p, g, d, act)
        self.pcnorm = PCNorm(c2)

    def forward(self, x):
        return self.pcnorm(self.conv4(x)) + self.conv3(self.conv2(self.conv1(x)))


class SCBlock(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True) -> None:
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, s, p, g, d, act)
        self.conv2 = Conv(c2, c2, k, s, p, g, d, act)
        self.conv3 = Conv(c2, c2, 1, s, p, g, d, act)
        self.bn = nn.BatchNorm2d(c2)
        self.scnorm = SCNorm(c2)

    def forward(self, x):
        return self.scnorm(x + self.bn(self.conv3(self.conv2(self.conv1(x)))))