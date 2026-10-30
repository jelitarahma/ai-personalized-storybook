"""
DCGAN (Deep Convolutional Generative Adversarial Network) Module.
Implements Generator and Discriminator architectures tailored for 64x64 children's story illustrations,
incorporating TTUR, one-sided label smoothing, and mode collapse detection.
References: Modul Praktikum 2 & Lecture Notes 04 (Solanki & Khublani 2024).
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional
from src.utils.reproducibility import set_seed


def weights_init(m):
    """
    Custom weight initialization for DCGAN (Normal(0.0, 0.02) for Conv/BatchNorm).
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    """
    DCGAN Generator: Transforms latent vector z (shape: nz x 1 x 1)
    into 64x64x3 synthetic RGB image with Tanh activation [-1, 1].
    """

    def __init__(self, nz: int = 100, ngf: int = 64, nc: int = 3):
        super(Generator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc

        self.main = nn.Sequential(
            # Input: nz x 1 x 1 -> ngf*8 x 4 x 4
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # ngf*8 x 4 x 4 -> ngf*4 x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # ngf*4 x 8 x 8 -> ngf*2 x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # ngf*2 x 16 x 16 -> ngf x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # ngf x 32 x 32 -> nc x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            z = z.view(z.size(0), self.nz, 1, 1)
        return self.main(z)


class Discriminator(nn.Module):
    """
    DCGAN Discriminator: Evaluates 64x64x3 RGB image into real (1) or fake (0) probability.
    """

    def __init__(self, nc: int = 3, ndf: int = 64):
        super(Discriminator, self).__init__()
        self.nc = nc
        self.ndf = ndf

        self.main = nn.Sequential(
            # Input: nc x 64 x 64 -> ndf x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # ndf x 32 x 32 -> ndf*2 x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # ndf*2 x 16 x 16 -> ndf*4 x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # ndf*4 x 8 x 8 -> ndf*8 x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # ndf*8 x 4 x 4 -> 1 x 1 x 1
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x).view(-1, 1).squeeze(1)


def build_dcgan_models(
    nz: int = 100,
    ngf: int = 64,
    ndf: int = 64,
    device: str = "cpu"
) -> Tuple[Generator, Discriminator]:
    """
    Helper to construct initialized Generator and Discriminator.
    """
    netG = Generator(nz=nz, ngf=ngf, nc=3).to(device)
    netD = Discriminator(nc=3, ndf=ndf).to(device)
    return netG, netD

