"""
Variational Autoencoder (VAE) Module.
Implements probabilistic latent representation, reparameterization trick, ELBO loss,
and latent manifold navigation for visual illustration synthesis.
References: Modul Praktikum 4 & Lecture Notes 03 (Solanki & Khublani 2024).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Optional
from src.utils.reproducibility import set_seed


class ConvVAE(nn.Module):
    """
    Convolutional Variational Autoencoder tailored for 64x64 illustration patches.
    Encoder maps (3, 64, 64) -> (mu, logvar) of latent dimension nz.
    Decoder maps nz -> (3, 64, 64) reconstructed visual.
    """

    def __init__(self, latent_dim: int = 32, in_channels: int = 3):
        super(ConvVAE, self).__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels

        # Encoder: 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),           # 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),          # 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),         # 4x4
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        self.flatten_dim = 256 * 4 * 4
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, self.flatten_dim)
        self.decoder_deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1), # 64x64
            nn.Sigmoid()  # output normalized between [0, 1]
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Maps input image tensor to Gaussian parameters: mean (mu) and log-variance (logvar).
        """
        h = self.encoder_conv(x)
        h_flat = h.view(h.size(0), -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization Trick: z = mu + eps * std, where eps ~ N(0, I).
        Enables backpropagation through stochastic sampling.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs image from latent vector z.
        """
        h = self.decoder_fc(z)
        h = h.view(h.size(0), 256, 4, 4)
        return self.decoder_deconv(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


def vae_loss_function(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kld_weight: float = 0.001
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ELBO Loss = Reconstruction Loss (MSE/BCE) + KLD Regularization.
    """
    recon_loss = F.mse_loss(recon_x, x, reduction="sum") / x.size(0)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    total_loss = recon_loss + kld_weight * kld_loss
    return total_loss, recon_loss, kld_loss


def generate_latent_manifold_grid(
    vae_model: ConvVAE,
    device: str = "cpu",
    grid_size: int = 8,
    save_path: Optional[str] = "outputs/visual_evaluations/vae_manifold_grid.png"
) -> np.ndarray:
    """
    Generates a 2D manifold grid from the VAE latent space (referencing Modul Praktikum 4 page 42).
    Traverses dimensions 0 and 1 while holding others at 0.
    """
    vae_model.eval()
    x = np.linspace(-2.0, 2.0, grid_size)
    y = np.linspace(-2.0, 2.0, grid_size)
    canvas = np.zeros((grid_size * 64, grid_size * 64, 3))

    with torch.no_grad():
        for i, yi in enumerate(y):
            for j, xi in enumerate(x):
                z = torch.zeros(1, vae_model.latent_dim, device=device)
                z[0, 0] = float(xi)
                if vae_model.latent_dim > 1:
                    z[0, 1] = float(yi)
                sample = vae_model.decode(z).cpu().squeeze(0).permute(1, 2, 0).numpy()
                canvas[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64, :] = sample

    if save_path:
        plt.figure(figsize=(8, 8))
        plt.imshow(np.clip(canvas, 0, 1))
        plt.title("VAE 2D Latent Manifold Grid Traversal")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    return canvas

