"""
TerraGAN Loss Functions
=======================
Composite loss for training the TerraGAN generator:

    L_total = L1(fake, hr)
            + 0.01  × L_perceptual(VGG19 features)
            + 0.005 × L_adversarial(BCEWithLogits)
            + 0.01  × L_LPIPS(AlexNet)

Weights are configurable via the config YAML.
"""

import torch
import torch.nn as nn
import lpips
from torchvision.models import vgg19, VGG19_Weights


class PerceptualLoss(nn.Module):
    """VGG19-based perceptual loss.

    Computes L1 distance between VGG19 feature maps (up to relu5_4)
    of the generated and ground-truth images.

    Args:
        device: Torch device string or object.
    """

    def __init__(self, device: str = "cpu"):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:35].eval().to(device)
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.l1 = nn.L1Loss()

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        return self.l1(self.vgg(fake), self.vgg(real))


class TerraGANLoss(nn.Module):
    """Composite generator loss combining pixel, perceptual, adversarial, and LPIPS terms.

    Args:
        device:          Torch device.
        w_pixel:         Weight for pixel (L1) loss.       Default: 1.0
        w_perceptual:    Weight for VGG19 perceptual loss. Default: 0.01
        w_adversarial:   Weight for adversarial loss.      Default: 0.005
        w_lpips:         Weight for LPIPS loss.            Default: 0.01
    """

    def __init__(
        self,
        device: str = "cpu",
        w_pixel: float = 1.0,
        w_perceptual: float = 0.01,
        w_adversarial: float = 0.005,
        w_lpips: float = 0.01,
    ):
        super().__init__()
        self.w_pixel = w_pixel
        self.w_perceptual = w_perceptual
        self.w_adversarial = w_adversarial
        self.w_lpips = w_lpips

        self.pixel_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss(device=device)
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.lpips_loss = lpips.LPIPS(net="alex").to(device)

    def forward(
        self,
        fake: torch.Tensor,
        real: torch.Tensor,
        fake_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            fake:      Generator output, shape (B, 3, H, W).
            real:      HR ground truth, shape (B, 3, H, W).
            fake_pred: Discriminator logits for fake images, shape (B, 1, h, w).

        Returns:
            Scalar total generator loss.
        """
        l_pixel = self.pixel_loss(fake, real)
        l_perceptual = self.perceptual_loss(fake, real)
        l_adv = self.adversarial_loss(fake_pred, torch.ones_like(fake_pred))
        l_lpips = self.lpips_loss(fake, real).mean()

        return (
            self.w_pixel * l_pixel
            + self.w_perceptual * l_perceptual
            + self.w_adversarial * l_adv
            + self.w_lpips * l_lpips
        )


class DiscriminatorLoss(nn.Module):
    """Standard adversarial discriminator loss (real vs. fake BCE)."""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            real_pred: Discriminator logits for real HR images.
            fake_pred: Discriminator logits for fake SR images.

        Returns:
            Scalar discriminator loss.
        """
        l_real = self.bce(real_pred, torch.ones_like(real_pred))
        l_fake = self.bce(fake_pred, torch.zeros_like(fake_pred))
        return l_real + l_fake
