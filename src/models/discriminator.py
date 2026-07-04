"""
TerraGAN Discriminator
======================
PatchGAN-style discriminator that classifies overlapping patches of an image
as real or fake, encouraging local texture fidelity.

Architecture: 7 convolutional blocks (Conv + LeakyReLU) with alternating
strides, ending in a single-channel logit map.
"""

import torch
import torch.nn as nn


def _conv_block(in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
    """Single discriminator block: Conv2d + LeakyReLU(0.2)."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
    )


class Discriminator(nn.Module):
    """TerraGAN Discriminator (PatchGAN-style).

    Takes a 3-channel HR or SR image and outputs a spatial logit map.
    Used with BCEWithLogitsLoss during adversarial training.

    Input shape:  (B, 3, 256, 256)
    Output shape: (B, 1, 32, 32)  — logit map over 32×32 patches
    """

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            _conv_block(3,   64,  stride=1),   # (B, 64,  256, 256)
            _conv_block(64,  64,  stride=2),   # (B, 64,  128, 128)
            _conv_block(64,  128, stride=1),   # (B, 128, 128, 128)
            _conv_block(128, 128, stride=2),   # (B, 128, 64,  64)
            _conv_block(128, 256, stride=1),   # (B, 256, 64,  64)
            _conv_block(256, 256, stride=2),   # (B, 256, 32,  32)
            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1),  # (B, 1, 32, 32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor of shape (B, 3, H, W).

        Returns:
            Logit map of shape (B, 1, H/8, W/8).
        """
        return self.model(x)
