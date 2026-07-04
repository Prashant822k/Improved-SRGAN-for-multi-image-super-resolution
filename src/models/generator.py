"""
TerraGAN Generator
==================
Architecture:
  - Input:  12-channel tensor (4 concatenated LR images, each 3-channel)
  - Body:   16 RRDB-style Residual Blocks + LSK Multi-Scale Attention
  - Output: 3-channel SR image at 4× spatial resolution via 2× PixelShuffle stages
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """RRDB-style residual block with scaled residual connection.

    Args:
        channels (int): Number of feature channels. Default: 64.
    """

    def __init__(self, channels: int = 64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.1 * self.block(x)


class LSKAttention(nn.Module):
    """Large Selective Kernel (LSK) Attention Module.

    Aggregates multi-scale spatial context using parallel depthwise convolutions
    at kernel sizes 3, 5, and 7 to selectively emphasise texture-rich regions.

    Args:
        channels (int): Number of feature channels.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(channels, channels, kernel_size=7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.conv3(x)
        b = self.conv5(x)
        c = self.conv7(x)
        return x + a + b + c


class Generator(nn.Module):
    """TerraGAN Generator.

    Takes a 12-channel input (4 LR satellite frames concatenated along the
    channel dimension) and produces a 3-channel 4× super-resolved output.

    Architecture summary:
        Conv(12→64, 9×9)
        → 16× ResidualBlock(64)
        → LSKAttention(64)
        → Conv(64→64, 3×3)          [global skip connection]
        → PixelShuffle(×2) ×2       [64→256 spatial upsampling]
        → Conv(64→3, 9×9) + Tanh
    """

    def __init__(self):
        super().__init__()

        # Initial feature extraction from 12-channel multi-frame input
        self.initial = nn.Conv2d(12, 64, kernel_size=9, stride=1, padding=4)

        # 16 RRDB-style residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(16)]
        )

        # LSK multi-scale attention
        self.lsk = LSKAttention(64)

        # Post-residual convolution (before global skip)
        self.conv_mid = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Upsampling stage 1: 64×64 → 128×128
        self.up1 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
        )

        # Upsampling stage 2: 128×128 → 256×256
        self.up2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
        )

        # Output convolution
        self.final = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, 12, H, W) — 4 LR frames concatenated.

        Returns:
            Tensor of shape (B, 3, 4H, 4W) — super-resolved output in [-1, 1].
        """
        x1 = self.initial(x)      # (B, 64, H, W)
        x = self.res_blocks(x1)   # (B, 64, H, W)
        x = self.lsk(x)           # (B, 64, H, W)
        x = self.conv_mid(x)      # (B, 64, H, W)
        x = x + x1                # global skip connection
        x = self.up1(x)           # (B, 64, 2H, 2W)
        x = self.up2(x)           # (B, 64, 4H, 4W)
        x = self.final(x)         # (B, 3, 4H, 4W)
        return torch.tanh(x)
