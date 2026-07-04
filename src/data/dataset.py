"""
WorldStrat Dataset Loader
=========================
Loads curated HR–4LR image pairs from the TerraGAN-preprocessed WorldStrat dataset.

Each sample folder contains:
  HR.png    — 1 High-Resolution ground truth image
  LR_01.png — Best LR frame (selected by pixel-wise similarity scoring)
  LR_02.png
  LR_03.png
  LR_04.png

Curation notes (applied offline before training):
  - Original dataset: ~3000 folders, each with 1 HR + 16 LR images.
  - LR selection: 4 best frames chosen per folder via pixel-wise similarity
    scoring against the HR reference.
  - Quality filtering: removed cloud-covered, noisy, corrupted, and
    poor-quality samples → final ~1800 high-quality groups retained.
  - All images converted to RGB (3-channel) format.
"""

import os
from typing import Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class WorldStratDataset(Dataset):
    """PyTorch Dataset for TerraGAN multi-image super-resolution training.

    Args:
        root_dir (str): Path to the curated dataset root.
            Expected structure::

                root_dir/
                ├── tile_0001/
                │   ├── HR.png
                │   ├── LR_01.png
                │   ├── LR_02.png
                │   ├── LR_03.png
                │   └── LR_04.png
                ├── tile_0002/
                │   └── ...
    """

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.samples = sorted([
            os.path.join(root_dir, folder)
            for folder in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, folder))
        ])

        # HR transform: resize to 256×256, normalize to [-1, 1]
        self.hr_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

        # LR transform: resize to 64×64, normalize to [-1, 1]
        self.lr_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (lr_stack, hr) pair.

        Returns:
            lr_stack: Tensor of shape (12, 64, 64) — 4 LR images concatenated
                      along the channel dimension.
            hr:       Tensor of shape (3, 256, 256) — HR ground truth.
        """
        folder = self.samples[idx]

        # Load HR image
        hr_path = os.path.join(folder, "HR.png")
        hr = Image.open(hr_path).convert("RGB")
        hr = self.hr_transform(hr)

        # Load 4 LR images and concatenate along channel dim → (12, 64, 64)
        lr_imgs = []
        for i in range(1, 5):
            lr_path = os.path.join(folder, f"LR_0{i}.png")
            img = Image.open(lr_path).convert("RGB")
            img = self.lr_transform(img)
            lr_imgs.append(img)

        lr_stack = torch.cat(lr_imgs, dim=0)  # (12, 64, 64)
        return lr_stack, hr


def build_splits(
    root_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Split the full dataset into train / val / test subsets.

    Args:
        root_dir:    Path to curated dataset.
        train_ratio: Fraction for training (default 0.70).
        val_ratio:   Fraction for validation (default 0.15).
        seed:        Random seed for reproducibility.

    Returns:
        Tuple of (train_set, val_set, test_set) torch Datasets.

    Example split (on ~1797 samples):
        Train: 1257 | Val: 269 | Test: 271
    """
    from torch.utils.data import random_split
    import torch

    dataset = WorldStratDataset(root_dir)
    n = len(dataset)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    n_test = n - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )
    return train_set, val_set, test_set
