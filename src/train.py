"""
TerraGAN Training Script
========================
Trains the TerraGAN generator and discriminator in Phase 3 (joint GAN training).

Usage:
    python src/train.py --config configs/default.yaml
    python src/train.py --config configs/default.yaml --resume /path/to/checkpoint.pth

The script supports resuming from a full checkpoint (generator + discriminator +
optimiser states) or a generator-only weights file.
"""

import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from src.models.generator import Generator
from src.models.discriminator import Discriminator
from src.losses.losses import TerraGANLoss, DiscriminatorLoss
from src.data.dataset import build_splits


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TerraGAN")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint (.pth) to resume from",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: dict, resume_path: str | None = None) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    train_set, val_set, _ = build_splits(
        root_dir=cfg["dataset"]["path"],
        train_ratio=cfg["dataset"]["train_ratio"],
        val_ratio=cfg["dataset"]["val_ratio"],
        seed=cfg["dataset"]["seed"],
    )
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"].get("num_workers", 0),
    )

    # ── Models ────────────────────────────────────────────────────────────
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # ── Losses ────────────────────────────────────────────────────────────
    g_criterion = TerraGANLoss(
        device=device,
        w_pixel=cfg["loss"]["w_pixel"],
        w_perceptual=cfg["loss"]["w_perceptual"],
        w_adversarial=cfg["loss"]["w_adversarial"],
        w_lpips=cfg["loss"]["w_lpips"],
    )
    d_criterion = DiscriminatorLoss()

    # ── Optimizers ────────────────────────────────────────────────────────
    g_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=cfg["training"]["lr"],
        betas=(0.9, 0.999),
    )
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=cfg["training"]["lr"],
        betas=(0.9, 0.999),
    )

    # ── Checkpoint Resume ─────────────────────────────────────────────────
    start_epoch = 0
    checkpoint_path = resume_path or cfg["training"].get("checkpoint_path")
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        if isinstance(ckpt, dict) and "generator" in ckpt:
            generator.load_state_dict(ckpt["generator"])
            discriminator.load_state_dict(ckpt["discriminator"])
            g_optimizer.load_state_dict(ckpt["g_optimizer"])
            d_optimizer.load_state_dict(ckpt["d_optimizer"])
            start_epoch = ckpt.get("epoch", 0) + 1
            print(f"Resuming from epoch {start_epoch}")
        else:
            generator.load_state_dict(ckpt)
            print("Loaded generator-only weights")

    # ── Output dirs ───────────────────────────────────────────────────────
    sample_dir = cfg["training"]["sample_dir"]
    ckpt_dir = cfg["training"]["checkpoint_dir"]
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    total_epochs = cfg["training"]["epochs"]

    # ── Training Loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, total_epochs):
        generator.train()
        discriminator.train()
        loop = tqdm(train_loader, desc=f"Phase3 [{epoch + 1}/{total_epochs}]")

        for lr_stack, hr in loop:
            lr_stack = lr_stack.to(device)
            hr = hr.to(device)

            # ── Discriminator step ──────────────────────────────────────
            fake = generator(lr_stack)
            real_pred = discriminator(hr)
            fake_pred = discriminator(fake.detach())
            d_loss = d_criterion(real_pred, fake_pred)

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # ── Generator step ──────────────────────────────────────────
            fake_pred_for_g = discriminator(fake)
            g_loss = g_criterion(fake, hr, fake_pred_for_g)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            loop.set_postfix(g_loss=f"{g_loss.item():.4f}", d_loss=f"{d_loss.item():.4f}")

        # ── Save sample ─────────────────────────────────────────────────
        save_image(
            fake[:1],
            os.path.join(sample_dir, f"sample_epoch_{epoch:03d}.png"),
            normalize=True,
        )

        # ── Save checkpoint ─────────────────────────────────────────────
        ckpt_file = os.path.join(ckpt_dir, "terragan_latest.pth")
        try:
            torch.save(
                {
                    "epoch": epoch,
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "g_optimizer": g_optimizer.state_dict(),
                    "d_optimizer": d_optimizer.state_dict(),
                },
                ckpt_file,
            )
        except Exception as e:
            print(f"Checkpoint save failed: {e}")

    print("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    train(cfg, resume_path=args.resume)
