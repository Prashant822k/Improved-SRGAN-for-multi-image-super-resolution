"""
TerraGAN Evaluation Script
===========================
Computes PSNR, SSIM, LPIPS, and Edge Score on the test split.

Usage:
    python scripts/evaluate.py \\
        --dataset_path /path/to/SRGAN_FINAL_V1 \\
        --checkpoint   checkpoints/terragan_latest.pth
"""

import argparse

import lpips
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.generator import Generator
from src.data.dataset import build_splits


def edge_score(img: np.ndarray) -> float:
    """Compute mean gradient magnitude as a proxy for edge sharpness.

    Args:
        img: Numpy array of shape (H, W, 3), values in [0, 1].

    Returns:
        Mean Sobel-like gradient magnitude (scalar).
    """
    gray = img.mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    return float(np.mean(gx) + np.mean(gy))


def evaluate(dataset_path: str, checkpoint_path: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── Model ──────────────────────────────────────────────────────────────
    generator = Generator().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "generator" in ckpt:
        generator.load_state_dict(ckpt["generator"])
    else:
        generator.load_state_dict(ckpt)
    generator.eval()

    # ── Data ───────────────────────────────────────────────────────────────
    _, _, test_set = build_splits(dataset_path)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    lpips_fn = lpips.LPIPS(net="alex").to(device)

    # ── Metrics accumulators ───────────────────────────────────────────────
    psnr_scores, ssim_scores, lpips_scores, edge_scores = [], [], [], []

    with torch.no_grad():
        for lr_stack, hr in tqdm(test_loader, desc="Evaluating"):
            lr_stack = lr_stack.to(device)
            hr = hr.to(device)

            fake = generator(lr_stack)

            # Convert tensors from [-1,1] → [0,1] numpy for pixel metrics
            fake_np = ((fake[0].permute(1, 2, 0).cpu().numpy() + 1) / 2).clip(0, 1)
            hr_np = ((hr[0].permute(1, 2, 0).cpu().numpy() + 1) / 2).clip(0, 1)

            psnr_scores.append(psnr(hr_np, fake_np, data_range=1.0))
            ssim_scores.append(ssim(hr_np, fake_np, channel_axis=2, data_range=1.0))
            lpips_scores.append(lpips_fn(fake, hr).item())
            edge_scores.append(edge_score(fake_np))

    print("\n" + "=" * 30)
    print("FINAL METRICS")
    print("=" * 30)
    print(f"PSNR:           {np.mean(psnr_scores):.4f} dB")
    print(f"SSIM:           {np.mean(ssim_scores):.4f}")
    print(f"LPIPS:          {np.mean(lpips_scores):.4f}")
    print(f"Edge Score:     {np.mean(edge_scores):.6f}")
    print("=" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TerraGAN Evaluation")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    evaluate(args.dataset_path, args.checkpoint)
