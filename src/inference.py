"""
TerraGAN Inference Script
=========================
Runs the trained TerraGAN generator on a folder of 4 LR satellite images
and saves the 4× super-resolved output.

Usage:
    python src/inference.py \\
        --lr_dir  /path/to/tile_folder \\
        --checkpoint checkpoints/terragan_latest.pth \\
        --output  output/sr_result.png

The tile_folder must contain: LR_01.png, LR_02.png, LR_03.png, LR_04.png
"""

import argparse
import os
import time

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from src.models.generator import Generator


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TerraGAN Inference")
    parser.add_argument(
        "--lr_dir",
        type=str,
        required=True,
        help="Directory containing LR_01.png … LR_04.png",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint (.pth)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/sr_result.png",
        help="Output path for the super-resolved image",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(lr_dir: str, checkpoint_path: str, output_path: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── Load model ─────────────────────────────────────────────────────────
    generator = Generator().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "generator" in ckpt:
        generator.load_state_dict(ckpt["generator"])
    else:
        generator.load_state_dict(ckpt)
    generator.eval()
    print(f"Checkpoint loaded: {checkpoint_path}")

    # ── Load LR images ─────────────────────────────────────────────────────
    lr_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    lr_imgs = []
    for i in range(1, 5):
        path = os.path.join(lr_dir, f"LR_0{i}.png")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing LR image: {path}")
        img = Image.open(path).convert("RGB")
        lr_imgs.append(lr_transform(img))

    lr_stack = torch.cat(lr_imgs, dim=0).unsqueeze(0).to(device)  # (1, 12, 64, 64)

    # ── Run generator ──────────────────────────────────────────────────────
    start = time.perf_counter()
    with torch.no_grad():
        sr = generator(lr_stack)
    elapsed = time.perf_counter() - start
    print(f"Inference time: {elapsed:.4f} sec")

    # ── Save output ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_image(sr, output_path, normalize=True)
    print(f"Super-resolved image saved: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    run_inference(
        lr_dir=args.lr_dir,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
    )
