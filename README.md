<div align="center">

# 🛰️ TerraGAN

### Multi-Image Super-Resolution for Remote Sensing Satellite Imagery

[![Status](https://img.shields.io/badge/status-active%20development-brightgreen?style=flat-square)](https://github.com/Prashant822k/Improved-SRGAN-for-multi-image-super-resolution)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/Prashant822k/Improved-SRGAN-for-multi-image-super-resolution?style=flat-square)](https://github.com/Prashant822k/Improved-SRGAN-for-multi-image-super-resolution/commits/main)
[![Lint](https://github.com/Prashant822k/Improved-SRGAN-for-multi-image-super-resolution/actions/workflows/lint.yml/badge.svg)](https://github.com/Prashant822k/Improved-SRGAN-for-multi-image-super-resolution/actions/workflows/lint.yml)

</div>

---

## Overview

TerraGAN is a **multi-image super-resolution framework** for remote sensing satellite imagery. It addresses a core limitation of single-image SR methods — the inability to exploit complementary spatial information across multiple observations of the same scene — by **fusing 4 aligned Low-Resolution (LR) frames** into a single 4× High-Resolution (HR) output.

At its core, TerraGAN combines an **RRDB-style deep generator** with **LSK (Large Selective Kernel) multi-scale attention** and a **multi-component GAN loss** (pixel + perceptual + adversarial + LPIPS), trained end-to-end on the curated WorldStrat satellite dataset.

---

## Motivation & Research Gap

Existing SR models — SRGAN, ESRGAN, Real-ESRGAN, SRFlow — are designed for single-image inputs. When applied to multi-temporal satellite imagery they suffer from:

- **Texture loss** — hallucinated fine details that don't correspond to the real scene
- **Misalignment artifacts** — when multiple LR frames are naively concatenated without alignment
- **Poor structural consistency** — especially over repetitive terrain textures (agriculture, water, urban grids)

TerraGAN addresses these limitations by:
1. **Multi-frame fusion** — concatenating 4 curated LR frames as a 12-channel input tensor
2. **RRDB backbone** — deep residual feature learning with scaled skip connections
3. **LSK attention** — multi-scale (3×3, 5×5, 7×7) convolutions that selectively weight texture-rich regions
4. **Composite loss** — jointly optimising for pixel fidelity (L1), perceptual structure (VGG19), realism (adversarial), and perceptual distance (LPIPS)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        TerraGAN Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   [4 LR Frames]  ──→  Concat (12-ch)  ──→  Conv(12→64, 9×9)  │
│   64×64 each                                       │            │
│                                              16× ResidualBlock  │
│                                                    │            │
│                                            LSK Attention        │
│                                       (conv3 + conv5 + conv7)  │
│                                                    │            │
│                                          PixelShuffle ×2        │
│                                         (64→128→256 spatial)   │
│                                                    │            │
│                                          Conv(64→3, 9×9) + Tanh │
│                                                    │            │
│                                          [SR Output 256×256]   │
│                                                    │            │
│   ┌────────────────────────────────────────────────┘            │
│   │                 GAN Training Loop                           │
│   ├─ Generator Loss = L1 + 0.01×VGG19 + 0.005×Adv + 0.01×LPIPS│
│   └─ Discriminator Loss = BCE(real=1, fake=0)                   │
└─────────────────────────────────────────────────────────────────┘
```

See [`docs/architecture.md`](docs/architecture.md) for the full Mermaid diagram and [`docs/methodology.md`](docs/methodology.md) for a complete methodology write-up.

---

## Key Features

| Feature | TerraGAN | Baseline SRGAN |
|---|---|---|
| Input frames | **4 LR images (12-channel)** | 1 LR image (3-channel) |
| Backbone | **16× RRDB blocks** | 16× ResNet blocks |
| Attention | **LSK multi-scale (3,5,7)** | None |
| Loss | **L1 + VGG19 + Adv + LPIPS** | MSE + Adv |
| Dataset | **WorldStrat (multi-temporal)** | DIV2K / ImageNet |
| Upscale factor | **4×** (64→256) | 4× |

---

## Dataset

TerraGAN is trained on a curated subset of the **[WorldStrat](https://github.com/worldstrat/worldstrat) satellite dataset**.

**Original dataset:**
- ~3,000 scene folders, each containing 1 HR image + 16 LR observations

**Curation pipeline:**
1. Convert all images to RGB (3-channel)
2. Compute pixel-wise similarity scores between each LR frame and the HR reference
3. Select the **top 4 LR frames** per scene (best similarity score)
4. Quality filter: remove cloud-covered, noisy, corrupted, and low-quality samples
5. Result: **~1,800 high-quality HR–4LR pairs**

**Preprocessing:**
- LR: resized to **64×64**, normalised to [-1, 1]
- HR: resized to **256×256**, normalised to [-1, 1]

**Train / Val / Test split (70 / 15 / 15):**
| Split | Samples |
|-------|---------|
| Train | 1,257   |
| Val   | 269     |
| Test  | 271     |

---

## Results

### Quantitative Metrics (100 Epochs, Test Set)

| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Edge Score | Inference Time |
|-------|--------|--------|---------|------------|---------------|
| **TerraGAN (ours)** | **20.79 dB** | **0.6577** | **0.3037** | 0.0145 | ~0.92 sec/tile |

> Baseline comparison metrics (SRGAN, ESRGAN) are currently being evaluated and will be added here soon.

### Visual Comparisons

> 📂 Sample comparison images (LR input | TerraGAN output | HR ground truth) will be added to [`assets/results/`](assets/results/) — see the folder README for upload instructions.

---

## Installation & Usage

### 1. Clone the repo

```bash
git clone https://github.com/Prashant822k/Improved-SRGAN-for-multi-image-super-resolution.git
cd Improved-SRGAN-for-multi-image-super-resolution
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** A GPU (CUDA) is strongly recommended for training. Inference runs on CPU but will be slower.

### 3. Training

```bash
python src/train.py --config configs/default.yaml
```

To resume from a checkpoint:

```bash
python src/train.py --config configs/default.yaml --resume checkpoints/terragan_latest.pth
```

Edit [`configs/default.yaml`](configs/default.yaml) to set your `dataset.path` and other hyperparameters.

### 4. Inference

```bash
python src/inference.py \
    --lr_dir  /path/to/tile_folder \
    --checkpoint checkpoints/terragan_latest.pth \
    --output  output/sr_result.png
```

The `tile_folder` must contain: `LR_01.png`, `LR_02.png`, `LR_03.png`, `LR_04.png`

### 5. Evaluation

```bash
python scripts/evaluate.py \
    --dataset_path /path/to/SRGAN_FINAL_V1 \
    --checkpoint   checkpoints/terragan_latest.pth
```

---

## Project Structure

```
Improved-SRGAN-for-multi-image-super-resolution/
├── src/
│   ├── models/
│   │   ├── generator.py       # RRDB + LSK generator (12-ch → 3-ch, 4× upscale)
│   │   └── discriminator.py   # PatchGAN discriminator
│   ├── data/
│   │   └── dataset.py         # WorldStratDataset + train/val/test split
│   ├── losses/
│   │   └── losses.py          # Composite GAN loss (L1 + VGG19 + Adv + LPIPS)
│   ├── train.py               # Training entry point
│   └── inference.py           # Single-tile inference entry point
├── scripts/
│   └── evaluate.py            # PSNR / SSIM / LPIPS / Edge Score evaluation
├── configs/
│   └── default.yaml           # All hyperparameters (lr, batch_size, loss weights…)
├── docs/
│   ├── architecture.md        # Mermaid pipeline diagram
│   └── methodology.md         # Full methodology write-up
├── assets/
│   └── results/               # Visual comparison images (upload manually)
├── checkpoints/               # Model weights (not committed — see checkpoints/README.md)
├── notebooks/                 # Training notebooks (see Google Colab link)
├── .github/
│   └── workflows/
│       └── lint.yml           # CI: flake8 + black on every push
├── requirements.txt
├── CHANGELOG.md
├── CONTRIBUTING.md
└── LICENSE
```

---

## Roadmap

Active improvements in progress:

- [ ] **Deformable alignment** — learnable frame registration before multi-frame fusion ([#1](https://github.com/Prashant822k/Improved-SRGAN-for-multi-image-super-resolution/issues/1))
- [ ] **Relativistic discriminator** — RaGAN-style adversarial training for more stable GAN dynamics ([#2](https://github.com/Prashant822k/Improved-SRGAN-for-multi-image-super-resolution/issues/2))
- [ ] **Baseline comparison metrics** — PSNR/SSIM/LPIPS table vs. SRGAN/ESRGAN/Real-ESRGAN ([#3](https://github.com/Prashant822k/Improved-SRGAN-for-multi-image-super-resolution/issues/3))
- [ ] **Reference-guided texture transfer** — TTSR-style texture matching from reference frames ([#4](https://github.com/Prashant822k/Improved-SRGAN-for-multi-image-super-resolution/issues/4))
- [ ] **Public model checkpoint + Colab inference demo** — one-click inference for anyone ([#5](https://github.com/Prashant822k/Improved-SRGAN-for-multi-image-super-resolution/issues/5))

See [CHANGELOG.md](CHANGELOG.md) for version history.

---

## Team & Acknowledgements

**Developed by** (NMIT, Dept. of Information Science and Engineering):
- Manisha Chaubey
- Prashant Kumar
- Priyanshu Harshvardhan
- Rakshitha G

**Project Guide:** Dr. Deepthi K

**Dataset:** [WorldStrat — Cornebise et al.](https://github.com/worldstrat/worldstrat)

---

## References

- Ledig et al. — [Photo-Realistic Single Image Super-Resolution Using a GAN (SRGAN)](https://arxiv.org/abs/1609.04802), CVPR 2017
- Wang et al. — [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/abs/1809.00219), ECCVW 2018
- Wang et al. — [Real-ESRGAN: Training Real-World Blind Super-Resolution](https://arxiv.org/abs/2107.10833), ICCV 2021
- Chan et al. — [BasicVSR: The Search for Essential Components in Video Super-Resolution](https://arxiv.org/abs/2012.02181), CVPR 2021
- Li et al. — [Large Selective Kernel Network for Remote Sensing Object Detection](https://arxiv.org/abs/2303.09030)
- Zhang et al. — [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS)](https://arxiv.org/abs/1801.03924), CVPR 2018
- Cornebise et al. — [Open High-Resolution Satellite Imagery: The WorldStrat Dataset](https://arxiv.org/abs/2207.06418), NeurIPS 2022
- Yang et al. — [TTSR: Learning Texture Transformer Network for Image Super-Resolution](https://arxiv.org/abs/2006.04139), CVPR 2020

---

## License

This project is licensed under the [MIT License](LICENSE).