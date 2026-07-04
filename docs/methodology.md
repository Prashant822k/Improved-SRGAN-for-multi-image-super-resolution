# TerraGAN Methodology

## Full Methodology Write-Up

---

## 1. Problem Statement

Satellite remote sensing produces large volumes of Low-Resolution (LR) imagery
from sensors like Sentinel-2 (10m/pixel). A single overpass provides limited
spatial detail. However, multiple LR observations of the same scene across
different acquisition dates exist and are publicly available. The research
challenge is to **fuse these multi-temporal LR frames** into a single
photorealistic High-Resolution (HR) image, recovering structural and textural
detail lost during downsampling.

---

## 2. Literature Gap

| Model | Input | Backbone | Key Limitation |
|-------|-------|----------|----------------|
| SRGAN (Ledig et al., 2017) | Single LR | ResNet-16 | Single-image only; no multi-frame fusion |
| ESRGAN (Wang et al., 2018) | Single LR | RRDB | No multi-frame; limited satellite domain transfer |
| Real-ESRGAN (Wang et al., 2021) | Single LR | RRDB + degradation model | Focused on blind SR for natural images |
| SRFlow (Lugmayr et al., 2020) | Single LR | Normalising flow | Single-image; high compute |
| BasicVSR (Chan et al., 2021) | Video frames | Optical flow + EDVR | Designed for video SR; requires temporal sequence ordering |

**Gap**: No existing model combines **multi-frame satellite input** with
**deep RRDB feature extraction** and **multi-scale attention** tuned for
remote sensing texture patterns.

---

## 3. TerraGAN Framework — 9-Step Pipeline

### Step 1 — Data Acquisition
Download the WorldStrat dataset (~3,000 scene folders). Each scene contains:
- 1 HR image (SPOT-6/7, ~1.5m/pixel, resampled to 256×256)
- 16 LR images (Sentinel-2, 10m/pixel, 64×64)

### Step 2 — LR Frame Curation
For each scene, compute pixel-wise similarity (L2 distance / structural similarity)
between every LR image and the HR reference. Select the **top 4 LR frames**
with the highest similarity scores. This removes temporally distant or
atmospherically degraded observations.

### Step 3 — Quality Filtering
Apply manual and automatic filters to remove:
- Cloud-covered images (high pixel intensity + low variance)
- Corrupted samples (missing channels, file errors)
- Noisy images (high frequency noise above threshold)
- Result: ~3,000 → **~1,800 high-quality scene groups**

### Step 4 — Preprocessing
- Resize LR frames to **64×64** (bilinear interpolation)
- Resize HR images to **256×256**
- Convert all images to **RGB** (3-channel)
- Normalise pixel values to **[-1, 1]**: `x = (x - 0.5) / 0.5`

### Step 5 — Dataset Splitting
Random train/val/test split with fixed seed (42):
- **Train**: 1,257 scenes (70%)
- **Val**: 269 scenes (15%)
- **Test**: 271 scenes (15%)

### Step 6 — Multi-Frame Input Tensor
Stack 4 LR frames along the channel dimension:
```
lr_stack = concat([LR_01, LR_02, LR_03, LR_04], dim=channels)
         → shape: (B, 12, 64, 64)
```

### Step 7 — Generator Training (Phase 3: Joint GAN)
TerraGAN is trained end-to-end using a joint GAN objective:

**Discriminator update:**
```
L_D = BCE(D(HR), 1) + BCE(D(SR).detach(), 0)
```

**Generator update:**
```
SR  = G(lr_stack)
L_G = 1.0  × L1(SR, HR)
    + 0.01 × L_VGG19(SR, HR)
    + 0.005 × L_BCE(D(SR), 1)
    + 0.01 × L_LPIPS(SR, HR)
```

Optimiser: **Adam** (lr=1e-4, β=(0.9, 0.999)) for both G and D.
Batch size: 8 | Epochs: 100

### Step 8 — Evaluation
Metrics computed on the held-out test set:

| Metric | Formula | Meaning |
|--------|---------|---------|
| PSNR | 20·log₁₀(MAX/RMSE) | Pixel-level fidelity (dB) |
| SSIM | Structural similarity index | Luminance, contrast, structure |
| LPIPS | Deep feature distance (AlexNet) | Perceptual quality |
| Edge Score | Mean Sobel gradient magnitude | Sharpness of reconstructed edges |

### Step 9 — Results Analysis

**TerraGAN (100 epochs, test set):**

| PSNR | SSIM | LPIPS | Edge Score | Inference Time |
|------|------|-------|------------|---------------|
| 20.79 dB | 0.6577 | 0.3037 | 0.0145 | ~0.92 sec/tile |

---

## 4. Architecture Design Decisions

### Why 12-channel input?
Concatenating 4 RGB LR frames (4 × 3 = 12 channels) allows the network to
learn cross-frame complementarity directly from data, without requiring explicit
optical flow estimation. The generator learns which frame contributes most
useful texture information for each spatial region.

### Why RRDB blocks?
RRDB (Residual-in-Residual Dense Block) from ESRGAN provides:
- Deeper feature representations than standard ResNet blocks
- Scaled residual connections (×0.1) for training stability
- Better gradient flow in 16-block deep networks

### Why LSK attention?
Large Selective Kernel (LSK) attention applies **parallel convolutions at
multiple scales** (3×3, 5×5, 7×7) and aggregates responses. In satellite imagery,
texture patterns span multiple spatial scales simultaneously (e.g., a building
roof, its surrounding roads, and the wider urban block). LSK enables
the network to simultaneously attend to all these scales.

### Why composite loss?
- **L1 alone**: produces blurry outputs (over-smoothed pixel averages)
- **+ VGG19**: recovers perceptual structure and edge sharpness
- **+ Adversarial**: forces realistic high-frequency texture
- **+ LPIPS**: additional perceptual constraint aligned with human perception

---

## 5. Known Limitations

1. **No explicit alignment**: The 4 LR frames are naively concatenated. Scenes
   with significant inter-frame misalignment (e.g., seasonal changes, moving
   objects) may introduce artifacts. Deformable alignment is on the roadmap.

2. **Compute constraints**: Training on a single GPU limits batch size (8)
   and total epochs (100). Longer training may improve PSNR/SSIM further.

3. **Dataset scope**: WorldStrat covers global terrain types but is biased
   toward locations sampled in the original dataset curation. Performance
   on extremely rare terrain types (polar ice, salt flats) is not evaluated.

4. **Fixed scale factor**: The current architecture is fixed at 4× upscaling.
   Arbitrary-scale SR is a future extension.
