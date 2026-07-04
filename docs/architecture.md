# TerraGAN Architecture

## Pipeline Diagram

```mermaid
flowchart TD
    A["🛰️ 4 LR Frames\n(64×64×3 each)"] --> B["Channel Concatenation\n→ 12-channel tensor\n(64×64×12)"]
    B --> C["Initial Conv\n9×9, 12→64 channels"]
    C --> D["16× Residual Blocks\nRRDB-style with scaled skip\n(×0.1 residual)"]
    D --> E["LSK Multi-Scale Attention\nconv3×3 + conv5×5 + conv7×7"]
    E --> F["Conv 3×3, 64→64\n+ Global Skip Connection"]
    F --> G["PixelShuffle ×2\n64→128 spatial\n(64→256 channels → shuffle)"]
    G --> H["PixelShuffle ×2\n128→256 spatial"]
    H --> I["Output Conv 9×9, 64→3\n+ Tanh activation"]
    I --> J["🖼️ SR Output\n(256×256×3)"]

    J --> K["Discriminator\n7 Conv blocks + LeakyReLU\n→ Logit map (32×32)"]
    L["🖼️ HR Ground Truth\n(256×256×3)"] --> K

    subgraph loss ["Generator Loss"]
        M["L1 Pixel Loss\n(w = 1.0)"]
        N["VGG19 Perceptual\n(w = 0.01)"]
        O["Adversarial BCE\n(w = 0.005)"]
        P["LPIPS AlexNet\n(w = 0.01)"]
    end

    J --> M
    J --> N
    J --> O
    J --> P
    L --> M
    L --> N
    L --> P
    K --> O
```

## Component Descriptions

### Input Processing
- 4 LR satellite frames (64×64 RGB) selected from 16 candidates via pixel-wise similarity scoring
- Concatenated along the channel dimension → **12-channel input tensor**
- Normalised to [-1, 1]

### Generator

| Layer | In Channels | Out Channels | Kernel | Notes |
|-------|-------------|--------------|--------|-------|
| Initial Conv | 12 | 64 | 9×9 | Feature extraction entry |
| ResidualBlock ×16 | 64 | 64 | 3×3 | RRDB-style, scaled skip (×0.1) |
| LSK Attention | 64 | 64 | 3,5,7×3,5,7 | Multi-scale parallel convolutions |
| Mid Conv | 64 | 64 | 3×3 | + global skip from initial |
| PixelShuffle ×2 | 64 | 256→64 | 3×3 | 64×64 → 128×128 |
| PixelShuffle ×2 | 64 | 256→64 | 3×3 | 128×128 → 256×256 |
| Output Conv | 64 | 3 | 9×9 | + Tanh |

### LSK Attention
Three parallel depthwise convolutions (k=3, k=5, k=7) are summed with the input,
providing multi-scale receptive fields that selectively amplify texture-rich regions
in satellite imagery (vegetation edges, building boundaries, road networks).

### Discriminator (PatchGAN)
7 convolutional blocks with alternating strides produce a **32×32 logit map**
over patches of the 256×256 input. This encourages local texture fidelity
rather than global image-level judgement.

### Loss Function

```
L_total = 1.0  × L1(fake, hr)
        + 0.01 × L_VGG19(features(fake), features(hr))
        + 0.005 × L_BCE(D(fake), 1)
        + 0.01 × L_LPIPS(fake, hr)
```
