# TerraGAN Architecture Diagrams & Pipeline

This page contains the system flow chart, system use case diagram, and the detailed layer-level model pipeline.

---

## 1. System Flow Chart

This diagram illustrates the overall processing flow from input low-resolution images to the final super-resolved output, including the preprocessing phase and the GAN training loop.

```mermaid
flowchart TD
    subgraph Preprocessing ["Preprocessing Stage"]
        A(["Input Low Resolution Images"]) --> B["Align Frames"]
        B --> C["Fuse Frames"]
        C --> D["Add Reference Features"]
    end

    D --> E

    subgraph GAN ["GAN Loop"]
        E["Generator"] --> F["Discriminator"]
        F --> G["Loss Calculation"]
        G --> H{"Done Decision?"}
        H -- No --> E
    end

    H -- Yes --> I

    subgraph Output ["Output Stage"]
        I(["Output Super Resolution Image"])
    end

    style D fill:#00796B,stroke:#004D40,color:#fff
    style H fill:#FFECB3,stroke:#FFA000
    style I fill:#C8E6C9,stroke:#388E3C
```

---

## 2. System Use Case Diagram

This diagram displays the interaction flow between the **User** and **Researcher** roles and the core functions of the SRGAN system.

```mermaid
flowchart LR
    subgraph Actors ["Users"]
        U(["User"])
        R(["Researcher"])
    end

    subgraph System ["SRGAN System"]
        LR["Upload LR Images"]
        Proc["Run SR Processing"]
        Out["View Super-Resolved Output"]
        Ref["Upload Reference Image"]
    end

    U -- "Initiate upload of low-resolution images" --> LR
    R -- "Initiate upload of reference image" --> Ref

    LR -- "After images uploaded" --> Proc
    Ref -- "After reference uploaded" --> Proc
    Proc -- "Processing complete" --> Out

    Out -- "Upload more images (optional)" --> LR
    Out -- "Upload new reference (optional)" --> Ref

    style U fill:#B2EBF2,stroke:#00ACC1
    style R fill:#C8E6C9,stroke:#4CAF50
    style LR fill:#D1C4E9,stroke:#5E35B1
    style Proc fill:#FFE0B2,stroke:#FB8C00
    style Out fill:#F8BBD0,stroke:#E91E63
    style Ref fill:#C8E6C9,stroke:#4CAF50
```

---

## 3. Layer-Level Model Pipeline

This diagram shows the tensor shapes and layer operations inside the Generator, Discriminator, and Loss functions.

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
