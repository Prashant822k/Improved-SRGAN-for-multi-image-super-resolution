# Changelog

All notable changes to TerraGAN are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- Deformable alignment module for learnable frame registration
- Relativistic discriminator (RaGAN-style)
- Reference-guided texture transfer (TTSR-style)
- Temporal consistency metric integration
- Public model checkpoint release + Colab demo

---

## [v0.1.0] — 2026-07-04

### Added
- **Core model architecture**: Generator (RRDB × 16 + LSK attention, 12-ch → 3-ch, 4×)
  and Discriminator (PatchGAN-style, 7 conv blocks)
- **WorldStrat dataset loader**: 4 LR + 1 HR per tile, 70/15/15 split,
  normalisation to [-1, 1]
- **Training pipeline**: Phase 3 joint GAN training, checkpoint save/resume,
  config-driven via `configs/default.yaml`
- **Composite loss**: L1 (w=1.0) + VGG19 perceptual (w=0.01)
  + adversarial BCE (w=0.005) + LPIPS-AlexNet (w=0.01)
- **Inference script**: single-tile 4× SR from 4 LR inputs
- **Evaluation script**: PSNR / SSIM / LPIPS / Edge Score on test split
- **Benchmark results** (100 epochs, test set):
  PSNR 20.79 dB | SSIM 0.6577 | LPIPS 0.3037 | Edge Score 0.0145
- **CI**: GitHub Actions flake8 + black lint workflow
- **Docs**: README, methodology, architecture diagram, CHANGELOG, CONTRIBUTING

[Unreleased]: https://github.com/Prashant822k/Improved-SRGAN-for-multi-image-super-resolution/compare/v0.1.0...HEAD
[v0.1.0]: https://github.com/Prashant822k/Improved-SRGAN-for-multi-image-super-resolution/releases/tag/v0.1.0
