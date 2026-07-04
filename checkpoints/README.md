# Checkpoints

Model checkpoints are **not committed to this repository** due to file size constraints (GitHub recommends <50 MB per file).

## Downloading Trained Weights

| Checkpoint | Epochs | PSNR | SSIM | LPIPS | Link |
|------------|--------|------|------|-------|------|
| `terragan_v0.1.0.pth` | 100 | 20.79 dB | 0.6577 | 0.3037 | _Link coming soon_ |

> The checkpoint will be hosted on Google Drive / GitHub Releases once the model is publicly released (see [Issue #5](https://github.com/Prashant822k/Improved-SRGAN-for-multi-image-super-resolution/issues/5)).

## Using a Checkpoint

```bash
python src/inference.py \
    --lr_dir  /path/to/tile_folder \
    --checkpoint checkpoints/terragan_v0.1.0.pth \
    --output  output/sr_result.png
```

```bash
python scripts/evaluate.py \
    --dataset_path /path/to/SRGAN_FINAL_V1 \
    --checkpoint   checkpoints/terragan_v0.1.0.pth
```
