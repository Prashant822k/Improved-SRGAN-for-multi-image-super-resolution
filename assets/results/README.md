# Sample Results

Place comparison images here as you generate them during training and evaluation.

## Recommended naming convention

```
comparison_<scene_id>_<terrain_type>.png
```

Examples:
- `comparison_01_urban.png`
- `comparison_02_agricultural.png`
- `comparison_03_forest.png`

## Image grid format

Each comparison image should be a horizontal grid:

```
[ LR Input (64×64) ] | [ TerraGAN SR (256×256) ] | [ HR Ground Truth (256×256) ]
```

Use `torchvision.utils.make_grid` or any image editing tool to compose the grid.

## Uploading via GitHub web UI

1. Go to this folder on GitHub: `assets/results/`
2. Click **Add file → Upload files**
3. Drag and drop your comparison PNG files
4. Commit with message: `assets: add visual comparison results`
