# Stage 2 ShapeNetInternals Dataset Report

- ShapeNet root: `/workspace/TRELLIS/ShapeNet`
- Output dataset: `/workspace/TRELLIS/datasets/ShapeNetInternals`
- Report path: `/workspace/TRELLIS/stage_2/dataset_report.md`
- Run mode: `dry-run/preflight`

## Summary

- Metadata objects: `1949`
- Fully successful objects: `0`
- Failed/incomplete objects: `1949`

## Artifact Counts

- `voxelized`: `0 / 1949`
- `cond_rendered`: `0 / 1949`
- `rendered`: `0 / 1949`
- `feature_dinov2_vitl14_reg`: `0 / 1949`
- `ss_latent_ss_enc_conv3d_16l8_fp16`: `0 / 1949`
- `latent_dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16`: `0 / 1949`

## Splits

- `test`: `98`
- `train`: `1753`
- `val`: `98`

## Categories

- `bus`: `299`
- `cabinet`: `500`
- `cars`: `1000`
- `file_cabinet`: `150`

## Voxel Statistics

- Min voxels: `0`
- Mean voxels: `0.0`
- Max voxels: `0`

## Output Folder Sizes

- `metadata.csv`: `1.2M`
- `splits`: `1.1M`
- `voxels`: `0B`
- `renders_cond`: `0B`
- `renders`: `0B`
- `features/dinov2_vitl14_reg`: `0B`
- `ss_latents/ss_enc_conv3d_16l8_fp16`: `0B`
- `latents/dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16`: `0B`
- `logs`: `979K`

## Failed Objects

- `surface binvox to voxel PLY missing/failed`: `1949`

Full failure table: `/workspace/TRELLIS/datasets/ShapeNetInternals/failures_stage2.csv`

## Invalid Source Objects

- Excluded before preprocessing: `1`
- Manifest: `/workspace/TRELLIS/datasets/ShapeNetInternals/invalid_objects.csv`

## Commands

Full dataset command:

```bash
/workspace/venv/bin/python stage_2/run_full_dataset_pipeline.py --output-dir /workspace/TRELLIS/datasets/ShapeNetInternals
```

Useful resume command:

```bash
/workspace/venv/bin/python stage_2/run_full_dataset_pipeline.py --output-dir /workspace/TRELLIS/datasets/ShapeNetInternals --render-workers 4 --cond-render-workers 4 --voxel-workers 8
```
