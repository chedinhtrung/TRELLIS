# ShapeNet to TRELLIS preparation

`run_pipeline.sh` is the single project entrypoint for converting the four
ShapeNet categories into the TRELLIS dataset used by both final stages. It
delegates to `dataset_toolkits/prepare_shapenet.sh`, which creates deterministic
train/validation/test manifests and produces 40 ordinary renders, 40
conditioning renders, surface voxels, DINOv2 features, sparse-structure
latents, and SLAT latents.

```bash
env \
  SHAPENET_ROOT=/workspace/TRELLIS/ShapeNet \
  SHAPENET_PROCESSED=/workspace/TRELLIS/datasets/ShapeNetTRELLIS_full \
  MAX_WORKERS=8 \
  bash interior_reconstruction/data_preparation/run_pipeline.sh
```

The output must pass:

```bash
python interior_reconstruction/trellis_finetuning_interior/validate_dataset.py \
  --dataset-root /workspace/TRELLIS/datasets/ShapeNetTRELLIS_full
```
