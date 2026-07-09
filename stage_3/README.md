# Stage 3: Uniform vs Weighted SLAT LoRA

Stage 3 reuses the Stage 2 finetuning, export, and evaluation. Trains: 

- `weighted_<scalar>`: reuse trained `ss_flow` from stage 2 and trains `slat_flow` with `--invisible_weight_scalar` enabled.

The weighting only affects `slat_flow` because visibility of voxel only makes sense in SLAT flow, not in the sparse structure flow.

## Configure

```bash
source stage_3/config.sh

# Optional overrides
export DATA_DIR=/path/to/ShapeNetInternals_small
export STAGE3_OUT_ROOT=results/stage_3_weighted_slat
export WEIGHTED_SCALAR=1.5
```

## Train

```bash
bash stage_3/run_finetune_with_weight.sh
```

This writes:

- `results/stage_3_weighted_slat/weighted_1p5/slat_flow`


## Evaluate

```bash
bash stage_3/run_eval.sh
```

Results should be in
- `results/stage_3_weighted_slat/weighted_1p5/eval/comparison.csv`