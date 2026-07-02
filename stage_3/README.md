# Stage 3: Uniform vs Weighted SLAT LoRA

Stage 3 reuses the Stage 2 finetuning, export, and evaluation infrastructure to compare two LoRA runs:

- `uniform`: trains `ss_flow` and `slat_flow` with the original uniform SLAT flow loss.
- `weighted_<scalar>`: trains `ss_flow` normally and trains `slat_flow` with `--invisible_weight_scalar` enabled.

The weighting only affects `slat_flow` because it is implemented in `SparseFlowMatchingTrainer`.

## Configure

```bash
source stage_3/config.sh

# Optional overrides
export DATA_DIR=/path/to/ShapeNetInternals_small
export STAGE3_OUT_ROOT=results/stage_3_weighted_slat
export WEIGHTED_SCALAR=1.5
export NUM_GPUS=1
```

## Train

```bash
bash stage_3/run_lora_finetune_pair.sh
```

This writes:

- `results/stage_3_weighted_slat/uniform/ss_flow`
- `results/stage_3_weighted_slat/uniform/slat_flow`
- `results/stage_3_weighted_slat/weighted_1p5/ss_flow`
- `results/stage_3_weighted_slat/weighted_1p5/slat_flow`

## Export

```bash
bash stage_3/export_pair.sh
```

## Evaluate

```bash
bash stage_3/run_eval_pair.sh
```

Each case gets its own comparison table:

- `results/stage_3_weighted_slat/uniform/eval/comparison.csv`
- `results/stage_3_weighted_slat/weighted_1p5/eval/comparison.csv`