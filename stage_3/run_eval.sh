#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/stage_3/config.sh"

cd "$REPO_ROOT"

OUT_DIR="$STAGE3_OUT_ROOT/weighted_${WEIGHTED_SCALAR//./p}"
PRED_ROOT="$OUT_DIR/predictions"

python stage_2/export_full_pipeline_voxels.py \
    --mode lora \
    --dataset-dir "$DATA_DIR" \
    --pred-root "$PRED_ROOT" \
    --ss-lora-ckpt "$STAGE2_OUT_ROOT/ss_flow/ckpts/denoiser_lora_step0002000.pt" \
    --slat-lora-ckpt "$OUT_DIR/slat_flow/ckpts/denoiser_lora_step0002000.pt" \
    --skip-existing

python stage_2/compare_internals.py \
    --gt-voxels "$GT_VOXELS" \
    --pred-root "$PRED_ROOT" \
    --output "$OUT_DIR/eval/comparison.csv"

echo "Finished Stage 3 evaluation under $OUT_DIR"