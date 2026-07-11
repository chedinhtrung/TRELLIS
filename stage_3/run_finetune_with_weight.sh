#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/stage_3/config.sh"

cd "$REPO_ROOT"

echo "training with weighted loss: invisible_weight_scalar=$WEIGHTED_SCALAR)"

python train.py \
    --config configs/finetune/slat_flow_img_shapenet_internals_lora.json \
    --data_dir "$DATA_DIR" \
    --output_dir "$STAGE3_OUT_ROOT/weighted_${WEIGHTED_SCALAR//./p}/slat_flow" \
    --ckpt none \
    --invisible_weight_scalar "$WEIGHTED_SCALAR"

echo "Finished Stage 3 finetuning runs under $STAGE3_OUT_ROOT"