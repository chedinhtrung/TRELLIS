#!/usr/bin/env bash
set -euo pipefail
trap 'status=$?; echo "[interior-finetuning] Pipeline failed with exit code $status"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/config.sh"

cd "$REPO_ROOT"

echo "fine tuning flow models"

python train.py \
    --config configs/finetune/ss_flow_img_shapenet_internals_lora.json \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUT_DIR/ss_flow" \
    --num_gpus "$NUM_GPUS" \
    --ckpt none 

python train.py \
    --config configs/finetune/slat_flow_img_shapenet_internals_lora.json \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUT_DIR/slat_flow" \
    --num_gpus "$NUM_GPUS" \
    --ckpt none
