#!/usr/bin/env bash
set -euo pipefail
trap 'status=$?; echo "[stage2] Evaluation pipeline failed with exit code $status"; read -r -p "Press Enter to exit..."' ERR

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/stage_2/config.sh"

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
