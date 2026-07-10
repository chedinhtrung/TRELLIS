#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/stage_2/config.sh"

cd "$REPO_ROOT"

echo "fine tuning slat decoder"

python train.py \
    --config configs/finetune/slat_vae_enc_dec_mesh_shapenet_internals_lora.json \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUT_DIR/slat_vae" \
    --num_gpus 1 \
    --ckpt none \
    --auto_retry 0