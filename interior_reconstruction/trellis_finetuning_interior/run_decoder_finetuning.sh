#!/usr/bin/env bash
set -euo pipefail
trap 'status=$?; echo "[interior-finetuning] Pipeline failed with exit code $status"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/config.sh"

cd "$REPO_ROOT"

echo "fine tuning slat decoder"

python train.py \
    --config configs/finetune/slat_vae_enc_dec_mesh_shapenet_internals_lora.json \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUT_DIR/decoder" \
    --num_gpus 1 \
    --ckpt none \
    --auto_retry 0
