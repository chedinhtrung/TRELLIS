#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_DIR="${TRAIN_DIR:-$REPO_ROOT/datasets/ShapeNetTRELLIS_full/train}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/decoder_aware_ss_flow/ss_flow}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="$REPO_ROOT/configs/finetune/ss_flow_img_shapenet_decoder_aware_lora.json"

export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"
export SPARSE_ATTN_BACKEND="${SPARSE_ATTN_BACKEND:-sdpa}"
export SPCONV_ALGO="${SPCONV_ALGO:-native}"
if [[ "${TRELLIS_USE_DINOV2_XFORMERS:-0}" != "1" ]]; then
    export XFORMERS_DISABLED="${XFORMERS_DISABLED:-1}"
fi

for path in \
    "$CONFIG" \
    "$TRAIN_DIR/metadata.csv" \
    "$TRAIN_DIR/voxels" \
    "$TRAIN_DIR/ss_latents/ss_enc_conv3d_16l8_fp16" \
    "$TRAIN_DIR/renders_cond"; do
    if [[ ! -e "$path" ]]; then
        echo "Required input not found: $path" >&2
        exit 1
    fi
done

if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit)" ]]; then
    echo "Output directory is not empty: $OUTPUT_DIR" >&2
    echo "Use a fresh OUTPUT_DIR; adapter-only training cannot resume." >&2
    exit 1
fi

cd "$REPO_ROOT"
"$PYTHON_BIN" train.py \
    --config "$CONFIG" \
    --data_dir "$TRAIN_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs 3 \
    --i_save 780 \
    --num_gpus 1 \
    --ckpt none \
    --auto_retry 0
