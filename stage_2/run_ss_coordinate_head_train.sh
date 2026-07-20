#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_DIR="${TRAIN_DIR:-$REPO_ROOT/datasets/ShapeNetTRELLIS_full/train}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/ss_coordinate_head}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EPOCHS="${EPOCHS:-3}"
RESUME="${RESUME:-0}"

CONFIG="$REPO_ROOT/configs/finetune/ss_flow_img_shapenet_coordinate_head.json"
OBJECTIVE1_SS_CKPT="$REPO_ROOT/results/objective1_full/ss_flow/ckpts/denoiser_lora_final.pt"

export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"
export SPARSE_ATTN_BACKEND="${SPARSE_ATTN_BACKEND:-sdpa}"
if [[ "${TRELLIS_USE_DINOV2_XFORMERS:-0}" != "1" ]]; then
    export XFORMERS_DISABLED="${XFORMERS_DISABLED:-1}"
fi

for path in \
    "$CONFIG" \
    "$OBJECTIVE1_SS_CKPT" \
    "$TRAIN_DIR/metadata.csv" \
    "$TRAIN_DIR/voxels" \
    "$TRAIN_DIR/renders_cond" \
    "$TRAIN_DIR/ss_latents/ss_enc_conv3d_16l8_fp16"; do
    if [[ ! -e "$path" ]]; then
        echo "Required input not found: $path" >&2
        exit 1
    fi
done

if [[ "$RESUME" == "1" ]]; then
    if [[ ! -d "$OUTPUT_DIR/ckpts" ]] || [[ -z "$(find "$OUTPUT_DIR/ckpts" -name 'misc_step*.pt' -print -quit)" ]]; then
        echo "No resumable checkpoint found under: $OUTPUT_DIR/ckpts" >&2
        exit 1
    fi
    CKPT="latest"
else
    if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit)" ]]; then
        echo "Output directory is not empty: $OUTPUT_DIR" >&2
        echo "Use a fresh OUTPUT_DIR, or set RESUME=1 to continue this run." >&2
        exit 1
    fi
    CKPT="none"
fi

cd "$REPO_ROOT"
"$PYTHON_BIN" train.py \
    --config "$CONFIG" \
    --data_dir "$TRAIN_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --i_save 250 \
    --num_gpus 1 \
    --ckpt "$CKPT" \
    --auto_retry 0
