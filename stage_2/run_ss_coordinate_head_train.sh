#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_DIR="${TRAIN_DIR:-$REPO_ROOT/datasets/ShapeNetTRELLIS_full/train}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/ss_coordinate_head_v2/model}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EPOCHS="${EPOCHS:-3}"
I_SAVE="${I_SAVE:-780}"

CONFIG="$REPO_ROOT/configs/finetune/ss_flow_img_shapenet_coordinate_head.json"
OBJECTIVE1_SS_CKPT="$REPO_ROOT/results/objective1_full/ss_flow/ckpts/denoiser_lora_final.pt"
VIEW_INDEX=18
LATENT_NAME="o1_generated_view18_seed42"
CACHE_DIR="$TRAIN_DIR/ss_latents/$LATENT_NAME"

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
    "$CACHE_DIR/cache_config.json"; do
    if [[ ! -e "$path" ]]; then
        echo "Required input not found: $path" >&2
        exit 1
    fi
done

if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit)" ]]; then
    echo "Output directory is not empty: $OUTPUT_DIR" >&2
    echo "Use a fresh OUTPUT_DIR. This head-only run intentionally saves adapter-only checkpoints." >&2
    exit 1
fi

echo "Validating the complete Objective-1 endpoint cache before training"
"$PYTHON_BIN" "$REPO_ROOT/stage_2/cache_ss_coordinate_endpoints.py" \
    --dataset-dir "$TRAIN_DIR" \
    --ss-lora-ckpt "$OBJECTIVE1_SS_CKPT" \
    --latent-name "$LATENT_NAME" \
    --view-index "$VIEW_INDEX" \
    --seed 42 \
    --skip-existing

if [[ ! -f "$CACHE_DIR/cache_complete.json" ]]; then
    echo "Endpoint cache did not complete: $CACHE_DIR/cache_complete.json" >&2
    exit 1
fi

cd "$REPO_ROOT"
"$PYTHON_BIN" train.py \
    --config "$CONFIG" \
    --data_dir "$TRAIN_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --i_save "$I_SAVE" \
    --num_gpus 1 \
    --ckpt none \
    --auto_retry 0
