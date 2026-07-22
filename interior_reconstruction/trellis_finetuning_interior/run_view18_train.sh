#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRAIN_DIR="${TRAIN_DIR:-$REPO_ROOT/datasets/ShapeNetTRELLIS_full/train}"
OBJECTIVE1_DIR="${OBJECTIVE1_DIR:-$REPO_ROOT/results/objective1_full}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/objective1_view18_train}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_GPUS="${NUM_GPUS:-1}"
SEED=42
COORDINATE_THRESHOLD="${COORDINATE_THRESHOLD:--8}"

SS_CKPT="$OBJECTIVE1_DIR/ss_flow/ckpts/denoiser_lora_final.pt"
SLAT_CKPT="$OBJECTIVE1_DIR/slat_flow/ckpts/denoiser_lora_final.pt"
DECODER_CKPT="$OBJECTIVE1_DIR/decoder/ckpts/decoder_lora_final.pt"
IDS_FILE="$OUTPUT_DIR/selected_ids.txt"
PRED_DIR="$OUTPUT_DIR/predictions/objective1/seed_$SEED"

export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"
export SPARSE_ATTN_BACKEND="${SPARSE_ATTN_BACKEND:-sdpa}"
export SPCONV_ALGO="${SPCONV_ALGO:-native}"
if [[ "${TRELLIS_USE_DINOV2_XFORMERS:-0}" != "1" ]]; then
    export XFORMERS_DISABLED="${XFORMERS_DISABLED:-1}"
fi

cd "$REPO_ROOT"

if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_GPUS must be a positive integer" >&2
    exit 1
fi
AVAILABLE_GPUS="$("$PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count())')"
if (( NUM_GPUS > AVAILABLE_GPUS )); then
    echo "Requested NUM_GPUS=$NUM_GPUS, but PyTorch sees only $AVAILABLE_GPUS GPU(s)" >&2
    exit 1
fi
for path in \
    "$TRAIN_DIR/metadata.csv" \
    "$TRAIN_DIR/voxels" \
    "$TRAIN_DIR/renders_cond" \
    "$SS_CKPT" \
    "$SLAT_CKPT" \
    "$DECODER_CKPT"; do
    if [[ ! -e "$path" ]]; then
        echo "Required input not found: $path" >&2
        exit 1
    fi
done
if ! [[ "$COORDINATE_THRESHOLD" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
    echo "COORDINATE_THRESHOLD must be numeric, got: $COORDINATE_THRESHOLD" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
echo "[1/2] Writing the complete train ID list"
"$PYTHON_BIN" "$SCRIPT_DIR/select_ids.py" \
    --metadata "$TRAIN_DIR/metadata.csv" \
    --output "$IDS_FILE" \
    --per-category 0

echo "[2/2] Generating train-set Objective-1 predictions from view 18 on $NUM_GPUS GPU(s)"
pids=()
for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$SCRIPT_DIR/generate_predictions.py" \
        --dataset-dir "$TRAIN_DIR" \
        --output-dir "$PRED_DIR" \
        --ids-file "$IDS_FILE" \
        --view-index 18 \
        --seed "$SEED" \
        --slat-seed "$SEED" \
        --coordinate-threshold "$COORDINATE_THRESHOLD" \
        --ss-lora-ckpt "$SS_CKPT" \
        --slat-lora-ckpt "$SLAT_CKPT" \
        --decoder-lora-ckpt "$DECODER_CKPT" \
        --num-shards "$NUM_GPUS" \
        --shard-index "$gpu" \
        --skip-mesh-write \
        --skip-existing &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done
if [[ "$failed" != "0" ]]; then
    echo "At least one Objective-1 train worker failed" >&2
    exit 1
fi

expected="$(wc -l < "$IDS_FILE" | tr -d ' ')"
actual="$(find "$PRED_DIR/voxels" -maxdepth 1 -name '*.ply' -type f | wc -l | tr -d ' ')"
if [[ "$actual" != "$expected" ]]; then
    echo "Incomplete train prediction cache: expected $expected voxel files, found $actual" >&2
    exit 1
fi
echo "Train Objective-1 cache complete: $actual shapes in $PRED_DIR"
