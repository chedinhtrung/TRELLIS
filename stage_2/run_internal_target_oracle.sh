#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/datasets/ShapeNetTRELLIS_full/train}"
PILOT_DIR="${PILOT_DIR:-$REPO_ROOT/results/geometry_expert/pilot_targets}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/geometry_expert/pilot_oracle}"
DECODER_LORA_CKPT="${DECODER_LORA_CKPT:-$REPO_ROOT/results/objective1_full/decoder/ckpts/decoder_lora_final.pt}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export SPCONV_ALGO="${SPCONV_ALGO:-native}"
export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_DIR/metrics"

if [[ ! -f "$DECODER_LORA_CKPT" ]]; then
    echo "Decoder LoRA checkpoint not found: $DECODER_LORA_CKPT" >&2
    exit 1
fi

echo "[1/2] Decoding old and internal-aware GT SLAT targets"
"$PYTHON_BIN" stage_2/decode_internal_target_oracle.py \
    --data_dir "$DATA_DIR" \
    --ids_file "$PILOT_DIR/pilot_ids.txt" \
    --output_dir "$OUTPUT_DIR" \
    --decoder_lora_ckpt "$DECODER_LORA_CKPT"

echo "[2/2] Computing reconstruction metrics at margins 1-4"
"$PYTHON_BIN" stage_2/compare_internals.py \
    --gt-voxels "$DATA_DIR/voxels" \
    --pred-root "$OUTPUT_DIR/predictions" \
    --metadata "$DATA_DIR/metadata.csv" \
    --ids-file "$PILOT_DIR/pilot_ids.txt" \
    --methods old_target new_target \
    --margins 1 2 3 4 \
    --output "$OUTPUT_DIR/metrics/summary.csv" \
    --per-sample-output "$OUTPUT_DIR/metrics/per_sample.csv"

echo "Oracle comparison complete: $OUTPUT_DIR/metrics/summary.csv"
