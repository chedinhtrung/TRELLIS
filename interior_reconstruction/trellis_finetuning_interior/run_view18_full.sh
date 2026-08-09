#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TEST_DIR="${TEST_DIR:-$REPO_ROOT/datasets/ShapeNetTRELLIS_full/test}"
OBJECTIVE1_DIR="${OBJECTIVE1_DIR:-$REPO_ROOT/results/objective1_full}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/objective1_view18_full}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED=42
VIEW_INDEX=18
RESUME="${RESUME:-0}"
COORDINATE_THRESHOLD="${COORDINATE_THRESHOLD:--8}"

SS_CKPT="$OBJECTIVE1_DIR/ss_flow/ckpts/denoiser_lora_final.pt"
SLAT_CKPT="$OBJECTIVE1_DIR/slat_flow/ckpts/denoiser_lora_final.pt"
DECODER_CKPT="$OBJECTIVE1_DIR/decoder/ckpts/decoder_lora_final.pt"
IDS_FILE="$OUTPUT_DIR/selected_ids.txt"
PRED_ROOT="$OUTPUT_DIR/predictions"
PRED_DIR="$PRED_ROOT/objective1/seed_$SEED"

export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"
export SPARSE_ATTN_BACKEND="${SPARSE_ATTN_BACKEND:-sdpa}"
export SPCONV_ALGO="${SPCONV_ALGO:-native}"
if [[ "${TRELLIS_USE_DINOV2_XFORMERS:-0}" != "1" ]]; then
    export XFORMERS_DISABLED="${XFORMERS_DISABLED:-1}"
fi

cd "$REPO_ROOT"

for path in \
    "$TEST_DIR/metadata.csv" \
    "$TEST_DIR/voxels" \
    "$TEST_DIR/renders_cond" \
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

if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit)" ]] && [[ "$RESUME" != "1" ]]; then
    echo "Output directory is not empty: $OUTPUT_DIR" >&2
    echo "Use a new OUTPUT_DIR, or set RESUME=1 to continue this exact run." >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

if [[ "$RESUME" == "1" ]] && [[ -f "$IDS_FILE" ]]; then
    echo "[1/4] Reusing all-object ID list from $IDS_FILE"
else
    echo "[1/4] Selecting all held-out test objects"
    "$PYTHON_BIN" "$SCRIPT_DIR/select_ids.py" \
        --metadata "$TEST_DIR/metadata.csv" \
        --output "$IDS_FILE" \
        --per-category 0
fi

echo "[2/4] Generating Objective-1 outputs from conditioning view $VIEW_INDEX"
"$PYTHON_BIN" "$SCRIPT_DIR/generate_predictions.py" \
    --dataset-dir "$TEST_DIR" \
    --output-dir "$PRED_DIR" \
    --ids-file "$IDS_FILE" \
    --view-index "$VIEW_INDEX" \
    --seed "$SEED" \
    --slat-seed "$SEED" \
    --coordinate-threshold "$COORDINATE_THRESHOLD" \
    --ss-lora-ckpt "$SS_CKPT" \
    --slat-lora-ckpt "$SLAT_CKPT" \
    --decoder-lora-ckpt "$DECODER_CKPT" \
    --skip-existing

echo "[3/4] Evaluating raw sparse-structure coordinates"
"$PYTHON_BIN" "$SCRIPT_DIR/evaluate_internals.py" \
    --gt-voxels "$TEST_DIR/voxels" \
    --pred-root "$PRED_ROOT" \
    --prediction-subdir structure_voxels \
    --metadata "$TEST_DIR/metadata.csv" \
    --ids-file "$IDS_FILE" \
    --methods objective1 \
    --margins 1 2 3 4 \
    --output "$OUTPUT_DIR/structure_metrics/summary.csv" \
    --per-sample-output "$OUTPUT_DIR/structure_metrics/per_sample.csv"

echo "[4/4] Evaluating decoded mesh voxels"
"$PYTHON_BIN" "$SCRIPT_DIR/evaluate_internals.py" \
    --gt-voxels "$TEST_DIR/voxels" \
    --pred-root "$PRED_ROOT" \
    --metadata "$TEST_DIR/metadata.csv" \
    --ids-file "$IDS_FILE" \
    --methods objective1 \
    --margins 1 2 3 4 \
    --output "$OUTPUT_DIR/mesh_metrics/summary.csv" \
    --per-sample-output "$OUTPUT_DIR/mesh_metrics/per_sample.csv"

echo "Objective-1 view-18 evaluation complete."
echo "Structure metrics: $OUTPUT_DIR/structure_metrics/summary.csv"
echo "Mesh metrics:      $OUTPUT_DIR/mesh_metrics/summary.csv"
echo "Mesh voxels:       $PRED_DIR/voxels"
