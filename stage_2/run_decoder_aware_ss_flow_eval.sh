#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_DIR="${TEST_DIR:-$REPO_ROOT/datasets/ShapeNetTRELLIS_full/test}"
OBJECTIVE1_DIR="${OBJECTIVE1_DIR:-$REPO_ROOT/results/objective1_full}"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/results/decoder_aware_ss_flow/ss_flow}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/decoder_aware_ss_flow/eval_test}"
NEW_SS_CKPT="${NEW_SS_CKPT:-$MODEL_DIR/ckpts/denoiser_lora_final.pt}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SAMPLES_PER_CATEGORY="${SAMPLES_PER_CATEGORY:-5}"
RESUME="${RESUME:-0}"

BASE_SS_CKPT="$OBJECTIVE1_DIR/ss_flow/ckpts/denoiser_lora_final.pt"
SLAT_CKPT="$OBJECTIVE1_DIR/slat_flow/ckpts/denoiser_lora_final.pt"
DECODER_CKPT="$OBJECTIVE1_DIR/decoder/ckpts/decoder_lora_final.pt"
IDS_FILE="$OUTPUT_DIR/selected_ids.txt"
PRED_ROOT="$OUTPUT_DIR/predictions"
SEED=42
VIEW_INDEX=18

export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"
export SPARSE_ATTN_BACKEND="${SPARSE_ATTN_BACKEND:-sdpa}"
export SPCONV_ALGO="${SPCONV_ALGO:-native}"
if [[ "${TRELLIS_USE_DINOV2_XFORMERS:-0}" != "1" ]]; then
    export XFORMERS_DISABLED="${XFORMERS_DISABLED:-1}"
fi

for path in \
    "$TEST_DIR/metadata.csv" \
    "$TEST_DIR/voxels" \
    "$TEST_DIR/renders" \
    "$TEST_DIR/renders_cond" \
    "$BASE_SS_CKPT" \
    "$SLAT_CKPT" \
    "$DECODER_CKPT" \
    "$NEW_SS_CKPT"; do
    if [[ ! -e "$path" ]]; then
        echo "Required input not found: $path" >&2
        exit 1
    fi
done

if [[ -d "$OUTPUT_DIR" ]] \
    && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit)" ]] \
    && [[ "$RESUME" != "1" ]]; then
    echo "Output directory is not empty: $OUTPUT_DIR" >&2
    echo "Use a fresh OUTPUT_DIR, or set RESUME=1 to continue an interrupted evaluation." >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

cd "$REPO_ROOT"

echo "[1/7] Selecting held-out test objects"
if [[ "$RESUME" == "1" ]] && [[ -f "$IDS_FILE" ]]; then
    echo "Reusing $IDS_FILE"
else
    "$PYTHON_BIN" stage_2/select_internal_target_pilot.py \
        --metadata "$TEST_DIR/metadata.csv" \
        --output "$IDS_FILE" \
        --per-category "$SAMPLES_PER_CATEGORY"
fi

generate() {
    local method="$1"
    local ss_ckpt="$2"
    "$PYTHON_BIN" stage_2/export_full_pipeline_voxels.py \
        --dataset-dir "$TEST_DIR" \
        --output-dir "$PRED_ROOT/$method/seed_$SEED" \
        --ids-file "$IDS_FILE" \
        --view-index "$VIEW_INDEX" \
        --seed "$SEED" \
        --slat-seed "$SEED" \
        --coordinate-threshold 0 \
        --ss-lora-ckpt "$ss_ckpt" \
        --slat-lora-ckpt "$SLAT_CKPT" \
        --decoder-lora-ckpt "$DECODER_CKPT" \
        --skip-existing
}

echo "[2/7] Generating Objective-1 baseline"
generate objective1 "$BASE_SS_CKPT"

echo "[3/7] Generating decoder-aware SS-flow result"
generate decoder_aware "$NEW_SS_CKPT"

METHODS=(objective1 decoder_aware)

echo "[4/7] Comparing Stage-1 sparse coordinates"
"$PYTHON_BIN" stage_2/compare_internals.py \
    --gt-voxels "$TEST_DIR/voxels" \
    --pred-root "$PRED_ROOT" \
    --prediction-subdir structure_voxels \
    --metadata "$TEST_DIR/metadata.csv" \
    --ids-file "$IDS_FILE" \
    --methods "${METHODS[@]}" \
    --margins 1 2 3 4 \
    --output "$OUTPUT_DIR/structure_metrics/summary.csv" \
    --per-sample-output "$OUTPUT_DIR/structure_metrics/per_sample.csv"

echo "[5/7] Comparing final mesh voxels"
"$PYTHON_BIN" stage_2/compare_internals.py \
    --gt-voxels "$TEST_DIR/voxels" \
    --pred-root "$PRED_ROOT" \
    --metadata "$TEST_DIR/metadata.csv" \
    --ids-file "$IDS_FILE" \
    --methods "${METHODS[@]}" \
    --margins 1 2 3 4 \
    --output "$OUTPUT_DIR/mesh_metrics/summary.csv" \
    --per-sample-output "$OUTPUT_DIR/mesh_metrics/per_sample.csv"

echo "[6/7] Rendering deterministic GT cutaways"
"$PYTHON_BIN" dataset_toolkits/render_internal_kiui.py \
    --data_dir "$TEST_DIR" \
    --instances "$IDS_FILE"

echo "[7/7] Comparing cutaway masks and depth"
"$PYTHON_BIN" stage_2/evaluate_internal_cutaways.py \
    --data_dir "$TEST_DIR" \
    --ids_file "$IDS_FILE" \
    --output_dir "$OUTPUT_DIR/cutaway_metrics" \
    --method "objective1=$PRED_ROOT/objective1/seed_$SEED/mesh" \
    --method "decoder_aware=$PRED_ROOT/decoder_aware/seed_$SEED/mesh"

echo "Evaluation complete: $OUTPUT_DIR"
