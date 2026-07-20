#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_DIR="${TEST_DIR:-$REPO_ROOT/datasets/ShapeNetTRELLIS_full/test}"
OBJECTIVE1_DIR="${OBJECTIVE1_DIR:-$REPO_ROOT/results/objective1_full}"
COORDINATE_HEAD_DIR="${COORDINATE_HEAD_DIR:-$REPO_ROOT/results/ss_coordinate_head_v2/model}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/ss_coordinate_head_v2/eval_test}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SAMPLES_PER_CATEGORY="${SAMPLES_PER_CATEGORY:-5}"
SEED=42
VIEW_INDEX=18
RESUME="${RESUME:-0}"

BASE_SS_CKPT="$OBJECTIVE1_DIR/ss_flow/ckpts/denoiser_lora_final.pt"
HEAD_SS_CKPT="$COORDINATE_HEAD_DIR/ckpts/denoiser_lora_best.pt"
THRESHOLD_FILE="$COORDINATE_HEAD_DIR/coordinate_threshold.txt"
BASE_THRESHOLD_FILE="$COORDINATE_HEAD_DIR/objective1_threshold.txt"
SLAT_CKPT="$OBJECTIVE1_DIR/slat_flow/ckpts/denoiser_lora_final.pt"
DECODER_CKPT="$OBJECTIVE1_DIR/decoder/ckpts/decoder_lora_final.pt"
IDS_FILE="$OUTPUT_DIR/selected_ids.txt"
PRED_ROOT="$OUTPUT_DIR/predictions"

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
    "$TEST_DIR/renders" \
    "$TEST_DIR/renders_cond" \
    "$OBJECTIVE1_DIR/ss_flow/config.json" \
    "$OBJECTIVE1_DIR/slat_flow/config.json" \
    "$OBJECTIVE1_DIR/decoder/config.json" \
    "$COORDINATE_HEAD_DIR/config.json" \
    "$BASE_SS_CKPT" \
    "$HEAD_SS_CKPT" \
    "$THRESHOLD_FILE" \
    "$BASE_THRESHOLD_FILE" \
    "$SLAT_CKPT" \
    "$DECODER_CKPT"; do
    if [[ ! -e "$path" ]]; then
        echo "Required input not found: $path" >&2
        exit 1
    fi
done

IFS= read -r COORDINATE_THRESHOLD < "$THRESHOLD_FILE"
IFS= read -r OBJECTIVE1_THRESHOLD < "$BASE_THRESHOLD_FILE"
if [[ -z "$COORDINATE_THRESHOLD" ]]; then
    echo "Coordinate threshold is empty: $THRESHOLD_FILE" >&2
    exit 1
fi
if [[ -z "$OBJECTIVE1_THRESHOLD" ]]; then
    echo "Objective-1 threshold is empty: $BASE_THRESHOLD_FILE" >&2
    exit 1
fi

if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit)" ]] && [[ "$RESUME" != "1" ]]; then
    echo "Output directory is not empty: $OUTPUT_DIR" >&2
    echo "Use a new OUTPUT_DIR, or set RESUME=1 to continue this run." >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

if [[ "$RESUME" == "1" ]] && [[ -f "$IDS_FILE" ]]; then
    echo "[1/7] Reusing selected test objects from $IDS_FILE"
else
    echo "[1/7] Selecting $SAMPLES_PER_CATEGORY held-out objects per category"
    "$PYTHON_BIN" stage_2/select_internal_target_pilot.py \
        --metadata "$TEST_DIR/metadata.csv" \
        --output "$IDS_FILE" \
        --per-category "$SAMPLES_PER_CATEGORY"
fi

while IFS= read -r sample_id; do
    [[ -n "$sample_id" ]] || continue
    for path in \
        "$TEST_DIR/renders/$sample_id/mesh.ply" \
        "$TEST_DIR/renders_cond/$sample_id/transforms.json" \
        "$TEST_DIR/voxels/$sample_id.ply"; do
        if [[ ! -f "$path" ]]; then
            echo "Selected sample is incomplete: $path" >&2
            exit 1
        fi
    done
done < "$IDS_FILE"

generate() {
    local method="$1"
    local ss_ckpt="$2"
    local coordinate_threshold="$3"
    "$PYTHON_BIN" stage_2/export_full_pipeline_voxels.py \
        --dataset-dir "$TEST_DIR" \
        --output-dir "$PRED_ROOT/$method/seed_$SEED" \
        --ids-file "$IDS_FILE" \
        --view-index "$VIEW_INDEX" \
        --seed "$SEED" \
        --slat-seed "$SEED" \
        --coordinate-threshold "$coordinate_threshold" \
        --ss-lora-ckpt "$ss_ckpt" \
        --slat-lora-ckpt "$SLAT_CKPT" \
        --decoder-lora-ckpt "$DECODER_CKPT" \
        --skip-existing
}

echo "[2/7] Generating calibrated Objective-1 baseline (threshold=$OBJECTIVE1_THRESHOLD)"
generate objective1 "$BASE_SS_CKPT" "$OBJECTIVE1_THRESHOLD"

echo "[3/7] Generating corrected coordinate-head model (threshold=$COORDINATE_THRESHOLD)"
generate coordinate_head "$HEAD_SS_CKPT" "$COORDINATE_THRESHOLD"

METHODS=(objective1 coordinate_head)

echo "[4/7] Comparing raw Stage-1 sparse coordinates"
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

echo "[5/7] Comparing full-pipeline mesh voxels"
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
    --method "coordinate_head=$PRED_ROOT/coordinate_head/seed_$SEED/mesh"

echo "Coordinate-head evaluation complete."
echo "Structure metrics: $OUTPUT_DIR/structure_metrics/summary.csv"
echo "Mesh metrics:      $OUTPUT_DIR/mesh_metrics/summary.csv"
echo "Cutaway metrics:   $OUTPUT_DIR/cutaway_metrics/summary.csv"
