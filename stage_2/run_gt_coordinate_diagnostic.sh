#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_DIR="${TEST_DIR:-$REPO_ROOT/datasets/ShapeNetTRELLIS_full/test}"
OBJECTIVE1_DIR="${OBJECTIVE1_DIR:-$REPO_ROOT/results/objective1_full}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/gt_coordinate_diagnostic}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SAMPLES_PER_CATEGORY="${SAMPLES_PER_CATEGORY:-5}"
SEED="${SEED:-42}"
RESUME="${RESUME:-0}"
GT_LATENT_NAME="dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16"

SS_CKPT="$OBJECTIVE1_DIR/ss_flow/ckpts/denoiser_lora_final.pt"
SLAT_CKPT="$OBJECTIVE1_DIR/slat_flow/ckpts/denoiser_lora_final.pt"
DECODER_CKPT="$OBJECTIVE1_DIR/decoder/ckpts/decoder_lora_final.pt"
IDS_FILE="$OUTPUT_DIR/selected_ids.txt"
PRED_ROOT="$OUTPUT_DIR/predictions"

export SPCONV_ALGO="${SPCONV_ALGO:-native}"
export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"
export SPARSE_ATTN_BACKEND="${SPARSE_ATTN_BACKEND:-sdpa}"
if [[ "${TRELLIS_USE_DINOV2_XFORMERS:-0}" != "1" ]]; then
    export XFORMERS_DISABLED="${XFORMERS_DISABLED:-1}"
fi

cd "$REPO_ROOT"

for path in \
    "$TEST_DIR/metadata.csv" \
    "$TEST_DIR/voxels" \
    "$TEST_DIR/renders" \
    "$TEST_DIR/renders_cond" \
    "$TEST_DIR/latents/$GT_LATENT_NAME" \
    "$OBJECTIVE1_DIR/ss_flow/config.json" \
    "$OBJECTIVE1_DIR/slat_flow/config.json" \
    "$OBJECTIVE1_DIR/decoder/config.json" \
    "$SS_CKPT" \
    "$SLAT_CKPT" \
    "$DECODER_CKPT"; do
    if [[ ! -e "$path" ]]; then
        echo "Required input not found: $path" >&2
        exit 1
    fi
done

if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit)" ]] && [[ "$RESUME" != "1" ]]; then
    echo "Output directory is not empty: $OUTPUT_DIR" >&2
    echo "Use a new OUTPUT_DIR, or set RESUME=1 only to continue the same run." >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

if [[ "$RESUME" == "1" ]] && [[ -f "$IDS_FILE" ]]; then
    echo "[1/6] Reusing selected test objects from $IDS_FILE"
else
    echo "[1/6] Selecting $SAMPLES_PER_CATEGORY held-out test objects per category"
    "$PYTHON_BIN" stage_2/select_internal_target_pilot.py \
        --metadata "$TEST_DIR/metadata.csv" \
        --output "$IDS_FILE" \
        --per-category "$SAMPLES_PER_CATEGORY"
fi

while IFS= read -r sample_id; do
    [[ -n "$sample_id" ]] || continue
    for path in \
        "$TEST_DIR/renders/$sample_id/mesh.ply" \
        "$TEST_DIR/renders_cond/$sample_id/000.png" \
        "$TEST_DIR/voxels/$sample_id.ply" \
        "$TEST_DIR/latents/$GT_LATENT_NAME/$sample_id.npz"; do
        if [[ ! -f "$path" ]]; then
            echo "Selected sample is incomplete: $path" >&2
            exit 1
        fi
    done
done < "$IDS_FILE"

generate() {
    local method="$1"
    shift
    "$PYTHON_BIN" stage_2/export_full_pipeline_voxels.py \
        --dataset-dir "$TEST_DIR" \
        --output-dir "$PRED_ROOT/$method/seed_$SEED" \
        --ids-file "$IDS_FILE" \
        --seed "$SEED" \
        --slat-seed "$SEED" \
        --ss-lora-ckpt "$SS_CKPT" \
        --slat-lora-ckpt "$SLAT_CKPT" \
        --decoder-lora-ckpt "$DECODER_CKPT" \
        --skip-existing \
        "$@"
}

echo "[2/6] Generating Objective-1 outputs with predicted coordinates"
generate pred_coords_generated_slat

echo "[3/6] Generating Objective-1 outputs with GT coordinates"
generate gt_coords_generated_slat \
    --gt-coords-latent-name "$GT_LATENT_NAME"

METHODS=(pred_coords_generated_slat gt_coords_generated_slat)

echo "[4/6] Computing voxel metrics at margins 1-4"
"$PYTHON_BIN" stage_2/compare_internals.py \
    --gt-voxels "$TEST_DIR/voxels" \
    --pred-root "$PRED_ROOT" \
    --metadata "$TEST_DIR/metadata.csv" \
    --ids-file "$IDS_FILE" \
    --methods "${METHODS[@]}" \
    --margins 1 2 3 4 \
    --output "$OUTPUT_DIR/voxel_metrics/summary.csv" \
    --per-sample-output "$OUTPUT_DIR/voxel_metrics/per_sample.csv"

echo "[5/6] Preparing deterministic GT cutaways"
"$PYTHON_BIN" dataset_toolkits/render_internal_kiui.py \
    --data_dir "$TEST_DIR" \
    --instances "$IDS_FILE"

echo "[6/6] Computing cutaway mask and depth metrics"
"$PYTHON_BIN" stage_2/evaluate_internal_cutaways.py \
    --data_dir "$TEST_DIR" \
    --ids_file "$IDS_FILE" \
    --output_dir "$OUTPUT_DIR/cutaway_metrics" \
    --method "pred_coords_generated_slat=$PRED_ROOT/pred_coords_generated_slat/seed_$SEED/mesh" \
    --method "gt_coords_generated_slat=$PRED_ROOT/gt_coords_generated_slat/seed_$SEED/mesh"

echo "GT-coordinate diagnostic complete."
echo "Voxel metrics:   $OUTPUT_DIR/voxel_metrics/summary.csv"
echo "Cutaway metrics: $OUTPUT_DIR/cutaway_metrics/summary.csv"
