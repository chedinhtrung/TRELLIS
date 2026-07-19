#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_DIR="${TRAIN_DIR:-$REPO_ROOT/datasets/ShapeNetTRELLIS_full/train}"
TEST_DIR="${TEST_DIR:-$REPO_ROOT/datasets/ShapeNetTRELLIS_full/test}"
OBJECTIVE1_DIR="$REPO_ROOT/results/objective1_full"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$REPO_ROOT/results/geometry_expert/corrected_interior_expert}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DECODER_EPOCHS="${DECODER_EPOCHS:-50}"
EXPERT_EPOCHS="${EXPERT_EPOCHS:-50}"
SAMPLES_PER_CATEGORY="${SAMPLES_PER_CATEGORY:-5}"
SEED="${SEED:-42}"

TARGET_DIR="$EXPERIMENT_DIR/internal_targets"
PREP_DIR="$EXPERIMENT_DIR/data"
DECODER_DIR="$EXPERIMENT_DIR/decoder"
EXPERT_DIR="$EXPERIMENT_DIR/slat_flow"
EVAL_DIR="$EXPERIMENT_DIR/eval_test"
IDS_FILE="$EVAL_DIR/selected_ids.txt"

DECODER_CONFIG="$REPO_ROOT/configs/finetune/slat_decoder_internal_target_pilot_lora.json"
EXPERT_CONFIG="$REPO_ROOT/configs/finetune/slat_flow_img_shapenet_interior_expert.json"
SS_CKPT="$OBJECTIVE1_DIR/ss_flow/ckpts/denoiser_lora_final.pt"
SLAT_CKPT="$OBJECTIVE1_DIR/slat_flow/ckpts/denoiser_lora_final.pt"
OBJECTIVE1_DECODER_CKPT="$OBJECTIVE1_DIR/decoder/ckpts/decoder_lora_final.pt"
ADAPTED_DECODER_CKPT="$DECODER_DIR/ckpts/decoder_lora_final.pt"
EXPERT_CKPT="$EXPERT_DIR/ckpts/denoiser_lora_final.pt"

export SPCONV_ALGO="${SPCONV_ALGO:-native}"
export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"
export SPARSE_ATTN_BACKEND="${SPARSE_ATTN_BACKEND:-sdpa}"
if [[ "${TRELLIS_USE_DINOV2_XFORMERS:-0}" != "1" ]]; then
    export XFORMERS_DISABLED="${XFORMERS_DISABLED:-1}"
fi

cd "$REPO_ROOT"
for path in \
    "$TRAIN_DIR/metadata.csv" \
    "$TEST_DIR/metadata.csv" \
    "$DECODER_CONFIG" \
    "$EXPERT_CONFIG" \
    "$OBJECTIVE1_DIR/ss_flow/config.json" \
    "$OBJECTIVE1_DIR/slat_flow/config.json" \
    "$OBJECTIVE1_DIR/decoder/config.json" \
    "$SS_CKPT" \
    "$SLAT_CKPT" \
    "$OBJECTIVE1_DECODER_CKPT"; do
    if [[ ! -f "$path" ]]; then
        echo "Required input not found: $path" >&2
        exit 1
    fi
done

echo "[1/9] Building and validating internal_v1 targets for 20 training objects"
DATA_DIR="$TRAIN_DIR" \
RUN_DIR="$TARGET_DIR" \
PYTHON_BIN="$PYTHON_BIN" \
OVERRIDE="${OVERRIDE_TARGETS:-1}" \
bash stage_2/run_internal_target_pilot.sh

echo "[2/9] Preparing the deterministic 16-object training subset"
"$PYTHON_BIN" stage_2/prepare_internal_decoder_pilot.py \
    --data_dir "$TRAIN_DIR" \
    --pilot_ids "$TARGET_DIR/pilot_ids.txt" \
    --output_dir "$PREP_DIR"

echo "[3/9] Adapting the mesh decoder to internal_v1 SLAT targets"
"$PYTHON_BIN" train.py \
    --config "$DECODER_CONFIG" \
    --data_dir "$PREP_DIR/train" \
    --output_dir "$DECODER_DIR" \
    --epochs "$DECODER_EPOCHS" \
    --i_save 100 \
    --num_gpus 1 \
    --ckpt latest \
    --auto_retry 0

if [[ ! -f "$ADAPTED_DECODER_CKPT" ]]; then
    echo "Adapted decoder checkpoint not found: $ADAPTED_DECODER_CKPT" >&2
    exit 1
fi

echo "[4/9] Training the routed expert on internal_v1 SLAT targets"
"$PYTHON_BIN" train.py \
    --config "$EXPERT_CONFIG" \
    --data_dir "$PREP_DIR/train" \
    --output_dir "$EXPERT_DIR" \
    --epochs "$EXPERT_EPOCHS" \
    --i_save 100 \
    --num_gpus 1 \
    --ckpt latest \
    --auto_retry 0

if [[ ! -f "$EXPERT_CKPT" ]]; then
    echo "Interior-expert checkpoint not found: $EXPERT_CKPT" >&2
    exit 1
fi

echo "[5/9] Selecting a deterministic category-balanced held-out test subset"
"$PYTHON_BIN" stage_2/select_internal_target_pilot.py \
    --metadata "$TEST_DIR/metadata.csv" \
    --output "$IDS_FILE" \
    --per-category "$SAMPLES_PER_CATEGORY"

generate() {
    local method="$1"
    local slat_ckpt="$2"
    local decoder_ckpt="$3"
    "$PYTHON_BIN" stage_2/export_full_pipeline_voxels.py \
        --dataset-dir "$TEST_DIR" \
        --output-dir "$EVAL_DIR/predictions/$method/seed_$SEED" \
        --ids-file "$IDS_FILE" \
        --seed "$SEED" \
        --ss-lora-ckpt "$SS_CKPT" \
        --slat-lora-ckpt "$slat_ckpt" \
        --decoder-lora-ckpt "$decoder_ckpt"
}

echo "[6/9] Generating the 2x2 SLAT/decoder comparison"
generate objective1_slat_objective1_decoder "$SLAT_CKPT" "$OBJECTIVE1_DECODER_CKPT"
generate expert_slat_objective1_decoder "$EXPERT_CKPT" "$OBJECTIVE1_DECODER_CKPT"
generate objective1_slat_adapted_decoder "$SLAT_CKPT" "$ADAPTED_DECODER_CKPT"
generate expert_slat_adapted_decoder "$EXPERT_CKPT" "$ADAPTED_DECODER_CKPT"

METHODS=(
    objective1_slat_objective1_decoder
    expert_slat_objective1_decoder
    objective1_slat_adapted_decoder
    expert_slat_adapted_decoder
)

echo "[7/9] Computing voxel metrics at margins 1-4"
"$PYTHON_BIN" stage_2/compare_internals.py \
    --gt-voxels "$TEST_DIR/voxels" \
    --pred-root "$EVAL_DIR/predictions" \
    --metadata "$TEST_DIR/metadata.csv" \
    --ids-file "$IDS_FILE" \
    --methods "${METHODS[@]}" \
    --margins 1 2 3 4 \
    --output "$EVAL_DIR/voxel_metrics/summary.csv" \
    --per-sample-output "$EVAL_DIR/voxel_metrics/per_sample.csv"

echo "[8/9] Rendering deterministic GT cutaways"
"$PYTHON_BIN" dataset_toolkits/render_internal_kiui.py \
    --data_dir "$TEST_DIR" \
    --instances "$IDS_FILE"

echo "[9/9] Computing cutaway depth and mask metrics"
"$PYTHON_BIN" stage_2/evaluate_internal_cutaways.py \
    --data_dir "$TEST_DIR" \
    --ids_file "$IDS_FILE" \
    --output_dir "$EVAL_DIR/cutaway_metrics" \
    --method "objective1_slat_objective1_decoder=$EVAL_DIR/predictions/objective1_slat_objective1_decoder/seed_$SEED/mesh" \
    --method "expert_slat_objective1_decoder=$EVAL_DIR/predictions/expert_slat_objective1_decoder/seed_$SEED/mesh" \
    --method "objective1_slat_adapted_decoder=$EVAL_DIR/predictions/objective1_slat_adapted_decoder/seed_$SEED/mesh" \
    --method "expert_slat_adapted_decoder=$EVAL_DIR/predictions/expert_slat_adapted_decoder/seed_$SEED/mesh"

echo "Corrected interior-expert experiment complete."
echo "Voxel metrics:   $EVAL_DIR/voxel_metrics/summary.csv"
echo "Cutaway metrics: $EVAL_DIR/cutaway_metrics/summary.csv"
