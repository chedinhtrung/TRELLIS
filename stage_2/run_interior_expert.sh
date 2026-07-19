#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_DIR="${TRAIN_DIR:-$REPO_ROOT/datasets/ShapeNetTRELLIS_full/train}"
TEST_DIR="${TEST_DIR:-$REPO_ROOT/datasets/ShapeNetTRELLIS_full/test}"
OBJECTIVE1_DIR="$REPO_ROOT/results/objective1_full"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/results/geometry_expert/interior_expert}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EPOCHS="${EPOCHS:-3}"
SAMPLES_PER_CATEGORY="${SAMPLES_PER_CATEGORY:-5}"
SEED="${SEED:-42}"

CONFIG="$REPO_ROOT/configs/finetune/slat_flow_img_shapenet_interior_expert.json"
SS_CKPT="$OBJECTIVE1_DIR/ss_flow/ckpts/denoiser_lora_final.pt"
SLAT_CKPT="$OBJECTIVE1_DIR/slat_flow/ckpts/denoiser_lora_final.pt"
DECODER_CKPT="$OBJECTIVE1_DIR/decoder/ckpts/decoder_lora_final.pt"
EXPERT_CKPT="$RUN_DIR/slat_flow/ckpts/denoiser_lora_final.pt"
EVAL_DIR="$RUN_DIR/eval_test"
IDS_FILE="$EVAL_DIR/selected_ids.txt"

export SPCONV_ALGO="${SPCONV_ALGO:-native}"
export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"

cd "$REPO_ROOT"
for path in \
    "$TRAIN_DIR/metadata.csv" \
    "$TEST_DIR/metadata.csv" \
    "$OBJECTIVE1_DIR/ss_flow/config.json" \
    "$OBJECTIVE1_DIR/slat_flow/config.json" \
    "$OBJECTIVE1_DIR/decoder/config.json" \
    "$SS_CKPT" \
    "$SLAT_CKPT" \
    "$DECODER_CKPT"; do
    if [[ ! -f "$path" ]]; then
        echo "Required input not found: $path" >&2
        exit 1
    fi
done

echo "[1/7] Training the margin-2 interior expert; Objective-1 SLAT LoRA stays frozen"
"$PYTHON_BIN" train.py \
    --config "$CONFIG" \
    --data_dir "$TRAIN_DIR" \
    --output_dir "$RUN_DIR/slat_flow" \
    --epochs "$EPOCHS" \
    --i_save 250 \
    --num_gpus 1 \
    --ckpt latest \
    --auto_retry 0

if [[ ! -f "$EXPERT_CKPT" ]]; then
    echo "Interior-expert checkpoint not found after training: $EXPERT_CKPT" >&2
    exit 1
fi

echo "[2/7] Selecting a deterministic category-balanced test subset"
"$PYTHON_BIN" stage_2/select_internal_target_pilot.py \
    --metadata "$TEST_DIR/metadata.csv" \
    --output "$IDS_FILE" \
    --per-category "$SAMPLES_PER_CATEGORY"

echo "[3/7] Generating the Objective-1 control"
"$PYTHON_BIN" stage_2/export_full_pipeline_voxels.py \
    --dataset-dir "$TEST_DIR" \
    --output-dir "$EVAL_DIR/predictions/objective1/seed_$SEED" \
    --ids-file "$IDS_FILE" \
    --seed "$SEED" \
    --ss-lora-ckpt "$SS_CKPT" \
    --slat-lora-ckpt "$SLAT_CKPT" \
    --decoder-lora-ckpt "$DECODER_CKPT" \
    --skip-existing

echo "[4/7] Generating with the routed SLAT expert"
"$PYTHON_BIN" stage_2/export_full_pipeline_voxels.py \
    --dataset-dir "$TEST_DIR" \
    --output-dir "$EVAL_DIR/predictions/interior_expert/seed_$SEED" \
    --ids-file "$IDS_FILE" \
    --seed "$SEED" \
    --ss-lora-ckpt "$SS_CKPT" \
    --slat-lora-ckpt "$EXPERT_CKPT" \
    --decoder-lora-ckpt "$DECODER_CKPT"

echo "[5/7] Computing voxel metrics at margins 1-4"
"$PYTHON_BIN" stage_2/compare_internals.py \
    --gt-voxels "$TEST_DIR/voxels" \
    --pred-root "$EVAL_DIR/predictions" \
    --metadata "$TEST_DIR/metadata.csv" \
    --ids-file "$IDS_FILE" \
    --methods objective1 interior_expert \
    --margins 1 2 3 4 \
    --output "$EVAL_DIR/voxel_metrics/summary.csv" \
    --per-sample-output "$EVAL_DIR/voxel_metrics/per_sample.csv"

echo "[6/7] Rendering deterministic GT cutaways for the same objects"
"$PYTHON_BIN" dataset_toolkits/render_internal_kiui.py \
    --data_dir "$TEST_DIR" \
    --instances "$IDS_FILE"

echo "[7/7] Computing cutaway depth and mask metrics"
"$PYTHON_BIN" stage_2/evaluate_internal_cutaways.py \
    --data_dir "$TEST_DIR" \
    --ids_file "$IDS_FILE" \
    --output_dir "$EVAL_DIR/cutaway_metrics" \
    --method "objective1=$EVAL_DIR/predictions/objective1/seed_$SEED/mesh" \
    --method "interior_expert=$EVAL_DIR/predictions/interior_expert/seed_$SEED/mesh"

echo "Interior-expert experiment complete."
echo "Voxel metrics: $EVAL_DIR/voxel_metrics/summary.csv"
echo "Cutaway metrics: $EVAL_DIR/cutaway_metrics/summary.csv"
