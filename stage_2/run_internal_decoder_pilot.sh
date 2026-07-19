#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/datasets/ShapeNetTRELLIS_full/train}"
TARGET_PILOT_DIR="${TARGET_PILOT_DIR:-$REPO_ROOT/results/geometry_expert/pilot_targets}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/results/geometry_expert/decoder_adapter_pilot}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EPOCHS="${EPOCHS:-50}"

CONFIG="$REPO_ROOT/configs/finetune/slat_decoder_internal_target_pilot_lora.json"
OBJECTIVE1_CKPT="$REPO_ROOT/results/objective1_full/decoder/ckpts/decoder_lora_final.pt"
PREP_DIR="$RUN_DIR/data"
TRAIN_DIR="$RUN_DIR/decoder"
EVAL_DIR="$RUN_DIR/eval_val"
ADAPTED_CKPT="$TRAIN_DIR/ckpts/decoder_lora_final.pt"

export SPCONV_ALGO="${SPCONV_ALGO:-native}"
export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"

cd "$REPO_ROOT"

if [[ ! -f "$TARGET_PILOT_DIR/pilot_ids.txt" ]]; then
    echo "Pilot IDs not found: $TARGET_PILOT_DIR/pilot_ids.txt" >&2
    exit 1
fi
if [[ ! -f "$OBJECTIVE1_CKPT" ]]; then
    echo "Objective-1 decoder LoRA not found: $OBJECTIVE1_CKPT" >&2
    exit 1
fi

echo "[1/6] Preparing deterministic 16-train/4-validation pilot"
"$PYTHON_BIN" stage_2/prepare_internal_decoder_pilot.py \
    --data_dir "$DATA_DIR" \
    --pilot_ids "$TARGET_PILOT_DIR/pilot_ids.txt" \
    --output_dir "$PREP_DIR"

echo "[2/6] Adapting the decoder on internal-aware SLAT targets"
"$PYTHON_BIN" train.py \
    --config "$CONFIG" \
    --data_dir "$PREP_DIR/train" \
    --output_dir "$TRAIN_DIR" \
    --epochs "$EPOCHS" \
    --i_save 100 \
    --num_gpus 1 \
    --ckpt latest \
    --auto_retry 0

if [[ ! -f "$ADAPTED_CKPT" ]]; then
    echo "Adapted decoder LoRA not found after training: $ADAPTED_CKPT" >&2
    exit 1
fi

echo "[3/6] Decoding held-out targets with the Objective-1 decoder"
"$PYTHON_BIN" stage_2/decode_internal_target_oracle.py \
    --data_dir "$DATA_DIR" \
    --ids_file "$PREP_DIR/val_ids.txt" \
    --output_dir "$EVAL_DIR/objective1_decoder" \
    --decoder_lora_ckpt "$OBJECTIVE1_CKPT"

echo "[4/6] Decoding held-out targets with the adapted decoder"
"$PYTHON_BIN" stage_2/decode_internal_target_oracle.py \
    --data_dir "$DATA_DIR" \
    --ids_file "$PREP_DIR/val_ids.txt" \
    --output_dir "$EVAL_DIR/adapted_decoder" \
    --decoder_lora_ckpt "$ADAPTED_CKPT"

COMBINED_PREDICTIONS="$EVAL_DIR/combined_predictions"
mkdir -p "$COMBINED_PREDICTIONS" "$EVAL_DIR/voxel_metrics"

safe_relative_link() {
    local target="$1"
    local destination="$2"
    if [[ ! -d "$(dirname "$destination")/$target" ]]; then
        echo "Prediction directory not found: $(dirname "$destination")/$target" >&2
        exit 1
    fi
    if [[ -L "$destination" ]]; then
        if [[ "$(readlink "$destination")" != "$target" ]]; then
            echo "Refusing to replace different symlink: $destination" >&2
            exit 1
        fi
    elif [[ -e "$destination" ]]; then
        echo "Refusing to replace existing path: $destination" >&2
        exit 1
    else
        ln -s "$target" "$destination"
    fi
}

safe_relative_link "../objective1_decoder/predictions/old_target" "$COMBINED_PREDICTIONS/objective1_old_target"
safe_relative_link "../objective1_decoder/predictions/new_target" "$COMBINED_PREDICTIONS/objective1_new_target"
safe_relative_link "../adapted_decoder/predictions/old_target" "$COMBINED_PREDICTIONS/adapted_old_target"
safe_relative_link "../adapted_decoder/predictions/new_target" "$COMBINED_PREDICTIONS/adapted_new_target"

echo "[5/6] Computing held-out voxel metrics at margins 1-4"
"$PYTHON_BIN" stage_2/compare_internals.py \
    --gt-voxels "$DATA_DIR/voxels" \
    --pred-root "$COMBINED_PREDICTIONS" \
    --metadata "$DATA_DIR/metadata.csv" \
    --ids-file "$PREP_DIR/val_ids.txt" \
    --methods objective1_old_target objective1_new_target adapted_old_target adapted_new_target \
    --margins 1 2 3 4 \
    --output "$EVAL_DIR/voxel_metrics/summary.csv" \
    --per-sample-output "$EVAL_DIR/voxel_metrics/per_sample.csv"

echo "[6/6] Computing exact held-out cutaway depth and mask metrics"
"$PYTHON_BIN" stage_2/evaluate_internal_cutaways.py \
    --data_dir "$DATA_DIR" \
    --ids_file "$PREP_DIR/val_ids.txt" \
    --output_dir "$EVAL_DIR/cutaway_metrics" \
    --method "objective1_old_target=$EVAL_DIR/objective1_decoder/predictions/old_target/seed_0/mesh" \
    --method "objective1_new_target=$EVAL_DIR/objective1_decoder/predictions/new_target/seed_0/mesh" \
    --method "adapted_old_target=$EVAL_DIR/adapted_decoder/predictions/old_target/seed_0/mesh" \
    --method "adapted_new_target=$EVAL_DIR/adapted_decoder/predictions/new_target/seed_0/mesh"

echo "Decoder pilot complete."
echo "Cutaway metrics: $EVAL_DIR/cutaway_metrics/summary.csv"
echo "Voxel metrics: $EVAL_DIR/voxel_metrics/summary.csv"
