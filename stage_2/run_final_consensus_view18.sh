#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/datasets/ShapeNetTRELLIS_full}"
OBJECTIVE1_DIR="${OBJECTIVE1_DIR:-$REPO_ROOT/results/objective1_view18_full}"
DINO_DIR="${DINO_DIR:-$REPO_ROOT/results/dino_retrieval_view18}"
CALIBRATION_DIR="${CALIBRATION_DIR:-$REPO_ROOT/results/dino_consensus_calibration_view18}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/final_consensus_view18}"
PYTHON_BIN="${PYTHON_BIN:-python}"

TRAIN_DIR="$DATASET_ROOT/train"
EMBEDDINGS="$DINO_DIR/embeddings/train_view018_dinov2_vitl14_reg.npz"
RANKINGS="$DINO_DIR/rankings.csv"
IDS_FILE="$OBJECTIVE1_DIR/selected_ids.txt"
OBJECTIVE1_VOXELS="$OBJECTIVE1_DIR/predictions/objective1/seed_42/voxels"
POLICY="$CALIBRATION_DIR/policy.json"

cd "$REPO_ROOT"

for path in \
    "$TRAIN_DIR/metadata.csv" \
    "$TRAIN_DIR/voxels" \
    "$DATASET_ROOT/test/metadata.csv" \
    "$DATASET_ROOT/test/voxels" \
    "$EMBEDDINGS" \
    "$RANKINGS" \
    "$IDS_FILE" \
    "$OBJECTIVE1_VOXELS"; do
    if [[ ! -e "$path" ]]; then
        echo "Required input not found: $path" >&2
        exit 1
    fi
done

echo "[1/2] Calibrating consensus with training-only leave-one-out retrieval"
"$PYTHON_BIN" stage_2/calibrate_dino_consensus.py \
    --train-dir "$TRAIN_DIR" \
    --embeddings "$EMBEDDINGS" \
    --output-dir "$CALIBRATION_DIR" \
    --view-index 18 \
    --model dinov2_vitl14_reg \
    --resolution 64 \
    --margin 2 \
    --consensus 5:2 5:3 20:2 20:3 20:5

echo "[2/2] Applying the frozen policy to all held-out view-18 predictions"
"$PYTHON_BIN" stage_2/evaluate_calibrated_consensus.py \
    --dataset-root "$DATASET_ROOT" \
    --objective1-voxels "$OBJECTIVE1_VOXELS" \
    --rankings "$RANKINGS" \
    --policy "$POLICY" \
    --ids-file "$IDS_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --view-index 18 \
    --resolution 64 \
    --transplant-margin 2 \
    --margins 1 2 3 4 \
    --global-reference 20:5 \
    --visualizations-per-category 3

echo "Final run complete. Copy both directories back:"
echo "  $CALIBRATION_DIR"
echo "  $OUTPUT_DIR"
echo "Then open visualize_interior_comparisons.ipynb and run all cells."
