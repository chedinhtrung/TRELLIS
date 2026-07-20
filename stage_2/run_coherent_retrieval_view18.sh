#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/datasets/ShapeNetTRELLIS_full}"
OBJECTIVE1_DIR="${OBJECTIVE1_DIR:-$REPO_ROOT/results/objective1_view18_full}"
DINO_DIR="${DINO_DIR:-$REPO_ROOT/results/dino_retrieval_view18}"
CALIBRATION_DIR="${CALIBRATION_DIR:-$REPO_ROOT/results/coherent_retrieval_calibration_view18}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/coherent_retrieval_view18}"
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

echo "[1/2] Calibrating conservative coherent component transfer on training shapes"
echo "This is CPU/RAM work; no additional model inference is required."
"$PYTHON_BIN" stage_2/calibrate_coherent_retrieval.py \
    --train-dir "$TRAIN_DIR" \
    --embeddings "$EMBEDDINGS" \
    --output-dir "$CALIBRATION_DIR" \
    --view-index 18 \
    --model dinov2_vitl14_reg \
    --resolution 64 \
    --margin 2 \
    --support-k 20 \
    --selection-k 1 5 20 \
    --exterior-weight 0 0.25 0.5 1.0 \
    --max-calibration-internal-ratio 1.15

echo "[2/2] Applying the frozen policy to all held-out view-18 predictions"
"$PYTHON_BIN" stage_2/evaluate_coherent_retrieval.py \
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
    --gallery-categories bus car \
    --gallery-per-category 15

echo "Coherent retrieval run complete. Copy these directories back:"
echo "  $CALIBRATION_DIR"
echo "  $OUTPUT_DIR"
echo "Then open visualize_interior_comparisons.ipynb and run all cells."
