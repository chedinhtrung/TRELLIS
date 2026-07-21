#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/datasets/ShapeNetTRELLIS_full}"
OBJECTIVE1_TEST_DIR="${OBJECTIVE1_TEST_DIR:-$REPO_ROOT/results/objective1_view18_full}"
OBJECTIVE1_TRAIN_DIR="${OBJECTIVE1_TRAIN_DIR:-$REPO_ROOT/results/objective1_view18_train}"
DINO_DIR="${DINO_DIR:-$REPO_ROOT/results/dino_retrieval_view18}"
CALIBRATION_DIR="${CALIBRATION_DIR:-$REPO_ROOT/results/retrieval_v2_calibration_view18}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/retrieval_v2_view18}"
PYTHON_BIN="${PYTHON_BIN:-python}"

TRAIN_DIR="$DATASET_ROOT/train"
TEST_DIR="$DATASET_ROOT/test"
EMBEDDINGS="$DINO_DIR/embeddings/train_view018_dinov2_vitl14_reg.npz"
RANKINGS="$DINO_DIR/rankings.csv"
IDS_FILE="$OBJECTIVE1_TEST_DIR/selected_ids.txt"
OBJECTIVE1_TRAIN_VOXELS="$OBJECTIVE1_TRAIN_DIR/predictions/objective1/seed_42/voxels"
OBJECTIVE1_TEST_VOXELS="$OBJECTIVE1_TEST_DIR/predictions/objective1/seed_42/voxels"
OBJECTIVE1_TEST_MESHES="$OBJECTIVE1_TEST_DIR/predictions/objective1/seed_42/mesh"
POLICY="$CALIBRATION_DIR/policy.json"

cd "$REPO_ROOT"

if ! "$PYTHON_BIN" -c 'import trimesh' >/dev/null 2>&1; then
    echo "Missing retrieval-v2 dependency. Run: $PYTHON_BIN -m pip install trimesh" >&2
    exit 1
fi
"$PYTHON_BIN" stage_2/test_retrieval_v2.py

for path in \
    "$TRAIN_DIR/metadata.csv" \
    "$TRAIN_DIR/voxels" \
    "$TRAIN_DIR/renders" \
    "$TEST_DIR/metadata.csv" \
    "$TEST_DIR/voxels" \
    "$TEST_DIR/renders" \
    "$EMBEDDINGS" \
    "$RANKINGS" \
    "$IDS_FILE" \
    "$OBJECTIVE1_TRAIN_VOXELS" \
    "$OBJECTIVE1_TEST_VOXELS" \
    "$OBJECTIVE1_TEST_MESHES"; do
    if [[ ! -e "$path" ]]; then
        echo "Required input not found: $path" >&2
        exit 1
    fi
done

if [[ -f "$POLICY" ]]; then
    echo "[1/2] Reusing frozen retrieval-v2 policy at $POLICY"
else
    echo "[1/2] Fitting the top-20 reranker and hybrid gate on train-only splits"
    "$PYTHON_BIN" stage_2/calibrate_retrieval_v2.py \
        --train-dir "$TRAIN_DIR" \
        --objective1-voxels "$OBJECTIVE1_TRAIN_VOXELS" \
        --embeddings "$EMBEDDINGS" \
        --output-dir "$CALIBRATION_DIR" \
        --view-index 18 \
        --model dinov2_vitl14_reg \
        --resolution 64 \
        --margin 2 \
        --top-k 20 \
        --ridge 1.0 \
        --max-internal-ratio 1.15
fi

echo "[2/2] Applying the frozen policy and exporting voxel plus smooth triangle meshes"
"$PYTHON_BIN" stage_2/evaluate_retrieval_v2.py \
    --dataset-root "$DATASET_ROOT" \
    --objective1-voxels "$OBJECTIVE1_TEST_VOXELS" \
    --objective1-meshes "$OBJECTIVE1_TEST_MESHES" \
    --rankings "$RANKINGS" \
    --policy "$POLICY" \
    --ids-file "$IDS_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --view-index 18 \
    --resolution 64 \
    --transplant-margin 2 \
    --margins 1 2 3 4 \
    --gallery-categories bus car \
    --gallery-per-category 15 \
    --save-smooth-meshes \
    --resume

echo "Retrieval-v2 run complete. Copy these directories back:"
echo "  $CALIBRATION_DIR"
echo "  $OUTPUT_DIR"
echo "Then open visualize_interior_comparisons.ipynb and run all cells."
