#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/datasets/ShapeNetTRELLIS_full}"
OBJECTIVE1_TEST_DIR="${OBJECTIVE1_TEST_DIR:-$REPO_ROOT/results/objective1_view18_full}"
OBJECTIVE1_TRAIN_DIR="${OBJECTIVE1_TRAIN_DIR:-$REPO_ROOT/results/objective1_view18_train}"
DINO_DIR="${DINO_DIR:-$REPO_ROOT/results/dino_retrieval_view18}"
V2_CALIBRATION_DIR="${V2_CALIBRATION_DIR:-$REPO_ROOT/results/retrieval_v2_calibration_view18}"
V21_CALIBRATION_DIR="${V21_CALIBRATION_DIR:-$REPO_ROOT/results/retrieval_v21_calibration_view18}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/retrieval_v21_view18}"
PYTHON_BIN="${PYTHON_BIN:-python}"
FUSION_QUERIES_PER_CATEGORY="${FUSION_QUERIES_PER_CATEGORY:-24}"
CALIBRATION_WORKERS="${CALIBRATION_WORKERS:-4}"
GALLERY_PER_CATEGORY="${GALLERY_PER_CATEGORY:-15}"

TRAIN_DIR="$DATASET_ROOT/train"
TEST_DIR="$DATASET_ROOT/test"
EMBEDDINGS="$DINO_DIR/embeddings/train_view018_dinov2_vitl14_reg.npz"
RANKINGS="$DINO_DIR/rankings.csv"
IDS_FILE="${IDS_FILE:-$OBJECTIVE1_TEST_DIR/selected_ids.txt}"
OBJECTIVE1_TRAIN_VOXELS="$OBJECTIVE1_TRAIN_DIR/predictions/objective1/seed_42/voxels"
OBJECTIVE1_TEST_VOXELS="$OBJECTIVE1_TEST_DIR/predictions/objective1/seed_42/voxels"
OBJECTIVE1_TEST_MESHES="$OBJECTIVE1_TEST_DIR/predictions/objective1/seed_42/mesh"
POLICY="$V21_CALIBRATION_DIR/policy.json"

cd "$REPO_ROOT"

if ! "$PYTHON_BIN" -c 'import numpy, trimesh, utils3d' >/dev/null 2>&1; then
    echo "Missing retrieval dependency. Activate the TRELLIS environment first." >&2
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
    "$OBJECTIVE1_TEST_MESHES" \
    "$V2_CALIBRATION_DIR/policy.json" \
    "$V2_CALIBRATION_DIR/reranker_fit.csv"; do
    if [[ ! -e "$path" ]]; then
        echo "Required input not found: $path" >&2
        exit 1
    fi
done

if [[ -f "$POLICY" ]]; then
    echo "[1/2] Reusing frozen retrieval-v2.1 policy at $POLICY"
else
    echo "[1/2] Calibrating structural non-car policies with durable per-query caches"
    "$PYTHON_BIN" stage_2/calibrate_retrieval_v21.py \
        --train-dir "$TRAIN_DIR" \
        --objective1-voxels "$OBJECTIVE1_TRAIN_VOXELS" \
        --embeddings "$EMBEDDINGS" \
        --v2-calibration-dir "$V2_CALIBRATION_DIR" \
        --output-dir "$V21_CALIBRATION_DIR" \
        --categories bus cabinet file_cabinet \
        --view-index 18 \
        --model dinov2_vitl14_reg \
        --resolution 64 \
        --margin 2 \
        --top-k 20 \
        --ridge 1.0 \
        --max-internal-ratio 1.15 \
        --fusion-queries-per-category "$FUSION_QUERIES_PER_CATEGORY" \
        --workers "$CALIBRATION_WORKERS" \
        --resume
fi

echo "[2/2] Applying v2.1 and exporting coherent voxel plus triangle meshes"
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
    --gallery-categories bus cabinet car file_cabinet \
    --gallery-per-category "$GALLERY_PER_CATEGORY" \
    --save-smooth-meshes \
    --resume

echo "Retrieval-v2.1 run complete. Copy these directories back:"
echo "  $V21_CALIBRATION_DIR"
echo "  $OUTPUT_DIR"
echo "Then open visualize_interior_comparisons.ipynb and run all cells."
