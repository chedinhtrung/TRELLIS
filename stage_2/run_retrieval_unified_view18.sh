#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/datasets/ShapeNetTRELLIS_full}"
OBJECTIVE1_TEST_DIR="${OBJECTIVE1_TEST_DIR:-$REPO_ROOT/results/objective1_view18_full}"
OBJECTIVE1_TRAIN_DIR="${OBJECTIVE1_TRAIN_DIR:-$REPO_ROOT/results/objective1_view18_train}"
DINO_DIR="${DINO_DIR:-$REPO_ROOT/results/dino_retrieval_view18}"
V2_CALIBRATION_DIR="${V2_CALIBRATION_DIR:-$REPO_ROOT/results/retrieval_v2_calibration_view18}"
V21_CALIBRATION_DIR="${V21_CALIBRATION_DIR:-$REPO_ROOT/results/retrieval_v21_calibration_view18}"
UNIFIED_CALIBRATION_DIR="${UNIFIED_CALIBRATION_DIR:-$REPO_ROOT/results/retrieval_unified_calibration_view18}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/retrieval_unified_view18}"
PYTHON_BIN="${PYTHON_BIN:-python}"
UNIFIED_QUERIES_PER_CATEGORY="${UNIFIED_QUERIES_PER_CATEGORY:-160}"
CALIBRATION_WORKERS="${CALIBRATION_WORKERS:-8}"
UNIFIED_FOLDS="${UNIFIED_FOLDS:-5}"
SELECTOR_NEIGHBORS="${SELECTOR_NEIGHBORS:-8}"
DONOR_SHORTLIST="${DONOR_SHORTLIST:-8}"
MINIMUM_HEADLINE_DELTA="${MINIMUM_HEADLINE_DELTA:-0.005}"
UNCERTAINTY_WEIGHT="${UNCERTAINTY_WEIGHT:-0.0}"
GALLERY_PER_CATEGORY="${GALLERY_PER_CATEGORY:-15}"

TRAIN_DIR="$DATASET_ROOT/train"
TEST_DIR="$DATASET_ROOT/test"
EMBEDDINGS="$DINO_DIR/embeddings/train_view018_dinov2_vitl14_reg.npz"
RANKINGS="$DINO_DIR/rankings.csv"
IDS_FILE="${IDS_FILE:-$OBJECTIVE1_TEST_DIR/selected_ids.txt}"
OBJECTIVE1_TRAIN_VOXELS="$OBJECTIVE1_TRAIN_DIR/predictions/objective1/seed_42/voxels"
OBJECTIVE1_TEST_VOXELS="$OBJECTIVE1_TEST_DIR/predictions/objective1/seed_42/voxels"
OBJECTIVE1_TEST_MESHES="$OBJECTIVE1_TEST_DIR/predictions/objective1/seed_42/mesh"
V2_POLICY="$V2_CALIBRATION_DIR/policy.json"
RERANKER_FIT="$V2_CALIBRATION_DIR/reranker_fit.csv"
V21_POLICY="$V21_CALIBRATION_DIR/policy.json"
POLICY="$UNIFIED_CALIBRATION_DIR/policy.json"

cd "$REPO_ROOT"

if ! "$PYTHON_BIN" -c 'import numpy, trimesh, utils3d' >/dev/null 2>&1; then
    echo "Missing retrieval dependency. Activate the TRELLIS environment first." >&2
    exit 1
fi
"$PYTHON_BIN" stage_2/test_retrieval_v2.py
"$PYTHON_BIN" stage_2/test_retrieval_unified.py

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
    "$V2_POLICY" \
    "$RERANKER_FIT" \
    "$V21_POLICY"; do
    if [[ ! -e "$path" ]]; then
        echo "Required input not found: $path" >&2
        exit 1
    fi
done

if [[ -f "$POLICY" ]]; then
    echo "[1/2] Reusing frozen unified policy at $POLICY"
else
    echo "[1/2] Training one category-blind selector with durable per-query caches"
    "$PYTHON_BIN" stage_2/calibrate_retrieval_unified.py \
        --train-dir "$TRAIN_DIR" \
        --objective1-voxels "$OBJECTIVE1_TRAIN_VOXELS" \
        --embeddings "$EMBEDDINGS" \
        --reranker-fit "$RERANKER_FIT" \
        --v2-policy "$V2_POLICY" \
        --v21-policy "$V21_POLICY" \
        --output-dir "$UNIFIED_CALIBRATION_DIR" \
        --view-index 18 \
        --model dinov2_vitl14_reg \
        --resolution 64 \
        --margin 2 \
        --top-k 20 \
        --folds "$UNIFIED_FOLDS" \
        --selector-neighbors "$SELECTOR_NEIGHBORS" \
        --donor-ridge 1.0 \
        --donor-shortlist "$DONOR_SHORTLIST" \
        --donor-reranker-queries-per-category 48 \
        --max-internal-ratio 1.15 \
        --minimum-headline-delta "$MINIMUM_HEADLINE_DELTA" \
        --queries-per-category "$UNIFIED_QUERIES_PER_CATEGORY" \
        --workers "$CALIBRATION_WORKERS" \
        --uncertainty-weight "$UNCERTAINTY_WEIGHT" \
        --resume
fi

echo "[2/2] Applying the unified selector and exporting voxel plus triangle meshes"
"$PYTHON_BIN" stage_2/evaluate_retrieval_unified.py \
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

echo "Unified retrieval run complete. Copy these directories back:"
echo "  $UNIFIED_CALIBRATION_DIR"
echo "  $OUTPUT_DIR"
echo "Then open visualize_interior_comparisons.ipynb and run all cells."
