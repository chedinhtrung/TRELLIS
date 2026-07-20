#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/datasets/ShapeNetTRELLIS_full}"
OBJECTIVE1_DIR="${OBJECTIVE1_DIR:-$REPO_ROOT/results/objective1_view18_full}"
DINO_DIR="${DINO_DIR:-$REPO_ROOT/results/dino_retrieval_view18}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/dino_rerank_consensus_view18}"
PYTHON_BIN="${PYTHON_BIN:-python}"

IDS_FILE="$OBJECTIVE1_DIR/selected_ids.txt"
OBJECTIVE1_VOXELS="$OBJECTIVE1_DIR/predictions/objective1/seed_42/voxels"
RANKINGS="$DINO_DIR/rankings.csv"

cd "$REPO_ROOT"

for path in \
    "$DATASET_ROOT/train/metadata.csv" \
    "$DATASET_ROOT/train/voxels" \
    "$DATASET_ROOT/test/metadata.csv" \
    "$DATASET_ROOT/test/voxels" \
    "$IDS_FILE" \
    "$OBJECTIVE1_VOXELS" \
    "$RANKINGS"; do
    if [[ ! -e "$path" ]]; then
        echo "Required input not found: $path" >&2
        exit 1
    fi
done

echo "Evaluating view-18 DINO reranking and multi-neighbor consensus"
echo "This stage is voxel-only and does not require a GPU."

"$PYTHON_BIN" stage_2/evaluate_dino_rerank_consensus.py \
    --dataset-root "$DATASET_ROOT" \
    --objective1-voxels "$OBJECTIVE1_VOXELS" \
    --rankings "$RANKINGS" \
    --ids-file "$IDS_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --view-index 18 \
    --resolution 64 \
    --transplant-margin 2 \
    --margins 1 2 3 4 \
    --rerank-k 20 \
    --rerank-lambdas 0.25 0.5 1.0 2.0 \
    --consensus 5:2 5:3 20:2 20:3 20:5 \
    --visualizations-per-category 3

echo "Run complete. Copy this directory back:"
echo "  $OUTPUT_DIR"
echo "Primary tables:"
echo "  $OUTPUT_DIR/summary.csv"
echo "  $OUTPUT_DIR/paired_summary.csv"
echo "  $OUTPUT_DIR/category_summary.csv"
