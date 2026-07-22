#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/datasets/ShapeNetTRELLIS_full}"
OBJECTIVE1_TEST_DIR="${OBJECTIVE1_TEST_DIR:-$REPO_ROOT/results/objective1_view18_full}"
DINO_DIR="${DINO_DIR:-$REPO_ROOT/results/dino_retrieval_view18}"
FROZEN_RESULTS_DIR="${FROZEN_RESULTS_DIR:-$REPO_ROOT/results/retrieval_unified_view18}"
SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-$REPO_ROOT/results/retrieval_unified_smoke_view18}"
PYTHON_BIN="${PYTHON_BIN:-python}"

POLICY="${POLICY:-$FROZEN_RESULTS_DIR/policy.json}"
RANKINGS="$DINO_DIR/rankings.csv"
IDS_FILE="$REPO_ROOT/stage_2/final_retrieval/smoke_ids.txt"
REFERENCE="$REPO_ROOT/stage_2/final_retrieval/smoke_reference.csv"
OBJECTIVE1_VOXELS="$OBJECTIVE1_TEST_DIR/predictions/objective1/seed_42/voxels"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/stage_2${PYTHONPATH:+:$PYTHONPATH}"

if ! "$PYTHON_BIN" -c 'import numpy, utils3d' >/dev/null 2>&1; then
    echo "Missing retrieval dependency. Activate the TRELLIS environment first." >&2
    exit 1
fi

"$PYTHON_BIN" -m final_retrieval.test_pipeline

for path in \
    "$DATASET_ROOT/train/metadata.csv" \
    "$DATASET_ROOT/train/voxels" \
    "$DATASET_ROOT/test/metadata.csv" \
    "$DATASET_ROOT/test/voxels" \
    "$OBJECTIVE1_VOXELS" \
    "$RANKINGS" \
    "$POLICY" \
    "$IDS_FILE" \
    "$REFERENCE"; do
    if [[ ! -e "$path" ]]; then
        echo "Required smoke-test input not found: $path" >&2
        exit 1
    fi
done

echo "Applying the frozen policy to 8 samples (one accepted and one fallback per category)"
"$PYTHON_BIN" -m final_retrieval.evaluate \
    --dataset-root "$DATASET_ROOT" \
    --objective1-voxels "$OBJECTIVE1_VOXELS" \
    --rankings "$RANKINGS" \
    --policy "$POLICY" \
    --ids-file "$IDS_FILE" \
    --output-dir "$SMOKE_OUTPUT_DIR" \
    --view-index 18 \
    --resolution 64 \
    --transplant-margin 2 \
    --margins 2 \
    --gallery-categories bus cabinet car file_cabinet \
    --gallery-per-category 0 \
    --resume

"$PYTHON_BIN" -m final_retrieval.smoke_check \
    --actual "$SMOKE_OUTPUT_DIR/per_sample.csv" \
    --reference "$REFERENCE"

echo "Smoke-test outputs: $SMOKE_OUTPUT_DIR"
