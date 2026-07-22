#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/datasets/ShapeNetTRELLIS_full}"
OBJECTIVE1_DIR="${OBJECTIVE1_DIR:-$REPO_ROOT/results/objective1_full}"
OBJECTIVE1_TRAIN_DIR="${OBJECTIVE1_TRAIN_DIR:-$REPO_ROOT/results/objective1_view18_train}"
OBJECTIVE1_TEST_DIR="${OBJECTIVE1_TEST_DIR:-$REPO_ROOT/results/objective1_view18_full}"
DINO_DIR="${DINO_DIR:-$REPO_ROOT/results/dino_retrieval_view18}"
FROZEN_RESULTS_DIR="${FROZEN_RESULTS_DIR:-$REPO_ROOT/results/retrieval_unified_view18}"
POLICY="${POLICY:-$FROZEN_RESULTS_DIR/policy.json}"
FULL_DATASET_VALIDATION="${FULL_DATASET_VALIDATION:-1}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"

FIXTURE_DIR="$SCRIPT_DIR/fixtures/retrieval_refinement"
CREATED_TEMP_OUTPUT=0
if [[ -n "${TEST_OUTPUT_DIR:-}" ]]; then
    REGRESSION_OUTPUT_DIR="$TEST_OUTPUT_DIR"
else
    REGRESSION_OUTPUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/trellis-retrieval-regression.XXXXXX")"
    CREATED_TEMP_OUTPUT=1
fi

cleanup() {
    status=$?
    if [[ "$CREATED_TEMP_OUTPUT" == "1" && "$status" == "0" ]]; then
        rm -rf -- "$REGRESSION_OUTPUT_DIR"
    elif [[ "$status" != "0" ]]; then
        echo "Regression outputs retained for debugging: $REGRESSION_OUTPUT_DIR" >&2
    fi
}
trap cleanup EXIT

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

echo "[1/7] Checking Python and retrieval dependencies"
"$PYTHON_BIN" - <<'PY'
import importlib
import sys

required = (
    "numpy",
    "open3d",
    "pandas",
    "PIL",
    "plyfile",
    "scipy",
    "torch",
    "tqdm",
    "trimesh",
    "utils3d",
)
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {exc}")
if missing:
    raise SystemExit("Missing required Python dependencies:\n  " + "\n  ".join(missing))

import torch
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
PY

if [[ "$REQUIRE_CUDA" == "1" ]]; then
    "$PYTHON_BIN" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else "CUDA is required for full pipeline reproduction")'
elif [[ "$REQUIRE_CUDA" != "0" ]]; then
    echo "REQUIRE_CUDA must be 0 or 1" >&2
    exit 1
fi

echo "[2/7] Checking shell syntax"
while IFS= read -r -d '' shell_file; do
    bash -n "$shell_file"
done < <(find interior_reconstruction tests -type f -name '*.sh' -print0)
bash -n dataset_toolkits/prepare_shapenet.sh

echo "[3/7] Running dependency-free repository tests"
"$PYTHON_BIN" -m unittest -v tests.test_repository_structure

echo "[4/7] Running fine-tuning and retrieval unit tests"
"$PYTHON_BIN" -m unittest -v \
    tests.test_finetuning_interior \
    tests.test_retrieval_refinement

if [[ "$FULL_DATASET_VALIDATION" == "1" ]]; then
    echo "[5/7] Validating the complete TRELLIS-format dataset"
    "$PYTHON_BIN" \
        interior_reconstruction/trellis_finetuning_interior/validate_dataset.py \
        --dataset-root "$DATASET_ROOT"
elif [[ "$FULL_DATASET_VALIDATION" == "0" ]]; then
    echo "[5/7] Skipping full dataset validation by request"
else
    echo "FULL_DATASET_VALIDATION must be 0 or 1" >&2
    exit 1
fi

echo "[6/7] Checking checkpoints, prediction caches, DINO index, and policy"
"$PYTHON_BIN" tests/check_reproduction_inputs.py \
    --dataset-root "$DATASET_ROOT" \
    --checkpoint-root "$OBJECTIVE1_DIR" \
    --train-output "$OBJECTIVE1_TRAIN_DIR" \
    --test-output "$OBJECTIVE1_TEST_DIR" \
    --dino-dir "$DINO_DIR" \
    --policy "$POLICY" \
    --fixture-ids "$FIXTURE_DIR/ids.txt" \
    --fixture-expected "$FIXTURE_DIR/expected.csv"

echo "[7/7] Replaying eight frozen-policy retrieval cases"
"$PYTHON_BIN" -m interior_reconstruction.retrieval_refinement.evaluate \
    --dataset-root "$DATASET_ROOT" \
    --objective1-voxels \
        "$OBJECTIVE1_TEST_DIR/predictions/objective1/seed_42/voxels" \
    --rankings "$DINO_DIR/rankings.csv" \
    --policy "$POLICY" \
    --ids-file "$FIXTURE_DIR/ids.txt" \
    --output-dir "$REGRESSION_OUTPUT_DIR" \
    --view-index 18 \
    --resolution 64 \
    --transplant-margin 2 \
    --margins 2 \
    --gallery-categories bus cabinet car file_cabinet \
    --gallery-per-category 0

"$PYTHON_BIN" tests/verify_retrieval_regression.py \
    --actual "$REGRESSION_OUTPUT_DIR/per_sample.csv" \
    --expected "$FIXTURE_DIR/expected.csv"

echo
echo "PASS: all final reproducibility checks completed successfully"
