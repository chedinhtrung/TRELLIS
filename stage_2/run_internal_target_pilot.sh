#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/datasets/ShapeNetTRELLIS_full/train}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/results/geometry_expert/pilot_targets}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ROUTE_MARGIN="${ROUTE_MARGIN:-2}"
DINO_BATCH_SIZE="${DINO_BATCH_SIZE:-4}"
OVERRIDE_ARGS=()
if [[ "${OVERRIDE:-0}" == "1" ]]; then
    OVERRIDE_ARGS=(--override)
fi

export SPCONV_ALGO="${SPCONV_ALGO:-native}"
export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"

mkdir -p "$RUN_DIR"
cd "$REPO_ROOT"

echo "[1/5] Selecting 20 training objects"
"$PYTHON_BIN" stage_2/select_internal_target_pilot.py \
    --metadata "$DATA_DIR/metadata.csv" \
    --output "$RUN_DIR/pilot_ids.txt" \
    --per-category 5

echo "[2/5] Rendering 18 internal cross-sections per object"
"$PYTHON_BIN" dataset_toolkits/render_internal_kiui.py \
    --data_dir "$DATA_DIR" \
    --instances "$RUN_DIR/pilot_ids.txt" \
    "${OVERRIDE_ARGS[@]}"

echo "[3/5] Extracting visibility-filtered DINO features"
"$PYTHON_BIN" dataset_toolkits/extract_internal_feature.py \
    --data_dir "$DATA_DIR" \
    --instances "$RUN_DIR/pilot_ids.txt" \
    --batch_size "$DINO_BATCH_SIZE" \
    --depth_tolerance_voxels 1.5 \
    --diagnostics_dir "$RUN_DIR/projection_overlays" \
    "${OVERRIDE_ARGS[@]}"

echo "[4/5] Building stitched SLAT targets"
"$PYTHON_BIN" dataset_toolkits/build_internal_slat_targets.py \
    --data_dir "$DATA_DIR" \
    --instances "$RUN_DIR/pilot_ids.txt" \
    --route_margin "$ROUTE_MARGIN" \
    "${OVERRIDE_ARGS[@]}"

echo "[5/5] Validating invariants and coverage"
"$PYTHON_BIN" stage_2/validate_internal_targets.py \
    --data_dir "$DATA_DIR" \
    --instances "$RUN_DIR/pilot_ids.txt" \
    --output_dir "$RUN_DIR/validation" \
    --route_margin "$ROUTE_MARGIN" \
    --min_coverage 0.90 \
    --min_two_view_coverage 0.70 \
    --min_category_coverage 0.80

echo "Pilot passed. Read $RUN_DIR/validation/summary.json"
