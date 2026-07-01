#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/stage_2/config.sh"

cd "$REPO_ROOT"

if [ ! -d "$PRED_ROOT" ]; then
    echo "Prediction root not found: $PRED_ROOT"
    echo "Edit stage_2/config.sh or run with PRED_ROOT=/path/to/predictions"
    exit 1
fi

python stage_2/compare_internals.py \
    --gt-voxels "$GT_VOXELS" \
    --pred-root "$PRED_ROOT" \
    --output "$OUT_CSV"
