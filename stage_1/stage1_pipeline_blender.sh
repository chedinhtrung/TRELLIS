#!/usr/bin/env bash
set -euo pipefail
trap 'status=$?; echo "[stage1] Pipeline failed with exit code $status"; read -r -p "Press Enter to exit..."' ERR

REPO_ROOT="/workspace/TRELLIS"
SHAPENET_PROCESSED="$REPO_ROOT/datasets/ShapeNetTRELLIS_blender_10views/train"
export SPCONV_ALGO="${SPCONV_ALGO:-native}"

echo "[stage1] Running reconstruction evaluation"
# run reconstruction evaluation of the sparse structure VAE
cd "$REPO_ROOT/stage_1"

python run_mesh_reconstruction_eval.py --dataset-dir "$SHAPENET_PROCESSED" \
    --output-dir "$REPO_ROOT/results/reconstruction_eval_blender_10views" \

python run_reconstruction_eval.py --dataset-dir "$SHAPENET_PROCESSED" \
    --output-dir "$REPO_ROOT/results/reconstruction_eval_blender_10views" \




