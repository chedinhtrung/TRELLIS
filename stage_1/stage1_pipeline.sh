#!/usr/bin/env bash
set -euo pipefail
trap 'status=$?; echo "[stage1] Pipeline failed with exit code $status"; read -r -p "Press Enter to exit..."' ERR

REPO_ROOT="/workspace/TRELLIS"
SHAPENET_PROCESSED="$REPO_ROOT/datasets/ShapeNetTRELLIS_nano/train"
export SPCONV_ALGO="${SPCONV_ALGO:-native}"

echo "[stage1] Running reconstruction evaluation"
# run reconstruction evaluation of the sparse structure VAE
cd "$REPO_ROOT/stage_1"
#python run_reconstruction_eval.py --dataset-dir "$SHAPENET_PROCESSED"

python run_mesh_reconstruction_eval.py --dataset-dir "$SHAPENET_PROCESSED" \
    --output-dir "$REPO_ROOT/results/mesh_reconstruction_eval" \
    #--decoder-lora-dir "$REPO_ROOT/results/shapenet_internals_lora/slat_vae" 


