#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_DIR="$REPO_ROOT/datasets/ShapeNetInternals_small"
NUM_GPUS=1
STAGE2_OUT_ROOT="$REPO_ROOT/results/shapenet_internals_lora"
STAGE3_OUT_ROOT="$REPO_ROOT/results/stage_3_weighted_slat"
WEIGHTED_SCALAR=1.8 # invisible voxels get 1.8 times more loss

export ATTN_BACKEND=sdpa
export SPARSE_ATTN_BACKEND=sdpa
export SPCONV_ALGO=native

GT_VOXELS="$DATA_DIR/voxels"