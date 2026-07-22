#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DATA_DIR="${DATA_DIR:-$REPO_ROOT/datasets/ShapeNetTRELLIS_full/train}"
VAL_DIR="${VAL_DIR:-$REPO_ROOT/datasets/ShapeNetTRELLIS_full/val}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/results/objective1_full}"
NUM_GPUS="${NUM_GPUS:-1}"

# Avoid xFormers kernels during interior fine-tuning by default. PyTorch SDPA is
# much less sensitive to third-party wheel/GPU architecture mismatches.
export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"
export SPARSE_ATTN_BACKEND="${SPARSE_ATTN_BACKEND:-sdpa}"

# The SLAT flow uses spconv sparse 3D convolutions. On the current Blackwell/CUDA
# stack, spconv's auto-selected implicit_gemm path can crash with SIGFPE.
export SPCONV_ALGO="${SPCONV_ALGO:-native}"

# DINOv2 auto-enables xFormers when it is importable. Some prebuilt xFormers
# wheels select CUDA kernels that are not runnable on newer/older GPUs, causing
# "no kernel image is available for execution on the device" before training
# reaches TRELLIS proper. Disable only DINOv2's optional xFormers fast path by
# default; export TRELLIS_USE_DINOV2_XFORMERS=1 to opt back in.
if [[ "${TRELLIS_USE_DINOV2_XFORMERS:-0}" != "1" ]]; then
    export XFORMERS_DISABLED="${XFORMERS_DISABLED:-1}"
fi
