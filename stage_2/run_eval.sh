#!/usr/bin/env bash
set -euo pipefail
trap 'status=$?; echo "[stage2] Evaluation pipeline failed with exit code $status"; read -r -p "Press Enter to exit..."' ERR

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/stage_2/config.sh"

cd "$REPO_ROOT"

# Default LoRA checkpoints used for export when not provided via env vars.
SS_LORA_CKPT="${SS_LORA_CKPT:-$OUT_DIR/ss_flow/ckpts/denoiser_lora_step0002000.pt}"
SLAT_LORA_CKPT="${SLAT_LORA_CKPT:-$OUT_DIR/slat_flow/ckpts/denoiser_lora_step0002000.pt}"

# 1) Sparse-structure flow export: baseline and LoRA
python stage_2/export_ss_flow_voxels.py \
    --dataset-dir "$DATA_DIR" \
    --output-dir "$PRED_ROOT/base_ss_flow_voxels" \
    --skip-existing

python stage_2/export_ss_flow_voxels.py \
    --dataset-dir "$DATA_DIR" \
    --output-dir "$PRED_ROOT/ss_flow_voxels" \
    --lora-ckpt "$SS_LORA_CKPT" \
    --skip-existing

# 2) Full pipeline export: baseline and LoRA (ss + slat)
python stage_2/export_full_pipeline_voxels.py \
    --dataset-dir "$DATA_DIR" \
    --output-dir "$PRED_ROOT/base_ss_slat_voxelized" \
    --skip-existing

python stage_2/export_full_pipeline_voxels.py \
    --dataset-dir "$DATA_DIR" \
    --output-dir "$PRED_ROOT/lora_ss_slat_voxelized" \
    --ss-lora-ckpt "$SS_LORA_CKPT" \
    --slat-lora-ckpt "$SLAT_LORA_CKPT" \
    --skip-existing

if [ ! -d "$PRED_ROOT" ]; then
    echo "Prediction root not found: $PRED_ROOT"
    echo "Edit stage_2/config.sh or run with PRED_ROOT=/path/to/predictions"
    exit 1
fi

python stage_2/compare_internals.py \
    --gt-voxels "$GT_VOXELS" \
    --pred-root "$PRED_ROOT" \
    --output "$OUT_CSV"
