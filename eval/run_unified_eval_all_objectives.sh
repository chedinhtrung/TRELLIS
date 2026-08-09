#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/workspace/TRELLIS"
EVAL_SCRIPT="$REPO_ROOT/eval/run_unified_eval.py"
DATASET_TEST="$REPO_ROOT/datasets/ShapeNetTRELLIS_full/test"

mkdir -p \
  "$REPO_ROOT/results/baseline/eval" \
  "$REPO_ROOT/results/objective1_full/eval" \
  "$REPO_ROOT/results/objective2_category/eval" \
  "$REPO_ROOT/results/objective3_full/eval"

# Baseline: pure pretrained pipeline (no LoRA checkpoints).
python "$EVAL_SCRIPT" \
  --dataset-dir "$DATASET_TEST" \
  --output-dir "$REPO_ROOT/results/baseline/eval" \
  --view-index 18 \
  --resolution 64 \
  --outer-layers 2 \
  --seed 42 \
  2>&1 | tee "$REPO_ROOT/results/baseline/eval/unified_eval.log" &
PID_BASELINE=$!

# Objective 1: uses its own ss/slat/decoder LoRA checkpoints.
python "$EVAL_SCRIPT" \
  --dataset-dir "$DATASET_TEST" \
  --output-dir "$REPO_ROOT/results/objective1_full/eval" \
  --ss-lora-ckpt "$REPO_ROOT/results/objective1_full/ss_flow/ckpts/denoiser_lora_final.pt" \
  --slat-lora-ckpt "$REPO_ROOT/results/objective1_full/slat_flow/ckpts/denoiser_lora_final.pt" \
  --decoder-lora-ckpt "$REPO_ROOT/results/objective1_full/decoder/ckpts/decoder_lora_final.pt" \
  --view-index 18 \
  --resolution 64 \
  --outer-layers 2 \
  --seed 42 \
  2>&1 | tee "$REPO_ROOT/results/objective1_full/eval/unified_eval.log" &
PID_OBJ1=$!

wait "$PID_BASELINE"
wait "$PID_OBJ1"

# Objective 2: no decoder training; reuse objective1 decoder LoRA checkpoint.
python "$EVAL_SCRIPT" \
  --dataset-dir "$DATASET_TEST" \
  --output-dir "$REPO_ROOT/results/objective2_category/eval" \
  --ss-lora-ckpt "$REPO_ROOT/results/objective2_category/ss_flow/ckpts/denoiser_lora_final.pt" \
  --slat-lora-ckpt "$REPO_ROOT/results/objective2_category/slat_flow/ckpts/denoiser_lora_final.pt" \
  --decoder-lora-ckpt "$REPO_ROOT/results/objective1_full/decoder/ckpts/decoder_lora_final.pt" \
  --view-index 18 \
  --resolution 64 \
  --outer-layers 2 \
  --seed 42 \
  2>&1 | tee "$REPO_ROOT/results/objective2_category/eval/unified_eval.log" &
PID_OBJ2=$!

# Objective 3: only SLAT flow was retrained; reuse objective1 ss and decoder LoRA checkpoints.
python "$EVAL_SCRIPT" \
  --dataset-dir "$DATASET_TEST" \
  --output-dir "$REPO_ROOT/results/objective3_full/eval" \
  --ss-lora-ckpt "$REPO_ROOT/results/objective1_full/ss_flow/ckpts/denoiser_lora_final.pt" \
  --slat-lora-ckpt "$REPO_ROOT/results/objective3_full/slat_flow/ckpts/denoiser_lora_final.pt" \
  --decoder-lora-ckpt "$REPO_ROOT/results/objective1_full/decoder/ckpts/decoder_lora_final.pt" \
  --view-index 18 \
  --resolution 64 \
  --outer-layers 2 \
  --seed 42 \
  2>&1 | tee "$REPO_ROOT/results/objective3_full/eval/unified_eval.log" &
PID_OBJ3=$!

wait "$PID_OBJ2"
wait "$PID_OBJ3"
