# Semantic Interior Reconstruction with TRELLIS

This repository reconstructs semantically meaningful internal 3D geometry from
a single image. The final method has two stages: interior-aware TRELLIS
fine-tuning followed by a category-blind retrieval refinement. All commands
below run from the repository root inside the TRELLIS environment.

## Layout

```text
interior_reconstruction/
├── data_preparation/
├── trellis_finetuning_interior/
└── retrieval_refinement/
notebooks/
└── visualize_results.ipynb
tests/
└── run_reproducibility_tests.sh
```

## Data preparation

The retained ShapeNet objects are converted to TRELLIS format with 40 ordinary
and conditioning views, surface voxels, DINOv2 features, sparse-structure
latents, and SLAT latents.

```bash
cd /workspace/TRELLIS

env \
  SHAPENET_ROOT=/workspace/TRELLIS/ShapeNet \
  SHAPENET_PROCESSED=/workspace/TRELLIS/datasets/ShapeNetTRELLIS_full \
  MAX_WORKERS=8 \
  bash interior_reconstruction/data_preparation/run_pipeline.sh
```

## 1. TRELLIS interior fine-tuning

### Method

LoRA fine-tuning adapts the sparse-structure flow, structured-latent flow, and
mesh decoder to internal-rich ShapeNet geometry. Inference is fixed to
conditioning view 18, seed 42, and sparse-coordinate threshold `-8`. Outputs
are voxelized at resolution 64 and measured with interior precision, recall,
F1, and F0.5.

### Implementation

- `run_flow_finetuning.sh` trains the structure-flow and SLAT-flow LoRAs.
- `run_decoder_finetuning.sh` trains the mesh-decoder LoRA.
- `generate_predictions.py` loads all three LoRAs and exports meshes and
  voxel predictions.
- `evaluate_internals.py` separates exterior and interior voxels and computes
  the final metrics.
- `run_view18_full.sh` generates and evaluates the held-out split.
- `run_view18_train.sh` generates the train prediction cache required when the
  retrieval policy is calibrated from scratch.

### Train

```bash
cd /workspace/TRELLIS

env \
  DATA_DIR=/workspace/TRELLIS/datasets/ShapeNetTRELLIS_full/train \
  OUT_DIR=/workspace/TRELLIS/results/objective1_full \
  NUM_GPUS=1 \
  bash interior_reconstruction/trellis_finetuning_interior/run_flow_finetuning.sh

env \
  DATA_DIR=/workspace/TRELLIS/datasets/ShapeNetTRELLIS_full/train \
  OUT_DIR=/workspace/TRELLIS/results/objective1_full \
  bash interior_reconstruction/trellis_finetuning_interior/run_decoder_finetuning.sh
```

Expected checkpoints:

```text
results/objective1_full/ss_flow/ckpts/denoiser_lora_final.pt
results/objective1_full/slat_flow/ckpts/denoiser_lora_final.pt
results/objective1_full/decoder/ckpts/decoder_lora_final.pt
```

### Generate and evaluate

```bash
cd /workspace/TRELLIS

time env \
  PYTHONUNBUFFERED=1 \
  OBJECTIVE1_DIR=/workspace/TRELLIS/results/objective1_full \
  OUTPUT_DIR=/workspace/TRELLIS/results/objective1_view18_full \
  bash interior_reconstruction/trellis_finetuning_interior/run_view18_full.sh
```

Set `RESUME=1` to continue an interrupted held-out run. Generate the train
prediction cache for retrieval calibration with:

```bash
time env PYTHONUNBUFFERED=1 NUM_GPUS=1 \
  bash interior_reconstruction/trellis_finetuning_interior/run_view18_train.sh
```

## 2. Retrieval refinement

### Method

DINOv2 features from view 18 retrieve the top 20 same-category training
shapes. Their interiors are aligned to the fine-tuned TRELLIS exterior and
clipped to a conservative enclosed volume. The pipeline creates a
component-hybrid hypothesis and a structural-replacement hypothesis. One
category-blind KNN selector scores all donor/operator candidates, while one
global confidence gate accepts predicted improvements and otherwise preserves
the fine-tuned result unchanged.

The donor models and selector use train-only ground truth. Held-out ground
truth is read only after selection, for reporting metrics.

### Implementation

- `dino_index.py` builds the view-18 DINOv2 retrieval index.
- `geometry.py` aligns donor geometry and defines safe transfer regions.
- `pipeline.py` creates and selects coherent refinement hypotheses.
- `calibrate.py` trains the donor models and shared selector with resumable
  per-query caches.
- `evaluate.py` applies a frozen policy and exports voxels, smooth meshes, and
  metrics.
- `run_view18.sh` is the complete retrieval entrypoint.

### Apply the frozen policy

```bash
cd /workspace/TRELLIS

time env \
  PYTHONUNBUFFERED=1 \
  POLICY=/workspace/TRELLIS/results/retrieval_unified_view18/policy.json \
  OUTPUT_DIR=/workspace/TRELLIS/results/retrieval_unified_view18 \
  bash interior_reconstruction/retrieval_refinement/run_view18.sh
```

### Calibrate from scratch

This requires the fine-tuned train and test predictions. Missing DINO caches
are built automatically.

```bash
cd /workspace/TRELLIS

time env \
  PYTHONUNBUFFERED=1 \
  UNIFIED_CALIBRATION_DIR=/workspace/TRELLIS/results/retrieval_unified_calibration_fresh \
  OUTPUT_DIR=/workspace/TRELLIS/results/retrieval_unified_fresh \
  CALIBRATION_WORKERS=8 \
  bash interior_reconstruction/retrieval_refinement/run_view18.sh
```

## Reproducibility check

The final test command validates the repository, training configs,
checkpoints, dataset, cached predictions, DINO index, frozen policy, unit
behavior, and eight fixed end-to-end retrieval cases. It does not retrain the
selector. The eight-case replay takes about one minute; complete dataset
validation can make the full command take several minutes depending on storage.

```bash
cd /workspace/TRELLIS

time env \
  PYTHONUNBUFFERED=1 \
  POLICY=/workspace/TRELLIS/results/retrieval_unified_view18/policy.json \
  bash tests/run_reproducibility_tests.sh
```

Final metrics are stored in `results/retrieval_unified_view18/*.csv`. Open
`notebooks/visualize_results.ipynb` for ground-truth, fine-tuned TRELLIS, and
retrieval-refined cutaway comparisons.
