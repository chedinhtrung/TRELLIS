# Final retrieval refinement (view 18)

This package is the self-contained final Stage 2 method. It does not import an
exploratory retrieval module and does not read a v2 or v2.1 policy or feature
table.

The pipeline has three explicit stages:

1. `dino_index.py` extracts view-18 DINOv2 embeddings and builds a
   same-category top-20 donor index.
2. `calibrate.py` rebuilds the donor rankers from train-only ground truth,
   generates the component and structural hypotheses, and trains one
   category-blind selector with a global fail-closed gate.
3. `evaluate.py` freezes that policy before reading held-out ground truth,
   applies it to Objective-1 predictions, and exports voxels, smooth meshes,
   metrics, and the visualization manifest.

The fixed operator hyperparameters are in `config.py`. They are now part of
the final method definition rather than being loaded from earlier experiment
artifacts. Durable per-query caches make interrupted calibration resumable.

Run the public entrypoint from the repository root:

```bash
bash stage_2/run_retrieval_unified_view18.sh
```

If a frozen final `policy.json` exists, the runner reuses it and only performs
held-out inference. To prove complete reproducibility from raw final inputs,
set `UNIFIED_CALIBRATION_DIR` and `OUTPUT_DIR` to new empty directories; the
same command then rebuilds DINO data if needed and trains every final model.

The supported inputs are the ShapeNetTRELLIS train/test split, Objective-1
train/test outputs, view-18 conditioning renders, and (optionally) the cached
DINO index. No exploratory retrieval result is an input.

