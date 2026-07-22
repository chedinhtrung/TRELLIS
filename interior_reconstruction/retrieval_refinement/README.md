# Retrieval refinement

This package contains the complete final retrieval method. It is self-contained:
it neither imports exploratory retrieval code nor reads an earlier experimental
policy or feature table.

The pipeline has three stages:

1. `dino_index.py` extracts view-18 DINOv2 embeddings and builds a same-category
   top-20 donor index.
2. `calibrate.py` rebuilds the donor models from train-only ground truth,
   generates component and structural hypotheses, and trains one category-blind
   selector with a global fail-closed gate.
3. `evaluate.py` freezes that policy before reading held-out ground truth,
   applies it to fine-tuned TRELLIS predictions, and exports voxels, smooth
   meshes, metrics, and the visualization manifest.

The fixed operator hyperparameters are defined in `config.py`. Durable
per-query caches make interrupted calibration resumable.

Run the complete method from the repository root:

```bash
bash interior_reconstruction/retrieval_refinement/run_view18.sh
```

If `POLICY` points to an existing final `policy.json`, the runner reuses it and
performs only held-out inference. With a new calibration directory and no
policy, the same command rebuilds the DINO cache when needed and trains every
final retrieval model from the fine-tuned train predictions.

The supported inputs are the ShapeNetTRELLIS train/test split, fine-tuned
TRELLIS train/test outputs, view-18 conditioning renders, and optionally the
cached DINO index. No exploratory retrieval result is an input.
