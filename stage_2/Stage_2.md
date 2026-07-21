# Stage 2: TRELLIS LoRA Finetuning and Evaluation

## What Was Finetuned

We finetune the two image-conditioned TRELLIS flow transformers:

- `SparseStructureFlowModel` (`ss_flow`)
- `ElasticSLatFlowModel` (`slat_flow`)

Both start from `microsoft/TRELLIS-image-large` pretrained weights. The base weights are frozen; only LoRA weights are trained.

LoRA settings:

- Rank: `8`
- Alpha: `8.0`
- Dropout: `0.0`
- Target modules: Every `nn.Linear` whose module name starts with `blocks.`

So LoRA is added only inside the transformer block stack, including attention and MLP linear layers, not IO/projection/decoder code outside `blocks.*`.

## Implementation

LoRA is implemented in `trellis/modules/lora.py`:

- `LoRALinear` wraps an existing `nn.Linear`.
- The original linear weight/bias are frozen.
- Trainable matrices `lora_down` and `lora_up` add the low-rank residual.
- Sparse tensors are supported by applying LoRA to `SparseTensor.feats`.

Training integration:

- `train.py` reads an optional model config field named `lora`.
- If present, it calls `apply_lora(...)` before training.
- `trellis/trainers/basic.py` saves both normal checkpoints and small `*_lora_step*.pt` checkpoints.

Relevant files added/modified:

- Added `trellis/modules/lora.py`
- Modified `train.py`
- Modified `trellis/trainers/basic.py`
- Modified `trellis/datasets/structured_latent.py` for this dataset/config path
- Added `configs/finetune/*.json`
- Added `stage_2/*.py`, `stage_2/*.sh`

Finetune configs:

- `configs/finetune/ss_flow_img_shapenet_internals_lora.json`
- `configs/finetune/slat_flow_img_shapenet_internals_lora.json`

Stage 2 scripts:

- `stage_2/config.sh`: Shared paths/env defaults.
- `stage_2/run_lora_finetune.sh`: Trains `ss_flow`, then `slat_flow`.
- `stage_2/export_ss_flow_voxels.py`: Exports sparse-structure voxel predictions.
- `stage_2/export_full_pipeline_voxels.py`: Exports full TRELLIS mesh outputs, voxelized to PLY.
- `stage_2/compare_internals.py`: Produces the final comparison table.
- `stage_2/run_eval.sh`: Wrapper for `compare_internals.py`.

## Evaluation

Evaluation compares predicted voxel PLYs against ground-truth voxel PLYs in `datasets/ShapeNetInternals_small/voxels`.

Metrics:

- `voxel_iou`: Untersection over union of occupied voxels.
- `interior_recall`: Recall on GT voxels inside the GT bounding box after a margin crop.
- `matched_samples`: Number of matching GT/prediction filenames.

The final table compares:

- `base_ss_flow`
- `lora_ss_flow`
- `base_ss+slat_voxelized`
- `lora_ss+slat_voxelized`

Output:

```bash
results/shapenet_internals_lora/eval/comparison.csv
```

## Run End-to-End

From repo root:

```bash
cd /workspace/TRELLIS
source stage_2/config.sh
```

Train LoRA adapters:

```bash
bash stage_2/run_lora_finetune.sh
```

Export sparse-structure predictions:

```bash
python stage_2/export_ss_flow_voxels.py \
  --no-lora \
  --output-dir results/shapenet_internals_lora/predictions/base_ss_flow_voxels \
  --skip-existing

python stage_2/export_ss_flow_voxels.py \
  --output-dir results/shapenet_internals_lora/predictions/ss_flow_voxels \
  --skip-existing
```

Export full-pipeline voxelized predictions:

```bash
python stage_2/export_full_pipeline_voxels.py --mode base --skip-existing
python stage_2/export_full_pipeline_voxels.py --mode lora --skip-existing
```

Evaluate:

```bash
bash stage_2/run_eval.sh
```

## Final View-18 Coherent Retrieval

The old voxel-consensus postprocessor is retained only as an experiment. It is
not the final method: independently unioning votes from several shapes can blur
surfaces and overfill interiors.

The corrected postprocessor chooses one DINO-retrieved training shape using
view 18 plus Objective-1 exterior compatibility. Neighbor votes may accept or
reject a complete connected donor surface component, but never create voxels.
Tiny, heavily clipped, fragmented, and unsupported components are rejected.
The Objective-1 outer shell is preserved, uncertain Objective-1 internals are
replaced rather than accumulated, and a training-derived category budget guards
against overfill. All policy choices are frozen with training-only leave-one-out
calibration using a precision-weighted objective and an explicit overfill limit.

Run calibration and held-out evaluation together:

```bash
bash stage_2/run_coherent_retrieval_view18.sh
```

The command writes:

- `results/coherent_retrieval_calibration_view18/policy.json`: frozen train-only policy.
- `results/coherent_retrieval_view18/predictions/coherent_retrieval/`: all final voxel PLYs.
- `results/coherent_retrieval_view18/summary.csv`: overlap, overfill, and topology metrics.
- `results/coherent_retrieval_view18/paired_summary.csv`: paired F1 gains and confidence intervals.
- `results/coherent_retrieval_view18/references/ground_truth/`: portable voxel GT for fair visualization.
- `results/coherent_retrieval_view18/visualization_manifest.csv`: 15 best buses and 15 best cars by precision-weighted semantic improvement.

After copying both result directories back, run
`visualize_interior_comparisons.ipynb`. Ground truth and both predictions use the
same voxel-to-surface conversion and the same display-only smoothing before the
synchronized half cut.

## Retrieval v2: Reranked, Gated Hybrid Meshes

Retrieval v2 fixes the remaining all-or-nothing replacement failure. A learned
category reranker scores every DINO top-20 candidate with view-18 similarity,
tolerant exterior fit, agreement with Objective-1 internals, and neighborhood
medoid agreement. The selected donor receives only a small robust-extent affine
fit. Supported donor components replace matching Objective-1 regions, while
large surface-like Objective-1 components outside those regions are preserved.
A separately calibrated coverage and quality gate falls back to Objective 1.

The reranker and fusion policy use disjoint deterministic subsets of the 1,560
training shapes. Held-out test labels never influence selection or gating.

First cache view-18 Objective-1 predictions for the training split. This command
can shard inference across GPUs and does not retain the large train meshes:

```bash
NUM_GPUS=4 bash stage_2/run_objective1_view18_train.sh
```

Then calibrate, evaluate, and export both 64³ voxels and smooth hybrid triangle
meshes:

```bash
bash stage_2/run_retrieval_v2_view18.sh
```

Outputs are written to `results/retrieval_v2_calibration_view18` and
`results/retrieval_v2_view18`. The updated
`visualize_interior_comparisons.ipynb` shows the 15 strongest buses and 15
strongest cars as synchronized smooth half cuts; it also contains an optional
four-way voxel diagnostic for Objective 1, retrieval v1, and retrieval v2.

## Retrieval v2.1: Category-Aware Structural Transfer

Retrieval v2.1 preserves the frozen v2 car branch and corrects the geometry
assumptions that caused other categories to fall back. Bus, cabinet, and file
cabinet donors use a query-centered top-20 reranker. Shell-connected donor
geometry may be split into large supported fragments after safe clipping, but
every transferred voxel and triangle still comes from the single selected
donor. Cabinet-like categories use a side-and-vertical enclosure mask so an
open front does not erase shelves and drawers. Coverage is measured against
usable donor content rather than an often-overfilled Objective-1 interior.

All v2.1 choices are calibrated on the original train-only reranker/fusion
split. The successful car policy is copied byte-for-byte from v2. Prepared
non-car calibration queries are cached separately, so an interrupted run can
resume and later policy sweeps do not repeat alignment work. If no safe
improving structural policy exists for a category, calibration retains its
frozen v2 policy instead of aborting.

Run the category-aware calibration and held-out export with:

```bash
bash stage_2/run_retrieval_v21_view18.sh
```

Outputs are written to `results/retrieval_v21_calibration_view18` and
`results/retrieval_v21_view18`. The visualization notebook reads the v2.1
manifest and shows the strongest available bus, cabinet, car, and file-cabinet
half cuts with explicit fallback labels.

## Unified Adaptive Retrieval

The unified method removes every category-specific inference branch. For each
view-18 query it aligns the same DINO top-20 donor set. Four frozen donor
experts, a shared donor ranker, and fixed diversity anchors choose eight donors;
every expert is applied to every query, with no category branch. Both proven hypotheses are
created for each: component-preserving hybrid fusion and coherent
structural-fragment replacement. One shared category-blind selector ranks the
16 donor/operator hypotheses, and one global train-calibrated confidence gate either
accepts the best hypothesis or returns Objective 1 unchanged. Category is used
only to restrict the donor gallery to semantically compatible training shapes;
it is not a selector feature.

Calibration uses train-only ground truth. The default run makes one durable
geometry-cache pass over up to 160 selector queries per category. It excludes
the 48 labeled queries per class used to fit the donor rankers, leaving 552
disjoint selector queries with the current data (160 bus, 160 cabinet, 160 car,
and 72 file cabinets). It fits five query-level cross-validation models and
refuses to write a deployable policy unless both car and bus improve under the
same global gate without degrading precision or internal density. Set
`UNIFIED_QUERIES_PER_CATEGORY=0` to use every remaining disjoint query. The
expensive cache survives interruption and a failed preflight.

Run training, held-out export, and smooth visualization preparation with:

```bash
bash stage_2/run_retrieval_unified_view18.sh
```

Outputs are written to `results/retrieval_unified_calibration_view18` and
`results/retrieval_unified_view18`. The visualization notebook defaults to the
unified manifest and renders ground truth, Objective 1, and unified retrieval
with synchronized smooth half cuts.
