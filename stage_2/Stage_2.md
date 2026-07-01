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
