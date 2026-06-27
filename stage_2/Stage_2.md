# Stage 2: Full ShapeNetInternals Dataset Generation

## Goal

Stage 2 turns the ShapeNet internal-rich source data into a full TRELLIS-compatible training dataset.

The source data is:

```text
/workspace/TRELLIS/ShapeNet
```

The target dataset is:

```text
/workspace/TRELLIS/datasets/ShapeNetInternals
```

The purpose is not LoRA training yet. Stage 2 prepares the full training data needed for the next stage by coordinating TRELLIS' existing preprocessing tools across every valid ShapeNet object. The full dataset should eventually contain metadata, splits, conditioning renders, multi-view renders, sparse voxel targets, DINO feature caches, sparse-structure latents, and SLAT latents.

The guiding rule for Stage 2 was:

```text
reuse TRELLIS preprocessing, add only ShapeNet orchestration
```

No TRELLIS core preprocessing script or model code was modified.

## Dataset Generation Pipeline

The Stage 2 pipeline converts raw ShapeNet folders into TRELLIS training artifacts:

```text
/workspace/TRELLIS/ShapeNet
        |
        v
metadata.csv + train/val/test split
        |
        +--> model_normalized.surface.binvox
        |       -> OR downsample to 64^3
        |       -> voxels/<id>.ply
        |
        +--> model_normalized.obj
        |       -> dataset_toolkits/render_cond.py
        |       -> renders_cond/<id>/transforms.json + RGBA PNGs
        |
        +--> model_normalized.obj
                -> dataset_toolkits/render.py
                -> renders/<id>/transforms.json + RGBA PNGs + mesh.ply
                        |
                        v
                dataset_toolkits/extract_feature.py
                        |
                        v
                features/dinov2_vitl14_reg/<id>.npz
                        |
                        v
                dataset_toolkits/encode_latent.py
                        |
                        v
                latents/dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16/<id>.npz

voxels/<id>.ply
        |
        v
dataset_toolkits/encode_ss_latent.py
        |
        v
ss_latents/ss_enc_conv3d_16l8_fp16/<id>.npz
```

Each stage is restartable. The orchestration script refreshes artifact flags in `metadata.csv` and skips stages that are already complete unless explicitly told to rerun them.

The object id format is:

```text
<category>__<shapenet_object_id>
```

Example:

```text
bus__2f9aecaf8c5d8dffc90887194d8d151d
```

This avoids collisions between categories and keeps the original ShapeNet id visible.

## Stage 2 Scripts

All new Stage 2 code lives under:

```text
/workspace/TRELLIS/stage_2/
```

### `stage_2/run_full_dataset_pipeline.py`

This is the main orchestration script. It coordinates the full preprocessing pipeline without duplicating TRELLIS' internal rendering, feature extraction, or latent encoding logic.

Its responsibilities are:

- scan `/workspace/TRELLIS/ShapeNet`
- build `metadata.csv`
- exclude invalid objects with missing required files
- create deterministic train/val/test splits
- convert `model_normalized.surface.binvox` into 64^3 sparse voxel PLY files
- call `dataset_toolkits/render_cond.py`
- call `dataset_toolkits/render.py`
- call `dataset_toolkits/extract_feature.py`
- call `dataset_toolkits/encode_ss_latent.py`
- call `dataset_toolkits/encode_latent.py`
- refresh artifact flags in `metadata.csv`
- write logs
- write failure tables
- update `stage_2/dataset_report.md`

Important inputs:

```text
/workspace/TRELLIS/ShapeNet
/workspace/TRELLIS/dataset_toolkits/render_cond.py
/workspace/TRELLIS/dataset_toolkits/render.py
/workspace/TRELLIS/dataset_toolkits/extract_feature.py
/workspace/TRELLIS/dataset_toolkits/encode_ss_latent.py
/workspace/TRELLIS/dataset_toolkits/encode_latent.py
```

Important outputs:

```text
/workspace/TRELLIS/datasets/ShapeNetInternals/metadata.csv
/workspace/TRELLIS/datasets/ShapeNetInternals/splits/
/workspace/TRELLIS/datasets/ShapeNetInternals/voxels/
/workspace/TRELLIS/datasets/ShapeNetInternals/renders_cond/
/workspace/TRELLIS/datasets/ShapeNetInternals/renders/
/workspace/TRELLIS/datasets/ShapeNetInternals/features/
/workspace/TRELLIS/datasets/ShapeNetInternals/ss_latents/
/workspace/TRELLIS/datasets/ShapeNetInternals/latents/
/workspace/TRELLIS/stage_2/dataset_report.md
```

The script supports resuming after interruption. Existing files are detected through artifact scans, and TRELLIS' own preprocessing scripts already skip completed outputs based on metadata and file existence.

Useful options include:

- `--render-workers`
- `--cond-render-workers`
- `--voxel-workers`
- `--feature-batch-size`
- `--overwrite-metadata`
- `--overwrite-voxels`
- `--rerun-completed-stages`
- `--skip-render-cond`
- `--skip-render`
- `--skip-features`
- `--skip-ss-latents`
- `--skip-slat-latents`
- `--report-only`
- `--dry-run`

### `stage_2/datasets/ShapeNetInternals.py`

This is a small dataset adapter used by TRELLIS' render scripts.

TRELLIS invokes render scripts like this:

```bash
dataset_toolkits/render.py ShapeNetInternals ...
dataset_toolkits/render_cond.py ShapeNetInternals ...
```

Those scripts import `datasets.ShapeNetInternals`. The adapter provides the functions TRELLIS expects:

- `add_args(parser)`
- `get_metadata(output_dir, **kwargs)`
- `foreach_instance(metadata, output_dir, func, max_workers, desc)`

It does not implement rendering. It only maps `metadata.csv` rows to ShapeNet OBJ paths and lets TRELLIS' existing Blender-based renderer do the actual work.

Input:

```text
/workspace/TRELLIS/datasets/ShapeNetInternals/metadata.csv
```

Output:

```text
records returned to dataset_toolkits/render.py
records returned to dataset_toolkits/render_cond.py
```

### `stage_2/dataset_report.md`

This is the generated Stage 2 status report. It is updated by `run_full_dataset_pipeline.py`.

It records:

- number of metadata objects
- number of fully successful objects
- number of failed or incomplete objects
- artifact counts
- split counts
- category counts
- voxel statistics
- output folder sizes
- failure reasons
- commands for full generation and resume

## Generated Dataset

The target dataset is:

```text
/workspace/TRELLIS/datasets/ShapeNetInternals
```

The expected final folder structure is:

```text
datasets/ShapeNetInternals/
  metadata.csv
  all_ids.txt
  invalid_objects.csv
  failures_stage2.csv
  binvox_surface_to_voxels.csv
  logs/
    stage2_pipeline.log
  splits/
    train.txt
    val.txt
    test.txt
  voxels/
    <id>.ply
  renders_cond/
    <id>/
      transforms.json
      000.png
      ...
  renders/
    <id>/
      transforms.json
      000.png
      ...
      mesh.ply
  features/
    dinov2_vitl14_reg/
      <id>.npz
  ss_latents/
    ss_enc_conv3d_16l8_fp16/
      <id>.npz
  latents/
    dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16/
      <id>.npz
```

Current metadata statistics from the Stage 2 preflight:

- valid metadata rows: `1949`
- excluded invalid source objects: `1`
- excluded object: `bus/a3841f9c67d82a24193d7855ecfc1bd3`
- reason: missing `models/model_normalized.surface.binvox`
- split counts:
  - `train`: `1753`
  - `val`: `98`
  - `test`: `98`
- category counts:
  - `bus`: `299`
  - `cabinet`: `500`
  - `cars`: `1000`
  - `file_cabinet`: `150`

At the time this document was written, `metadata.csv`, splits, logs, and the full `voxels/` folder existed on disk. A small number of `renders_cond/` outputs were also present from an interrupted or partial render run. The remaining heavy artifacts, especially full renders, DINO features, sparse-structure latents, and SLAT latents, are generated by running the full command below and can be resumed safely.

## Design Decisions

Stage 2 deliberately reuses TRELLIS' preprocessing scripts instead of reimplementing them.

This matters because TRELLIS' data format is not just a set of filenames. The render scripts generate camera metadata in the expected `transforms.json` format; feature extraction projects DINO tokens onto sparse voxel positions; latent encoders save tensors using the exact folder names expected by training configs. Reusing those scripts keeps the generated dataset compatible with the repository's training code.

We wrapped existing scripts rather than modifying them for three reasons:

1. It keeps TRELLIS core behavior unchanged.
2. It makes this ShapeNet work easier to inspect and remove.
3. It reduces the risk of accidentally changing Objaverse or other dataset behavior.

The only custom logic in Stage 2 is ShapeNet-specific:

- scanning category/object folders
- building metadata
- writing deterministic splits
- converting ShapeNet `surface.binvox` to TRELLIS 64^3 sparse voxel PLY
- coordinating existing TRELLIS scripts
- summarizing failures and progress

All new code was placed under `stage_2/` so the project history is clear:

```text
stage_1/  feasibility and reconstruction experiments
stage_2/  full dataset generation orchestration
```

## Reproducibility

Run from the repository root:

```bash
cd /workspace/TRELLIS
```

Full dataset generation command:

```bash
/workspace/venv/bin/python stage_2/run_full_dataset_pipeline.py \
  --output-dir /workspace/TRELLIS/datasets/ShapeNetInternals \
  --render-workers 4 \
  --cond-render-workers 4 \
  --voxel-workers 8 \
  --feature-batch-size 16
```

The same command can be rerun after interruption. The pipeline refreshes `metadata.csv`, detects completed outputs, and resumes missing stages.

Useful report-only command:

```bash
/workspace/venv/bin/python stage_2/run_full_dataset_pipeline.py \
  --output-dir /workspace/TRELLIS/datasets/ShapeNetInternals \
  --report-only
```

Useful dry-run command:

```bash
/workspace/venv/bin/python stage_2/run_full_dataset_pipeline.py \
  --output-dir /workspace/TRELLIS/datasets/ShapeNetInternals \
  --dry-run
```

Expected completed artifact counts after a successful full run:

```text
metadata rows: 1949
voxels: 1949 / 1949
conditioning renders: 1949 / 1949
multi-view renders: 1949 / 1949
DINO features: 1949 / 1949
sparse-structure latents: 1949 / 1949
SLAT latents: 1949 / 1949
```

After the run, check:

```text
/workspace/TRELLIS/stage_2/dataset_report.md
/workspace/TRELLIS/datasets/ShapeNetInternals/failures_stage2.csv
/workspace/TRELLIS/datasets/ShapeNetInternals/logs/stage2_pipeline.log
```

## Next Stage

Stage 3 should begin LoRA fine-tuning, but it should do so in stages rather than adapting every component at once.

Recommended order:

```text
1. Add LoRA adapters to SparseStructureFlowModel
2. Fine-tune structure flow only
3. Validate generated sparse structures and internal voxel cross-sections
4. Add LoRA adapters to ElasticSLatFlowModel
5. Fine-tune SLAT flow after structure generation is reliable
```

The reason for this order is that TRELLIS generation is staged. The sparse-structure flow first predicts where occupied structure exists. The SLAT flow then predicts latent features on that structure. If the generated sparse structure does not contain internal partitions, shelves, seats, or cavities, the SLAT stage cannot reliably invent them later.

Fine-tuning `SparseStructureFlowModel` first isolates the most important question:

```text
Can the image-conditioned structure prior learn ShapeNet internal-rich occupancy?
```

Only after that works should we fine-tune `ElasticSLatFlowModel`, which controls the richer latent features and final geometry details. This avoids confounding two failure modes at once and keeps validation simple.

Initial Stage 3 assumptions:

- keep DINO frozen
- keep the pretrained VAE/decoder frozen initially
- train only LoRA adapter parameters
- validate exterior and internal geometry separately
- use raw decoder meshes for internal checks, not only postprocessed GLB output
