# Stage 1: ShapeNet -> TRELLIS Data Understanding and Reconstruction Feasibility

## 1. Goal

The original research question was:

```text
Can TRELLIS be adapted to generate internal geometry by LoRA fine-tuning on ShapeNet-style data?
```

TRELLIS was trained mostly on Objaverse assets. Those assets often behave like exterior-only objects: they contain visible outer surfaces, but hidden shelves, walls, partitions, seats, vehicle interiors, or cabinet internals are not consistently represented or supervised. This matters because LoRA fine-tuning the TRELLIS flow models can only adapt the learned generators. It cannot recover internal geometry if TRELLIS' existing representation, latent encoders, or decoders discard that information.

The key feasibility question before LoRA was therefore:

```text
Can the existing TRELLIS latent representations and pretrained decoders preserve ShapeNet internal geometry at all?
```

If internals disappeared during encoding or decoding, LoRA on `SparseStructureFlowModel` or `ElasticSLatFlowModel` alone would not be enough. We would first need decoder/VAE fine-tuning with internal-aware supervision. Stage 1 was designed to answer that representation question before starting any LoRA work.

## 2. ShapeNet Investigation

The downloaded data lives under:

```text
/workspace/TRELLIS/ShapeNet
```

The source dataset contains 1,950 objects across four categories:

- `bus`: 300 objects
- `cabinet`: 500 objects
- `cars`: 1000 objects
- `file_cabinet`: 150 objects

Each object is stored in a category/object id folder:

```text
ShapeNet/<category>/<object_id>/
ShapeNet/bus/101fe6e34502a89952470de2774d6099/
```

Important files inside an object folder include:

```text
models/model_normalized.obj
models/model_normalized.mtl
models/model_normalized.surface.binvox
models/model_normalized.solid.binvox
images/texture0.jpg
```

The `.obj` and `.mtl` files define the normalized mesh and its materials. The `images/` folder contains material texture maps referenced by the `.mtl`; these are not external conditioning renders because they do not depict the object from a camera viewpoint and have no `transforms.json` camera metadata.

The two voxel files have different meanings:

- `model_normalized.surface.binvox` marks surface voxels, including internal surfaces when present. This is the correct target for internal-rich geometry experiments.
- `model_normalized.solid.binvox` marks filled volume. It can be useful as an ablation, but it is not equivalent to internal surface geometry.

The dataset initially had no TRELLIS-ready metadata or render structure:

```text
no metadata.csv
no train/val/test split
no renders/<id>/transforms.json
no renders_cond/<id>/transforms.json
no cached DINO features
no sparse-structure latents
no SLAT latents
```

The category score CSV files, for example `ShapeNet/cabinet_center_box_scores.csv`, include inner-face and inner-edge statistics. These statistics indicated that the selected ShapeNet subset is internal-rich enough to be useful for the LoRA feasibility study.

## 3. ShapeNet -> TRELLIS Conversion

TRELLIS training does not consume raw ShapeNet object folders directly. It expects a standardized preprocessed dataset directory.

The target layout is:

```text
datasets/<dataset_name>/
  metadata.csv
  splits/
    train.txt
    val.txt
    test.txt
  renders/<id>/
    transforms.json
    000.png
    ...
  renders_cond/<id>/
    transforms.json
    000.png
    ...
  voxels/<id>.ply
  features/dinov2_vitl14_reg/<id>.npz
  ss_latents/ss_enc_conv3d_16l8_fp16/<id>.npz
  latents/dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16/<id>.npz
```

Each component has a specific role:

- `metadata.csv` records sample ids, source paths, category labels, split labels, and artifact availability flags.
- `renders_cond/` contains external RGBA conditioning views used by image-conditioned training.
- `renders/` contains multi-view RGBA renders used by TRELLIS feature extraction and decoder preprocessing.
- `voxels/` contains 64^3 sparse voxel PLY files used as sparse-structure targets.
- `features/` contains DINOv2 features projected and averaged onto sparse voxel coordinates.
- `ss_latents/` contains sparse-structure VAE latent targets.
- `latents/` contains SLAT latent targets for the second-stage latent flow.

The complete preprocessing pipeline is:

```text
ShapeNet category/object folders
        |
        v
metadata.csv + splits
        |
        +--> model_normalized.surface.binvox
        |       -> OR downsample 128^3 to 64^3
        |       -> voxels/<id>.ply
        |
        +--> model_normalized.obj
        |       -> dataset_toolkits/render_cond.py
        |       -> renders_cond/<id>/*.png + transforms.json
        |
        +--> model_normalized.obj
                -> dataset_toolkits/render.py
                -> renders/<id>/*.png + transforms.json + mesh.ply
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

The important implementation choice was to reuse TRELLIS' existing preprocessing scripts whenever possible. The only custom conversion logic needed was adapting the ShapeNet folder layout and converting `surface.binvox` into TRELLIS-compatible sparse voxel PLY files.

## 4. Stage 1 Implementation

All Stage 1 code was placed under:

```text
/workspace/TRELLIS/stage_1/
```

No TRELLIS core model code was modified.

### `stage_1/common.py`

This file holds shared constants and utility functions used by the Stage 1 scripts. It defines the default ShapeNet root, the small converted dataset path, model names used by TRELLIS artifact folders, and helpers for metadata IO, split writing, binvox decoding, OR downsampling, sparse voxel PLY IO, and subprocess execution.

It is not a training component. Its purpose is to keep the Stage 1 wrappers small and consistent.

### `stage_1/datasets/ShapeNetInternalsSmall.py`

TRELLIS' `dataset_toolkits/render.py` and `dataset_toolkits/render_cond.py` expect a dataset adapter module under a `datasets` package. This adapter teaches those existing render scripts how to read our ShapeNet-derived `metadata.csv` and how to map each row to `local_path`.

Input:

```text
datasets/ShapeNetInternals_small/metadata.csv
```

Output:

```text
records returned to TRELLIS render scripts
```

It does not render anything itself. It only connects our metadata to TRELLIS' existing rendering tools.

### `stage_1/prepare_subset.py`

This script builds the small Stage 1 dataset manifest. It scans `/workspace/TRELLIS/ShapeNet`, selects a small subset across all four categories, records source OBJ/binvox paths, copies useful inner-geometry statistics from the category score CSV files, writes `metadata.csv`, and creates provisional train/val/test split files.

Input:

```text
/workspace/TRELLIS/ShapeNet/<category>/<object_id>/
```

Output:

```text
datasets/ShapeNetInternals_small/metadata.csv
datasets/ShapeNetInternals_small/splits/{train,val,test}.txt
datasets/ShapeNetInternals_small/selected_ids.txt
```

The actual Stage 1 run selected 12 objects: 3 each from `bus`, `cabinet`, `cars`, and `file_cabinet`.

### `stage_1/convert_binvox_to_voxels.py`

This script converts ShapeNet binvox files into TRELLIS sparse voxel PLY files. The main target is `model_normalized.surface.binvox`. It downsamples from 128^3 to 64^3 using OR/max pooling so thin internal surfaces are preserved as much as possible.

Input:

```text
models/model_normalized.surface.binvox
```

Output:

```text
datasets/ShapeNetInternals_small/voxels/<id>.ply
datasets/ShapeNetInternals_small/binvox_surface_to_voxels.csv
```

It can also write a solid-volume ablation folder from `model_normalized.solid.binvox`, but the main geometry target remains `surface.binvox`.

### `stage_1/refresh_metadata.py`

This script scans the converted dataset and refreshes artifact flags in `metadata.csv`. It checks whether each expected output exists: voxels, conditioning renders, multi-view renders, DINO features, sparse-structure latents, and SLAT latents.

Input:

```text
datasets/ShapeNetInternals_small/
```

Output:

```text
updated metadata.csv
statistics_stage1.txt
```

It makes the pipeline restartable because each TRELLIS preprocessing stage can skip samples already marked complete.

### `stage_1/run_stage1_pipeline.py`

This is the Stage 1 conversion orchestrator. It calls the Stage 1 adapter scripts and the existing TRELLIS preprocessing scripts in sequence.

It executes:

```text
prepare_subset.py
convert_binvox_to_voxels.py
dataset_toolkits/render_cond.py
dataset_toolkits/render.py
dataset_toolkits/extract_feature.py
dataset_toolkits/encode_ss_latent.py
dataset_toolkits/encode_latent.py
refresh_metadata.py
```

Input:

```text
/workspace/TRELLIS/ShapeNet
```

Output:

```text
/workspace/TRELLIS/datasets/ShapeNetInternals_small
```

The run used 12 samples with 8 multi-view renders and 1 conditioning render per object for a fast feasibility test.

### `stage_1/run_reconstruction_eval.py`

This script evaluates the sparse-structure VAE round trip:

```text
GT 64^3 surface voxels
        |
        v
SparseStructureEncoder
        |
        v
ss_latent
        |
        v
SparseStructureDecoder
        |
        v
thresholded reconstructed voxels
```

It writes reconstructed voxel PLYs, compressed reconstruction grids, cross-section visualizations, `metrics.csv`, and a Markdown report.

Output:

```text
/workspace/TRELLIS/results/shapenet_internals_stage1_reconstruction/
```

### `stage_1/run_mesh_reconstruction_eval.py`

This script evaluates the full SLAT-to-mesh reconstruction path:

```text
ShapeNet renders/features
        |
        v
ElasticSLatEncoder
        |
        v
SLAT latent
        |
        v
SLatMeshDecoder
        |
        v
SparseFeatures2Mesh + FlexiCubes
        |
        v
MeshExtractResult
```

It uses the pretrained mesh decoder checkpoint:

```text
microsoft/TRELLIS-image-large/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16
```

The script saves raw decoder meshes before TRELLIS GLB postprocessing because postprocessing may remove invisible/internal faces. It computes sampled mesh metrics, surface-voxel metrics, approximate external/internal recall, and cross-section visualizations.

Output:

```text
/workspace/TRELLIS/results/shapenet_mesh_reconstruction/
```

## 5. Reconstruction Experiments

### Experiment 1: Sparse-Structure Reconstruction

The first experiment tested whether TRELLIS' sparse-structure representation preserves ShapeNet surface voxels, including internal surfaces.

Pipeline:

```text
model_normalized.surface.binvox
        |
        v
64^3 sparse voxel PLY
        |
        v
SparseStructureEncoder
        |
        v
sparse-structure latent
        |
        v
SparseStructureDecoder
        |
        v
reconstructed voxel grid
```

Dataset:

```text
/workspace/TRELLIS/datasets/ShapeNetInternals_small
```

Subset:

- 12 objects total
- `bus`: 3
- `cabinet`: 3
- `cars`: 3
- `file_cabinet`: 3

All conversion artifacts were created successfully for all 12 objects:

- `voxelized`: 12 / 12
- `rendered`: 12 / 12
- `cond_rendered`: 12 / 12
- `feature_dinov2_vitl14_reg`: 12 / 12
- `ss_latent_ss_enc_conv3d_16l8_fp16`: 12 / 12
- `latent_dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16`: 12 / 12

Results:

- Mean voxel IoU: `0.9996`
- Mean precision: `0.9997`
- Mean recall: `0.9999`
- Mean F1: `0.9998`
- Mean GT voxels: `11638.2`
- Mean reconstructed voxels: `11639.6`

Conclusion:

The sparse-structure VAE is not the bottleneck. It preserved the 64^3 ShapeNet surface voxel targets almost perfectly. Cross-section visualizations showed internal-looking shelf and partition structures surviving the sparse-structure round trip.

### Experiment 2: Full Mesh Reconstruction

The second experiment tested the complete SLAT latent and mesh decoder path.

Pipeline:

```text
ShapeNet object
        |
        v
TRELLIS renders + DINO features
        |
        v
ElasticSLatEncoder
        |
        v
SLAT latent
        |
        v
SLatMeshDecoder
        |
        v
SparseFeatures2Mesh + FlexiCubes
        |
        v
raw reconstructed mesh
```

The decoder path used TRELLIS' normal mesh extraction implementation. The evaluator called the same decoder that `TrellisImageTo3DPipeline.decode_slat` would use for mesh output.

Results on the same 12-object subset:

- Successful mesh decodes: `12 / 12`
- Mean predicted vertices: `248580.8`
- Mean predicted faces: `496879.3`
- Mean Chamfer L1: `0.018869`
- Mean F-score @ `0.01`: `0.7989`
- Mean surface-voxel IoU: `0.9234`
- Mean surface-voxel precision: `0.9779`
- Mean surface-voxel recall: `0.9428`
- Mean surface-voxel F1: `0.9598`
- Mean external recall: `0.9895`
- Mean internal-candidate recall: `0.8865`

Visual findings:

- Raw decoder meshes retained major internal structures in cabinets and file cabinets.
- Voxel cross-sections showed mostly true-positive overlap for internal shelf and partition surfaces.
- Some local misses and extra surfaces remain, especially in detailed internal layouts, but the decoder did not collapse objects to exterior-only shells.

Important caveat:

The internal/external split is approximate. External voxels were estimated as the first and last occupied GT surface voxels along the six cardinal grid directions. The remaining occupied GT surface voxels were treated as internal candidates. This is useful for screening, but not a perfect semantic internal-surface label.

## 6. Key Findings

The Stage 1 experiments changed the project from a speculative LoRA idea into a feasible fine-tuning plan.

Main findings:

- ShapeNet contains the assets needed for this project: normalized meshes, surface binvox grids, solid binvox grids, textures, and inner-geometry statistics.
- Texture maps are material assets, not image-conditioning views. TRELLIS-compatible external renders must be generated.
- `model_normalized.surface.binvox` is the correct geometry target for preserving internal surfaces.
- OR/max pooling from 128^3 to 64^3 preserves thin occupied structures better than averaging.
- TRELLIS' sparse-structure VAE preserves ShapeNet internal-rich surface voxels extremely well: mean IoU about `0.9996`.
- TRELLIS' full SLAT mesh decoder also preserves internal-candidate geometry reasonably well: mean surface-voxel F1 about `0.9598` and internal-candidate recall about `0.8865`.
- The decoder is probably not the primary bottleneck for internal geometry in this ShapeNet adaptation.
- The next bottleneck is likely the generative priors: the structure flow and SLAT flow have not been trained to sample these internal-rich targets from conditioning images.

## 7. Final Conclusion

Stage 1 supports proceeding to LoRA fine-tuning.

The original concern was that TRELLIS' pretrained representation might erase internal geometry before any LoRA training could learn it. The sparse-structure and full mesh reconstruction experiments did not show that failure mode. Instead:

```text
surface voxels -> sparse-structure latent -> reconstructed voxels
```

preserved internals almost perfectly, and:

```text
ShapeNet renders/features -> SLAT latent -> pretrained mesh decoder -> reconstructed mesh
```

preserved internal-candidate geometry well enough to justify LoRA on the flow models.

The recommended next training direction is:

1. Keep the pretrained encoders/decoders frozen initially.
2. Add LoRA adapters to `SparseStructureFlowModel`.
3. Fine-tune structure generation first.
4. Validate generated sparse structures and internal voxel cross-sections.
5. Add LoRA adapters to `ElasticSLatFlowModel` only after structure generation is working.

Remaining risks:

- The experiments used only 12 objects, so the result may not cover all ShapeNet failure cases.
- Internal/external metrics are approximate and should be improved with better internal-surface labels if possible.
- Reconstruction from ground-truth latents is easier than generation from image conditioning. LoRA still has to learn the conditional distribution over internal-rich geometry.
- Exterior render quality is not sufficient validation. Internal cross-sections and voxel/mesh metrics must remain part of the evaluation loop.
- TRELLIS GLB postprocessing can remove invisible/internal faces, so internal evaluations should use raw decoder meshes whenever possible.
