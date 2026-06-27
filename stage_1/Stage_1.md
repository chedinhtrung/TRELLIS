# Stage 1: ShapeNet -> TRELLIS Data Understanding and Reconstruction Feasibility

## 1. Goal

The original research question was:

```text
Can TRELLIS be adapted to generate internal geometry by LoRA fine-tuning on ShapeNet-style data?
```

The problem is: If internals disappeared during encoding or decoding, LoRA on `SparseStructureFlowModel` or `ElasticSLatFlowModel` alone would not be enough. We would first need decoder/sparse VAE fine-tuning with internal-aware supervision. Stage 1 was designed to answer that representation question before starting any LoRA work.

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

`.obj` stores the actual 3D mesh geometry. `.mtl` stores material information referenced by `.obj` file, which tells the renderer the material name, base color, which texture image to use, ... . The `images/` folder contains material texture maps referenced by the `.mtl`. `.binvox` files store a voxelized version of the 3D object. Instead of triangles like `.obj`, they represent the object as a 3D occupancy grid. `model_normalized.surface.binvox` marks surface voxels, including internal surfaces when present. This is the correct target for internal-rich geometry experiments. `model_normalized.solid.binvox` marks filled volume. It can be useful as an ablation, but it is not equivalent to internal surface geometry.

The `.csv` files under ShapeNet/ rank/filter ShapeNet objects by how much internal geometry they seem to have. `inner-face` and `inner-edge` statistics mean: How many mesh faces / edges are inside the object's bounding box, not just on the outside shell?

## 3. ShapeNet -> TRELLIS Conversion

TRELLIS training does not train directly from raw .obj files. It expects a standardized preprocessed dataset directory. The target layout is:

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

- `metadata.csv`: Reecords sample ids, source paths, category labels, split labels (belong to train, val or test), and preprocessed file availability flags (does renders/, renders_cond/, voxels/, DINO features, structure latent and SLAT latent exist for that sample).
- `renders_cond/`: Conditioning images. These are input views for image-to-3D training. 
- `renders/`: Multi-view renders of the object. Used to extract DINOv2 visual features.
- `voxels/`: 64^3 sparse voxel PLY files used as sparse-structure targets.
- `features/`: DINOv2 features projected and averaged onto the sparse 3D voxels. Instead of image features at pixel, TRELLIS gets feature at 3D voxel. 
- `ss_latents/`: Latent code from the structure VAE. This is the target for the first flow Transformer. 
- `latents/`: Latent code from the SLAT VAE. This is the target the for second flow Transformer. 

## 4. Stage 1 Implementation

All Stage 1 code was placed under:

```text
/workspace/TRELLIS/stage_1/
```

No TRELLIS core model code was modified.

### `stage_1/common.py`

This file holds shared constants and utility functions used by the Stage 1 scripts. 

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

Before finetuning the Flow Transformer G_S, we first check if the sparse structure VAE can preserve internal geometry. 

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

Output:

```text
/workspace/TRELLIS/results/shapenet_sparse_reconstruction/
```

### `stage_1/run_mesh_reconstruction_eval.py`

This script evaluates the full SLAT-to-mesh reconstruction path. It loads cached SLAT latents from the converted dataset and checks: If TRELLIS is given the correct latent for this exact object, can its pretrained decoder reconstruct the object well?

```text
cached SLAT latent
        |
        v
SLatMeshDecoder
        |
        v
SparseFeatures2Mesh + FlexiCubes
        |
        v
raw mesh in TRELLIS/binvox grid frame
        |
        v
ShapeNet OBJ-frame transform using surface.binvox metadata
        |
        v
MeshExtractResult in ShapeNet OBJ frame
```

It uses the pretrained mesh decoder checkpoint:

```text
microsoft/TRELLIS-image-large/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16
```

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
- Mean true positives: `11637.0`
- Mean false positives: `2.6`
- Mean false negatives: `1.2`

Category means:

```text
category       IoU       precision  recall    F1
bus            0.999947  0.999947   1.000000  0.999974
cabinet        1.000000  1.000000   1.000000  1.000000
cars           0.998805  0.999153   0.999652  0.999402
file_cabinet   0.999781  0.999869   0.999912  0.999891
```

Conclusion:

The sparse-structure VAE is not the bottleneck. It preserved the 64^3 ShapeNet surface voxel targets almost perfectly. Cross-section visualizations showed internal-looking shelf and partition structures surviving the sparse-structure round trip.

### Experiment 2: SLAT-to-Mesh Reconstruction

The second experiment tested the full pretrained mesh decoding path from cached SLAT latents to triangle meshes. This answers a different question from Experiment 1: if TRELLIS is given the cached latent for the exact ShapeNet object, can the pretrained mesh decoder reconstruct a mesh close to the original source OBJ?

Pipeline:

```text
cached SLAT latent
        |
        v
SLatMeshDecoder
        |
        v
SparseFeatures2Mesh + FlexiCubes
        |
        v
raw reconstructed mesh in TRELLIS/binvox grid frame
        |
        v
transform with surface.binvox axis, translate, and scale metadata
        |
        v
reconstructed mesh in ShapeNet OBJ frame
        |
        v
surface sampling and Chamfer/F-score evaluation against source OBJ
```

Output:

```text
/workspace/TRELLIS/results/shapenet_mesh_reconstruction/
```

Artifacts:

- `metrics.csv`: mesh surface metrics for 12 samples
- `recon_meshes_grid/*.ply`: 12 raw decoder meshes in TRELLIS/binvox grid frame
- `recon_meshes_obj_frame/*.ply`: 12 transformed meshes in ShapeNet OBJ frame

Metrics were computed with 50,000 sampled surface points per mesh and an F-score threshold of `0.01`.

Results:

- Mean Chamfer-L1: `0.0189`
- Mean Chamfer-L2: `0.0006`
- Mean precision @ `0.01`: `0.8601`
- Mean recall @ `0.01`: `0.7545`
- Mean F-score @ `0.01`: `0.7989`
- Mean GT vertices: `399676.3`
- Mean GT faces: `376270.2`
- Mean reconstructed vertices: `248580.8`
- Mean reconstructed faces: `496879.3`

Category means:

```text
category       Chamfer-L1  Chamfer-L2  precision  recall    F-score
bus            0.018156    0.000354    0.790847   0.705147  0.745193
cabinet        0.031864    0.001668    0.832900   0.554240  0.664964
cars           0.015774    0.000305    0.866313   0.796987  0.829691
file_cabinet   0.009683    0.000082    0.950207   0.961767  0.955833
```

Conclusion:

After restoring the binvox-grid to ShapeNet-OBJ coordinate transform, the pretrained SLAT mesh decoder reconstructs the source meshes much more faithfully than the earlier raw-coordinate evaluation suggested. The mean F-score is `0.7989`, with strongest performance on `file_cabinet` (`0.9558`) and weakest performance on `cabinet` (`0.6650`). The earlier near-zero mesh metrics were therefore mainly a coordinate-frame evaluation bug, not evidence that the decoder completely failed.

## 6. Key Findings and Next Steps

The original concern was that TRELLIS' pretrained representation might erase internal geometry before any LoRA training could learn it. The sparse-structure reconstruction experiment did not show that failure mode. Instead:

```text
surface voxels -> sparse-structure latent -> reconstructed voxels
```

preserved internals almost perfectly. The updated mesh reconstruction experiment also tests:

```text
cached SLAT latent -> pretrained mesh decoder -> reconstructed mesh
```

After converting decoded meshes from TRELLIS/binvox grid coordinates back into ShapeNet OBJ coordinates, the mesh path also gives useful reconstructions. The remaining risk is no longer basic coordinate alignment; it is whether generated latents from fine-tuned flow models will preserve the same internal structures when they are not copied from the cached encoders.

The recommended next training direction is:

1. Keep the pretrained encoders/decoders frozen initially.
2. Add LoRA adapters to `SparseStructureFlowModel`.
3. Fine-tune structure generation first.
4. Validate generated sparse structures and internal voxel cross-sections.
5. Add LoRA adapters to `ElasticSLatFlowModel` after structure generation is working.
6. Validate generated SLAT latents through the OBJ-frame mesh reconstruction evaluator.
