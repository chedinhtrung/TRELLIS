# **Single-Image Internal 3D Geometry Generation with TRELLIS**

This project adapts image-conditioned TRELLIS to reconstruct semantically
meaningful `internal 3D geometry` from a single exterior image. The final system
contains two methods:

1. `Interior-aware TRELLIS fine-tuning`: Generates the complete shape,
   including its exterior and predicted internal structure.
2. `Retrieval refinement`: Uses structurally similar training shapes to replace
   or supplement uncertain generated internals with coherent geometry.

The final implementation supports four ShapeNet categories: `bus`, `cabinet`,
`car`, and `file_cabinet`. The quantitative results below report all four
categories. Buses and cars are used as the two headline visual categories: the
visualization notebook presents the 10 strongest buses and 10 strongest cars as
half-cut meshes in two stages. The first compares ground truth, original
TRELLIS, and fine-tuned TRELLIS; the second compares ground truth, fine-tuned
TRELLIS, and retrieval refinement on the same objects.

## **Repository Layout**

```text
interior_reconstruction/
├── data_preparation/               # ShapeNet -> TRELLIS-format entrypoint
├── trellis_finetuning_interior/    # Fine-tuning, generation, and evaluation
└── retrieval_refinement/           # DINO retrieval, calibration, and refinement

configs/finetune/                   # The three final LoRA configurations
dataset_toolkits/                   # Rendering, voxelization, and latent encoding
notebooks/visualize_results.ipynb   # Visualization
```

## **1. Setup**

### **Requirements**

- Linux with an NVIDIA CUDA GPU
- Python 3.11
- Git and Git LFS
- A CUDA-compatible C/C++ build toolchain
- Enough storage for ShapeNet, TRELLIS preprocessing outputs, and checkpoints (around 100 - 150GB)

The final project was validated on Python 3.11, PyTorch 2.7.0, CUDA 12.8, and an
NVIDIA A100. Other recent NVIDIA GPUs should work, but sparse-convolution and
CUDA-extension compatibility depends on the installed PyTorch/CUDA versions.

### **Clone the Repository**

```bash
git clone --recursive <repository-url> TRELLIS
cd TRELLIS
git submodule update --init --recursive
git lfs pull
```

### **Create the Environment**

On a CUDA 12 machine, the provided setup script creates `/workspace/venv` with
Python 3.11 and the project’s PyTorch environment:

```bash
python3 -m pip install --upgrade uv

bash setup.sh \
  --basic \
  --train \
  --xformers \
  --flash-attn \
  --diffoctreerast \
  --vox2seq \
  --spconv \
  --mipgaussian \
  --kaolin \
  --nvdiffrast \
  --demo

source /workspace/venv/bin/activate
uv pip install jupyterlab kiui
```

Verify the essential environment:

```bash
python - <<'PY'
import torch
import numpy
import open3d
import trimesh
import utils3d

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

TRELLIS base checkpoints and DINOv2 weights are downloaded automatically when
first used, so the first run requires network access to the model providers.

## **2. Prepare ShapeNet in TRELLIS Format**

The expected raw ShapeNet organization is:

```text
ShapeNet/
├── bus/<object-id>/models/model_normalized.obj
├── cabinet/<object-id>/models/model_normalized.obj
├── car/<object-id>/models/model_normalized.obj
└── file_cabinet/<object-id>/models/model_normalized.obj
```

Run the retained conversion pipeline:

```bash
cd /workspace/TRELLIS

env \
  SHAPENET_ROOT=/workspace/TRELLIS/ShapeNet \
  SHAPENET_PROCESSED=/workspace/TRELLIS/datasets/ShapeNetTRELLIS_full \
  MAX_WORKERS=8 \
  bash interior_reconstruction/data_preparation/run_pipeline.sh
```

This creates deterministic train/validation/test manifests and, for every
shape, produces:

- 40 ordinary renders
- 40 conditioning renders
- normalized meshes and 64^3 surface voxels
- DINOv2 features
- sparse-structure latents
- structured-latent (SLAT) features

The public wrapper is
`interior_reconstruction/data_preparation/run_pipeline.sh`. It delegates to
`dataset_toolkits/prepare_shapenet.sh`, while the individual conversion tools
remain under `dataset_toolkits/`.

Validate the completed dataset before training:

```bash
python interior_reconstruction/trellis_finetuning_interior/validate_dataset.py \
  --dataset-root /workspace/TRELLIS/datasets/ShapeNetTRELLIS_full
```

## **3. Pipeline 1: Interior-Aware TRELLIS Fine-Tuning**

### **Methodology**

The original image-conditioned TRELLIS pipeline generates a shape in three
relevant stages:

```text
conditioning image
    -> sparse-structure flow
    -> occupied sparse coordinates
    -> SLAT flow
    -> per-coordinate latent features
    -> mesh decoder
    -> triangle mesh
```

Internal geometry can disappear at any of these stages. Fine-tuning only the
last decoder cannot recover coordinates that were never generated, while
fine-tuning only the flow models cannot guarantee that latent internal surfaces
survive mesh decoding. We therefore adapt all three stages:

1. `Sparse-structure flow LoRA`: Learns where exterior and interior geometry
   should exist. Its final configuration uses rank 8.
2. `SLAT-flow LoRA`: Learns interior-aware latent features on the generated
   sparse coordinates. Its final configuration uses rank 16.
3. `Mesh-decoder LoRA`: Teaches the decoder to convert those latent features
   into visible internal triangle surfaces. Its final configuration uses rank
   32 and an explicit internal-geometry loss.

#### **Interior-Aware Decoder Loss**

TRELLIS's original decoder loss renders the complete predicted mesh and compares
it with a render of the complete ground-truth mesh using the silhouette, depth,
and surface normals. This provides strong supervision for the exterior, but a
normal render shows only the first surface hit by each camera ray. The outer
shell therefore hides most internal surfaces, so a missing floor, seat, shelf,
or partition may contribute little or nothing to the original rendering loss.

We add a second, interior-aware comparison during decoder training. A plane is
placed through the center of both the predicted mesh and the ground-truth mesh,
and the half facing the training camera is temporarily removed. We then render
the two cut-open meshes and compare their silhouettes, depths, and surface
normals. This exposes internal surfaces to the renderer and gives the decoder a
direct penalty when its predicted interior differs from the ground truth. The
original full-mesh loss is still used, so the decoder must preserve the exterior
while also learning the interior. In the final configuration, the additional
interior loss has weight `lambda_internal_geometry = 1.0`.

The mesh is cut open only while calculating this training loss. Nothing is cut
away during inference; the decoder still produces the complete object.

### **Quantitative Results**

The baseline uses the original pretrained `microsoft/TRELLIS-image-large`
weights without any LoRA adapters. Both models are evaluated on the same 195
held-out shapes with conditioning view 18, seed 42, sparse-coordinate threshold
`-8`, a 64^3 voxel grid, and interior margin 2. The comparison therefore
isolates the effect of the three fine-tuned adapters while keeping inference
settings fixed.

| Category | Samples | Precision (original -> fine-tuned) | Recall (original -> fine-tuned) | F1 (original -> fine-tuned) | F1 gain | F0.5 (original -> fine-tuned) | Predicted/GT internal volume (original -> fine-tuned) |
|---|---:|---:|---:|---:|---:|---:|---:|
| All | 195 | 0.2437 -> 0.2994 | 0.0920 -> 0.3346 | 0.1185 -> 0.2922 | **+0.1737 (+146.58%)** | 0.1594 -> 0.2880 | 0.471 -> 1.673 |
| Bus | 30 | 0.2761 -> 0.2938 | 0.0827 -> 0.4312 | 0.1129 -> 0.3340 | **+0.2211 (+195.87%)** | 0.1598 -> 0.3069 | 0.388 -> 2.011 |
| Cabinet | 50 | 0.1308 -> 0.1751 | 0.0711 -> 0.2295 | 0.0733 -> 0.1602 | **+0.0869 (+118.55%)** | 0.0882 -> 0.1567 | 0.605 -> 2.589 |
| Car | 100 | 0.3009 -> 0.3850 | 0.1093 -> 0.3866 | 0.1485 -> 0.3719 | **+0.2234 (+150.46%)** | 0.2032 -> 0.3743 | 0.397 -> 1.091 |
| File cabinet | 15 | 0.1733 -> 0.1539 | 0.0657 -> 0.1449 | 0.0805 -> 0.1172 | **+0.0367 (+45.66%)** | 0.1040 -> 0.1120 | 0.676 -> 1.817 |


### **Train the three LoRAs**

Train the sparse-structure and SLAT-flow adapters:

```bash
cd /workspace/TRELLIS

time env \
  PYTHONUNBUFFERED=1 \
  DATA_DIR=/workspace/TRELLIS/datasets/ShapeNetTRELLIS_full/train \
  OUT_DIR=/workspace/TRELLIS/results/objective1_full \
  NUM_GPUS=1 \
  bash interior_reconstruction/trellis_finetuning_interior/run_flow_finetuning.sh
```

Train the mesh-decoder adapter:

```bash
time env \
  PYTHONUNBUFFERED=1 \
  DATA_DIR=/workspace/TRELLIS/datasets/ShapeNetTRELLIS_full/train \
  OUT_DIR=/workspace/TRELLIS/results/objective1_full \
  bash interior_reconstruction/trellis_finetuning_interior/run_decoder_finetuning.sh
```

The expected final checkpoints are:

```text
results/objective1_full/ss_flow/ckpts/denoiser_lora_final.pt
results/objective1_full/slat_flow/ckpts/denoiser_lora_final.pt
results/objective1_full/decoder/ckpts/decoder_lora_final.pt
```

### **Generate and Evaluate the Held-Out Split**

Use a new output directory for a fresh run. Set `RESUME=1` only when continuing
the same interrupted run.

```bash
time env \
  PYTHONUNBUFFERED=1 \
  OBJECTIVE1_DIR=/workspace/TRELLIS/results/objective1_full \
  OUTPUT_DIR=/workspace/TRELLIS/results/objective1_view18_full \
  bash interior_reconstruction/trellis_finetuning_interior/run_view18_full.sh
```

The retrieval selector is calibrated from train-set predictions, not from
held-out predictions. Generate that cache with:

```bash
time env \
  PYTHONUNBUFFERED=1 \
  OBJECTIVE1_DIR=/workspace/TRELLIS/results/objective1_full \
  OUTPUT_DIR=/workspace/TRELLIS/results/objective1_view18_train \
  NUM_GPUS=1 \
  bash interior_reconstruction/trellis_finetuning_interior/run_view18_train.sh
```

## **4. Pipeline 2: Retrieval Refinement**

### **Motivation**

Fine-tuning improves internal generation, but a single image still contains no
direct observation of hidden geometry. The interiors generated by Pipeline 1
are acceptable, but there is room for improvement. Training shapes provide a
useful prior: Objects with similar visible structure often share coherent
internal layouts. The retrieval method uses this prior while preserving the
fine-tuned exterior and rejecting uncertain changes.

### **Methodology**

Pipeline 2 does not generate a second mesh from scratch. It treats the
fine-tuned output from Pipeline 1 as the base shape, retrieves complete
training shapes that look similar, and asks whether transferring one donor's
interior is safer than leaving the base prediction unchanged.

The complete test-time flow is:

```text
view-18 query image
    -> top-20 same-category DINO donors
    -> align every donor to the generated exterior
    -> shortlist 8 donors
    -> build 2 coherent candidates per donor
    -> predict each candidate's F1 change
    -> accept the best candidate or copy Pipeline 1 unchanged
```

#### **1. Separate the Generated Exterior and Interior**

The decoded Pipeline-1 mesh is voxelized at 64^3. With evaluation margin 2, a
surface voxel is called `internal` only when it lies at least two voxels behind
the visible extremum along all three axes. Everything else forms the generated
exterior.

The generated exterior is used to define a protected region inside the object.
Retrieved geometry is added only when it lies completely inside this region, so
it cannot extend beyond the predicted outer shell. The original exterior is
always kept unchanged; retrieval only modifies the internal geometry.

#### **2. Retrieve 20 Possible Training Donors**

DINOv2 embeds the conditioning render from `view 18` for every training and
test object. For each test image, the system retrieves the 20 most similar
training images from the same ShapeNet category. These 20 training shapes form
the donor pool from which interior geometry can be retrieved. No test shape is
used as a donor.

#### **3. Put Each Donor in the Right Place**

Similar objects can still have different positions and scales. We therefore make
a small adjustment to each donor so that its outer shell fits the generated
Pipeline-1 shell better (axis-aligned affine transformation of the donor's vertices).
The same adjustment is applied to the donor’s entire mesh, including its interior,
so the donor’s internal parts stay in the same relative arrangement. We try a few
conservative adjustment amounts and keep the one that gives the best exterior F1 score.


#### **4. Keep the Eight Most Promising Donors**

The image-nearest donor is not always the best geometric match. We score all 20
donors using image similarity, outer-shell agreement, agreement with the current
interior, agreement with the other donors, how much fits in the safe volume, and
whether the donor interior is a reasonable size. We keep `8 donors` that are
strong according to these different checks.

#### **5. Make Two Possible Repairs from Each Donor**

We only accept interiors from 1 donor at a time (not mixing information between different
donors). For each of the 8 shortlisted donors, there are 2 different ways of using its
interior, giving in total 16 possible outputs.

`component_hybrid`:

- Split the aligned donor interior into 26 components.
- Keep a donor component only if it has at least 24 voxels, at least 70% remains after
  clipping to the safe volume, at least 90% of the clipped part stays connected,
  and at least 20% of its voxels are contained in another aligned donor.
- Insert accepted donor components inside the safe volume. Throw away Pipeline 1 voxels that overlap
  or lie within 1 voxel of the donor components. Preserve the non-overlapping part of a Pipeline 1 component
  only if the original component has at least 64 voxels and at most 15% of the component's voxels has neighbors
  in all directions (prevent blobs). The motivation for combining components is to add the useful missing donor structure without
  throwing away every useful structure that pipeline 1 already generated.

`structural_replace`:

- Split the aligned donor interior into 26 components
- Keep a donor component only if it has at least 24 voxels, at most 50% of the component's voxels has neighbors in all
  directions, and at least 10% of its voxels are contained in another aligned donor.
- Remove all Pipeline 1 interior voxels inside the safe volume, then insert accepted donor components inside the safe volume.


#### **6. Learn Which Repair is Likely to Help**

Idea: After constructing all 16 candidate outputs, predict which complete
candidate is most likely to improve Pipeline 1. If none is predicted to help
confidently, use the unchanged Pipeline 1 output.

For every training query, we already have:

- The Pipeline-1 prediction
- Its ground-truth interior
- 16 candidate repairs

For each training repair, we store its actual F1 change relative to Pipeline 1
as the target label. We also store input features that remain available during
inference:

- Donor image similarity
- Donor/query shell agreement
- Repair method
- Donor support
- Amount of geometry transferred
- Amount of Pipeline-1 geometry removed
- Candidate interior size and density

We fit one kNN model to these labeled training repairs. During inference, ground
truth and the actual F1 change are unavailable, so we calculate only the input
features for each of the 16 test candidates. For one candidate, kNN finds the 8
most similar training repairs in feature space and computes a distance-weighted
average of their known F1 changes. This average is the candidate's predicted F1
improvement. Step 7 accepts the highest-scoring valid candidate only if its
predicted improvement is at least 0.020 and it beats the runner-up by at least
0.001; otherwise, the method falls back to Pipeline 1.

#### **7. Build the Final Smooth Mesh**

At this point, the voxel grid tells us which parts of the final object should
come from Pipeline 1 and which should come from the selected donor. The voxel
grid is only a guide; it is not the mesh that we show or save.

To build the final mesh, we remove the Pipeline-1 interior triangles from the
regions being replaced, while keeping its exterior and any interior regions
that the chosen repair preserves. We then take the donor's smooth interior
triangles from the accepted regions, apply the same alignment used earlier,
and add them to the remaining Pipeline-1 triangles. Therefore, the final output
is made from smooth triangles rather than one cube for every occupied voxel. If
the repair is rejected, we simply copy the original Pipeline-1 mesh without
changing it.


### **Quantitative improvements**

The retrieval method is paired against the fine-tuned TRELLIS output for the
same 195 held-out objects at margin 2. The values below are read from
`results/retrieval_unified_view18/summary.csv` and `category_summary.csv`:

| Category | Precision (fine-tuned -> retrieval) | F1 (fine-tuned -> retrieval) | F1 gain | F0.5 gain | Internal-volume ratio |
|---|---:|---:|---:|---:|---:|
| All | 0.2994 -> 0.3187 | 0.2922 -> 0.3014 | **+0.0092 (+3.14%)** | **+0.0155 (+5.39%)** | 1.673 -> 1.625 |
| Bus | 0.2938 -> 0.3573 | 0.3340 -> 0.3619 | **+0.0279 (+8.36%)** | **+0.0489 (+15.94%)** | 2.011 -> 1.713 |
| Car | 0.3850 -> 0.4024 | 0.3719 -> 0.3793 | **+0.0074 (+2.00%)** | **+0.0139 (+3.71%)** | 1.091 -> 1.031 |
| File cabinet | 0.1539 -> 0.1728 | 0.1172 -> 0.1347 | **+0.0175 (+14.92%)** | **+0.0187 (+16.65%)** | 1.817 -> 1.619 |
| Cabinet | 0.1751 -> 0.1717 | 0.1602 -> 0.1591 | -0.0011 (-0.67%) | -0.0022 (-1.39%) | 2.589 -> 2.763 |


### **Calibrate the Final Policy from Scratch**

```bash
cd /workspace/TRELLIS

time env \
  PYTHONUNBUFFERED=1 \
  DATASET_ROOT=/workspace/TRELLIS/datasets/ShapeNetTRELLIS_full \
  OBJECTIVE1_TRAIN_DIR=/workspace/TRELLIS/results/objective1_view18_train \
  OBJECTIVE1_TEST_DIR=/workspace/TRELLIS/results/objective1_view18_full \
  DINO_DIR=/workspace/TRELLIS/results/dino_retrieval_view18_fresh \
  UNIFIED_CALIBRATION_DIR=/workspace/TRELLIS/results/retrieval_unified_calibration_fresh \
  OUTPUT_DIR=/workspace/TRELLIS/results/retrieval_unified_fresh \
  CALIBRATION_WORKERS=8 \
  UNIFIED_QUERIES_PER_CATEGORY=160 \
  bash interior_reconstruction/retrieval_refinement/run_view18.sh
```
