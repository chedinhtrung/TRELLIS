# TRELLIS Research Code Map

Content: Understanding TRELLIS' inference path, sparse structure stage, SLAT latent stage, rectified-flow transformers, VAEs, and output decoders.

## Pipeline At A Glance

TRELLIS generation is a two-stage conditional flow pipeline:

1. Encode a text or image prompt into conditioning tokens.
2. Generate a sparse-structure latent with a rectified-flow transformer.
3. Decode that latent with the sparse-structure VAE decoder into occupied voxel coordinates.
4. Create a `SparseTensor` whose coordinates are the generated occupied voxels and whose features begin as Gaussian noise.
5. Generate SLAT features on those coordinates with a sparse rectified-flow transformer.
6. Denormalize the SLAT features (during training, SLAT features are normalized, so we have to denormalize them before decoding).
7. Decode the SLAT into mesh, Gaussian, or radiance-field representations.

The structure stage decides `where geometry can exist`. The SLAT stage decides `what latent feature lives at each occupied sparse voxel`. The final decoders translate those sparse latent features into `concrete renderable outputs`.

## Important Files And Folders

### `trellis/pipelines/trellis_image_to_3d.py`

What it contains: The image-conditioned inference pipeline.

Role in TRELLIS: Loads DINOv2 image features, samples sparse structure, samples SLAT on generated coordinates, then decodes into mesh/Gaussian/radiance outputs.

Read it for: The end-to-end inference order, image preprocessing, where image conditioning is created, how sparse coordinates are produced, and how the SLAT sampler is called.


### `trellis/pipelines/trellis_text_to_3d.py`

What it contains: The text-conditioned inference pipeline and a variant-generation path from an input mesh.

Role in TRELLIS: Encodes prompts with CLIP text tokens, uses those tokens as conditioning for both structure and SLAT flow models, and optionally uses voxelized mesh coordinates for variant generation.

Read it for: Text conditioning, classifier-free guidance inputs, the standard `run()` path, and how `run_variant()` bypasses structure sampling by supplying coordinates from a mesh.


### `trellis/pipelines/base.py`

What it contains: The base pipeline loader/device wrapper.

Role in TRELLIS: Loads `pipeline.json`, model configs, and safetensor weights through `models.from_pretrained`.

Read it for: How pretrained pipelines assemble named submodels such as `sparse_structure_flow_model`, `sparse_structure_decoder`, `slat_flow_model`, and output decoders.

Architecture modification relevance: Medium. Relevant when changing checkpoint layout or adding/removing model components.

### `trellis/pipelines/samplers/flow_euler.py`

What it contains: Euler sampling for rectified-flow models, including CFG variants.

Role in TRELLIS: Numerically integrates the learned velocity field from noise at `t=1` toward a sample at `t=0`.

Read it for: How the flow model prediction is interpreted as velocity, how timesteps are scheduled, and how samples move from noise to data.

Architecture modification relevance: Medium. Modify for new samplers, timestep schedules, or sampling diagnostics, but not for core model layer changes.

### `trellis/pipelines/samplers/classifier_free_guidance_mixin.py`

What it contains: Classifier-free guidance logic for samplers.

Role in TRELLIS: Runs the same flow model with positive and negative conditioning, then combines the two predictions.

Read it for: How `cond` and `neg_cond` are used at inference.

Architecture modification relevance: Medium. Relevant if conditioning behavior changes.

### `trellis/models/sparse_structure_flow.py`

What it contains: Dense rectified-flow transformer for sparse-structure latents.

Role in TRELLIS: Predicts flow velocity for a dense 3D latent grid, conditioned by text or image tokens through cross-attention. This is the model that generates the latent decoded into occupancy.

Read it for: Dense 3D patchification, timestep embedding, positional embedding, DiT-style adaptive layer norm, and cross-attention conditioning.

Architecture modification relevance: Very high. This is the main file for changing structure-generation architecture.

### `trellis/models/sparse_structure_vae.py`

What it contains: 3D convolutional encoder and decoder for sparse structure.

Role in TRELLIS: Encodes binary occupancy grids into lower-resolution structure latents during training and decodes generated structure latents back to occupancy logits during inference.

Read it for: Structure latent resolution, convolutional down/up-sampling, posterior sampling, and the threshold that ultimately creates sparse coordinates.

Architecture modification relevance: Very high for changing the structure representation, voxel resolution, or internal/external occupancy target.

### `trellis/models/structured_latent_flow.py`

What it contains: Sparse rectified-flow transformer for SLAT features.

Role in TRELLIS: Predicts flow velocity for per-voxel latent features on the sparse coordinates generated by the structure stage.

Read it for: Sparse `SparseTensor` inputs, per-coordinate features, sparse residual IO blocks, sparse transformer cross-attention, and the SLAT sampling architecture.

Architecture modification relevance: Very high. This is the main file for changing latent generation on occupied structure.

### `trellis/models/structured_latent_vae/base.py`

What it contains: Shared sparse transformer base for SLAT VAE encoder and decoders.

Role in TRELLIS: Provides the sparse transformer torso used by the SLAT encoder and the mesh/Gaussian/radiance decoders.

Read it for: Sparse positional encoding and attention mode choices such as full, shifted, serialized, and Swin-style windowed attention.

Architecture modification relevance: High. Modify here for shared SLAT VAE transformer architecture changes.

### `trellis/models/structured_latent_vae/encoder.py`

What it contains: SLAT VAE encoder.

Role in TRELLIS: Encodes sparse features from training data into posterior latent features at sparse coordinates.

Read it for: How sparse feature posteriors are parameterized as mean/log-variance per occupied coordinate.

Architecture modification relevance: High when changing the SLAT latent space or training encoder.

### `trellis/models/structured_latent_vae/decoder_mesh.py`

What it contains: SLAT-to-mesh decoder.

Role in TRELLIS: Transforms SLAT features, subdivides sparse voxels, predicts mesh extraction features, and calls `SparseFeatures2Mesh`.

Read it for: Mesh-specific decoding, sparse subdivision, and where the mesh representation is produced.

Architecture modification relevance: High for mesh output changes, surface extraction behavior, or extra per-surface attributes.

### `trellis/models/structured_latent_vae/decoder_gs.py`

What it contains: SLAT-to-3D-Gaussian decoder.

Role in TRELLIS: Converts each occupied sparse voxel into one or more Gaussian primitives with offsets, color, scale, rotation, and opacity.

Read it for: The Gaussian parameter layout and how voxel coordinates become world-space Gaussian centers.

Architecture modification relevance: High for Gaussian output changes and medium for upstream latent changes.

### `trellis/models/structured_latent_vae/decoder_rf.py`

What it contains: SLAT-to-radiance-field decoder.

Role in TRELLIS: Converts sparse latent features into a `Strivec` radiance-field representation.

Read it for: Radiance-field parameter layout, per-voxel feature unpacking, and `Strivec` construction.

Architecture modification relevance: High for radiance-field output changes.

### `trellis/modules/sparse/basic.py`

What it contains: TRELLIS's sparse tensor wrapper around TorchSparse or spconv.

Role in TRELLIS: Stores sparse coordinates and features, tracks batch layouts, preserves backend caches, and makes sparse tensors behave like model tensors.

Read it for: How sparse data is represented: `coords` are `[num_points, 4]` as `(batch, x, y, z)`, `feats` are `[num_points, channels]` or similar per-point feature shapes, and `layout` maps each batch item to a contiguous slice.

Architecture modification relevance: Medium. Critical to understand before modifying sparse models, but low-level enough that changes should be cautious.

### `trellis/modules/sparse/transformer/modulated.py`

What it contains: Sparse transformer blocks with adaptive layer norm and optional cross-attention.

Role in TRELLIS: Defines where timestep modulation and conditioning tokens enter the sparse SLAT flow model.

Read it for: Sparse self-attention, cross-attention to prompt/image context, and AdaLN-style modulation.

Architecture modification relevance: Very high for new conditioning mechanisms or transformer block changes.

### `trellis/modules/transformer/modulated.py`

What it contains: Dense transformer blocks with adaptive layer norm and optional cross-attention.

Role in TRELLIS: Defines where timestep modulation and conditioning tokens enter the dense structure flow model.

Read it for: The dense counterpart to the sparse transformer blocks.

Architecture modification relevance: Very high for structure-flow conditioning or DiT block changes.

### `trellis/modules/sparse/attention/`

What it contains: Sparse attention implementations: full, serialized, and windowed attention.

Role in TRELLIS: Supplies the attention kernels/modes used by sparse transformers.

Read it for: Scaling behavior, sparse attention windows, serialization order, and memory/performance details.

Architecture modification relevance: Medium to high. Important for attention changes, but more implementation-sensitive than model wiring.

### `trellis/modules/sparse/conv/`

What it contains: Sparse convolution wrappers for TorchSparse/spconv backends.

Role in TRELLIS: Provides sparse convolution operations used in sparse residual, subdivision, and decoder blocks.

Read it for: Backend behavior and sparse convolution construction.

Architecture modification relevance: Medium. Usually a utility layer unless adding new sparse convolution operators.

### `trellis/modules/spatial.py`

What it contains: Dense spatial utilities such as 3D patchify/unpatchify and 3D pixel shuffle.

Role in TRELLIS: Used by the structure flow model and structure VAE to move between grids, patches, and upsampled tensors.

Read it for: Tensor reshaping around dense voxel grids.

Architecture modification relevance: Medium. Relevant when changing patch size or dense 3D resolution flow.

### `trellis/trainers/flow_matching/flow_matching.py`

What it contains: Dense flow-matching training objective.

Role in TRELLIS: Trains dense flow denoisers by sampling `x_t` between data and noise and supervising predicted velocity.

Read it for: Rectified-flow objective, timestep sampling, velocity target, and snapshot sampling.

Architecture modification relevance: High when changing structure-flow training.

### `trellis/trainers/flow_matching/sparse_flow_matching.py`

What it contains: Sparse flow-matching training objective.

Role in TRELLIS: Same objective as the dense trainer, but operating on `SparseTensor.feats` while preserving sparse coordinates.

Read it for: How SLAT flow training differs from dense structure flow training.

Architecture modification relevance: High when changing SLAT-flow training.

### `trellis/trainers/flow_matching/mixins/text_conditioned.py`

What it contains: Text conditioning for training.

Role in TRELLIS: Encodes prompt strings with CLIP text and supplies null conditioning for CFG.

Read it for: How text tokens enter training.

Architecture modification relevance: High for alternative language encoders or additional text-conditioning paths.

### `trellis/trainers/flow_matching/mixins/image_conditioned.py`

What it contains: Image conditioning for training.

Role in TRELLIS: Encodes images with DINOv2 and supplies zero negative conditioning for CFG.

Read it for: How image tokens enter training.

Architecture modification relevance: High for alternative image encoders or multi-view/image-conditioning changes.

### `trellis/trainers/flow_matching/mixins/classifier_free_guidance.py`

What it contains: Classifier-free guidance conditioning dropout for training.

Role in TRELLIS: Randomly replaces conditioning with negative conditioning so the same model can be guided at inference.

Read it for: How CFG training data is formed.

Architecture modification relevance: Medium to high if changing conditioning or guidance.

### `trellis/trainers/vae/sparse_structure_vae.py`

What it contains: Training loss for the sparse-structure VAE.

Role in TRELLIS: Reconstructs occupancy and applies KL regularization to the structure latent.

Read it for: Structure occupancy loss and how the VAE is optimized.

Architecture modification relevance: High for internal-structure targets, occupancy supervision, or latent regularization.

### `trellis/trainers/vae/structured_latent_vae_*.py`

What it contains: Training logic for the SLAT VAE and its output-specific decoders.

Role in TRELLIS: Optimizes sparse latent encoders/decoders for Gaussian, mesh, and radiance-field outputs.

Read it for: Output-specific reconstruction losses and decoder training.

Architecture modification relevance: High for changing output representations or adding internal attributes.

### `configs/generation/*.json`

What it contains: Model names, architectural hyperparameters, pretrained components, and sampler arguments for generation models.

Role in TRELLIS: Defines the structure flow and SLAT flow variants for text/image conditioning.

Read it for: Actual resolution, channel counts, block counts, patch sizes, conditioning dimensions, sampler settings, and CFG strengths.

Architecture modification relevance: Very high. Most architecture edits require matching config edits.

### `configs/vae/*.json`

What it contains: Model configs for structure VAE and SLAT VAE decoders.

Role in TRELLIS: Defines VAE latent channels, decoder types, output representation settings, and sparse transformer architecture.

Read it for: The concrete VAE/decoder architecture used by checkpoints.

Architecture modification relevance: Very high.

### `trellis/datasets/`

What it contains: Dataset classes for sparse structures, sparse structure latents, SLAT features, and render supervision.

Role in TRELLIS: Supplies training tensors for the VAE and flow trainers.

Read it for: What data fields are expected by each trainer, especially `ss`, `x_0`, `cond`, sparse coordinates, and latent features.

Architecture modification relevance: High if changing targets, adding internal occupancy, or adding new conditioning data.

### `dataset_toolkits/`

What it contains: Offline preprocessing scripts for rendering, voxelizing, feature extraction, and latent encoding.

Role in TRELLIS: Builds the datasets consumed by trainers.

Read it for: How meshes/images are converted into training artifacts.

Architecture modification relevance: High when changing training targets or adding internal structure supervision.

### `trellis/representations/`

What it contains: Output representation classes for meshes, Gaussians, octrees, and radiance fields.

Role in TRELLIS: Defines the objects returned by decoders and renderers.

Read it for: Output semantics after SLAT decoding.

Architecture modification relevance: Medium to high for output-format changes.

### `trellis/renderers/`

What it contains: Renderers for mesh, Gaussian, and octree/radiance-field outputs.

Role in TRELLIS: Converts decoded representations into images/videos for visualization and losses.

Read it for: Rendering and evaluation behavior, not the core generative architecture.

Architecture modification relevance: Medium. Relevant when changing representation outputs or training render losses.

### `train.py`

What it contains: Entry point for training from config files.

Role in TRELLIS: Instantiates dataset, models, trainers, and distributed training setup.

Read it for: How training configs become actual model/trainer objects.

Architecture modification relevance: Medium. Mostly orchestration unless adding new trainer/model config conventions.

## Where Conditioning Enters

Conditioning is created in:

- `trellis/pipelines/trellis_image_to_3d.py`: DINOv2 patch tokens for image inference.
- `trellis/pipelines/trellis_text_to_3d.py`: CLIP text hidden states for text inference.
- `trellis/trainers/flow_matching/mixins/image_conditioned.py`: DINOv2 tokens for image-conditioned training.
- `trellis/trainers/flow_matching/mixins/text_conditioned.py`: CLIP text tokens for text-conditioned training.

Conditioning is injected into the denoisers through cross-attention in:

- `trellis/modules/transformer/modulated.py`: dense structure flow.
- `trellis/modules/sparse/transformer/modulated.py`: sparse SLAT flow.

Timestep conditioning enters through AdaLN-style modulation in the same transformer blocks. The timestep embedding is created in `trellis/models/sparse_structure_flow.py` and reused by the SLAT flow model.

## Where To Modify Architecture

For structure generation, start with:

- `trellis/models/sparse_structure_flow.py`
- `trellis/models/sparse_structure_vae.py`
- `configs/generation/ss_flow_*.json`
- `configs/vae/ss_vae_*.json`
- `trellis/trainers/flow_matching/flow_matching.py`
- `trellis/trainers/vae/sparse_structure_vae.py`

For SLAT latent generation, start with:

- `trellis/models/structured_latent_flow.py`
- `trellis/models/structured_latent_vae/base.py`
- `trellis/models/structured_latent_vae/encoder.py`
- `configs/generation/slat_flow_*.json`
- `configs/vae/slat_vae_*.json`
- `trellis/trainers/flow_matching/sparse_flow_matching.py`

For conditioning changes, start with:

- `trellis/pipelines/trellis_image_to_3d.py`
- `trellis/pipelines/trellis_text_to_3d.py`
- `trellis/trainers/flow_matching/mixins/image_conditioned.py`
- `trellis/trainers/flow_matching/mixins/text_conditioned.py`
- `trellis/modules/transformer/modulated.py`
- `trellis/modules/sparse/transformer/modulated.py`

For output representation changes, start with:

- `trellis/models/structured_latent_vae/decoder_mesh.py`
- `trellis/models/structured_latent_vae/decoder_gs.py`
- `trellis/models/structured_latent_vae/decoder_rf.py`
- `trellis/representations/`
- `trellis/renderers/`

## Notes For Predicting Internal Structure

Current TRELLIS generation is oriented around visible/external 3D assets: a sparse occupancy-like structure stage followed by surface/radiance/Gaussian decoding. To make TRELLIS predict internal structure, the most likely changes are:

- Change the structure target in preprocessing and datasets so occupancy labels include interior semantics or volumetric internal geometry, not only surface-derived occupancy.
- Modify `trellis/models/sparse_structure_vae.py` if the structure latent needs more channels, semantic labels, signed distance, density, or multi-class internal occupancy instead of binary occupancy logits.
- Modify `trellis/trainers/vae/sparse_structure_vae.py` to replace binary reconstruction loss with losses appropriate for internal structure, such as multi-class cross entropy, SDF regression, density regression, or multi-head losses.
- Modify `trellis/models/sparse_structure_flow.py` and `configs/generation/ss_flow_*.json` if the structure latent dimensionality or resolution changes.
- Modify `trellis/models/structured_latent_flow.py` if SLAT features must be generated for interior voxels as well as exterior/surface-adjacent voxels.
- Modify `trellis/models/structured_latent_vae/decoder_*` or add a new decoder if the internal structure is an output in its own right rather than only a support for external mesh/Gaussian/radiance decoding.
- Update `dataset_toolkits/voxelize.py`, `dataset_toolkits/encode_ss_latent.py`, `dataset_toolkits/encode_latent.py`, and the relevant `trellis/datasets/` classes so training data contains the new internal targets and coordinates.
