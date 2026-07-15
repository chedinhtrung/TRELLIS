set -euo pipefail
trap 'status=$?; echo "[stage1] Pipeline failed with exit code $status"; read -r -p "Press Enter to exit..."' ERR

REPO_ROOT="/workspace/TRELLIS"
SHAPENET_PROCESSED="$REPO_ROOT/datasets/ShapeNetTRELLIS_nano"
export SPCONV_ALGO="${SPCONV_ALGO:-native}"
export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"

cd "$REPO_ROOT/dataset_toolkits"
echo "[stage1] Preparing ShapeNetTRELLIS_nano subset"
# simlink ShapeNet to ShapeNetTRELLIS_nano
python shapenet/shapenet_to_trellis_raw.py \
    --shapenet-root "$REPO_ROOT/datasets/ShapeNet" \
    --categories car bus file_cabinet cabinet \
    --outdir "$SHAPENET_PROCESSED" \
    --limit 5

echo "[stage1] Rendering images"
for split in train val test; do
    echo "[stage1] Rendering $split split"

    python render_kiui.py ShapeNet \
        --output_dir "$SHAPENET_PROCESSED/$split" \
        --num_views 40 \
        --resolution 512 \
        --max_workers 1 & \

    python render_cond_kiui.py ShapeNet \
        --output_dir "$SHAPENET_PROCESSED/$split" \
        --num_views 40 \
        --resolution 512 \
        --max_workers 1 & 

    wait

    echo "[stage1] Ensuring metadata compliance"
    python shapenet/ensure_metadata_compliance.py \
        --metadata "$SHAPENET_PROCESSED/$split/metadata.csv"
    
    python build_metadata.py ShapeNet \
    --output_dir "$SHAPENET_PROCESSED/$split"

    echo "[stage1] Voxelizing $split split"
    python voxelize.py ShapeNet \
        --output_dir "$SHAPENET_PROCESSED/$split"
    
    python build_metadata.py ShapeNet \
    --output_dir "$SHAPENET_PROCESSED/$split"

    echo "[stage1] Extracting features for $split split"
    python extract_feature.py \
        --output_dir "$SHAPENET_PROCESSED/$split"

    python build_metadata.py ShapeNet \
        --output_dir "$SHAPENET_PROCESSED/$split"

    echo "[stage1] Encoding sparse-structure latents for $split split"
    python encode_ss_latent.py \
        --output_dir "$SHAPENET_PROCESSED/$split" &

    echo "[stage1] Encoding SLAT latents for $split split"
    python encode_latent.py \
        --output_dir "$SHAPENET_PROCESSED/$split" &
    
    wait
    python build_metadata.py ShapeNet \
        --output_dir "$SHAPENET_PROCESSED/$split"
done


# we dont train the VAEs again, only the flow matching
# so encode anyway to save time