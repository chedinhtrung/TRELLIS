set -euo pipefail
trap 'status=$?; echo "[stage1] Pipeline failed with exit code $status"; read -r -p "Press Enter to exit..."' ERR

REPO_ROOT="/workspace/TRELLIS"
SHAPENET_PROCESSED="$REPO_ROOT/datasets/ShapeNetTRELLIS_nano"
export SPCONV_ALGO="${SPCONV_ALGO:-native}"

cd "$REPO_ROOT/dataset_toolkits"
echo "[stage1] Preparing ShapeNetTRELLIS_nano subset"
# simlink ShapeNet to ShapeNetTRELLIS_nano
python shapenet/shapenet_to_trellis_raw.py \
    --shapenet-root "$REPO_ROOT/datasets/ShapeNet" \
    --categories car bus file_cabinet cabinet \
    --limit 3 \
    --outdir "$SHAPENET_PROCESSED"

echo "[stage1] Starting render and render_cond in parallel"
# run render and render_cond in parallel
# this is to create multiview images and conditional images
(
    python render.py ShapeNet \
        --output_dir "$SHAPENET_PROCESSED" \
        --num_views 150 \
        --max_workers 6 &

    python render_cond.py ShapeNet \
        --output_dir "$SHAPENET_PROCESSED" \
        --num_views 150 \
        --max_workers 6 &

    wait  # for the render and render_cond to finish before continue
)

# we dont train the VAEs again, only the flow matching
# so encode anyway to save time