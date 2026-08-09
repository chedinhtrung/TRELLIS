#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SHAPENET_ROOT="${SHAPENET_ROOT:-$REPO_ROOT/ShapeNet}"
SHAPENET_PROCESSED="${SHAPENET_PROCESSED:-$REPO_ROOT/datasets/ShapeNetTRELLIS_full}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MAX_WORKERS="${MAX_WORKERS:-1}"

export SPCONV_ALGO="${SPCONV_ALGO:-native}"
export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"

cd "$REPO_ROOT/dataset_toolkits"

echo "[data-preparation] Preparing all ShapeNet objects in the four project categories"
"$PYTHON_BIN" shapenet/shapenet_to_trellis_raw.py \
    --shapenet-root "$SHAPENET_ROOT" \
    --categories car bus file_cabinet cabinet \
    --outdir "$SHAPENET_PROCESSED"

for split in train val test; do
    split_dir="$SHAPENET_PROCESSED/$split"
    echo "[data-preparation] Rendering 40 ordinary and 40 conditioning views for $split"

    "$PYTHON_BIN" render_kiui.py ShapeNet \
        --output_dir "$split_dir" \
        --num_views 40 \
        --resolution 512 \
        --max_workers "$MAX_WORKERS" &
    render_pid=$!

    "$PYTHON_BIN" render_cond_kiui.py ShapeNet \
        --output_dir "$split_dir" \
        --num_views 40 \
        --resolution 512 \
        --max_workers "$MAX_WORKERS" &
    cond_render_pid=$!

    wait "$render_pid"
    wait "$cond_render_pid"

    echo "[data-preparation] Updating metadata for $split"
    "$PYTHON_BIN" shapenet/ensure_metadata_compliance.py \
        --metadata "$split_dir/metadata.csv"
    "$PYTHON_BIN" build_metadata.py ShapeNet \
        --output_dir "$split_dir"

    echo "[data-preparation] Voxelizing $split"
    "$PYTHON_BIN" voxelize.py ShapeNet \
        --output_dir "$split_dir"
    "$PYTHON_BIN" build_metadata.py ShapeNet \
        --output_dir "$split_dir"

    echo "[data-preparation] Extracting DINOv2 features for $split"
    "$PYTHON_BIN" extract_feature.py \
        --output_dir "$split_dir"
    "$PYTHON_BIN" build_metadata.py ShapeNet \
        --output_dir "$split_dir"

    echo "[data-preparation] Encoding SS and SLAT latents for $split"
    "$PYTHON_BIN" encode_ss_latent.py \
        --output_dir "$split_dir" &
    ss_latent_pid=$!

    "$PYTHON_BIN" encode_latent.py \
        --output_dir "$split_dir" &
    slat_latent_pid=$!

    wait "$ss_latent_pid"
    wait "$slat_latent_pid"

    "$PYTHON_BIN" build_metadata.py ShapeNet \
        --output_dir "$split_dir"
done

echo "[data-preparation] ShapeNet preparation complete: $SHAPENET_PROCESSED"
