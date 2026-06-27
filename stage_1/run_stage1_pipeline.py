from __future__ import annotations

import argparse
import os
from pathlib import Path

from common import (
    DEFAULT_DATASET_DIR,
    DEFAULT_SHAPENET_ROOT,
    REPO_ROOT,
    ensure_dir,
    python_cmd,
    run_command,
)


def parse_args() -> argparse.Namespace:
    """Parse options for the full Stage 1 preprocessing wrapper.

    The flags let you rerun only the parts you need: metadata creation, voxel
    conversion, rendering, feature extraction, and latent encoding can be
    skipped or overwritten independently.  This matters because rendering and
    encoding are expensive compared with CSV/voxel bookkeeping.
    """
    parser = argparse.ArgumentParser(description="Run Phase 1 ShapeNet -> TRELLIS small dataset conversion.")
    parser.add_argument("--shapenet-root", type=Path, default=DEFAULT_SHAPENET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--per-category", type=int, default=3)
    parser.add_argument("--render-views", type=int, default=8)
    parser.add_argument("--cond-views", type=int, default=1)
    parser.add_argument("--render-workers", type=int, default=1)
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--overwrite-metadata", action="store_true")
    parser.add_argument("--overwrite-voxels", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--skip-latents", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


def refresh(output_dir: Path, log_path: Path, keep_going: bool) -> None:
    """Run ``refresh_metadata.py`` after a preprocessing stage finishes.

    TRELLIS tools write artifacts to disk but do not know about our Stage 1
    metadata flags.  Refreshing between stages keeps ``metadata.csv`` truthful
    and makes the wrapper restartable after partial failures.
    """
    run_command(
        python_cmd(REPO_ROOT / "stage_1" / "refresh_metadata.py", "--output-dir", str(output_dir)),
        log_path=log_path,
        keep_going=keep_going,
    )


def main() -> None:
    """Execute the complete ShapeNet-to-TRELLIS Stage 1 conversion pipeline.

    The sequence mirrors the dependency graph: choose samples, build voxel
    targets, render conditioning and multiview images, extract DINO features,
    encode sparse-structure latents, encode SLAT latents, and refresh metadata
    after each major artifact-producing step.
    """
    args = parse_args()
    ensure_dir(args.output_dir)
    log_path = args.output_dir / "logs" / "stage1_pipeline.log"

    env = os.environ.copy()
    stage1_path = str(REPO_ROOT / "stage_1")
    # The render scripts import a dataset adapter by module name.  Adding
    # stage_1 to PYTHONPATH lets those scripts see our Stage 1 adapter package
    # when it is present in the checkout.
    env["PYTHONPATH"] = stage1_path + os.pathsep + env.get("PYTHONPATH", "")

    # Step 1: build metadata.csv and split files from raw ShapeNet folders.
    prepare_cmd = python_cmd(
        REPO_ROOT / "stage_1" / "prepare_subset.py",
        "--shapenet-root", str(args.shapenet_root),
        "--output-dir", str(args.output_dir),
        "--per-category", str(args.per_category),
    )
    if args.overwrite_metadata:
        prepare_cmd.append("--overwrite")
    run_command(prepare_cmd, env=env, log_path=log_path, keep_going=args.keep_going)

    # Step 2: convert the surface binvox grids into the canonical sparse voxel
    # targets used by TRELLIS' sparse-structure VAE.
    voxel_cmd = python_cmd(
        REPO_ROOT / "stage_1" / "convert_binvox_to_voxels.py",
        "--output-dir", str(args.output_dir),
        "--kind", "surface",
        "--resolution", "64",
    )
    if args.overwrite_voxels:
        voxel_cmd.append("--overwrite")
    run_command(voxel_cmd, env=env, log_path=log_path, keep_going=args.keep_going)

    # Optional ablation: also convert solid occupancy binvox grids, but keep
    # them separate from the main surface-voxel training target.
    solid_cmd = python_cmd(
        REPO_ROOT / "stage_1" / "convert_binvox_to_voxels.py",
        "--output-dir", str(args.output_dir),
        "--kind", "solid",
        "--resolution", "64",
        "--out-folder", "voxels_solid",
    )
    if args.overwrite_voxels:
        solid_cmd.append("--overwrite")
    run_command(solid_cmd, env=env, log_path=log_path, keep_going=args.keep_going)
    refresh(args.output_dir, log_path, args.keep_going)

    if not args.skip_render:
        # Conditioning renders are the input views for image-conditioned
        # generation.  Stage 1 keeps this small by default: one view per object.
        run_command(
            python_cmd(
                REPO_ROOT / "dataset_toolkits" / "render_cond.py",
                "ShapeNetInternalsSmall",
                "--output_dir", str(args.output_dir),
                "--num_views", str(args.cond_views),
                "--max_workers", str(args.render_workers),
            ),
            env=env,
            log_path=log_path,
            keep_going=args.keep_going,
        )
        refresh(args.output_dir, log_path, args.keep_going)

        # Multiview renders feed DINO feature extraction and latent encoding.
        run_command(
            python_cmd(
                REPO_ROOT / "dataset_toolkits" / "render.py",
                "ShapeNetInternalsSmall",
                "--output_dir", str(args.output_dir),
                "--num_views", str(args.render_views),
                "--max_workers", str(args.render_workers),
            ),
            env=env,
            log_path=log_path,
            keep_going=args.keep_going,
        )
        refresh(args.output_dir, log_path, args.keep_going)

    if not args.skip_features:
        # DINOv2 image features are projected onto the sparse 3D structure by
        # TRELLIS' existing feature extraction script.
        run_command(
            python_cmd(
                REPO_ROOT / "dataset_toolkits" / "extract_feature.py",
                "--output_dir", str(args.output_dir),
                "--batch_size", str(args.feature_batch_size),
            ),
            env=env,
            log_path=log_path,
            keep_going=args.keep_going,
        )
        refresh(args.output_dir, log_path, args.keep_going)

    if not args.skip_latents:
        # Sparse-structure latents are the supervision target for the first
        # TRELLIS flow model.
        run_command(
            python_cmd(
                REPO_ROOT / "dataset_toolkits" / "encode_ss_latent.py",
                "--output_dir", str(args.output_dir),
            ),
            env=env,
            log_path=log_path,
            keep_going=args.keep_going,
        )
        refresh(args.output_dir, log_path, args.keep_going)

        # SLAT latents are the supervision target for the second-stage latent
        # flow and for mesh decoder reconstruction checks.
        run_command(
            python_cmd(
                REPO_ROOT / "dataset_toolkits" / "encode_latent.py",
                "--output_dir", str(args.output_dir),
            ),
            env=env,
            log_path=log_path,
            keep_going=args.keep_going,
        )
        refresh(args.output_dir, log_path, args.keep_going)

    print(f"Stage 1 conversion pipeline finished. See {log_path}")


if __name__ == "__main__":
    main()
