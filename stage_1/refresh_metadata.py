from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import (
    DEFAULT_DATASET_DIR,
    FEATURE_MODEL,
    SLAT_LATENT_MODEL,
    SS_LATENT_MODEL,
    artifact_exists,
    read_metadata,
    read_ply_points,
    write_metadata,
)


def parse_args() -> argparse.Namespace:
    """Parse the dataset directory whose artifact flags should be refreshed."""
    parser = argparse.ArgumentParser(description="Refresh ShapeNetInternals_small metadata artifact flags.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATASET_DIR)
    return parser.parse_args()


def main() -> None:
    """Scan the converted dataset and update artifact availability columns.

    Each preprocessing stage writes files into a known TRELLIS folder layout.
    This function checks those locations for every sample id, records whether
    the artifact exists and is non-empty, and writes both the refreshed
    ``metadata.csv`` and a compact ``statistics_stage1.txt`` summary.
    """
    args = parse_args()
    metadata = read_metadata(args.output_dir)

    # Older or partially generated metadata files may not have every Stage 1
    # flag column yet.  Add missing columns before assigning refreshed values.
    for col, default in [
        ("rendered", False),
        ("voxelized", False),
        ("cond_rendered", False),
        (f"feature_{FEATURE_MODEL}", False),
        (f"ss_latent_{SS_LATENT_MODEL}", False),
        (f"latent_{SLAT_LATENT_MODEL}", False),
        ("num_voxels", 0),
    ]:
        if col not in metadata.columns:
            metadata[col] = default

    rendered = []
    cond_rendered = []
    voxelized = []
    num_voxels = []
    features = []
    ss_latents = []
    latents = []
    for _, row in metadata.iterrows():
        sample_id = row["sha256"]
        # Render completion is represented by transforms.json; it is the file
        # downstream TRELLIS tools need in order to interpret the view images.
        rendered.append(artifact_exists(args.output_dir / "renders" / sample_id / "transforms.json"))
        cond_rendered.append(artifact_exists(args.output_dir / "renders_cond" / sample_id / "transforms.json"))
        voxel_path = args.output_dir / "voxels" / f"{sample_id}.ply"
        has_voxels = artifact_exists(voxel_path)
        voxelized.append(has_voxels)
        if has_voxels:
            try:
                # Counting PLY vertices gives the number of occupied target
                # voxels because Stage 1 writes exactly one point per voxel.
                num_voxels.append(int(read_ply_points(voxel_path).shape[0]))
            except Exception:
                num_voxels.append(0)
        else:
            num_voxels.append(0)
        features.append(artifact_exists(args.output_dir / "features" / FEATURE_MODEL / f"{sample_id}.npz"))
        ss_latents.append(artifact_exists(args.output_dir / "ss_latents" / SS_LATENT_MODEL / f"{sample_id}.npz"))
        latents.append(artifact_exists(args.output_dir / "latents" / SLAT_LATENT_MODEL / f"{sample_id}.npz"))

    metadata["rendered"] = rendered
    metadata["cond_rendered"] = cond_rendered
    metadata["voxelized"] = voxelized
    metadata["num_voxels"] = np.asarray(num_voxels, dtype=np.int64)
    metadata[f"feature_{FEATURE_MODEL}"] = features
    metadata[f"ss_latent_{SS_LATENT_MODEL}"] = ss_latents
    metadata[f"latent_{SLAT_LATENT_MODEL}"] = latents
    write_metadata(args.output_dir, metadata)

    # The text summary is deliberately simple so it can be read quickly after a
    # long pipeline run or included in experiment notes.
    stats = {
        "samples": len(metadata),
        "rendered": int(np.sum(rendered)),
        "cond_rendered": int(np.sum(cond_rendered)),
        "voxelized": int(np.sum(voxelized)),
        f"feature_{FEATURE_MODEL}": int(np.sum(features)),
        f"ss_latent_{SS_LATENT_MODEL}": int(np.sum(ss_latents)),
        f"latent_{SLAT_LATENT_MODEL}": int(np.sum(latents)),
    }
    lines = ["Stage 1 metadata statistics:"]
    lines.extend(f"  - {key}: {value}" for key, value in stats.items())
    text = "\n".join(lines) + "\n"
    (args.output_dir / "statistics_stage1.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
