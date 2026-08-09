#!/usr/bin/env python3
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
import utils3d

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval.run_unified_eval import evaluate_sample, read_categories, read_ids_file, read_mesh, read_metadata, select_ids


def trimesh_to_voxel_points(mesh, resolution: int) -> np.ndarray:
    vertices = np.clip(np.asarray(mesh.vertices, dtype=np.float32), -0.5 + 1e-6, 0.5 - 1e-6)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    o3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(faces),
    )
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        o3d_mesh,
        voxel_size=1 / resolution,
        min_bound=(-0.5, -0.5, -0.5),
        max_bound=(0.5, 0.5, 0.5),
    )
    coords = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()], dtype=np.float32)
    if len(coords) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return ((coords + 0.5) / resolution - 0.5).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pre-generated TRELLIS meshes with the same metrics as run_unified_eval.py.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--pred-mesh-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--outer-layers", type=int, default=3)
    parser.add_argument("--mesh-sample-points", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--samples-per-category", type=int, default=0)
    parser.add_argument("--ids-file", type=Path, default=None)
    args = parser.parse_args()

    if args.resolution <= 1:
        parser.error("--resolution must be > 1")
    if args.outer_layers < 0:
        parser.error("--outer-layers must be non-negative")
    if args.outer_layers * 2 >= args.resolution:
        parser.error("--outer-layers is too large for --resolution")
    if args.mesh_sample_points <= 0:
        parser.error("--mesh-sample-points must be positive")
    if args.samples_per_category < 0:
        parser.error("--samples-per-category must be non-negative")
    if not args.pred_mesh_dir.is_dir():
        raise FileNotFoundError(f"Prediction mesh directory not found: {args.pred_mesh_dir}")

    metadata_path = args.dataset_dir / "metadata.csv"
    rows = read_metadata(metadata_path)
    all_ids = rows["sha256"].astype(str).tolist()
    categories = read_categories(rows)

    if args.ids_file is not None:
        ids = read_ids_file(args.ids_file, all_ids)
    else:
        ids = select_ids(rows, args.samples_per_category)

    if args.limit is not None:
        ids = ids[:args.limit]
    if not ids:
        raise ValueError("No sample IDs selected for evaluation")

    metrics_dir = args.output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    per_sample_rows = []
    with tempfile.TemporaryDirectory(prefix="trellis_eval_pred_vox_") as tmpdir:
        tmp_voxel_dir = Path(tmpdir)

        for sample_id in ids:
            gt_voxel_path = args.dataset_dir / "voxels" / f"{sample_id}.ply"
            gt_mesh_path = args.dataset_dir / "renders" / sample_id / "mesh.ply"
            pred_mesh_path = args.pred_mesh_dir / f"{sample_id}.ply"
            pred_voxel_path = tmp_voxel_dir / f"{sample_id}.ply"

            if not gt_voxel_path.is_file():
                raise FileNotFoundError(f"Missing GT voxel file: {gt_voxel_path}")
            if not gt_mesh_path.is_file():
                raise FileNotFoundError(f"Missing GT mesh file: {gt_mesh_path}")
            if not pred_mesh_path.is_file():
                raise FileNotFoundError(f"Missing prediction mesh file: {pred_mesh_path}")

            pred_mesh = read_mesh(pred_mesh_path)
            pred_voxel_points = trimesh_to_voxel_points(pred_mesh, args.resolution)
            utils3d.io.write_ply(pred_voxel_path, pred_voxel_points)

            metrics = evaluate_sample(
                gt_voxel_path,
                pred_voxel_path,
                gt_mesh_path,
                pred_mesh_path,
                resolution=args.resolution,
                outer_layers=args.outer_layers,
                mesh_sample_points=args.mesh_sample_points,
                mesh_sample_seed=args.seed,
            )
            per_sample_rows.append({
                "sample_id": sample_id,
                "category": categories[sample_id],
                **metrics,
            })

    summary = {
        "n_samples": len(per_sample_rows),
        "internal_precision": float(np.mean([row["internal_precision"] for row in per_sample_rows])),
        "internal_recall": float(np.mean([row["internal_recall"] for row in per_sample_rows])),
        "internal_f1": float(np.mean([row["internal_f1"] for row in per_sample_rows])),
        "mesh_chamfer_l1": float(np.mean([row["mesh_chamfer_l1"] for row in per_sample_rows])),
    }

    per_sample_df = pd.DataFrame(per_sample_rows)
    per_sample_path = metrics_dir / "per_sample.csv"
    per_sample_df.to_csv(per_sample_path, index=False)

    summary_path = metrics_dir / "summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print("\nMesh-only unified evaluation finished")
    print(f"Prediction meshes: {args.pred_mesh_dir}")
    print(f"Per-sample metrics: {per_sample_path}")
    print(f"Summary metrics: {summary_path}")


if __name__ == "__main__":
    main()
