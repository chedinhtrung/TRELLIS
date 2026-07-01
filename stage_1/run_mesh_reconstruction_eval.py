from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import trimesh
from scipy.spatial import cKDTree

from common import (
    DEFAULT_DATASET_DIR,
    REPO_ROOT,
    SLAT_LATENT_MODEL,
    artifact_exists,
    ensure_dir,
    read_metadata,
)

sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("SPCONV_ALGO", "native")

import trellis.models as models
import trellis.modules.sparse as sp


DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "shapenet_mesh_reconstruction"
DEFAULT_MESH_DECODER = "microsoft/TRELLIS-image-large/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16"


def parse_args() -> argparse.Namespace:
    """Parse options for SLAT-to-mesh reconstruction evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate TRELLIS SLAT latent -> mesh reconstruction on ShapeNet objects.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of objects; 0 means all available objects.")
    parser.add_argument("--slat-latent-model", default=SLAT_LATENT_MODEL)
    parser.add_argument("--mesh-decoder", default=DEFAULT_MESH_DECODER)
    parser.add_argument("--mesh-sample-points", type=int, default=50000)
    parser.add_argument("--fscore-threshold", type=float, default=0.01)
    parser.add_argument("--overwrite-meshes", action="store_true")
    return parser.parse_args()


def load_slat(sample_id: str, dataset_dir: Path, latent_model: str, device: torch.device) -> sp.SparseTensor:
    """Load a cached SLAT latent and add the batch column TRELLIS expects."""
    path = dataset_dir / "latents" / latent_model / f"{sample_id}.npz"
    data = np.load(path)
    coords = torch.cat([
        torch.zeros(data["coords"].shape[0], 1, dtype=torch.int32),
        torch.from_numpy(data["coords"]).int(),
    ], dim=1).to(device)
    feats = torch.from_numpy(data["feats"]).float().to(device)
    return sp.SparseTensor(coords=coords, feats=feats)


def validate_slat(sample_id: str, slat: sp.SparseTensor, resolution: int = 64) -> None:
    """Print SLAT stats and fail early on values likely to crash native decoders."""
    coords = slat.coords
    feats = slat.feats
    print(
        f"  SLAT coords={tuple(coords.shape)} feats={tuple(feats.shape)} "
        f"coord_dtype={coords.dtype} feat_dtype={feats.dtype}",
        flush=True,
    )

    problems = []
    if coords.numel() == 0:
        problems.append("empty coords")
    if feats.numel() == 0:
        problems.append("empty feats")
    if coords.ndim != 2 or coords.shape[1] != 4:
        problems.append(f"expected coords shape [N, 4], got {tuple(coords.shape)}")
    if feats.ndim != 2:
        problems.append(f"expected feats shape [N, C], got {tuple(feats.shape)}")
    if coords.shape[0] != feats.shape[0]:
        problems.append(f"coords/feats row mismatch: {coords.shape[0]} != {feats.shape[0]}")

    if coords.numel() > 0:
        coords_cpu = coords.detach().cpu()
        spatial = coords_cpu[:, 1:] if coords_cpu.ndim == 2 and coords_cpu.shape[1] >= 4 else coords_cpu
        print(
            f"  coord range: batch=[{int(coords_cpu[:, 0].min())}, {int(coords_cpu[:, 0].max())}] "
            f"xyz=[{int(spatial.min())}, {int(spatial.max())}]",
            flush=True,
        )
        if coords_cpu.ndim == 2 and coords_cpu.shape[1] == 4:
            if int(coords_cpu[:, 0].min()) != 0 or int(coords_cpu[:, 0].max()) != 0:
                problems.append("expected all batch coords to be 0 for single-sample decode")
            if int(spatial.min()) < 0 or int(spatial.max()) >= resolution:
                problems.append(f"spatial coords outside [0, {resolution - 1}]")
            unique_coords = torch.unique(coords_cpu, dim=0).shape[0]
            if unique_coords != coords_cpu.shape[0]:
                problems.append(f"duplicate coords: {coords_cpu.shape[0] - unique_coords}")

    if feats.numel() > 0:
        finite = torch.isfinite(feats)
        finite_count = int(finite.sum().item())
        total_count = feats.numel()
        if finite_count:
            finite_feats = feats[finite]
            print(
                f"  feat range: min={float(finite_feats.min()):.6g} "
                f"max={float(finite_feats.max()):.6g} mean={float(finite_feats.mean()):.6g} "
                f"finite={finite_count}/{total_count}",
                flush=True,
            )
        else:
            print(f"  feat range: no finite values finite=0/{total_count}", flush=True)
        if finite_count != total_count:
            problems.append(f"non-finite feats: {total_count - finite_count}")

    if problems:
        message = "; ".join(problems)
        print(f"  BAD SLAT {sample_id}: {message}", flush=True)
        raise RuntimeError(f"Invalid SLAT for {sample_id}: {message}")


def rendered_mesh_path(dataset_dir: Path, sample_id: str) -> Path:
    """Return the TRELLIS-normalized mesh produced by dataset_toolkits/render.py."""
    return dataset_dir / "renders" / sample_id / "mesh.ply"


def load_mesh(path: Path) -> trimesh.Trimesh:
    """Load a mesh without processing so evaluation uses the original geometry."""
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
        raise ValueError(f"Empty mesh: {path}")
    return mesh


def make_mesh(vertices: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    """Create a triangle mesh from raw arrays without trimesh repair."""
    return trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )


def export_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Save vertices and faces as a triangle PLY mesh."""
    ensure_dir(path.parent)
    make_mesh(vertices, faces).export(path)


def sample_mesh_points(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    """Sample deterministic surface points for mesh-to-mesh metrics."""
    if count <= 0:
        return np.asarray(mesh.vertices, dtype=np.float32)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        points = mesh.sample(count)
    finally:
        np.random.set_state(state)
    if points.shape[0] == 0:
        points = np.asarray(mesh.vertices, dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def mesh_metrics(gt_points: np.ndarray, pred_points: np.ndarray, threshold: float) -> dict[str, float]:
    """Compute Chamfer, precision, recall, and F-score from sampled surfaces."""
    if gt_points.shape[0] == 0 or pred_points.shape[0] == 0:
        return {
            "chamfer_l1": np.nan,
            "chamfer_l2": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "fscore": np.nan,
        }

    gt_tree = cKDTree(gt_points)
    pred_tree = cKDTree(pred_points)
    pred_to_gt = gt_tree.query(pred_points, k=1, workers=-1)[0]
    gt_to_pred = pred_tree.query(gt_points, k=1, workers=-1)[0]

    precision = float((pred_to_gt <= threshold).mean())
    recall = float((gt_to_pred <= threshold).mean())
    fscore = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "chamfer_l1": float(pred_to_gt.mean() + gt_to_pred.mean()),
        "chamfer_l2": float(np.square(pred_to_gt).mean() + np.square(gt_to_pred).mean()),
        "precision": precision,
        "recall": recall,
        "fscore": float(fscore),
    }


def main() -> None:
    """Decode cached SLAT latents into meshes and write mesh metrics."""
    args = parse_args()
    ensure_dir(args.output_dir)
    grid_mesh_dir = args.output_dir / "recon_meshes_grid"
    ensure_dir(grid_mesh_dir)

    metadata = read_metadata(args.dataset_dir)
    if "voxelized" in metadata.columns:
        metadata = metadata[metadata["voxelized"] == True].copy()

    selected = []
    latent_dir = args.dataset_dir / "latents" / args.slat_latent_model
    for _, row in metadata.iterrows():
        sample_id = row["sha256"]
        latent_path = latent_dir / f"{sample_id}.npz"
        gt_mesh_path = rendered_mesh_path(args.dataset_dir, sample_id)
        if artifact_exists(latent_path) and artifact_exists(gt_mesh_path):
            selected.append(row)
    if args.limit:
        selected = selected[:args.limit]
    if not selected:
        raise RuntimeError("No samples with rendered mesh.ply files and cached SLAT latents were found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("TRELLIS pretrained mesh decoder requires CUDA in this script.")

    decoder = models.from_pretrained(args.mesh_decoder).eval().to(device)

    rows = []
    for index, row in enumerate(selected):
        sample_id = row["sha256"]
        gt_mesh_path = rendered_mesh_path(args.dataset_dir, sample_id)
        grid_mesh_path = grid_mesh_dir / f"{sample_id}.ply"

        print(f"[{index + 1}/{len(selected)}] Decoding {sample_id}", flush=True)
        if grid_mesh_path.exists() and not args.overwrite_meshes:
            pred_mesh = load_mesh(grid_mesh_path)
        else:
            slat = load_slat(sample_id, args.dataset_dir, args.slat_latent_model, device)
            validate_slat(sample_id, slat)
            with torch.no_grad():
                decoded = decoder(slat)[0]
            if not decoded.success:
                raise RuntimeError(f"Mesh decoder produced an empty mesh for {sample_id}")

            vertices_grid = decoded.vertices.detach().cpu().numpy().astype(np.float32)
            faces = decoded.faces.detach().cpu().numpy().astype(np.int64)

            export_mesh(grid_mesh_path, vertices_grid, faces)
            pred_mesh = make_mesh(vertices_grid, faces)
            torch.cuda.empty_cache()

        gt_mesh = load_mesh(gt_mesh_path)

        # Mesh metrics compare nearest-neighbor distances between sampled GT
        # and reconstructed surface points.
        gt_points = sample_mesh_points(gt_mesh, args.mesh_sample_points, seed=1000 + index * 17)
        pred_points = sample_mesh_points(pred_mesh, args.mesh_sample_points, seed=1001 + index * 17)
        sample_metrics = mesh_metrics(gt_points, pred_points, args.fscore_threshold)
        sample_metrics.update({
            "sha256": sample_id,
            "category": row.get("category", ""),
            "shapenet_id": row.get("shapenet_id", ""),
            "gt_vertices": int(gt_mesh.vertices.shape[0]),
            "gt_faces": int(gt_mesh.faces.shape[0]),
            "pred_vertices": int(pred_mesh.vertices.shape[0]),
            "pred_faces": int(pred_mesh.faces.shape[0]),
        })
        rows.append(sample_metrics)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(args.output_dir / "metrics.csv", index=False)
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
