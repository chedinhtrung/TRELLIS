from __future__ import annotations

import argparse
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
    ensure_dir(args.output_dir / "recon_meshes")

    metadata = read_metadata(args.dataset_dir)
    if "voxelized" in metadata.columns:
        metadata = metadata[metadata["voxelized"] == True].copy()
    if "source_obj" not in metadata.columns:
        raise RuntimeError("metadata.csv must contain a source_obj column.")

    selected = []
    latent_dir = args.dataset_dir / "latents" / args.slat_latent_model
    for _, row in metadata.iterrows():
        sample_id = row["sha256"]
        source_obj = Path(str(row.get("source_obj", "")))
        latent_path = latent_dir / f"{sample_id}.npz"
        if source_obj.exists() and artifact_exists(latent_path):
            selected.append(row)
    if args.limit:
        selected = selected[:args.limit]
    if not selected:
        raise RuntimeError("No samples with source_obj and cached SLAT latents were found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("TRELLIS pretrained mesh decoder requires CUDA in this script.")

    decoder = models.from_pretrained(args.mesh_decoder).eval().to(device)

    rows = []
    for index, row in enumerate(selected):
        sample_id = row["sha256"]
        source_obj = Path(str(row["source_obj"]))
        mesh_path = args.output_dir / "recon_meshes" / f"{sample_id}.ply"

        print(f"[{index + 1}/{len(selected)}] Decoding {sample_id}", flush=True)
        if mesh_path.exists() and not args.overwrite_meshes:
            pred_mesh = load_mesh(mesh_path)
        else:
            # The cached SLAT stores sparse coordinates and features. The mesh
            # decoder turns that sparse latent directly into vertices and faces.
            slat = load_slat(sample_id, args.dataset_dir, args.slat_latent_model, device)
            with torch.no_grad():
                decoded = decoder(slat)[0]
            if not decoded.success:
                raise RuntimeError(f"Mesh decoder produced an empty mesh for {sample_id}")

            vertices = decoded.vertices.detach().cpu().numpy().astype(np.float32)
            faces = decoded.faces.detach().cpu().numpy().astype(np.int64)
            export_mesh(mesh_path, vertices, faces)
            pred_mesh = make_mesh(vertices, faces)
            torch.cuda.empty_cache()

        gt_mesh = load_mesh(source_obj)

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
