#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
import torch
import trimesh
import utils3d
from PIL import Image
from scipy.spatial import cKDTree
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def read_metadata(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path)
    if rows.empty:
        raise ValueError(f"No samples found in {path}")
    required = {"sha256", "category"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Metadata missing required columns {missing}: {path}")
    return rows


def select_ids(rows: pd.DataFrame, samples_per_category: int) -> list[str]:
    if samples_per_category <= 0:
        return rows["sha256"].astype(str).tolist()
    return (
        rows.groupby("category", sort=False, group_keys=False)
        .head(samples_per_category)["sha256"]
        .astype(str)
        .tolist()
    )


def read_ids_file(path: Path, dataset_ids: list[str]) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    unknown = sorted(set(ids) - set(dataset_ids))
    if unknown:
        raise ValueError(f"IDs file contains samples absent from dataset metadata: {unknown}")
    return ids


def read_categories(rows: pd.DataFrame) -> dict[str, str]:
    return dict(zip(rows["sha256"].astype(str), rows["category"].astype(str)))


def _load_model_cfg_from_run(ckpt_path: Path, model_key: str) -> dict:
    config_path = ckpt_path.parents[1] / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"LoRA run config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    model_cfg = cfg.get("models", {}).get(model_key, {})
    lora_cfg = model_cfg.get("lora")
    if not isinstance(lora_cfg, dict):
        raise ValueError(f"Missing models.{model_key}.lora configuration in {config_path}")
    required = {"rank", "alpha", "dropout", "target_patterns"}
    missing = sorted(required - set(lora_cfg))
    if missing:
        raise ValueError(f"Incomplete models.{model_key}.lora configuration in {config_path}: missing {missing}")
    return model_cfg


def load_lora(model, ckpt_path: Path, *, model_key: str = "denoiser"):
    from trellis.modules.lora import apply_lora

    model_cfg = _load_model_cfg_from_run(ckpt_path, model_key)
    lora_cfg = model_cfg["lora"]
    apply_lora(
        model,
        rank=lora_cfg["rank"],
        alpha=lora_cfg["alpha"],
        dropout=lora_cfg["dropout"],
        target_patterns=lora_cfg["target_patterns"],
    )
    categories = model_cfg.get("categories")
    if categories is not None:
        model.enable_category_conditioning(categories)
    state = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [key for key in missing if "lora_" in key or key.startswith("category_embedding.")]
    unexpected = [key for key in unexpected if "lora_" in key or key.startswith("category_embedding.")]
    if missing or unexpected:
        raise RuntimeError(f"LoRA checkpoint mismatch. missing={missing}, unexpected={unexpected}")
    model.eval()
    return categories


def mesh_to_voxel_points(mesh, resolution: int) -> np.ndarray:
    vertices = np.clip(mesh.vertices.detach().cpu().numpy(), -0.5 + 1e-6, 0.5 - 1e-6)
    faces = mesh.faces.detach().cpu().numpy()

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


def read_voxels(path: Path, resolution: int) -> set[tuple[int, int, int]]:
    points = np.asarray(utils3d.io.read_ply(str(path))[0], dtype=np.float32)
    if points.size == 0:
        return set()
    points = points.reshape(-1, 3)
    voxels = np.floor((points + 0.5) * resolution).astype(np.int32)
    voxels = np.clip(voxels, 0, resolution - 1)
    return {tuple(voxel) for voxel in voxels.tolist()}


def internal_voxels(voxels: set[tuple[int, int, int]], resolution: int, outer_layers: int) -> set[tuple[int, int, int]]:
    if outer_layers <= 0:
        return set(voxels)
    if not voxels:
        return set()

    external = set()

    # External voxels are the min/max K occupied voxels along each axis-aligned
    # line, where K = outer_layers.
    for axis in range(3):
        groups: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
        other_axes = [i for i in range(3) if i != axis]
        for voxel in voxels:
            key = (voxel[other_axes[0]], voxel[other_axes[1]])
            groups.setdefault(key, []).append(voxel)

        for line_voxels in groups.values():
            line_sorted = sorted(line_voxels, key=lambda v: v[axis])
            k = min(outer_layers, len(line_sorted))
            for i in range(k):
                external.add(line_sorted[i])
                external.add(line_sorted[-1 - i])

    return set(voxels) - external


def safe_ratio(numerator: int, denominator: int, both_empty: bool) -> float:
    if denominator:
        return numerator / denominator
    return 1.0 if both_empty else 0.0


def read_mesh(path: Path) -> trimesh.Trimesh:
    vertices, faces = utils3d.io.read_ply(str(path))
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
    if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
        raise ValueError(f"Empty mesh: {path}")
    return mesh


def sample_mesh_points(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        points = mesh.sample(count)
    finally:
        np.random.set_state(state)
    return np.asarray(points, dtype=np.float32)


def chamfer_l1(gt_points: np.ndarray, pred_points: np.ndarray) -> float:
    if gt_points.shape[0] == 0 or pred_points.shape[0] == 0:
        return float("nan")

    gt_tree = cKDTree(gt_points)
    pred_tree = cKDTree(pred_points)

    pred_to_gt = gt_tree.query(pred_points, k=1)[0]
    gt_to_pred = pred_tree.query(gt_points, k=1)[0]
    return float(pred_to_gt.mean() + gt_to_pred.mean())


def evaluate_sample(
    gt_voxel_path: Path,
    pred_voxel_path: Path,
    gt_mesh_path: Path,
    pred_mesh_path: Path,
    *,
    resolution: int,
    outer_layers: int,
    mesh_sample_points: int,
    mesh_sample_seed: int,
) -> dict[str, float | int]:
    gt_vox = read_voxels(gt_voxel_path, resolution)
    pred_vox = read_voxels(pred_voxel_path, resolution)

    gt_internal = internal_voxels(gt_vox, resolution, outer_layers)
    pred_internal = internal_voxels(pred_vox, resolution, outer_layers)

    true_positive = len(gt_internal & pred_internal)
    precision = safe_ratio(true_positive, len(pred_internal), not gt_internal)
    recall = safe_ratio(true_positive, len(gt_internal), not pred_internal)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    gt_mesh = read_mesh(gt_mesh_path)
    pred_mesh = read_mesh(pred_mesh_path)
    gt_points = sample_mesh_points(gt_mesh, mesh_sample_points, mesh_sample_seed)
    pred_points = sample_mesh_points(pred_mesh, mesh_sample_points, mesh_sample_seed)
    chamfer = chamfer_l1(gt_points, pred_points)

    return {
        "internal_precision": precision,
        "internal_recall": recall,
        "internal_f1": f1,
        "mesh_chamfer_l1": chamfer,
        "gt_voxels": len(gt_vox),
        "pred_voxels": len(pred_vox),
        "gt_internal_voxels": len(gt_internal),
        "pred_internal_voxels": len(pred_internal),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified TRELLIS generation + evaluation (internal voxel PRF + mesh Chamfer).")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pipeline", default="microsoft/TRELLIS-image-large")
    parser.add_argument("--ss-lora-ckpt", type=Path, default=None)
    parser.add_argument("--slat-lora-ckpt", type=Path, default=None)
    parser.add_argument("--decoder-lora-ckpt", type=Path, default=None)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--outer-layers", type=int, default=3)
    parser.add_argument("--mesh-sample-points", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--view-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--samples-per-category", type=int, default=0)
    parser.add_argument("--ids-file", type=Path, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if args.view_index < 0:
        parser.error("--view-index must be non-negative")
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

    if (args.ss_lora_ckpt is None) != (args.slat_lora_ckpt is None):
        parser.error("Provide both --ss-lora-ckpt and --slat-lora-ckpt, or neither")

    checkpoints = [args.ss_lora_ckpt, args.slat_lora_ckpt, args.decoder_lora_ckpt]
    for checkpoint in checkpoints:
        if checkpoint is not None and not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

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

    predictions_dir = args.output_dir / "predictions"
    mesh_dir = predictions_dir / "mesh"
    voxel_dir = predictions_dir / "voxels"
    metrics_dir = args.output_dir / "metrics"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    voxel_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    from trellis.pipelines import TrellisImageTo3DPipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.pipeline)
    pipeline.to(device)

    category_names = None
    if args.ss_lora_ckpt is not None:
        ss_categories = load_lora(pipeline.models["sparse_structure_flow_model"], args.ss_lora_ckpt, model_key="denoiser")
        slat_categories = load_lora(pipeline.models["slat_flow_model"], args.slat_lora_ckpt, model_key="denoiser")
        if ss_categories != slat_categories:
            raise ValueError("SS-flow and SLAT-flow category configurations do not match")
        category_names = ss_categories

    if args.decoder_lora_ckpt is not None:
        load_lora(pipeline.models["slat_decoder_mesh"], args.decoder_lora_ckpt, model_key="decoder")

    for sample_id in tqdm(ids, desc="Generating predictions"):
        pred_mesh_path = mesh_dir / f"{sample_id}.ply"
        pred_voxel_path = voxel_dir / f"{sample_id}.ply"
        if args.skip_existing and pred_mesh_path.exists() and pred_voxel_path.exists():
            continue

        image_path = args.dataset_dir / "renders_cond" / sample_id / f"{args.view_index:03d}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"Conditioning image not found: {image_path}")

        with Image.open(image_path) as image, torch.inference_mode():
            torch.manual_seed(args.seed)
            image = pipeline.preprocess_image(image)
            category = [categories[sample_id]] if category_names is not None else None
            cond = pipeline.get_cond([image], category=category)
            coords = pipeline.sample_sparse_structure(cond, num_samples=1)
            slat = pipeline.sample_slat(cond, coords)
            mesh = pipeline.decode_slat(slat, formats=["mesh"])["mesh"][0]

        utils3d.io.write_ply(pred_mesh_path, mesh.vertices.detach().cpu().numpy(), mesh.faces.detach().cpu().numpy())
        voxel_points = mesh_to_voxel_points(mesh, args.resolution)
        utils3d.io.write_ply(pred_voxel_path, voxel_points)

    per_sample_rows = []
    for sample_id in tqdm(ids, desc="Evaluating predictions"):
        gt_voxel_path = args.dataset_dir / "voxels" / f"{sample_id}.ply"
        gt_mesh_path = args.dataset_dir / "renders" / sample_id / "mesh.ply"
        pred_voxel_path = voxel_dir / f"{sample_id}.ply"
        pred_mesh_path = mesh_dir / f"{sample_id}.ply"

        if not gt_voxel_path.is_file():
            raise FileNotFoundError(f"Missing GT voxel file: {gt_voxel_path}")
        if not gt_mesh_path.is_file():
            raise FileNotFoundError(f"Missing GT mesh file: {gt_mesh_path}")
        if not pred_voxel_path.is_file():
            raise FileNotFoundError(f"Missing prediction voxel file: {pred_voxel_path}")
        if not pred_mesh_path.is_file():
            raise FileNotFoundError(f"Missing prediction mesh file: {pred_mesh_path}")

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

    print("\nUnified evaluation finished")
    print(f"Predicted meshes: {mesh_dir}")
    print(f"Predicted voxels: {voxel_dir}")
    print(f"Per-sample metrics: {per_sample_path}")
    print(f"Summary metrics: {summary_path}")


if __name__ == "__main__":
    main()
