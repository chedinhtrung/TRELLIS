#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

import numpy as np
import utils3d


REPO_ROOT = Path(__file__).resolve().parents[1]

METHODS = [
    ("base_ss_flow", "base_ss_flow_voxels"),
    ("lora_ss_flow", "ss_flow_voxels"),
    ("base_ss+slat_voxelized", "base_ss_slat_voxelized"),
    ("lora_ss+slat_voxelized", "lora_ss_slat_voxelized"),
]


def read_ply_points(path: Path) -> np.ndarray:
    points = utils3d.io.read_ply(str(path))[0]
    points = np.asarray(points, dtype=np.float32)
    return np.atleast_2d(points)


def read_voxels(path: Path, resolution: int) -> set[tuple[int, int, int]]:
    points = read_ply_points(path)
    voxels = np.floor((points + 0.5) * resolution).astype(np.int32)
    voxels = np.clip(voxels, 0, resolution - 1)
    return {tuple(voxel) for voxel in voxels.tolist()}


def interior(voxels: set[tuple[int, int, int]], margin: int) -> set[tuple[int, int, int]]:
    if not voxels:
        return set()

    arr = np.array(list(voxels), dtype=np.int32)
    surface = set()

    # For fixed (y, z), min/max x are surface voxels.
    yz_to_x = {}
    for x, y, z in arr.tolist():
        yz_to_x.setdefault((y, z), []).append(x)
    for (y, z), xs in yz_to_x.items():
        x_min = min(xs)
        x_max = max(xs)
        surface.add((x_min, y, z))
        surface.add((x_max, y, z))

    # For fixed (x, z), min/max y are surface voxels.
    xz_to_y = {}
    for x, y, z in arr.tolist():
        xz_to_y.setdefault((x, z), []).append(y)
    for (x, z), ys in xz_to_y.items():
        y_min = min(ys)
        y_max = max(ys)
        surface.add((x, y_min, z))
        surface.add((x, y_max, z))

    # For fixed (x, y), min/max z are surface voxels.
    xy_to_z = {}
    for x, y, z in arr.tolist():
        xy_to_z.setdefault((x, y), []).append(z)
    for (x, y), zs in xy_to_z.items():
        z_min = min(zs)
        z_max = max(zs)
        surface.add((x, y, z_min))
        surface.add((x, y, z_max))

    return set(voxels) - surface


def score_pair(gt_path: Path, pred_path: Path, resolution: int, interior_margin: int) -> tuple[float, float]:
    gt = read_voxels(gt_path, resolution)
    pred = read_voxels(pred_path, resolution)

    union = gt | pred
    gt_inside = interior(gt, interior_margin)

    voxel_iou = len(gt & pred) / len(union) if union else 1.0
    interior_recall = len(pred & gt_inside) / len(gt_inside) if gt_inside else 1.0
    return voxel_iou, interior_recall


def evaluate_method(gt_voxels: Path, pred_voxels: Path, resolution: int, interior_margin: int) -> dict[str, float | int]:
    voxel_dir = pred_voxels / "voxels"
    pred_base_dir = voxel_dir if voxel_dir.is_dir() else pred_voxels

    scores = []
    for gt_path in sorted(gt_voxels.glob("*.ply")):
        pred_path = pred_base_dir / gt_path.name
        if pred_path.is_file():
            scores.append(score_pair(gt_path, pred_path, resolution, interior_margin))

    if not scores:
        return {"voxel_iou": math.nan, "interior_recall": math.nan, "matched_samples": 0}

    ious, recalls = zip(*scores)
    return {
        "voxel_iou": float(np.mean(ious)),
        "interior_recall": float(np.mean(recalls)),
        "matched_samples": len(scores),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the four exported internal-aware TRELLIS voxel sets.")
    parser.add_argument("--gt-voxels", type=Path, default=REPO_ROOT / "datasets/ShapeNetInternals_small/voxels")
    parser.add_argument("--pred-root", type=Path, default=REPO_ROOT / "results/shapenet_internals_lora/predictions")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results/shapenet_internals_lora/eval/comparison.csv")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--interior-margin", type=int, default=3)
    args = parser.parse_args()

    rows = []
    for method, dirname in METHODS:
        metrics = evaluate_method(args.gt_voxels, args.pred_root / dirname, args.resolution, args.interior_margin)
        rows.append({"method": method, **metrics})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "voxel_iou", "interior_recall", "matched_samples"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'method':28s} {'voxel_iou':>10s} {'interior_recall':>16s} {'matched':>8s}")
    for row in rows:
        iou = "nan" if math.isnan(row["voxel_iou"]) else f"{row['voxel_iou']:.6f}"
        recall = "nan" if math.isnan(row["interior_recall"]) else f"{row['interior_recall']:.6f}"
        print(f"{row['method']:28s} {iou:>10s} {recall:>16s} {row['matched_samples']:8d}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
