#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

METHODS = [
    ("base_ss_flow", "base_ss_flow_voxels"),
    ("lora_ss_flow", "ss_flow_voxels"),
    ("base_ss+slat_voxelized", "base_ss_slat_voxelized"),
    ("lora_ss+slat_voxelized", "lora_ss_slat_voxelized"),
]


PLY_DTYPES = {
    "char": "i1",
    "uchar": "u1",
    "short": "i2",
    "ushort": "u2",
    "int": "i4",
    "uint": "u4",
    "float": "f4",
    "double": "f8",
    "int8": "i1",
    "uint8": "u1",
    "int16": "i2",
    "uint16": "u2",
    "int32": "i4",
    "uint32": "u4",
    "float32": "f4",
    "float64": "f8",
}


def read_ply_points(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"PLY header is incomplete: {path}")
            text = line.decode("ascii").strip()
            header.append(text)
            if text == "end_header":
                break

        fmt = None
        vertex_count = None
        vertex_properties = []
        current_element = None

        for line in header:
            parts = line.split()
            if not parts:
                continue
            if parts[:1] == ["format"]:
                fmt = parts[1]
            elif parts[:1] == ["element"]:
                current_element = parts[1]
                if current_element == "vertex":
                    vertex_count = int(parts[2])
            elif parts[:1] == ["property"] and current_element == "vertex":
                if parts[1] == "list":
                    raise ValueError(f"List vertex properties are not supported: {path}")
                vertex_properties.append((parts[2], parts[1]))

        if fmt not in {"ascii", "binary_little_endian", "binary_big_endian"}:
            raise ValueError(f"Unsupported PLY format {fmt!r}: {path}")
        if vertex_count is None:
            raise ValueError(f"PLY file has no vertex element: {path}")

        names = [name for name, _ in vertex_properties]
        xyz = [names.index(axis) for axis in ("x", "y", "z")]

        if fmt == "ascii":
            points = np.loadtxt(f, max_rows=vertex_count, usecols=xyz, dtype=np.float32)
            return np.atleast_2d(points)

        endian = "<" if fmt == "binary_little_endian" else ">"
        dtype = np.dtype([(name, endian + PLY_DTYPES[prop]) for name, prop in vertex_properties])
        vertices = np.fromfile(f, dtype=dtype, count=vertex_count)
        return np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)


def read_voxels(path: Path, resolution: int) -> set[tuple[int, int, int]]:
    points = read_ply_points(path)
    voxels = np.floor((points + 0.5) * resolution).astype(np.int32)
    voxels = np.clip(voxels, 0, resolution - 1)
    return {tuple(voxel) for voxel in voxels.tolist()}


def interior(voxels: set[tuple[int, int, int]], margin: int) -> set[tuple[int, int, int]]:
    if not voxels:
        return set()

    arr = np.array(list(voxels), dtype=np.int32)
    lo = arr.min(axis=0) + margin
    hi = arr.max(axis=0) - margin
    keep = np.all((arr > lo) & (arr < hi), axis=1)
    return {tuple(voxel) for voxel in arr[keep].tolist()}


def score_pair(gt_path: Path, pred_path: Path, resolution: int, interior_margin: int) -> tuple[float, float]:
    gt = read_voxels(gt_path, resolution)
    pred = read_voxels(pred_path, resolution)

    union = gt | pred
    gt_inside = interior(gt, interior_margin)

    voxel_iou = len(gt & pred) / len(union) if union else 1.0
    interior_recall = len(pred & gt_inside) / len(gt_inside) if gt_inside else 1.0
    return voxel_iou, interior_recall


def evaluate_method(gt_voxels: Path, pred_voxels: Path, resolution: int, interior_margin: int) -> dict[str, float | int]:
    scores = []
    for gt_path in sorted(gt_voxels.glob("*.ply")):
        pred_path = pred_voxels / gt_path.name
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
