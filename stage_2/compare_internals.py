#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np
import utils3d


def read_voxels(path: Path, resolution: int) -> set[tuple[int, int, int]]:
    points = np.asarray(utils3d.io.read_ply(str(path))[0], dtype=np.float32)
    if points.size == 0:
        return set()
    points = points.reshape(-1, 3)
    voxels = np.floor((points + 0.5) * resolution).astype(np.int32)
    voxels = np.clip(voxels, 0, resolution - 1)
    return {tuple(voxel) for voxel in voxels.tolist()}


def interior(voxels: set[tuple[int, int, int]]) -> set[tuple[int, int, int]]:
    """Return the axis-extrema proxy for internal voxels."""
    if not voxels:
        return set()

    surface = set()
    for axis in range(3):
        groups = {}
        other_axes = [i for i in range(3) if i != axis]
        for voxel in voxels:
            key = (voxel[other_axes[0]], voxel[other_axes[1]])
            groups.setdefault(key, []).append(voxel)
        for group in groups.values():
            surface.add(min(group, key=lambda voxel: voxel[axis]))
            surface.add(max(group, key=lambda voxel: voxel[axis]))
    return voxels - surface


def safe_ratio(numerator: int, denominator: int, both_empty: bool) -> float:
    if denominator:
        return numerator / denominator
    return 1.0 if both_empty else 0.0


def score_pair(gt_path: Path, pred_path: Path, resolution: int) -> dict[str, float | int]:
    gt = read_voxels(gt_path, resolution)
    pred = read_voxels(pred_path, resolution)
    gt_internal = interior(gt)
    pred_internal = interior(pred)

    true_positive = len(gt_internal & pred_internal)
    precision = safe_ratio(true_positive, len(pred_internal), not gt_internal)
    recall = safe_ratio(true_positive, len(gt_internal), not pred_internal)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = gt | pred

    return {
        "voxel_iou": len(gt & pred) / len(union) if union else 1.0,
        "internal_precision": precision,
        "internal_recall": recall,
        "internal_f1": f1,
        "gt_voxels": len(gt),
        "pred_voxels": len(pred),
        "gt_internal_voxels": len(gt_internal),
        "pred_internal_voxels": len(pred_internal),
    }


def read_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not ids:
        raise ValueError(f"No sample IDs found in {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate sample IDs found in {path}")
    return ids


def read_categories(path: Path) -> dict[str, str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {row["sha256"]: row["category"] for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Objective 1 voxel predictions.")
    parser.add_argument("--gt-voxels", type=Path, required=True)
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--ids-file", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-sample-output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=64)
    args = parser.parse_args()

    ids = read_ids(args.ids_file)
    categories = read_categories(args.metadata)
    missing_categories = [sample_id for sample_id in ids if sample_id not in categories]
    if missing_categories:
        raise ValueError(f"IDs missing from metadata: {missing_categories[:5]}")

    per_sample_rows = []
    summary_rows = []
    metric_names = ["voxel_iou", "internal_precision", "internal_recall", "internal_f1"]

    for method in args.methods:
        method_dir = args.pred_root / method
        seed_dirs = sorted(path for path in method_dir.glob("seed_*") if path.is_dir())
        if not seed_dirs:
            raise FileNotFoundError(f"No seed directories found under {method_dir}")

        for seed_dir in seed_dirs:
            seed = seed_dir.name.removeprefix("seed_")
            seed_rows = []
            for sample_id in ids:
                gt_path = args.gt_voxels / f"{sample_id}.ply"
                pred_path = seed_dir / "voxels" / f"{sample_id}.ply"
                if not gt_path.is_file():
                    raise FileNotFoundError(f"Missing ground-truth voxels: {gt_path}")
                if not pred_path.is_file():
                    raise FileNotFoundError(f"Missing prediction: {pred_path}")

                metrics = score_pair(gt_path, pred_path, args.resolution)
                row = {
                    "method": method,
                    "seed": seed,
                    "sample_id": sample_id,
                    "category": categories[sample_id],
                    **metrics,
                }
                per_sample_rows.append(row)
                seed_rows.append(row)

            summary_rows.append({
                "method": method,
                "seed": seed,
                **{name: float(np.mean([row[name] for row in seed_rows])) for name in metric_names},
                "matched_samples": len(seed_rows),
            })

    args.per_sample_output.parent.mkdir(parents=True, exist_ok=True)
    with args.per_sample_output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_sample_rows[0]))
        writer.writeheader()
        writer.writerows(per_sample_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    for row in summary_rows:
        print(
            f"{row['method']} seed={row['seed']}: "
            f"IoU={row['voxel_iou']:.4f}, internal F1={row['internal_f1']:.4f}"
        )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.per_sample_output}")


if __name__ == "__main__":
    main()
