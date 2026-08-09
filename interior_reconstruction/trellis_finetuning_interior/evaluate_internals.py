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


def interior(
    voxels: set[tuple[int, int, int]],
    margin: int = 1,
) -> set[tuple[int, int, int]]:
    """Return voxels at least ``margin`` grid cells behind every axis extremum."""
    if margin < 1:
        raise ValueError("margin must be at least 1")
    if not voxels:
        return set()

    internal = set(voxels)
    for axis in range(3):
        groups = {}
        other_axes = [i for i in range(3) if i != axis]
        for voxel in voxels:
            key = (voxel[other_axes[0]], voxel[other_axes[1]])
            groups.setdefault(key, []).append(voxel)
        for group in groups.values():
            minimum = min(voxel[axis] for voxel in group)
            maximum = max(voxel[axis] for voxel in group)
            for voxel in group:
                if voxel[axis] - minimum < margin or maximum - voxel[axis] < margin:
                    internal.discard(voxel)
    return internal


def safe_ratio(numerator: int, denominator: int, both_empty: bool) -> float:
    if denominator:
        return numerator / denominator
    return 1.0 if both_empty else 0.0


def score_pair(
    gt: set[tuple[int, int, int]],
    pred: set[tuple[int, int, int]],
    margin: int = 1,
) -> dict[str, float | int]:
    gt_internal = interior(gt, margin)
    pred_internal = interior(pred, margin)

    true_positive = len(gt_internal & pred_internal)
    precision = safe_ratio(true_positive, len(pred_internal), not gt_internal)
    recall = safe_ratio(true_positive, len(gt_internal), not pred_internal)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = gt | pred
    gt_exterior = gt - gt_internal
    pred_exterior = pred - pred_internal
    exterior_union = gt_exterior | pred_exterior

    return {
        "voxel_iou": len(gt & pred) / len(union) if union else 1.0,
        "exterior_iou": (
            len(gt_exterior & pred_exterior) / len(exterior_union)
            if exterior_union
            else 1.0
        ),
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
    parser.add_argument("--prediction-subdir", default="voxels")
    parser.add_argument("--margins", type=int, nargs="+", default=[1, 2, 3, 4])
    args = parser.parse_args()

    if any(margin < 1 for margin in args.margins):
        parser.error("--margins must contain positive integers")
    if len(args.margins) != len(set(args.margins)):
        parser.error("--margins must not contain duplicates")

    ids = read_ids(args.ids_file)
    categories = read_categories(args.metadata)
    missing_categories = [sample_id for sample_id in ids if sample_id not in categories]
    if missing_categories:
        raise ValueError(f"IDs missing from metadata: {missing_categories[:5]}")

    per_sample_rows = []
    summary_rows = []
    metric_names = [
        "voxel_iou",
        "exterior_iou",
        "internal_precision",
        "internal_recall",
        "internal_f1",
    ]

    for method in args.methods:
        method_dir = args.pred_root / method
        seed_dirs = sorted(path for path in method_dir.glob("seed_*") if path.is_dir())
        if not seed_dirs:
            raise FileNotFoundError(f"No seed directories found under {method_dir}")

        for seed_dir in seed_dirs:
            seed = seed_dir.name.removeprefix("seed_")
            seed_rows = {margin: [] for margin in args.margins}
            for sample_id in ids:
                gt_path = args.gt_voxels / f"{sample_id}.ply"
                pred_path = seed_dir / args.prediction_subdir / f"{sample_id}.ply"
                if not gt_path.is_file():
                    raise FileNotFoundError(f"Missing ground-truth voxels: {gt_path}")
                if not pred_path.is_file():
                    raise FileNotFoundError(f"Missing prediction: {pred_path}")

                gt = read_voxels(gt_path, args.resolution)
                pred = read_voxels(pred_path, args.resolution)
                for margin in args.margins:
                    row = {
                        "method": method,
                        "seed": seed,
                        "sample_id": sample_id,
                        "category": categories[sample_id],
                        "margin": margin,
                        **score_pair(gt, pred, margin),
                    }
                    per_sample_rows.append(row)
                    seed_rows[margin].append(row)

            for margin, rows in seed_rows.items():
                summary_rows.append({
                    "method": method,
                    "seed": seed,
                    "margin": margin,
                    **{name: float(np.mean([row[name] for row in rows])) for name in metric_names},
                    "matched_samples": len(rows),
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
            f"{row['method']} seed={row['seed']} margin={row['margin']}: "
            f"IoU={row['voxel_iou']:.4f}, exterior IoU={row['exterior_iou']:.4f}, "
            f"internal F1={row['internal_f1']:.4f}"
        )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.per_sample_output}")


if __name__ == "__main__":
    main()
