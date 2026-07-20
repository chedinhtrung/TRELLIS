#!/usr/bin/env python3
"""Evaluate DINO-retrieved interior transplantation into Objective-1 geometry.

This is a voxel-space diagnostic. It uses only the deployable DINO top-1
candidate, aligns that training shape to the Objective-1 bounding box, and
compares exterior-preserving replacement and union variants.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import utils3d

from compare_internals import interior, read_voxels, score_pair
from evaluate_retrieval_completion import aggregate, write_csv


EXPECTED_QUERY_MODE = "dino_view018"


def read_metadata(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing metadata: {path}")
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    required = {"sha256", "category"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return [
        {"sha256": row["sha256"].strip(), "category": row["category"].strip()}
        for row in rows
    ]


def read_selected_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not ids:
        raise ValueError(f"No IDs found in {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs found in {path}")
    return set(ids)


def read_dino_top1(
    path: Path,
    retrieval_margin: int,
    selected_ids: set[str] | None,
) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing DINO results: {path}")
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    selected_rows = [
        row
        for row in rows
        if row["query_mode"] == EXPECTED_QUERY_MODE
        and row["selection"] == "top1"
        and int(row["k"]) == 1
        and int(row["margin"]) == retrieval_margin
        and (selected_ids is None or row["sample_id"] in selected_ids)
    ]
    mapping = {}
    for row in selected_rows:
        sample_id = row["sample_id"]
        if sample_id in mapping:
            raise ValueError(f"Duplicate DINO top-1 row for {sample_id}")
        mapping[sample_id] = row["retrieved_id"]
    if not mapping:
        raise ValueError(
            f"No {EXPECTED_QUERY_MODE} top-1 margin-{retrieval_margin} rows in {path}"
        )
    return mapping


def bounding_box(voxels: set[tuple[int, int, int]]) -> tuple[np.ndarray, np.ndarray]:
    if not voxels:
        raise ValueError("Cannot compute a bounding box for an empty voxel set")
    coordinates = np.asarray(list(voxels), dtype=np.float32)
    return coordinates.min(axis=0), coordinates.max(axis=0)


def align_voxels(
    voxels: set[tuple[int, int, int]],
    source_reference: set[tuple[int, int, int]],
    target_reference: set[tuple[int, int, int]],
    resolution: int,
) -> set[tuple[int, int, int]]:
    """Anisotropically map source-reference AABB coordinates to the target AABB."""
    if not voxels:
        return set()
    source_min, source_max = bounding_box(source_reference)
    target_min, target_max = bounding_box(target_reference)
    source_span = source_max - source_min
    target_span = target_max - target_min
    coordinates = np.asarray(list(voxels), dtype=np.float32)
    mapped = np.empty_like(coordinates)
    for axis in range(3):
        if source_span[axis] > 0:
            normalized = (coordinates[:, axis] - source_min[axis]) / source_span[axis]
            mapped[:, axis] = target_min[axis] + normalized * target_span[axis]
        else:
            mapped[:, axis] = (target_min[axis] + target_max[axis]) / 2
    mapped = np.rint(mapped).astype(np.int32)
    mapped = np.clip(mapped, 0, resolution - 1)
    return {tuple(voxel) for voxel in mapped.tolist()}


def enclosed_volume(
    surface: set[tuple[int, int, int]],
    margin: int,
) -> set[tuple[int, int, int]]:
    """Return grid cells bracketed by the surface along all three axes."""
    if not surface:
        return set()
    axis_volumes = []
    for axis in range(3):
        other_axes = [index for index in range(3) if index != axis]
        groups = defaultdict(list)
        for voxel in surface:
            groups[(voxel[other_axes[0]], voxel[other_axes[1]])].append(voxel[axis])

        axis_volume = set()
        for key, positions in groups.items():
            start = min(positions) + margin
            stop = max(positions) - margin
            for position in range(start, stop + 1):
                voxel = [0, 0, 0]
                voxel[axis] = position
                voxel[other_axes[0]] = key[0]
                voxel[other_axes[1]] = key[1]
                axis_volume.add(tuple(voxel))
        axis_volumes.append(axis_volume)
    return axis_volumes[0] & axis_volumes[1] & axis_volumes[2]


def count_ratio(numerator: int, denominator: int) -> float:
    if denominator:
        return numerator / denominator
    return 1.0 if numerator == 0 else 0.0


def score_voxels(
    gt: set[tuple[int, int, int]],
    pred: set[tuple[int, int, int]],
    margin: int,
) -> dict[str, float | int]:
    metrics = score_pair(gt, pred, margin)
    metrics["pred_to_gt_voxel_ratio"] = count_ratio(
        int(metrics["pred_voxels"]), int(metrics["gt_voxels"])
    )
    metrics["pred_to_gt_internal_ratio"] = count_ratio(
        int(metrics["pred_internal_voxels"]),
        int(metrics["gt_internal_voxels"]),
    )
    return metrics


def voxel_points(
    voxels: set[tuple[int, int, int]], resolution: int
) -> np.ndarray:
    if not voxels:
        return np.zeros((0, 3), dtype=np.float32)
    coordinates = np.asarray(sorted(voxels), dtype=np.float32)
    return ((coordinates + 0.5) / resolution - 0.5).astype(np.float32)


def write_voxels(
    path: Path,
    voxels: set[tuple[int, int, int]],
    resolution: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    utils3d.io.write_ply(str(path), voxel_points(voxels, resolution))


def make_variants(
    objective1: set[tuple[int, int, int]],
    retrieved: set[tuple[int, int, int]],
    resolution: int,
    transplant_margin: int,
) -> tuple[dict[str, set[tuple[int, int, int]]], dict[str, int]]:
    objective1_internal = interior(objective1, transplant_margin)
    objective1_exterior = objective1 - objective1_internal
    retrieved_internal = interior(retrieved, transplant_margin)
    retrieved_aligned = align_voxels(
        retrieved, retrieved, objective1, resolution
    )
    internal_aligned = align_voxels(
        retrieved_internal, retrieved, objective1, resolution
    )
    safe_volume = enclosed_volume(objective1_exterior, transplant_margin)
    safe_internal = internal_aligned & safe_volume

    variants = {
        "objective1": objective1,
        "dino_direct": retrieved,
        "dino_aligned": retrieved_aligned,
        "replace_aligned": objective1_exterior | internal_aligned,
        "replace_safe": objective1_exterior | safe_internal,
        "union_safe": objective1 | safe_internal,
    }
    diagnostics = {
        "objective1_internal_voxels": len(objective1_internal),
        "retrieved_internal_voxels": len(retrieved_internal),
        "aligned_internal_voxels": len(internal_aligned),
        "safe_volume_voxels": len(safe_volume),
        "safe_inserted_voxels": len(safe_internal - objective1),
    }
    return variants, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate view-18 DINO top-1 interior transplantation."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--objective1-voxels", type=Path, required=True)
    parser.add_argument("--dino-per-sample", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--retrieval-margin", type=int, default=2)
    parser.add_argument("--transplant-margin", type=int, default=2)
    parser.add_argument("--margins", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--visualizations-per-category", type=int, default=3)
    parser.add_argument("--save-all-plys", action="store_true")
    args = parser.parse_args()

    if args.resolution < 1:
        raise ValueError("--resolution must be positive")
    if args.retrieval_margin < 1 or args.transplant_margin < 1:
        raise ValueError("Retrieval and transplant margins must be positive")
    if any(margin < 1 for margin in args.margins):
        raise ValueError("--margins must contain positive integers")
    if len(args.margins) != len(set(args.margins)):
        raise ValueError("--margins must not contain duplicates")
    if args.visualizations_per_category < 0:
        raise ValueError("--visualizations-per-category cannot be negative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}. Use a new directory."
        )

    selected_ids = read_selected_ids(args.ids_file)
    train_dir = args.dataset_root / "train"
    test_dir = args.dataset_root / "test"
    train_rows = read_metadata(train_dir / "metadata.csv")
    test_rows = read_metadata(test_dir / "metadata.csv")
    if selected_ids is not None:
        known_ids = {row["sha256"] for row in test_rows}
        missing = sorted(selected_ids - known_ids)
        if missing:
            raise ValueError(f"IDs missing from test metadata: {missing[:5]}")
        test_rows = [row for row in test_rows if row["sha256"] in selected_ids]

    train_categories = {row["sha256"]: row["category"] for row in train_rows}
    dino_top1 = read_dino_top1(
        args.dino_per_sample, args.retrieval_margin, selected_ids
    )
    expected_test_ids = {row["sha256"] for row in test_rows}
    missing_dino = sorted(expected_test_ids - set(dino_top1))
    extra_dino = sorted(set(dino_top1) - expected_test_ids)
    if missing_dino or extra_dino:
        raise ValueError(
            f"DINO/test ID mismatch: missing={missing_dino[:5]}, extra={extra_dino[:5]}"
        )

    retrieved_cache = {}
    per_sample_rows = []
    visualized = defaultdict(int)
    margins = sorted(args.margins)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for index, row in enumerate(test_rows, start=1):
        sample_id = row["sha256"]
        category = row["category"]
        retrieved_id = dino_top1[sample_id]
        if retrieved_id not in train_categories:
            raise ValueError(f"Retrieved ID is absent from training metadata: {retrieved_id}")
        if train_categories[retrieved_id] != category:
            raise ValueError(
                f"Category mismatch for {sample_id}: test={category}, "
                f"retrieved={train_categories[retrieved_id]}"
            )

        gt_path = test_dir / "voxels" / f"{sample_id}.ply"
        objective1_path = args.objective1_voxels / f"{sample_id}.ply"
        retrieved_path = train_dir / "voxels" / f"{retrieved_id}.ply"
        for required_path in (gt_path, objective1_path, retrieved_path):
            if not required_path.is_file():
                raise FileNotFoundError(f"Missing voxel PLY: {required_path}")

        gt = read_voxels(gt_path, args.resolution)
        objective1 = read_voxels(objective1_path, args.resolution)
        if retrieved_id not in retrieved_cache:
            retrieved_cache[retrieved_id] = read_voxels(
                retrieved_path, args.resolution
            )
        retrieved = retrieved_cache[retrieved_id]
        variants, diagnostics = make_variants(
            objective1,
            retrieved,
            args.resolution,
            args.transplant_margin,
        )

        for method, prediction in variants.items():
            for margin in margins:
                per_sample_rows.append({
                    "method": method,
                    "sample_id": sample_id,
                    "category": category,
                    "retrieved_id": retrieved_id,
                    "margin": margin,
                    **score_voxels(gt, prediction, margin),
                    **diagnostics,
                })
            if args.save_all_plys:
                write_voxels(
                    args.output_dir / "predictions" / method / f"{sample_id}.ply",
                    prediction,
                    args.resolution,
                )

        if visualized[category] < args.visualizations_per_category:
            sample_dir = args.output_dir / "visualizations" / sample_id
            write_voxels(sample_dir / "ground_truth.ply", gt, args.resolution)
            for method, prediction in variants.items():
                write_voxels(
                    sample_dir / f"{method}.ply", prediction, args.resolution
                )
            visualized[category] += 1

        if index % 25 == 0 or index == len(test_rows):
            print(f"Processed {index}/{len(test_rows)}")

    summary_rows = aggregate(per_sample_rows, ("method", "margin"))
    category_summary_rows = aggregate(
        per_sample_rows, ("method", "category", "margin")
    )
    write_csv(args.output_dir / "per_sample.csv", per_sample_rows)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_csv(args.output_dir / "category_summary.csv", category_summary_rows)

    print("\nPrimary margin results")
    for row in summary_rows:
        if row["margin"] != args.transplant_margin:
            continue
        print(
            f"  {row['method']}: IoU={row['voxel_iou']:.4f}, "
            f"exterior IoU={row['exterior_iou']:.4f}, "
            f"internal F1={row['internal_f1']:.4f}, "
            f"internal ratio={row['pred_to_gt_internal_ratio']:.3f}"
        )
    print(f"\nWrote transplant results to {args.output_dir}")


if __name__ == "__main__":
    main()
