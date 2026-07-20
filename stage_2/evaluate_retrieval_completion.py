#!/usr/bin/env python3
"""Evaluate category-restricted shape retrieval as an interior baseline.

The script retrieves complete training shapes by exterior voxel IoU. It evaluates
two query modes:

* ``gt_exterior``: retrieve from the test shape's ground-truth exterior.
* ``<pred-name>_exterior``: retrieve from an optional predicted voxel PLY.

Oracle@K chooses the candidate in the retrieved top K with the best ground-truth
internal F1. It is a ceiling measurement, not a deployable selection rule.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from compare_internals import interior, read_voxels


METRIC_NAMES = (
    "voxel_iou",
    "exterior_iou",
    "internal_precision",
    "internal_recall",
    "internal_f1",
    "pred_to_gt_voxel_ratio",
    "pred_to_gt_internal_ratio",
)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    category: str
    voxel_path: Path
    full: int
    internal: dict[int, int]
    exterior: dict[int, int]


def voxel_mask(voxels: set[tuple[int, int, int]], resolution: int) -> int:
    """Pack a voxel set into a Python integer for fast intersections."""
    if not voxels:
        return 0
    coordinates = np.asarray(list(voxels), dtype=np.int64)
    flat_indices = (
        (coordinates[:, 0] * resolution + coordinates[:, 1]) * resolution
        + coordinates[:, 2]
    )
    occupancy = np.zeros(resolution ** 3, dtype=np.uint8)
    occupancy[flat_indices] = 1
    packed = np.packbits(occupancy, bitorder="little")
    return int.from_bytes(packed.tobytes(), byteorder="little")


def make_sample(
    sample_id: str,
    category: str,
    voxel_path: Path,
    resolution: int,
    margins: list[int],
) -> Sample:
    voxels = read_voxels(voxel_path, resolution)
    full = voxel_mask(voxels, resolution)
    internal_masks = {
        margin: voxel_mask(interior(voxels, margin), resolution)
        for margin in margins
    }
    exterior_masks = {
        margin: full & ~internal_masks[margin]
        for margin in margins
    }
    return Sample(
        sample_id=sample_id,
        category=category,
        voxel_path=voxel_path,
        full=full,
        internal=internal_masks,
        exterior=exterior_masks,
    )


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
    return rows


def read_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not ids:
        raise ValueError(f"No IDs found in {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs found in {path}")
    return set(ids)


def load_split(
    split_dir: Path,
    resolution: int,
    margins: list[int],
    ids: set[str] | None = None,
) -> list[Sample]:
    rows = read_metadata(split_dir / "metadata.csv")
    if ids is not None:
        known_ids = {row["sha256"].strip() for row in rows}
        missing = sorted(ids - known_ids)
        if missing:
            raise ValueError(f"IDs missing from {split_dir / 'metadata.csv'}: {missing[:5]}")
        rows = [row for row in rows if row["sha256"].strip() in ids]

    samples = []
    seen = set()
    print(f"Loading {len(rows)} shapes from {split_dir}")
    for index, row in enumerate(rows, start=1):
        sample_id = row["sha256"].strip()
        category = row["category"].strip()
        if not sample_id or not category:
            raise ValueError(f"Empty sha256 or category in {split_dir / 'metadata.csv'}")
        if sample_id in seen:
            raise ValueError(f"Duplicate sha256 in {split_dir / 'metadata.csv'}: {sample_id}")
        seen.add(sample_id)
        voxel_path = split_dir / "voxels" / f"{sample_id}.ply"
        if not voxel_path.is_file():
            raise FileNotFoundError(f"Missing voxel file: {voxel_path}")
        samples.append(make_sample(sample_id, category, voxel_path, resolution, margins))
        if index % 100 == 0 or index == len(rows):
            print(f"  loaded {index}/{len(rows)}")
    return samples


def load_prediction_queries(
    test_samples: list[Sample],
    pred_voxels: Path,
    resolution: int,
    margins: list[int],
) -> list[Sample]:
    samples = []
    print(f"Loading {len(test_samples)} predicted query shapes from {pred_voxels}")
    for index, test_sample in enumerate(test_samples, start=1):
        voxel_path = pred_voxels / f"{test_sample.sample_id}.ply"
        if not voxel_path.is_file():
            raise FileNotFoundError(f"Missing predicted voxel file: {voxel_path}")
        samples.append(
            make_sample(
                test_sample.sample_id,
                test_sample.category,
                voxel_path,
                resolution,
                margins,
            )
        )
        if index % 100 == 0 or index == len(test_samples):
            print(f"  loaded {index}/{len(test_samples)}")
    return samples


def count_ratio(numerator: int, denominator: int) -> float:
    if denominator:
        return numerator / denominator
    return 1.0 if numerator == 0 else 0.0


def overlap_ratio(numerator: int, denominator: int, both_empty: bool) -> float:
    if denominator:
        return numerator / denominator
    return 1.0 if both_empty else 0.0


def jaccard(left: int, right: int) -> float:
    union = (left | right).bit_count()
    return (left & right).bit_count() / union if union else 1.0


def score_pair(gt: Sample, pred: Sample, margin: int) -> dict[str, float | int]:
    gt_voxels = gt.full.bit_count()
    pred_voxels = pred.full.bit_count()
    gt_internal = gt.internal[margin]
    pred_internal = pred.internal[margin]
    gt_internal_voxels = gt_internal.bit_count()
    pred_internal_voxels = pred_internal.bit_count()
    true_positive = (gt_internal & pred_internal).bit_count()
    precision = overlap_ratio(
        true_positive, pred_internal_voxels, gt_internal_voxels == 0
    )
    recall = overlap_ratio(
        true_positive, gt_internal_voxels, pred_internal_voxels == 0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "voxel_iou": jaccard(gt.full, pred.full),
        "exterior_iou": jaccard(gt.exterior[margin], pred.exterior[margin]),
        "internal_precision": precision,
        "internal_recall": recall,
        "internal_f1": f1,
        "gt_voxels": gt_voxels,
        "pred_voxels": pred_voxels,
        "gt_internal_voxels": gt_internal_voxels,
        "pred_internal_voxels": pred_internal_voxels,
        "pred_to_gt_voxel_ratio": count_ratio(pred_voxels, gt_voxels),
        "pred_to_gt_internal_ratio": count_ratio(
            pred_internal_voxels, gt_internal_voxels
        ),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict], group_names: tuple[str, ...]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[name] for name in group_names)].append(row)

    output = []
    for key in sorted(groups):
        group_rows = groups[key]
        output.append({
            **dict(zip(group_names, key)),
            **{
                metric: float(np.mean([float(row[metric]) for row in group_rows]))
                for metric in METRIC_NAMES
            },
            "matched_samples": len(group_rows),
        })
    return output


def retrieve(
    query: Sample,
    candidates: list[Sample],
    margin: int,
    maximum_k: int,
) -> list[tuple[Sample, float]]:
    ranked = [
        (candidate, jaccard(query.exterior[margin], candidate.exterior[margin]))
        for candidate in candidates
    ]
    ranked.sort(key=lambda item: (-item[1], item[0].sample_id))
    return ranked[:maximum_k]


def copy_visualizations(
    output_dir: Path,
    query_name: str,
    test_sample: Sample,
    query_sample: Sample,
    top1: Sample,
    oracle: Sample,
) -> None:
    sample_dir = output_dir / "visualizations" / query_name / (
        f"{test_sample.category}__{test_sample.sample_id}"
    )
    sample_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(test_sample.voxel_path, sample_dir / "ground_truth.ply")
    shutil.copyfile(query_sample.voxel_path, sample_dir / "query.ply")
    shutil.copyfile(top1.voxel_path, sample_dir / f"top1__{top1.sample_id}.ply")
    shutil.copyfile(oracle.voxel_path, sample_dir / f"oracle__{oracle.sample_id}.ply")


def evaluate_query_mode(
    query_name: str,
    test_samples: list[Sample],
    query_samples: list[Sample],
    train_by_category: dict[str, list[Sample]],
    margins: list[int],
    retrieval_margin: int,
    oracle_margin: int,
    k_values: list[int],
    output_dir: Path,
    visualizations_per_category: int,
) -> tuple[list[dict], list[dict]]:
    query_by_id = {sample.sample_id: sample for sample in query_samples}
    maximum_k = max(k_values)
    selection_rows = []
    ranking_rows = []
    visualized = defaultdict(int)

    print(f"Retrieving with {query_name}")
    for index, test_sample in enumerate(test_samples, start=1):
        query_sample = query_by_id[test_sample.sample_id]
        candidates = train_by_category.get(test_sample.category, [])
        if not candidates:
            raise ValueError(f"No training candidates for category {test_sample.category}")
        if maximum_k > len(candidates):
            raise ValueError(
                f"Requested K={maximum_k}, but category {test_sample.category} has only "
                f"{len(candidates)} training shapes"
            )

        ranking = retrieve(query_sample, candidates, retrieval_margin, maximum_k)
        oracle_scores = [
            float(score_pair(test_sample, candidate, oracle_margin)["internal_f1"])
            for candidate, _retrieval_score in ranking
        ]

        for rank, ((candidate, retrieval_score), oracle_score) in enumerate(
            zip(ranking, oracle_scores), start=1
        ):
            ranking_rows.append({
                "query_mode": query_name,
                "sample_id": test_sample.sample_id,
                "category": test_sample.category,
                "rank": rank,
                "retrieved_id": candidate.sample_id,
                "retrieval_exterior_iou": retrieval_score,
                f"internal_f1_margin_{oracle_margin}": oracle_score,
            })

        selections = [("top1", 1, 0)]
        for k in k_values:
            if k > 1:
                best_index = max(
                    range(k),
                    key=lambda candidate_index: (
                        oracle_scores[candidate_index],
                        ranking[candidate_index][1],
                        ranking[candidate_index][0].sample_id,
                    ),
                )
                selections.append(("oracle", k, best_index))

        for selection, k, candidate_index in selections:
            candidate, retrieval_score = ranking[candidate_index]
            for margin in margins:
                selection_rows.append({
                    "query_mode": query_name,
                    "selection": selection,
                    "k": k,
                    "sample_id": test_sample.sample_id,
                    "category": test_sample.category,
                    "retrieved_id": candidate.sample_id,
                    "retrieved_rank": candidate_index + 1,
                    "retrieval_exterior_iou": retrieval_score,
                    "margin": margin,
                    **score_pair(test_sample, candidate, margin),
                })

        if visualized[test_sample.category] < visualizations_per_category:
            oracle_index = max(
                range(maximum_k),
                key=lambda candidate_index: (
                    oracle_scores[candidate_index],
                    ranking[candidate_index][1],
                    ranking[candidate_index][0].sample_id,
                ),
            )
            copy_visualizations(
                output_dir,
                query_name,
                test_sample,
                query_sample,
                ranking[0][0],
                ranking[oracle_index][0],
            )
            visualized[test_sample.category] += 1

        if index % 25 == 0 or index == len(test_samples):
            print(f"  retrieved {index}/{len(test_samples)}")

    return selection_rows, ranking_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate category-restricted retrieval by exterior voxel IoU."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pred-voxels",
        type=Path,
        help="Optional directory containing one predicted <sample_id>.ply per test object.",
    )
    parser.add_argument("--pred-name", default="objective1")
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--margins", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--retrieval-margin", type=int, default=2)
    parser.add_argument("--oracle-margin", type=int, default=2)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 20])
    parser.add_argument("--visualizations-per-category", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resolution < 1:
        raise ValueError("--resolution must be positive")
    if any(margin < 1 for margin in args.margins):
        raise ValueError("--margins must contain positive integers")
    if len(args.margins) != len(set(args.margins)):
        raise ValueError("--margins must not contain duplicates")
    if args.retrieval_margin not in args.margins:
        raise ValueError("--retrieval-margin must be included in --margins")
    if args.oracle_margin not in args.margins:
        raise ValueError("--oracle-margin must be included in --margins")
    if any(k < 1 for k in args.k) or len(args.k) != len(set(args.k)):
        raise ValueError("--k must contain unique positive integers")
    if args.visualizations_per_category < 0:
        raise ValueError("--visualizations-per-category cannot be negative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}. Use a new directory."
        )

    margins = sorted(args.margins)
    k_values = sorted(args.k)
    required_margins = sorted(set(margins + [args.retrieval_margin, args.oracle_margin]))
    selected_ids = read_ids(args.ids_file)

    train_samples = load_split(
        args.dataset_root / "train", args.resolution, required_margins
    )
    test_samples = load_split(
        args.dataset_root / "test", args.resolution, required_margins, selected_ids
    )
    train_ids = {sample.sample_id for sample in train_samples}
    test_ids = {sample.sample_id for sample in test_samples}
    overlap = sorted(train_ids & test_ids)
    if overlap:
        raise ValueError(f"Train/test ID leakage detected: {overlap[:5]}")

    train_by_category = defaultdict(list)
    for sample in train_samples:
        train_by_category[sample.category].append(sample)

    query_modes = [("gt_exterior", test_samples)]
    if args.pred_voxels is not None:
        predicted_queries = load_prediction_queries(
            test_samples, args.pred_voxels, args.resolution, required_margins
        )
        query_modes.append((f"{args.pred_name}_exterior", predicted_queries))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection_rows = []
    ranking_rows = []
    for query_name, query_samples in query_modes:
        mode_selections, mode_rankings = evaluate_query_mode(
            query_name=query_name,
            test_samples=test_samples,
            query_samples=query_samples,
            train_by_category=train_by_category,
            margins=margins,
            retrieval_margin=args.retrieval_margin,
            oracle_margin=args.oracle_margin,
            k_values=k_values,
            output_dir=args.output_dir,
            visualizations_per_category=args.visualizations_per_category,
        )
        selection_rows.extend(mode_selections)
        ranking_rows.extend(mode_rankings)

    summary_rows = aggregate(
        selection_rows, ("query_mode", "selection", "k", "margin")
    )
    category_summary_rows = aggregate(
        selection_rows, ("query_mode", "selection", "k", "category", "margin")
    )

    write_csv(args.output_dir / "per_sample.csv", selection_rows)
    write_csv(args.output_dir / "rankings.csv", ranking_rows)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_csv(args.output_dir / "category_summary.csv", category_summary_rows)

    print("\nPrimary margin results")
    for row in summary_rows:
        if row["margin"] != args.oracle_margin:
            continue
        print(
            f"  {row['query_mode']} {row['selection']}@{row['k']}: "
            f"IoU={row['voxel_iou']:.4f}, exterior IoU={row['exterior_iou']:.4f}, "
            f"internal F1={row['internal_f1']:.4f}"
        )
    print(f"\nWrote retrieval results to {args.output_dir}")


if __name__ == "__main__":
    main()
