#!/usr/bin/env python3
"""Evaluate deployable DINO reranking and consensus interior completion.

The script reuses the saved view-18 DINO rankings. Candidate interiors stay in
ShapeNet's canonical frame, are clipped to Objective 1's conservative enclosed
volume, and are added without changing any Objective-1 voxel. It evaluates:

* direct safe union with DINO top-1;
* the previous AABB-aligned top-1 safe union as a checksum;
* top-K reranking by DINO similarity plus Objective-1 exterior compatibility;
* top-K consensus, which inserts only voxels supported by multiple candidates.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from compare_internals import interior, read_voxels
from evaluate_dino_interior_transplant import (
    align_voxels,
    enclosed_volume,
    read_metadata,
    read_selected_ids,
    score_voxels,
    write_voxels,
)
from evaluate_retrieval_completion import aggregate, write_csv


def parse_consensus_specs(values: list[str]) -> list[tuple[int, int]]:
    specs = []
    for value in values:
        try:
            k_text, support_text = value.split(":", 1)
            k, support = int(k_text), int(support_text)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"Invalid consensus specification {value!r}; expected K:SUPPORT"
            ) from error
        if k < 1 or support < 1 or support > k:
            raise argparse.ArgumentTypeError(
                f"Invalid consensus specification {value!r}; require 1 <= SUPPORT <= K"
            )
        specs.append((k, support))
    if len(specs) != len(set(specs)):
        raise argparse.ArgumentTypeError("Consensus specifications must be unique")
    return specs


def read_rankings(
    path: Path,
    query_mode: str,
    selected_ids: set[str] | None,
    maximum_k: int,
) -> dict[str, list[dict[str, str | int | float]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing DINO rankings: {path}")
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    required = {
        "query_mode",
        "sample_id",
        "category",
        "rank",
        "retrieved_id",
        "image_cosine_similarity",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    rankings = defaultdict(list)
    for row in rows:
        sample_id = row["sample_id"].strip()
        rank = int(row["rank"])
        if (
            row["query_mode"] != query_mode
            or rank > maximum_k
            or (selected_ids is not None and sample_id not in selected_ids)
        ):
            continue
        rankings[sample_id].append({
            "rank": rank,
            "category": row["category"].strip(),
            "retrieved_id": row["retrieved_id"].strip(),
            "image_similarity": float(row["image_cosine_similarity"]),
        })

    for sample_id, ranking in rankings.items():
        ranking.sort(key=lambda row: int(row["rank"]))
        ranks = [int(row["rank"]) for row in ranking]
        expected = list(range(1, maximum_k + 1))
        if ranks != expected:
            raise ValueError(
                f"Incomplete top-{maximum_k} ranking for {sample_id}: {ranks[:5]}..."
            )
        retrieved_ids = [str(row["retrieved_id"]) for row in ranking]
        if len(retrieved_ids) != len(set(retrieved_ids)):
            raise ValueError(f"Duplicate retrieved candidate for {sample_id}")
    if not rankings:
        raise ValueError(f"No {query_mode} rankings found in {path}")
    return dict(rankings)


def intersection_over_union(
    left: set[tuple[int, int, int]],
    right: set[tuple[int, int, int]],
) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def lambda_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def paired_summary(
    rows: list[dict],
    baseline_method: str,
    margin: int,
) -> list[dict]:
    by_method = defaultdict(dict)
    for row in rows:
        if int(row["margin"]) != margin:
            continue
        by_method[row["method"]][row["sample_id"]] = float(row["internal_f1"])

    baseline = by_method[baseline_method]
    output = []
    rng = np.random.default_rng(12345)
    for method in sorted(by_method):
        sample_ids = sorted(baseline.keys() & by_method[method].keys())
        deltas = np.asarray(
            [by_method[method][sample_id] - baseline[sample_id] for sample_id in sample_ids],
            dtype=np.float64,
        )
        if method == baseline_method:
            confidence_low = confidence_high = 0.0
        else:
            bootstrap_indices = rng.integers(
                0, len(deltas), size=(10000, len(deltas)), endpoint=False
            )
            bootstrap_means = deltas[bootstrap_indices].mean(axis=1)
            confidence_low, confidence_high = np.quantile(
                bootstrap_means, [0.025, 0.975]
            )
        output.append({
            "method": method,
            "margin": margin,
            "mean_internal_f1_delta": float(deltas.mean()),
            "median_internal_f1_delta": float(np.median(deltas)),
            "wins": int(np.count_nonzero(deltas > 1e-12)),
            "losses": int(np.count_nonzero(deltas < -1e-12)),
            "ties": int(np.count_nonzero(np.abs(deltas) <= 1e-12)),
            "bootstrap_95_low": float(confidence_low),
            "bootstrap_95_high": float(confidence_high),
            "matched_samples": len(sample_ids),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate view-18 DINO reranking and consensus safe union."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--objective1-voxels", type=Path, required=True)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--view-index", type=int, default=18)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--transplant-margin", type=int, default=2)
    parser.add_argument("--margins", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--rerank-k", type=int, default=20)
    parser.add_argument(
        "--rerank-lambdas", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0]
    )
    parser.add_argument(
        "--consensus",
        nargs="+",
        default=["5:2", "5:3", "20:2", "20:3", "20:5"],
        metavar="K:SUPPORT",
    )
    parser.add_argument("--visualizations-per-category", type=int, default=3)
    args = parser.parse_args()

    consensus_specs = parse_consensus_specs(args.consensus)
    if args.view_index < 0:
        raise ValueError("--view-index cannot be negative")
    if args.resolution < 1 or args.transplant_margin < 1:
        raise ValueError("Resolution and transplant margin must be positive")
    if any(margin < 1 for margin in args.margins):
        raise ValueError("--margins must contain positive integers")
    if len(args.margins) != len(set(args.margins)):
        raise ValueError("--margins must not contain duplicates")
    if args.transplant_margin not in args.margins:
        raise ValueError("--transplant-margin must be included in --margins")
    if args.rerank_k < 1:
        raise ValueError("--rerank-k must be positive")
    if any(value < 0 for value in args.rerank_lambdas):
        raise ValueError("--rerank-lambdas cannot contain negative values")
    if len(args.rerank_lambdas) != len(set(args.rerank_lambdas)):
        raise ValueError("--rerank-lambdas must be unique")
    if args.visualizations_per_category < 0:
        raise ValueError("--visualizations-per-category cannot be negative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}. Use a new directory."
        )

    maximum_k = max([args.rerank_k, *(k for k, _support in consensus_specs)])
    query_mode = f"dino_view{args.view_index:03d}"
    selected_ids = read_selected_ids(args.ids_file)
    rankings = read_rankings(
        args.rankings, query_mode, selected_ids, maximum_k
    )

    train_dir = args.dataset_root / "train"
    test_dir = args.dataset_root / "test"
    train_rows = read_metadata(train_dir / "metadata.csv")
    test_rows = read_metadata(test_dir / "metadata.csv")
    if selected_ids is not None:
        known_ids = {row["sha256"] for row in test_rows}
        missing_ids = sorted(selected_ids - known_ids)
        if missing_ids:
            raise ValueError(f"IDs missing from test metadata: {missing_ids[:5]}")
        test_rows = [row for row in test_rows if row["sha256"] in selected_ids]

    train_categories = {row["sha256"]: row["category"] for row in train_rows}
    expected_test_ids = {row["sha256"] for row in test_rows}
    missing_rankings = sorted(expected_test_ids - set(rankings))
    extra_rankings = sorted(set(rankings) - expected_test_ids)
    if missing_rankings or extra_rankings:
        raise ValueError(
            f"Ranking/test mismatch: missing={missing_rankings[:5]}, "
            f"extra={extra_rankings[:5]}"
        )

    candidate_cache = {}
    internal_cache = {}
    exterior_cache = {}

    def load_candidate(candidate_id: str):
        if candidate_id not in candidate_cache:
            path = train_dir / "voxels" / f"{candidate_id}.ply"
            if not path.is_file():
                raise FileNotFoundError(f"Missing training voxel PLY: {path}")
            voxels = read_voxels(path, args.resolution)
            candidate_cache[candidate_id] = voxels
            candidate_internal = interior(voxels, args.transplant_margin)
            internal_cache[candidate_id] = candidate_internal
            exterior_cache[candidate_id] = voxels - candidate_internal
        return candidate_cache[candidate_id]

    margins = sorted(args.margins)
    per_sample_rows = []
    visualized = defaultdict(int)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for index, test_row in enumerate(test_rows, start=1):
        sample_id = test_row["sha256"]
        category = test_row["category"]
        ranking = rankings[sample_id]
        if any(row["category"] != category for row in ranking):
            raise ValueError(f"Ranking category mismatch for {sample_id}")

        gt_path = test_dir / "voxels" / f"{sample_id}.ply"
        objective1_path = args.objective1_voxels / f"{sample_id}.ply"
        for required_path in (gt_path, objective1_path):
            if not required_path.is_file():
                raise FileNotFoundError(f"Missing voxel PLY: {required_path}")
        gt = read_voxels(gt_path, args.resolution)
        objective1 = read_voxels(objective1_path, args.resolution)
        objective1_internal = interior(objective1, args.transplant_margin)
        objective1_exterior = objective1 - objective1_internal
        safe_volume = enclosed_volume(
            objective1_exterior, args.transplant_margin
        )

        for candidate_row in ranking:
            candidate_id = str(candidate_row["retrieved_id"])
            if candidate_id not in train_categories:
                raise ValueError(f"Unknown training candidate: {candidate_id}")
            if train_categories[candidate_id] != category:
                raise ValueError(
                    f"Candidate category mismatch for {sample_id}: {candidate_id}"
                )
            load_candidate(candidate_id)
            candidate_row["exterior_compatibility"] = intersection_over_union(
                objective1_exterior, exterior_cache[candidate_id]
            )

        variants = {}

        def add_variant(
            method: str,
            prediction: set[tuple[int, int, int]],
            selection: dict[str, str | int | float] | None = None,
            consensus_k: int = 0,
            consensus_support: int = 0,
            supported_voxels: int = 0,
        ) -> None:
            selection = selection or {}
            variants[method] = {
                "prediction": prediction,
                "selected_id": selection.get("retrieved_id", ""),
                "selected_rank": selection.get("rank", 0),
                "image_similarity": selection.get("image_similarity", 0.0),
                "exterior_compatibility": selection.get(
                    "exterior_compatibility", 0.0
                ),
                "consensus_k": consensus_k,
                "consensus_support": consensus_support,
                "supported_voxels": supported_voxels,
                "safe_inserted_voxels": len(prediction - objective1),
            }

        add_variant("objective1", objective1)

        top1 = ranking[0]
        top1_id = str(top1["retrieved_id"])
        top1_internal = internal_cache[top1_id]
        top1_direct = objective1 | (top1_internal & safe_volume)
        add_variant("top1_union_direct_safe", top1_direct, top1)

        top1_aligned_internal = align_voxels(
            top1_internal,
            candidate_cache[top1_id],
            objective1,
            args.resolution,
        )
        top1_aligned = objective1 | (top1_aligned_internal & safe_volume)
        add_variant("top1_union_aligned_safe", top1_aligned, top1)

        rerank_pool = ranking[: args.rerank_k]
        for weight in args.rerank_lambdas:
            selected = max(
                rerank_pool,
                key=lambda row: (
                    float(row["image_similarity"])
                    + weight * float(row["exterior_compatibility"]),
                    float(row["image_similarity"]),
                    -int(row["rank"]),
                ),
            )
            selected_id = str(selected["retrieved_id"])
            prediction = objective1 | (
                internal_cache[selected_id] & safe_volume
            )
            method = (
                f"rerank_k{args.rerank_k}_lam{lambda_label(weight)}_union_direct_safe"
            )
            add_variant(method, prediction, selected)

        vote_counts_by_k = {}
        for k, support in consensus_specs:
            if k not in vote_counts_by_k:
                vote_counts = Counter()
                for candidate_row in ranking[:k]:
                    candidate_id = str(candidate_row["retrieved_id"])
                    vote_counts.update(internal_cache[candidate_id])
                vote_counts_by_k[k] = vote_counts
            vote_counts = vote_counts_by_k[k]
            supported = {
                voxel for voxel, count in vote_counts.items() if count >= support
            }
            prediction = objective1 | (supported & safe_volume)
            add_variant(
                f"consensus_k{k}_s{support}_union_direct_safe",
                prediction,
                consensus_k=k,
                consensus_support=support,
                supported_voxels=len(supported),
            )

        for method, variant in variants.items():
            prediction = variant["prediction"]
            for margin in margins:
                per_sample_rows.append({
                    "method": method,
                    "sample_id": sample_id,
                    "category": category,
                    "margin": margin,
                    **score_voxels(gt, prediction, margin),
                    "selected_id": variant["selected_id"],
                    "selected_rank": variant["selected_rank"],
                    "image_similarity": variant["image_similarity"],
                    "exterior_compatibility": variant["exterior_compatibility"],
                    "consensus_k": variant["consensus_k"],
                    "consensus_support": variant["consensus_support"],
                    "supported_voxels": variant["supported_voxels"],
                    "objective1_internal_voxels": len(objective1_internal),
                    "safe_volume_voxels": len(safe_volume),
                    "safe_inserted_voxels": variant["safe_inserted_voxels"],
                })

        if visualized[category] < args.visualizations_per_category:
            sample_dir = args.output_dir / "visualizations" / sample_id
            write_voxels(sample_dir / "ground_truth.ply", gt, args.resolution)
            for method, variant in variants.items():
                write_voxels(
                    sample_dir / f"{method}.ply",
                    variant["prediction"],
                    args.resolution,
                )
            visualized[category] += 1

        if index % 25 == 0 or index == len(test_rows):
            print(f"Processed {index}/{len(test_rows)}")

    primary_exterior = {
        row["sample_id"]: float(row["exterior_iou"])
        for row in per_sample_rows
        if row["method"] == "objective1"
        and int(row["margin"]) == args.transplant_margin
    }
    for row in per_sample_rows:
        if int(row["margin"]) != args.transplant_margin:
            continue
        expected = primary_exterior[row["sample_id"]]
        if abs(float(row["exterior_iou"]) - expected) > 1e-12:
            raise AssertionError(
                f"{row['method']} changed the margin-{args.transplant_margin} "
                f"exterior for {row['sample_id']}"
            )

    summary_rows = aggregate(per_sample_rows, ("method", "margin"))
    category_summary_rows = aggregate(
        per_sample_rows, ("method", "category", "margin")
    )
    paired_rows = paired_summary(
        per_sample_rows, "objective1", args.transplant_margin
    )
    write_csv(args.output_dir / "per_sample.csv", per_sample_rows)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_csv(args.output_dir / "category_summary.csv", category_summary_rows)
    write_csv(args.output_dir / "paired_summary.csv", paired_rows)
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "dataset_root": str(args.dataset_root),
                "objective1_voxels": str(args.objective1_voxels),
                "rankings": str(args.rankings),
                "view_index": args.view_index,
                "resolution": args.resolution,
                "transplant_margin": args.transplant_margin,
                "margins": margins,
                "rerank_k": args.rerank_k,
                "rerank_lambdas": args.rerank_lambdas,
                "consensus": [f"{k}:{support}" for k, support in consensus_specs],
            },
            file,
            indent=2,
        )
        file.write("\n")

    primary_rows = [
        row for row in summary_rows if row["margin"] == args.transplant_margin
    ]
    primary_rows.sort(key=lambda row: row["internal_f1"], reverse=True)
    paired_by_method = {row["method"]: row for row in paired_rows}
    print("\nPrimary margin results (best internal F1 first)")
    for row in primary_rows:
        paired = paired_by_method[row["method"]]
        print(
            f"  {row['method']}: IoU={row['voxel_iou']:.4f}, "
            f"exterior IoU={row['exterior_iou']:.4f}, "
            f"internal P/R/F1={row['internal_precision']:.4f}/"
            f"{row['internal_recall']:.4f}/{row['internal_f1']:.4f}, "
            f"delta={paired['mean_internal_f1_delta']:+.4f}, "
            f"W/L/T={paired['wins']}/{paired['losses']}/{paired['ties']}"
        )
    print(f"\nWrote reranking and consensus results to {args.output_dir}")


if __name__ == "__main__":
    main()
