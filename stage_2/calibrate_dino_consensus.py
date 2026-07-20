#!/usr/bin/env python3
"""Calibrate DINO consensus thresholds with training-only leave-one-out retrieval."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from compare_internals import interior, read_voxels
from evaluate_dino_interior_transplant import read_metadata
from evaluate_dino_rerank_consensus import parse_consensus_specs
from evaluate_retrieval_completion import write_csv


def load_train_embeddings(
    path: Path,
    rows: list[dict[str, str]],
    view_index: int,
    model_name: str,
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing DINO embedding cache: {path}")
    with np.load(path, allow_pickle=False) as data:
        sample_ids = data["sample_ids"].astype(str).tolist()
        categories = data["categories"].astype(str).tolist()
        cached_view = int(data["view_index"].item())
        cached_model = str(data["model_name"].item())
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)

    expected_ids = [row["sha256"] for row in rows]
    expected_categories = [row["category"] for row in rows]
    if sample_ids != expected_ids or categories != expected_categories:
        raise ValueError(f"Embedding order does not match training metadata: {path}")
    if cached_view != view_index or cached_model != model_name:
        raise ValueError(
            f"Embedding configuration mismatch: view={cached_view}, model={cached_model}"
        )
    if embeddings.ndim != 2 or embeddings.shape[0] != len(rows):
        raise ValueError(f"Invalid embedding shape: {embeddings.shape}")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError("Expected L2-normalized DINO embeddings")
    return embeddings


def internal_metrics(
    ground_truth: set[tuple[int, int, int]],
    prediction: set[tuple[int, int, int]],
) -> dict[str, float | int]:
    true_positive = len(ground_truth & prediction)
    if prediction:
        precision = true_positive / len(prediction)
    else:
        precision = 1.0 if not ground_truth else 0.0
    if ground_truth:
        recall = true_positive / len(ground_truth)
    else:
        recall = 1.0 if not prediction else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "internal_precision": precision,
        "internal_recall": recall,
        "internal_f1": f1,
        "gt_internal_voxels": len(ground_truth),
        "pred_internal_voxels": len(prediction),
        "pred_to_gt_internal_ratio": (
            len(prediction) / len(ground_truth)
            if ground_truth
            else (1.0 if not prediction else 0.0)
        ),
    }


def summarize(
    rows: list[dict],
    group_names: tuple[str, ...],
) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[name] for name in group_names)].append(row)
    output = []
    for key in sorted(groups):
        group = groups[key]
        output.append({
            **dict(zip(group_names, key)),
            **{
                name: float(np.mean([float(row[name]) for row in group]))
                for name in (
                    "internal_precision",
                    "internal_recall",
                    "internal_f1",
                    "pred_to_gt_internal_ratio",
                )
            },
            "matched_samples": len(group),
        })
    return output


def choose_policy(
    category_rows: list[dict],
) -> dict[str, dict[str, float | int | str]]:
    by_category = defaultdict(list)
    for row in category_rows:
        by_category[row["category"]].append(row)
    policy = {}
    for category, rows in sorted(by_category.items()):
        selected = max(
            rows,
            key=lambda row: (
                float(row["internal_f1"]),
                float(row["internal_precision"]),
                int(row["support"]),
                -int(row["k"]),
            ),
        )
        k, support = int(selected["k"]), int(selected["support"])
        policy[category] = {
            "k": k,
            "support": support,
            "source_method": f"consensus_k{k}_s{support}_union_direct_safe",
            "calibration_internal_precision": float(selected["internal_precision"]),
            "calibration_internal_recall": float(selected["internal_recall"]),
            "calibration_internal_f1": float(selected["internal_f1"]),
            "calibration_internal_ratio": float(
                selected["pred_to_gt_internal_ratio"]
            ),
            "calibration_samples": int(selected["matched_samples"]),
        }
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Training-only leave-one-out DINO consensus calibration."
    )
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--view-index", type=int, default=18)
    parser.add_argument("--model", default="dinov2_vitl14_reg")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--margin", type=int, default=2)
    parser.add_argument(
        "--consensus",
        nargs="+",
        default=["5:2", "5:3", "20:2", "20:3", "20:5"],
        metavar="K:SUPPORT",
    )
    args = parser.parse_args()

    specs = parse_consensus_specs(args.consensus)
    if args.view_index < 0 or args.resolution < 1 or args.margin < 1:
        raise ValueError("View index cannot be negative; resolution and margin must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}. Use a new directory."
        )

    metadata = read_metadata(args.train_dir / "metadata.csv")
    embeddings = load_train_embeddings(
        args.embeddings, metadata, args.view_index, args.model
    )
    maximum_k = max(k for k, _support in specs)

    print(f"Loading internals for {len(metadata)} training shapes")
    internals = {}
    for index, row in enumerate(metadata, start=1):
        sample_id = row["sha256"]
        voxel_path = args.train_dir / "voxels" / f"{sample_id}.ply"
        if not voxel_path.is_file():
            raise FileNotFoundError(f"Missing training voxel PLY: {voxel_path}")
        internals[sample_id] = interior(
            read_voxels(voxel_path, args.resolution), args.margin
        )
        if index % 100 == 0 or index == len(metadata):
            print(f"  loaded {index}/{len(metadata)}")

    category_indices = defaultdict(list)
    for index, row in enumerate(metadata):
        category_indices[row["category"]].append(index)

    per_sample_rows = []
    for category, indices in sorted(category_indices.items()):
        if len(indices) <= maximum_k:
            raise ValueError(
                f"Category {category} has {len(indices)} shapes; need more than {maximum_k}"
            )
        category_embeddings = embeddings[indices]
        similarities = category_embeddings @ category_embeddings.T
        np.fill_diagonal(similarities, -np.inf)
        orders = np.argsort(-similarities, axis=1, kind="stable")[:, :maximum_k]
        print(f"Calibrating {category}: {len(indices)} leave-one-out queries")

        for local_query_index, global_query_index in enumerate(indices):
            query = metadata[global_query_index]
            query_internal = internals[query["sha256"]]
            ranked_ids = [
                metadata[indices[local_candidate_index]]["sha256"]
                for local_candidate_index in orders[local_query_index]
            ]
            vote_counts_by_k = {}
            for k, support in specs:
                if k not in vote_counts_by_k:
                    counts = Counter()
                    for candidate_id in ranked_ids[:k]:
                        counts.update(internals[candidate_id])
                    vote_counts_by_k[k] = counts
                prediction = {
                    voxel
                    for voxel, count in vote_counts_by_k[k].items()
                    if count >= support
                }
                per_sample_rows.append({
                    "sample_id": query["sha256"],
                    "category": category,
                    "k": k,
                    "support": support,
                    **internal_metrics(query_internal, prediction),
                })

    summary_rows = summarize(per_sample_rows, ("k", "support"))
    category_summary_rows = summarize(
        per_sample_rows, ("category", "k", "support")
    )
    policy = choose_policy(category_summary_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_sample.csv", per_sample_rows)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_csv(args.output_dir / "category_summary.csv", category_summary_rows)
    with (args.output_dir / "policy.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "calibration": "training-only leave-one-out",
                "view_index": args.view_index,
                "model": args.model,
                "resolution": args.resolution,
                "margin": args.margin,
                "candidate_specs": [
                    {"k": k, "support": support} for k, support in specs
                ],
                "categories": policy,
            },
            file,
            indent=2,
        )
        file.write("\n")

    print("\nFrozen category policy")
    for category, selection in policy.items():
        print(
            f"  {category}: K={selection['k']}, support={selection['support']}, "
            f"leave-one-out F1={selection['calibration_internal_f1']:.4f}"
        )
    print(f"\nWrote frozen policy to {args.output_dir / 'policy.json'}")


if __name__ == "__main__":
    main()
