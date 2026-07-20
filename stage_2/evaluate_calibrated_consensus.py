#!/usr/bin/env python3
"""Apply a frozen train-calibrated DINO consensus policy to Objective 1."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from compare_internals import interior, read_voxels
from evaluate_dino_interior_transplant import (
    enclosed_volume,
    read_metadata,
    read_selected_ids,
    score_voxels,
    write_voxels,
)
from evaluate_dino_rerank_consensus import (
    paired_summary,
    parse_consensus_specs,
    read_rankings,
)
from evaluate_retrieval_completion import aggregate, write_csv


def read_policy(
    path: Path,
    view_index: int,
    resolution: int,
    margin: int,
) -> dict[str, tuple[int, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing calibrated policy: {path}")
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    for name, expected in (
        ("view_index", view_index),
        ("resolution", resolution),
        ("margin", margin),
    ):
        if int(payload.get(name, -1)) != expected:
            raise ValueError(
                f"Policy {name}={payload.get(name)!r}, expected {expected}"
            )
    categories = payload.get("categories")
    if not isinstance(categories, dict) or not categories:
        raise ValueError(f"Policy contains no category selections: {path}")
    output = {}
    for category, selection in categories.items():
        k, support = int(selection["k"]), int(selection["support"])
        if k < 1 or support < 1 or support > k:
            raise ValueError(f"Invalid policy for {category}: K={k}, support={support}")
        output[category] = (k, support)
    return output


def choose_visualizations(
    per_sample_rows: list[dict],
    count_per_category: int,
    margin: int,
) -> list[dict]:
    if count_per_category == 0:
        return []
    primary = defaultdict(dict)
    for row in per_sample_rows:
        if int(row["margin"]) == margin:
            primary[row["method"]][row["sample_id"]] = row
    by_category = defaultdict(list)
    for sample_id, final_row in primary["calibrated_consensus"].items():
        baseline = primary["objective1"][sample_id]
        by_category[final_row["category"]].append({
            "sample_id": sample_id,
            "category": final_row["category"],
            "objective1_internal_f1": float(baseline["internal_f1"]),
            "calibrated_internal_f1": float(final_row["internal_f1"]),
            "internal_f1_delta": (
                float(final_row["internal_f1"])
                - float(baseline["internal_f1"])
            ),
        })

    selected = []
    for category, rows in sorted(by_category.items()):
        rows.sort(key=lambda row: row["internal_f1_delta"], reverse=True)
        number = min(count_per_category, len(rows))
        if number == 1:
            indices = [len(rows) // 2]
        else:
            indices = [
                round(index * (len(rows) - 1) / (number - 1))
                for index in range(number)
            ]
        labels = (
            ["typical"]
            if number == 1
            else [
                "best" if index == 0 else "worst" if index == number - 1 else "typical"
                for index in range(number)
            ]
        )
        for index, label in zip(indices, labels):
            selected.append({**rows[index], "selection": label})
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen train-calibrated consensus policy."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--objective1-voxels", type=Path, required=True)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--view-index", type=int, default=18)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--transplant-margin", type=int, default=2)
    parser.add_argument("--margins", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--global-reference", default="20:5", metavar="K:SUPPORT")
    parser.add_argument("--visualizations-per-category", type=int, default=3)
    args = parser.parse_args()

    if args.view_index < 0 or args.resolution < 1 or args.transplant_margin < 1:
        raise ValueError("View index cannot be negative; resolution and margin must be positive")
    if any(margin < 1 for margin in args.margins):
        raise ValueError("--margins must contain positive integers")
    if len(args.margins) != len(set(args.margins)):
        raise ValueError("--margins must not contain duplicates")
    if args.transplant_margin not in args.margins:
        raise ValueError("--transplant-margin must be included in --margins")
    if args.visualizations_per_category < 0:
        raise ValueError("--visualizations-per-category cannot be negative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}. Use a new directory."
        )

    global_k, global_support = parse_consensus_specs([args.global_reference])[0]
    policy = read_policy(
        args.policy,
        args.view_index,
        args.resolution,
        args.transplant_margin,
    )
    maximum_k = max(global_k, *(k for k, _support in policy.values()))
    selected_ids = read_selected_ids(args.ids_file)
    query_mode = f"dino_view{args.view_index:03d}"
    rankings = read_rankings(
        args.rankings, query_mode, selected_ids, maximum_k
    )

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

    test_categories = {row["category"] for row in test_rows}
    if test_categories != set(policy):
        raise ValueError(
            f"Policy/test category mismatch: policy={sorted(policy)}, "
            f"test={sorted(test_categories)}"
        )
    train_categories = {row["sha256"]: row["category"] for row in train_rows}
    expected_test_ids = {row["sha256"] for row in test_rows}
    if set(rankings) != expected_test_ids:
        raise ValueError("Ranking IDs do not exactly match the selected test IDs")

    candidate_voxels = {}
    candidate_internals = {}

    def load_candidate(candidate_id: str):
        if candidate_id not in candidate_voxels:
            path = train_dir / "voxels" / f"{candidate_id}.ply"
            if not path.is_file():
                raise FileNotFoundError(f"Missing training voxel PLY: {path}")
            voxels = read_voxels(path, args.resolution)
            candidate_voxels[candidate_id] = voxels
            candidate_internals[candidate_id] = interior(
                voxels, args.transplant_margin
            )

    margins = sorted(args.margins)
    per_sample_rows = []
    policy_rows = []
    prediction_dir = args.output_dir / "predictions" / "calibrated_consensus"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for index, test_row in enumerate(test_rows, start=1):
        sample_id = test_row["sha256"]
        category = test_row["category"]
        calibrated_k, calibrated_support = policy[category]
        ranking = rankings[sample_id]
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

        required_ks = {global_k, calibrated_k}
        counts_by_k = {}
        for k in sorted(required_ks):
            counts = Counter()
            for candidate_row in ranking[:k]:
                candidate_id = str(candidate_row["retrieved_id"])
                if candidate_id not in train_categories:
                    raise ValueError(f"Unknown training candidate: {candidate_id}")
                if train_categories[candidate_id] != category:
                    raise ValueError(
                        f"Candidate category mismatch for {sample_id}: {candidate_id}"
                    )
                load_candidate(candidate_id)
                counts.update(candidate_internals[candidate_id])
            counts_by_k[k] = counts

        def consensus_prediction(k: int, support: int):
            supported = {
                voxel for voxel, count in counts_by_k[k].items() if count >= support
            }
            prediction = objective1 | (supported & safe_volume)
            return prediction, supported

        global_prediction, global_supported = consensus_prediction(
            global_k, global_support
        )
        final_prediction, final_supported = consensus_prediction(
            calibrated_k, calibrated_support
        )
        variants = {
            "objective1": (objective1, 0, 0, set()),
            f"global_consensus_k{global_k}_s{global_support}": (
                global_prediction,
                global_k,
                global_support,
                global_supported,
            ),
            "calibrated_consensus": (
                final_prediction,
                calibrated_k,
                calibrated_support,
                final_supported,
            ),
        }

        for method, (prediction, k, support, supported) in variants.items():
            for margin in margins:
                per_sample_rows.append({
                    "method": method,
                    "sample_id": sample_id,
                    "category": category,
                    "margin": margin,
                    **score_voxels(gt, prediction, margin),
                    "consensus_k": k,
                    "consensus_support": support,
                    "supported_voxels": len(supported),
                    "objective1_internal_voxels": len(objective1_internal),
                    "safe_volume_voxels": len(safe_volume),
                    "safe_inserted_voxels": len(prediction - objective1),
                })

        write_voxels(
            prediction_dir / f"{sample_id}.ply",
            final_prediction,
            args.resolution,
        )
        policy_rows.append({
            "sample_id": sample_id,
            "category": category,
            "k": calibrated_k,
            "support": calibrated_support,
            "prediction": str(prediction_dir / f"{sample_id}.ply"),
        })
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
        if abs(float(row["exterior_iou"]) - primary_exterior[row["sample_id"]]) > 1e-12:
            raise AssertionError(
                f"{row['method']} changed the primary exterior for {row['sample_id']}"
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
    write_csv(args.output_dir / "applied_policy.csv", policy_rows)

    visualization_rows = choose_visualizations(
        per_sample_rows,
        args.visualizations_per_category,
        args.transplant_margin,
    )
    for row in visualization_rows:
        sample_id = row["sample_id"]
        sample_dir = args.output_dir / "visualizations" / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            test_dir / "voxels" / f"{sample_id}.ply",
            sample_dir / "ground_truth.ply",
        )
        shutil.copyfile(
            args.objective1_voxels / f"{sample_id}.ply",
            sample_dir / "objective1.ply",
        )
        shutil.copyfile(
            prediction_dir / f"{sample_id}.ply",
            sample_dir / "calibrated_consensus.ply",
        )
    if visualization_rows:
        write_csv(args.output_dir / "visualization_manifest.csv", visualization_rows)
    shutil.copyfile(args.policy, args.output_dir / "policy.json")

    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "dataset_root": str(args.dataset_root),
                "objective1_voxels": str(args.objective1_voxels),
                "rankings": str(args.rankings),
                "policy": str(args.policy),
                "view_index": args.view_index,
                "resolution": args.resolution,
                "transplant_margin": args.transplant_margin,
                "margins": margins,
                "global_reference": {
                    "k": global_k,
                    "support": global_support,
                },
            },
            file,
            indent=2,
        )
        file.write("\n")

    print("\nFinal margin results")
    paired_by_method = {row["method"]: row for row in paired_rows}
    for row in summary_rows:
        if row["margin"] != args.transplant_margin:
            continue
        paired = paired_by_method[row["method"]]
        print(
            f"  {row['method']}: IoU={row['voxel_iou']:.4f}, "
            f"exterior IoU={row['exterior_iou']:.4f}, "
            f"internal P/R/F1={row['internal_precision']:.4f}/"
            f"{row['internal_recall']:.4f}/{row['internal_f1']:.4f}, "
            f"delta={paired['mean_internal_f1_delta']:+.4f}, "
            f"W/L/T={paired['wins']}/{paired['losses']}/{paired['ties']}"
        )
    print(f"\nWrote final calibrated results to {args.output_dir}")


if __name__ == "__main__":
    main()
