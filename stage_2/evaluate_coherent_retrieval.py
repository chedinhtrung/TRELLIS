#!/usr/bin/env python3
"""Apply a frozen coherent-retrieval policy to held-out Objective-1 shapes."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from coherent_retrieval import (
    TransferPreset,
    build_support_counts,
    choose_donor,
    component_metrics,
    connected_components,
    transfer_components,
)
from compare_internals import interior, read_voxels
from evaluate_dino_interior_transplant import (
    enclosed_volume,
    read_metadata,
    read_selected_ids,
    score_voxels,
    write_voxels,
)
from evaluate_dino_rerank_consensus import paired_summary, read_rankings
from evaluate_retrieval_completion import write_csv


BASE_METRICS = (
    "voxel_iou",
    "exterior_iou",
    "internal_precision",
    "internal_recall",
    "internal_f1",
    "internal_f05",
    "pred_to_gt_voxel_ratio",
    "pred_to_gt_internal_ratio",
    "internal_components_26",
    "small_internal_components_26",
    "small_internal_component_fraction",
    "volumetric_core_fraction",
    "objective1_shell_preservation",
)


def read_policy(
    path: Path,
    view_index: int,
    resolution: int,
    margin: int,
) -> dict[str, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing coherent-retrieval policy: {path}")
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if payload.get("method") != "coherent_single_donor_component_transfer_v1":
        raise ValueError(f"Unsupported policy method in {path}: {payload.get('method')!r}")
    for name, expected in (
        ("view_index", view_index),
        ("resolution", resolution),
        ("margin", margin),
    ):
        if int(payload.get(name, -1)) != expected:
            raise ValueError(f"Policy {name}={payload.get(name)!r}, expected {expected}")
    categories = payload.get("categories")
    if not isinstance(categories, dict) or not categories:
        raise ValueError(f"Policy contains no category selections: {path}")
    max_calibration_ratio = float(
        payload.get("max_calibration_internal_ratio", 1.15)
    )
    output = {}
    for category, selection in categories.items():
        preset = TransferPreset(**selection["preset"])
        selection_k = int(selection["selection_k"])
        support_k = int(selection["support_k"])
        exterior_weight = float(selection["exterior_weight"])
        budget_ratio = float(selection["max_internal_to_exterior_ratio"])
        if selection_k < 1 or support_k < selection_k:
            raise ValueError(f"Invalid retrieval K values for {category}")
        if exterior_weight < 0 or budget_ratio < 0:
            raise ValueError(f"Invalid weights or budget for {category}")
        if (
            float(selection["calibration_micro_internal_ratio"])
            > max_calibration_ratio + 1e-12
        ):
            raise ValueError(f"Overfilling policy was selected for {category}")
        output[category] = {
            "selection_k": selection_k,
            "support_k": support_k,
            "exterior_weight": exterior_weight,
            "budget_ratio": budget_ratio,
            "preset": preset,
        }
    return output


def add_derived_metrics(metrics: dict, predicted_internal: set) -> dict:
    precision = float(metrics["internal_precision"])
    recall = float(metrics["internal_recall"])
    beta_squared = 0.25
    metrics["internal_f05"] = (
        (1 + beta_squared) * precision * recall
        / (beta_squared * precision + recall)
        if beta_squared * precision + recall
        else 0.0
    )
    metrics.update(component_metrics(predicted_internal))
    return metrics


def aggregate(rows: list[dict], group_names: tuple[str, ...]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[name] for name in group_names)].append(row)
    output = []
    for key in sorted(groups):
        group = groups[key]
        output.append({
            **dict(zip(group_names, key)),
            **{
                metric: float(np.mean([float(row[metric]) for row in group]))
                for metric in BASE_METRICS
            },
            "matched_samples": len(group),
        })
    return output


def choose_gallery(
    rows: list[dict], categories: list[str], count_per_category: int, margin: int
) -> list[dict]:
    primary = defaultdict(dict)
    for row in rows:
        if int(row["margin"]) == margin:
            primary[row["method"]][row["sample_id"]] = row
    selected = []
    for category in categories:
        candidates = []
        for sample_id, final in primary["coherent_retrieval"].items():
            if final["category"] != category:
                continue
            baseline = primary["objective1"][sample_id]
            baseline_quality = (
                0.70 * float(baseline["internal_f05"])
                + 0.30 * float(baseline["internal_f1"])
                - 0.30
                * max(0.0, float(baseline["pred_to_gt_internal_ratio"]) - 1.0)
            )
            final_quality = (
                0.70 * float(final["internal_f05"])
                + 0.30 * float(final["internal_f1"])
                - 0.30
                * max(0.0, float(final["pred_to_gt_internal_ratio"]) - 1.0)
            )
            candidates.append({
                "sample_id": sample_id,
                "category": category,
                "semantic_quality_delta": final_quality - baseline_quality,
                "objective1_internal_f1": float(baseline["internal_f1"]),
                "coherent_internal_f1": float(final["internal_f1"]),
                "coherent_internal_precision": float(final["internal_precision"]),
                "coherent_internal_recall": float(final["internal_recall"]),
                "coherent_internal_ratio": float(
                    final["pred_to_gt_internal_ratio"]
                ),
            })
        candidates.sort(
            key=lambda row: (
                row["semantic_quality_delta"],
                row["coherent_internal_precision"],
            ),
            reverse=True,
        )
        selected.extend(candidates[:count_per_category])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate coherent single-donor component transfer."
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
    parser.add_argument("--gallery-categories", nargs="+", default=["bus", "car"])
    parser.add_argument("--gallery-per-category", type=int, default=15)
    args = parser.parse_args()

    if args.view_index < 0 or args.resolution < 1 or args.transplant_margin < 1:
        raise ValueError("View index cannot be negative; resolution and margin must be positive")
    if any(margin < 1 for margin in args.margins):
        raise ValueError("--margins must contain positive integers")
    if len(args.margins) != len(set(args.margins)):
        raise ValueError("--margins must not contain duplicates")
    if args.transplant_margin not in args.margins:
        raise ValueError("--transplant-margin must be included in --margins")
    if args.gallery_per_category < 0:
        raise ValueError("--gallery-per-category cannot be negative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}. Use a new directory."
        )

    policy = read_policy(
        args.policy, args.view_index, args.resolution, args.transplant_margin
    )
    maximum_k = max(selection["support_k"] for selection in policy.values())
    selected_ids = read_selected_ids(args.ids_file)
    rankings = read_rankings(
        args.rankings,
        f"dino_view{args.view_index:03d}",
        selected_ids,
        maximum_k,
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
    unknown_gallery = set(args.gallery_categories) - test_categories
    if unknown_gallery:
        raise ValueError(f"Unknown gallery categories: {sorted(unknown_gallery)}")
    train_categories = {row["sha256"]: row["category"] for row in train_rows}
    expected_test_ids = {row["sha256"] for row in test_rows}
    if set(rankings) != expected_test_ids:
        raise ValueError("Ranking IDs do not exactly match the selected test IDs")

    candidate_internals: dict[str, set] = {}
    candidate_exteriors: dict[str, set] = {}
    candidate_components: dict[str, list[set]] = {}

    def load_candidate(candidate_id: str) -> None:
        if candidate_id in candidate_internals:
            return
        path = train_dir / "voxels" / f"{candidate_id}.ply"
        if not path.is_file():
            raise FileNotFoundError(f"Missing training voxel PLY: {path}")
        voxels = read_voxels(path, args.resolution)
        sample_internal = interior(voxels, args.transplant_margin)
        candidate_internals[candidate_id] = sample_internal
        candidate_exteriors[candidate_id] = voxels - sample_internal
        candidate_components[candidate_id] = connected_components(sample_internal)

    output_prediction_dir = args.output_dir / "predictions" / "coherent_retrieval"
    reference_gt_dir = args.output_dir / "references" / "ground_truth"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_gt_dir.mkdir(parents=True, exist_ok=True)
    margins = sorted(args.margins)
    per_sample_rows = []
    applied_rows = []

    for index, test_row in enumerate(test_rows, start=1):
        sample_id = test_row["sha256"]
        category = test_row["category"]
        selection = policy[category]
        ranking = rankings[sample_id]
        if any(row["category"] != category for row in ranking):
            raise ValueError(f"Ranking category mismatch for {sample_id}")
        for candidate_row in ranking[: selection["support_k"]]:
            candidate_id = str(candidate_row["retrieved_id"])
            if train_categories.get(candidate_id) != category:
                raise ValueError(
                    f"Candidate category mismatch for {sample_id}: {candidate_id}"
                )
            load_candidate(candidate_id)

        gt_path = test_dir / "voxels" / f"{sample_id}.ply"
        objective1_path = args.objective1_voxels / f"{sample_id}.ply"
        for required_path in (gt_path, objective1_path):
            if not required_path.is_file():
                raise FileNotFoundError(f"Missing voxel PLY: {required_path}")
        gt = read_voxels(gt_path, args.resolution)
        objective1 = read_voxels(objective1_path, args.resolution)
        objective1_internal = interior(objective1, args.transplant_margin)
        objective1_exterior = objective1 - objective1_internal
        safe_volume = enclosed_volume(objective1_exterior, args.transplant_margin)

        selected, selected_exterior_iou = choose_donor(
            ranking,
            objective1_exterior,
            candidate_exteriors,
            selection["selection_k"],
            selection["exterior_weight"],
        )
        donor_id = str(selected["retrieved_id"])
        support_counts = build_support_counts(
            ranking, candidate_internals, selection["support_k"]
        )
        voxel_budget = int(
            round(selection["budget_ratio"] * len(objective1_exterior))
        )
        transferred, diagnostics = transfer_components(
            candidate_internals[donor_id],
            candidate_components[donor_id],
            safe_volume,
            support_counts,
            selection["preset"],
            voxel_budget,
        )
        used_fallback = not transferred
        coherent_prediction = (
            objective1 if used_fallback else objective1_exterior | transferred
        )
        if not objective1_exterior <= coherent_prediction:
            raise AssertionError(f"Objective-1 shell was not preserved for {sample_id}")
        if transferred - safe_volume:
            raise AssertionError(f"Transferred voxels escaped the safe volume: {sample_id}")

        common = {
            "sample_id": sample_id,
            "category": category,
            "selected_id": donor_id,
            "selected_rank": int(selected["rank"]),
            "image_similarity": float(selected["image_similarity"]),
            "selected_exterior_iou": selected_exterior_iou,
            "selection_k": selection["selection_k"],
            "support_k": selection["support_k"],
            "exterior_weight": selection["exterior_weight"],
            "preset": selection["preset"].name,
            "objective1_internal_voxels": len(objective1_internal),
            "objective1_exterior_voxels": len(objective1_exterior),
            "safe_volume_voxels": len(safe_volume),
            "removed_objective1_internal_voxels": len(
                objective1_internal - coherent_prediction
            ),
            "added_vs_objective1_voxels": len(coherent_prediction - objective1),
            "used_objective1_fallback": int(used_fallback),
            **diagnostics,
        }
        variants = {
            "objective1": objective1,
            "coherent_retrieval": coherent_prediction,
        }
        for method, prediction in variants.items():
            for margin in margins:
                predicted_internal = interior(prediction, margin)
                metrics = add_derived_metrics(
                    score_voxels(gt, prediction, margin), predicted_internal
                )
                metrics["objective1_shell_preservation"] = (
                    len(objective1_exterior & prediction) / len(objective1_exterior)
                    if objective1_exterior
                    else 1.0
                )
                per_sample_rows.append({
                    "method": method,
                    "margin": margin,
                    **common,
                    **metrics,
                })

        write_voxels(
            output_prediction_dir / f"{sample_id}.ply",
            coherent_prediction,
            args.resolution,
        )
        shutil.copyfile(gt_path, reference_gt_dir / f"{sample_id}.ply")
        applied_rows.append({
            "sample_id": sample_id,
            "category": category,
            "selected_id": donor_id,
            "selected_rank": int(selected["rank"]),
            "selection_k": selection["selection_k"],
            "support_k": selection["support_k"],
            "exterior_weight": selection["exterior_weight"],
            "preset": selection["preset"].name,
            "transferred_voxels": len(transferred),
            "kept_components": diagnostics["kept_components"],
            "voxel_budget": voxel_budget,
            "used_objective1_fallback": int(used_fallback),
        })
        if index % 25 == 0 or index == len(test_rows):
            print(f"Processed {index}/{len(test_rows)}")

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
    write_csv(args.output_dir / "applied_policy.csv", applied_rows)
    gallery_rows = choose_gallery(
        per_sample_rows,
        args.gallery_categories,
        args.gallery_per_category,
        args.transplant_margin,
    )
    if gallery_rows:
        write_csv(args.output_dir / "visualization_manifest.csv", gallery_rows)
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
                "gallery_categories": args.gallery_categories,
                "gallery_per_category": args.gallery_per_category,
            },
            file,
            indent=2,
        )
        file.write("\n")

    print("\nHeld-out margin-2 results")
    paired_by_method = {row["method"]: row for row in paired_rows}
    for row in summary_rows:
        if int(row["margin"]) != args.transplant_margin:
            continue
        paired = paired_by_method[row["method"]]
        print(
            f"  {row['method']}: internal P/R/F1/F0.5="
            f"{row['internal_precision']:.4f}/{row['internal_recall']:.4f}/"
            f"{row['internal_f1']:.4f}/{row['internal_f05']:.4f}, "
            f"ratio={row['pred_to_gt_internal_ratio']:.3f}, "
            f"components={row['internal_components_26']:.1f}, "
            f"small-component fraction={row['small_internal_component_fraction']:.3f}, "
            f"solid-core fraction={row['volumetric_core_fraction']:.3f}, "
            f"F1 delta={paired['mean_internal_f1_delta']:+.4f}, "
            f"W/L/T={paired['wins']}/{paired['losses']}/{paired['ties']}"
        )
    print(f"\nWrote coherent retrieval results to {args.output_dir}")


if __name__ == "__main__":
    main()
