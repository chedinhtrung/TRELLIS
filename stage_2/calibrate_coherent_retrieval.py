#!/usr/bin/env python3
"""Calibrate coherent single-donor interior transfer on training shapes only."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from calibrate_dino_consensus import load_train_embeddings
from coherent_retrieval import (
    TRANSFER_PRESETS,
    build_support_counts,
    choose_donor,
    connected_components,
    internal_metrics,
    prepare_clipped_components,
    preset_payload,
    transfer_components,
)
from compare_internals import interior, read_voxels
from evaluate_dino_interior_transplant import enclosed_volume, read_metadata
from evaluate_retrieval_completion import write_csv


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Choose a conservative coherent-retrieval policy with training-only "
            "leave-one-out calibration."
        )
    )
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--view-index", type=int, default=18)
    parser.add_argument("--model", default="dinov2_vitl14_reg")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--margin", type=int, default=2)
    parser.add_argument("--support-k", type=int, default=20)
    parser.add_argument("--selection-k", type=int, nargs="+", default=[1, 5, 20])
    parser.add_argument(
        "--exterior-weight", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0]
    )
    parser.add_argument("--max-calibration-internal-ratio", type=float, default=1.15)
    args = parser.parse_args()

    if args.view_index < 0 or args.resolution < 1 or args.margin < 1:
        raise ValueError("View index cannot be negative; resolution and margin must be positive")
    if args.support_k < 2:
        raise ValueError("--support-k must be at least 2")
    if any(k < 1 or k > args.support_k for k in args.selection_k):
        raise ValueError("--selection-k values must be between 1 and --support-k")
    if len(args.selection_k) != len(set(args.selection_k)):
        raise ValueError("--selection-k values must be unique")
    if any(weight < 0 for weight in args.exterior_weight):
        raise ValueError("--exterior-weight cannot be negative")
    if len(args.exterior_weight) != len(set(args.exterior_weight)):
        raise ValueError("--exterior-weight values must be unique")
    if args.max_calibration_internal_ratio <= 0:
        raise ValueError("--max-calibration-internal-ratio must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}. Use a new directory."
        )

    metadata = read_metadata(args.train_dir / "metadata.csv")
    embeddings = load_train_embeddings(
        args.embeddings, metadata, args.view_index, args.model
    )

    print(f"Loading surfaces for {len(metadata)} training shapes")
    internals: dict[str, set] = {}
    exteriors: dict[str, set] = {}
    category_indices: dict[str, list[int]] = defaultdict(list)
    category_internal_exterior_ratios: dict[str, list[float]] = defaultdict(list)
    for index, row in enumerate(metadata):
        sample_id = row["sha256"]
        path = args.train_dir / "voxels" / f"{sample_id}.ply"
        if not path.is_file():
            raise FileNotFoundError(f"Missing training voxel PLY: {path}")
        voxels = read_voxels(path, args.resolution)
        sample_internal = interior(voxels, args.margin)
        sample_exterior = voxels - sample_internal
        internals[sample_id] = sample_internal
        exteriors[sample_id] = sample_exterior
        category_indices[row["category"]].append(index)
        category_internal_exterior_ratios[row["category"]].append(
            len(sample_internal) / len(sample_exterior) if sample_exterior else 0.0
        )
        if (index + 1) % 100 == 0 or index + 1 == len(metadata):
            print(f"  loaded {index + 1}/{len(metadata)}")

    budget_ratios = {
        category: {
            preset.budget_quantile: float(np.quantile(ratios, preset.budget_quantile))
            for preset in TRANSFER_PRESETS
        }
        for category, ratios in category_internal_exterior_ratios.items()
    }
    component_cache: dict[str, list[set]] = {}

    def components(sample_id: str) -> list[set]:
        if sample_id not in component_cache:
            component_cache[sample_id] = connected_components(internals[sample_id])
        return component_cache[sample_id]

    # Each accumulator contains only scalar lists.  This keeps calibration
    # transparent without writing a very large all-policy per-sample table.
    accumulators: dict[tuple, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    maximum_selection_k = max(args.selection_k)
    for category, indices in sorted(category_indices.items()):
        if len(indices) <= args.support_k:
            raise ValueError(
                f"Category {category} has {len(indices)} shapes; "
                f"need more than support-k={args.support_k}"
            )
        category_embeddings = embeddings[indices]
        similarities = category_embeddings @ category_embeddings.T
        np.fill_diagonal(similarities, -np.inf)
        orders = np.argsort(-similarities, axis=1, kind="stable")[:, : args.support_k]
        print(f"Calibrating {category}: {len(indices)} leave-one-out queries")

        for local_query_index, global_query_index in enumerate(indices):
            query = metadata[global_query_index]
            query_id = query["sha256"]
            query_internal = internals[query_id]
            query_exterior = exteriors[query_id]
            safe_volume = enclosed_volume(query_exterior, args.margin)
            ranking = []
            for rank, local_candidate_index in enumerate(
                orders[local_query_index], start=1
            ):
                candidate_global_index = indices[int(local_candidate_index)]
                candidate = metadata[candidate_global_index]
                ranking.append({
                    "rank": rank,
                    "retrieved_id": candidate["sha256"],
                    "image_similarity": float(
                        similarities[local_query_index, int(local_candidate_index)]
                    ),
                })
            if len(ranking) < max(args.support_k, maximum_selection_k):
                raise AssertionError("Incomplete leave-one-out ranking")

            support_counts = build_support_counts(
                ranking, internals, args.support_k
            )
            prepared_cache = {}
            transfer_cache = {}
            for selection_k in args.selection_k:
                for exterior_weight in args.exterior_weight:
                    selected, exterior_iou = choose_donor(
                        ranking,
                        query_exterior,
                        exteriors,
                        selection_k,
                        exterior_weight,
                    )
                    donor_id = str(selected["retrieved_id"])
                    if donor_id not in prepared_cache:
                        prepared_cache[donor_id] = prepare_clipped_components(
                            components(donor_id), safe_volume
                        )
                    for preset in TRANSFER_PRESETS:
                        cache_key = (donor_id, preset.name)
                        if cache_key not in transfer_cache:
                            budget_ratio = budget_ratios[category][preset.budget_quantile]
                            voxel_budget = int(round(budget_ratio * len(query_exterior)))
                            transfer_cache[cache_key] = transfer_components(
                                internals[donor_id],
                                components(donor_id),
                                safe_volume,
                                support_counts,
                                preset,
                                voxel_budget,
                                prepared_cache[donor_id],
                            )
                        transferred, diagnostics = transfer_cache[cache_key]
                        metrics = internal_metrics(query_internal, transferred)
                        policy_key = (
                            category,
                            selection_k,
                            exterior_weight,
                            preset.name,
                        )
                        accumulator = accumulators[policy_key]
                        for metric in (
                            "internal_precision",
                            "internal_recall",
                            "internal_f1",
                            "internal_f05",
                            "pred_to_gt_internal_ratio",
                        ):
                            accumulator[metric].append(float(metrics[metric]))
                        accumulator["pred_internal_voxels"].append(
                            float(metrics["pred_internal_voxels"])
                        )
                        accumulator["gt_internal_voxels"].append(
                            float(metrics["gt_internal_voxels"])
                        )
                        accumulator["kept_components"].append(
                            float(diagnostics["kept_components"])
                        )
                        accumulator["empty"].append(float(not transferred))
                        accumulator["selected_rank"].append(float(selected["rank"]))
                        accumulator["exterior_iou"].append(float(exterior_iou))

            if (local_query_index + 1) % 50 == 0 or local_query_index + 1 == len(indices):
                print(f"  calibrated {local_query_index + 1}/{len(indices)}")

    summary_rows = []
    for (category, selection_k, exterior_weight, preset_name), values in sorted(
        accumulators.items()
    ):
        total_prediction = sum(values["pred_internal_voxels"])
        total_ground_truth = sum(values["gt_internal_voxels"])
        micro_ratio = (
            total_prediction / total_ground_truth if total_ground_truth else 1.0
        )
        row = {
            "category": category,
            "selection_k": selection_k,
            "exterior_weight": exterior_weight,
            "preset": preset_name,
            "internal_precision": mean(values["internal_precision"]),
            "internal_recall": mean(values["internal_recall"]),
            "internal_f1": mean(values["internal_f1"]),
            "internal_f05": mean(values["internal_f05"]),
            "mean_pred_to_gt_internal_ratio": mean(
                values["pred_to_gt_internal_ratio"]
            ),
            "micro_pred_to_gt_internal_ratio": micro_ratio,
            "median_pred_to_gt_internal_ratio": float(
                np.median(values["pred_to_gt_internal_ratio"])
            ),
            "mean_kept_components": mean(values["kept_components"]),
            "empty_fraction": mean(values["empty"]),
            "mean_selected_rank": mean(values["selected_rank"]),
            "mean_exterior_iou": mean(values["exterior_iou"]),
            "matched_samples": len(values["internal_f1"]),
        }
        # Precision-weighted overlap is the main target.  The explicit penalty
        # starts at ratio 1.0 so a recall gain cannot hide systematic overfill.
        row["selection_score"] = (
            0.70 * row["internal_f05"]
            + 0.30 * row["internal_f1"]
            - 0.30 * max(0.0, micro_ratio - 1.0)
            - 0.05 * row["empty_fraction"]
        )
        summary_rows.append(row)

    policy = {}
    for category in sorted(category_indices):
        rows = [row for row in summary_rows if row["category"] == category]
        feasible = [
            row
            for row in rows
            if row["micro_pred_to_gt_internal_ratio"]
            <= args.max_calibration_internal_ratio
            and row["internal_recall"] >= 0.15
            and row["empty_fraction"] <= 0.20
        ]
        conservative = [
            row
            for row in rows
            if row["micro_pred_to_gt_internal_ratio"]
            <= args.max_calibration_internal_ratio
        ]
        if not conservative:
            raise AssertionError(
                f"No non-overfilling calibration policy exists for {category}"
            )
        pool = feasible or conservative
        selected = max(
            pool,
            key=lambda row: (
                row["selection_score"],
                row["internal_precision"],
                row["internal_f1"],
                -row["micro_pred_to_gt_internal_ratio"],
            ),
        )
        preset = next(
            preset for preset in TRANSFER_PRESETS if preset.name == selected["preset"]
        )
        policy[category] = {
            "selection_k": int(selected["selection_k"]),
            "exterior_weight": float(selected["exterior_weight"]),
            "support_k": args.support_k,
            "preset": preset_payload(preset),
            "max_internal_to_exterior_ratio": budget_ratios[category][
                preset.budget_quantile
            ],
            "calibration_internal_precision": selected["internal_precision"],
            "calibration_internal_recall": selected["internal_recall"],
            "calibration_internal_f1": selected["internal_f1"],
            "calibration_internal_f05": selected["internal_f05"],
            "calibration_micro_internal_ratio": selected[
                "micro_pred_to_gt_internal_ratio"
            ],
            "calibration_empty_fraction": selected["empty_fraction"],
            "calibration_selection_score": selected["selection_score"],
            "calibration_samples": selected["matched_samples"],
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "policy_summary.csv", summary_rows)
    policy_path = args.output_dir / "policy.json"
    with policy_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "method": "coherent_single_donor_component_transfer_v1",
                "calibration": "training-only leave-one-out",
                "view_index": args.view_index,
                "model": args.model,
                "resolution": args.resolution,
                "margin": args.margin,
                "selection_k_candidates": args.selection_k,
                "exterior_weight_candidates": args.exterior_weight,
                "max_calibration_internal_ratio": (
                    args.max_calibration_internal_ratio
                ),
                "categories": policy,
            },
            file,
            indent=2,
        )
        file.write("\n")

    print("\nFrozen coherent-retrieval policy")
    for category, selection in policy.items():
        print(
            f"  {category}: donor top-{selection['selection_k']}, "
            f"exterior weight={selection['exterior_weight']:g}, "
            f"preset={selection['preset']['name']}, "
            f"P/R/F0.5={selection['calibration_internal_precision']:.4f}/"
            f"{selection['calibration_internal_recall']:.4f}/"
            f"{selection['calibration_internal_f05']:.4f}, "
            f"micro ratio={selection['calibration_micro_internal_ratio']:.3f}"
        )
    print(f"\nWrote frozen policy to {policy_path}")


if __name__ == "__main__":
    main()
