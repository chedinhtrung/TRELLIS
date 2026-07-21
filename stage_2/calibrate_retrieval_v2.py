#!/usr/bin/env python3
"""Train the retrieval-v2 reranker and fusion gate without held-out test labels."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import monotonic

import numpy as np

from calibrate_dino_consensus import load_train_embeddings
from coherent_retrieval import internal_metrics
from compare_internals import interior, read_voxels
from evaluate_dino_interior_transplant import enclosed_volume, read_metadata
from evaluate_retrieval_completion import write_csv
from retrieval_v2 import (
    COMPONENT_PRESETS,
    component_core_fraction,
    component_preset_payload,
    choose_conservative_policy,
    exact_f1,
    fit_ridge,
    hybrid_fusion,
    predict_quality,
    prepare_candidate_records,
    support_counts,
    transfer_supported_components,
)


BASE_PRESERVATION = (
    (24, 0.20),
    (64, 0.15),
    (128, 0.10),
)
BUDGET_QUANTILES = (0.50, 0.75, 0.90)
MAX_EXPANSIONS = (1.00, 1.25)
COVERAGE_GATES = (0.00, 0.25, 0.50, 0.75, 1.50)
QUALITY_GATES = (0.00, 0.15, 0.25)


def f05(precision: float, recall: float) -> float:
    return (
        1.25 * precision * recall / (0.25 * precision + recall)
        if 0.25 * precision + recall
        else 0.0
    )


def make_rankings(
    metadata: list[dict[str, str]], embeddings: np.ndarray, top_k: int
) -> dict[str, list[dict]]:
    category_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        category_indices[row["category"]].append(index)
    rankings = {}
    for category, indices in sorted(category_indices.items()):
        if len(indices) <= top_k:
            raise ValueError(
                f"Category {category} has {len(indices)} shapes; need more than top-k={top_k}"
            )
        similarities = embeddings[indices] @ embeddings[indices].T
        np.fill_diagonal(similarities, -np.inf)
        orders = np.argsort(-similarities, axis=1, kind="stable")[:, :top_k]
        for local_query, global_query in enumerate(indices):
            query_id = metadata[global_query]["sha256"]
            ranking = []
            for rank, local_candidate in enumerate(orders[local_query], start=1):
                global_candidate = indices[int(local_candidate)]
                ranking.append({
                    "rank": rank,
                    "category": category,
                    "retrieved_id": metadata[global_candidate]["sha256"],
                    "image_similarity": float(
                        similarities[local_query, int(local_candidate)]
                    ),
                })
            rankings[query_id] = ranking
    return rankings


def split_queries(metadata: list[dict[str, str]]) -> tuple[set[str], set[str]]:
    """Deterministic 75/25 split inside every category."""
    by_category: dict[str, list[str]] = defaultdict(list)
    for row in metadata:
        by_category[row["category"]].append(row["sha256"])
    reranker_fit: set[str] = set()
    fusion_calibration: set[str] = set()
    for ids in by_category.values():
        for index, sample_id in enumerate(sorted(ids)):
            (fusion_calibration if index % 4 == 0 else reranker_fit).add(sample_id)
    return reranker_fit, fusion_calibration


def balanced_cap(
    sample_ids: set[str], categories: dict[str, str], per_category: int
) -> set[str]:
    """Keep a deterministic, category-balanced subset of hash-randomized IDs."""
    if per_category == 0:
        return set(sample_ids)
    by_category: dict[str, list[str]] = defaultdict(list)
    for sample_id in sample_ids:
        by_category[categories[sample_id]].append(sample_id)
    selected: set[str] = set()
    for ids in by_category.values():
        selected.update(sorted(ids)[:per_category])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train-only calibration for retrieval-v2 reranking and hybrid fusion."
    )
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--objective1-voxels", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--view-index", type=int, default=18)
    parser.add_argument("--model", default="dinov2_vitl14_reg")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--margin", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--max-internal-ratio", type=float, default=1.15)
    parser.add_argument(
        "--reranker-queries-per-category",
        type=int,
        default=0,
        help="Deterministic per-category cap; zero uses the complete 75%% split.",
    )
    parser.add_argument(
        "--fusion-queries-per-category",
        type=int,
        default=0,
        help="Deterministic per-category cap; zero uses the complete 25%% split.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of shared-memory query workers.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the durable calibration checkpoint in output-dir.",
    )
    args = parser.parse_args()

    if args.view_index != 18:
        raise ValueError("retrieval-v2 is intentionally calibrated for view 18")
    if args.top_k < 2 or args.resolution < 1 or args.margin < 1:
        raise ValueError("top-k must be >=2; resolution and margin must be positive")
    if args.ridge <= 0 or args.max_internal_ratio <= 0 or args.workers < 1:
        raise ValueError("ridge, max-internal-ratio, and workers must be positive")
    if (
        args.reranker_queries_per_category < 0
        or args.fusion_queries_per_category < 0
    ):
        raise ValueError("query caps must be non-negative")
    if (
        args.output_dir.exists()
        and any(args.output_dir.iterdir())
        and not args.resume
    ):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}. Use --resume or a new directory."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = read_metadata(args.train_dir / "metadata.csv")
    embeddings = load_train_embeddings(
        args.embeddings, metadata, args.view_index, args.model
    )
    rankings = make_rankings(metadata, embeddings, args.top_k)
    categories = {row["sha256"]: row["category"] for row in metadata}
    reranker_pool_ids, fusion_pool_ids = split_queries(metadata)
    reranker_ids = balanced_cap(
        reranker_pool_ids, categories, args.reranker_queries_per_category
    )
    fusion_ids = balanced_cap(
        fusion_pool_ids, categories, args.fusion_queries_per_category
    )
    calibration_ids = reranker_ids | fusion_ids
    required_gt_ids = set(calibration_ids)
    for query_id in calibration_ids:
        required_gt_ids.update(
            str(row["retrieved_id"]) for row in rankings[query_id]
        )

    checkpoint_signature = {
        "version": 1,
        "view_index": args.view_index,
        "model": args.model,
        "resolution": args.resolution,
        "margin": args.margin,
        "top_k": args.top_k,
        "ridge": args.ridge,
        "max_internal_ratio": args.max_internal_ratio,
        "reranker_queries_per_category": args.reranker_queries_per_category,
        "fusion_queries_per_category": args.fusion_queries_per_category,
        "reranker_ids": sorted(reranker_ids),
        "fusion_ids": sorted(fusion_ids),
    }
    checkpoint_path = args.output_dir / "calibration_checkpoint.pkl"

    def category_counts(sample_ids: set[str]) -> str:
        counts = defaultdict(int)
        for sample_id in sample_ids:
            counts[categories[sample_id]] += 1
        return ", ".join(
            f"{category}={counts[category]}" for category in sorted(counts)
        )

    print(
        f"Calibration query subset: reranker {len(reranker_ids)} "
        f"({category_counts(reranker_ids)}); fusion {len(fusion_ids)} "
        f"({category_counts(fusion_ids)})"
    )

    print(
        f"Loading GT for {len(required_gt_ids)} required queries/donors and "
        f"Objective-1 for the {len(calibration_ids)} calibration queries"
    )
    gt_internals = {}
    gt_exteriors = {}
    objective1_internals = {}
    objective1_exteriors = {}
    for index, sample_id in enumerate(sorted(required_gt_ids), start=1):
        gt_path = args.train_dir / "voxels" / f"{sample_id}.ply"
        if not gt_path.is_file():
            raise FileNotFoundError(f"Missing calibration voxel PLY: {gt_path}")
        gt = read_voxels(gt_path, args.resolution)
        gt_internals[sample_id] = interior(gt, args.margin)
        gt_exteriors[sample_id] = gt - gt_internals[sample_id]
        if sample_id in calibration_ids:
            objective1_path = args.objective1_voxels / f"{sample_id}.ply"
            if not objective1_path.is_file():
                raise FileNotFoundError(
                    f"Missing calibration voxel PLY: {objective1_path}"
                )
            objective1 = read_voxels(objective1_path, args.resolution)
            objective1_internals[sample_id] = interior(objective1, args.margin)
            objective1_exteriors[sample_id] = (
                objective1 - objective1_internals[sample_id]
            )
        if index % 100 == 0 or index == len(required_gt_ids):
            print(f"  loaded {index}/{len(required_gt_ids)}")

    rerankers: dict[str, dict] = {}
    reranker_rows: list[dict] = []
    accumulators: dict[tuple, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    counts: dict[tuple, int] = defaultdict(int)
    selector_diagnostics: list[dict] = []
    baseline_by_category: dict[str, list[dict]] = defaultdict(list)
    processed_fusion_ids: set[str] = set()

    if args.resume and checkpoint_path.is_file():
        with checkpoint_path.open("rb") as file:
            checkpoint = pickle.load(file)
        if checkpoint.get("signature") != checkpoint_signature:
            raise ValueError(
                f"Calibration checkpoint configuration mismatch: {checkpoint_path}"
            )
        rerankers = checkpoint["rerankers"]
        reranker_rows = checkpoint["reranker_rows"]
        accumulators.update({
            key: defaultdict(float, value)
            for key, value in checkpoint["accumulators"].items()
        })
        counts.update(checkpoint["counts"])
        selector_diagnostics = checkpoint["selector_diagnostics"]
        baseline_by_category.update(checkpoint["baseline_by_category"])
        processed_fusion_ids = set(checkpoint["processed_fusion_ids"])
        print(
            f"Resuming checkpoint: {len(rerankers)}/4 rerankers and "
            f"{len(processed_fusion_ids)}/{len(fusion_ids)} fusion queries complete"
        )

    def save_checkpoint() -> None:
        checkpoint = {
            "signature": checkpoint_signature,
            "rerankers": rerankers,
            "reranker_rows": reranker_rows,
            "accumulators": {
                key: dict(value) for key, value in accumulators.items()
            },
            "counts": dict(counts),
            "selector_diagnostics": selector_diagnostics,
            "baseline_by_category": dict(baseline_by_category),
            "processed_fusion_ids": sorted(processed_fusion_ids),
        }
        temporary = checkpoint_path.with_suffix(".tmp")
        with temporary.open("wb") as file:
            pickle.dump(checkpoint, file, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(checkpoint_path)

    print("Fitting category rerankers on the 75% selector split")
    for category in sorted(set(categories.values())):
        if category in rerankers:
            print(f"  {category}: reusing completed reranker from checkpoint")
            continue
        feature_rows = []
        labels = []
        category_ids = sorted(
            sample_id
            for sample_id in reranker_ids
            if categories[sample_id] == category
        )

        def prepare_reranker_query(query_id: str) -> tuple[list, list, list]:
            safe = enclosed_volume(objective1_exteriors[query_id], args.margin)
            records = prepare_candidate_records(
                rankings[query_id],
                gt_internals,
                gt_exteriors,
                objective1_internals[query_id],
                objective1_exteriors[query_id],
                safe,
                args.resolution,
            )
            query_features = []
            query_labels = []
            query_rows = []
            for record in records:
                candidate_prediction = objective1_exteriors[query_id] | (
                    record["aligned_internal"] & safe
                )
                label = exact_f1(
                    gt_internals[query_id], interior(candidate_prediction, args.margin)
                )
                query_features.append(record["features"])
                query_labels.append(label)
                query_rows.append({
                    "split": "reranker_fit",
                    "category": category,
                    "sample_id": query_id,
                    "candidate_id": record["retrieved_id"],
                    "dino_rank": record["rank"],
                    **record["features"],
                    "target_internal_f1": label,
                })
            return query_features, query_labels, query_rows

        started = monotonic()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            prepared_queries = executor.map(prepare_reranker_query, category_ids)
            for query_index, prepared in enumerate(prepared_queries, start=1):
                query_features, query_labels, query_rows = prepared
                feature_rows.extend(query_features)
                labels.extend(query_labels)
                reranker_rows.extend(query_rows)
                if query_index % 12 == 0 or query_index == len(category_ids):
                    elapsed = monotonic() - started
                    remaining = elapsed / query_index * (len(category_ids) - query_index)
                    print(
                        f"  {category}: prepared {query_index}/{len(category_ids)} "
                        f"queries ({elapsed / 60:.1f} min elapsed, "
                        f"~{remaining / 60:.1f} min remaining)"
                    )
        rerankers[category] = fit_ridge(feature_rows, labels, args.ridge)
        print(
            f"  {category}: {len(category_ids)} queries, "
            f"RMSE={rerankers[category]['training_rmse']:.4f}, "
            f"corr={rerankers[category]['training_correlation']:.3f}"
        )
        save_checkpoint()

    for row in reranker_rows:
        row["predicted_internal_f1"] = predict_quality(
            row, rerankers[row["category"]]
        )

    category_budget_ratios: dict[str, dict[float, float]] = {}
    for category in sorted(set(categories.values())):
        ratios = [
            len(gt_internals[sample_id]) / len(gt_exteriors[sample_id])
            for sample_id in fusion_ids
            if categories[sample_id] == category and gt_exteriors[sample_id]
        ]
        category_budget_ratios[category] = {
            quantile: float(np.quantile(ratios, quantile))
            for quantile in BUDGET_QUANTILES
        }

    print("Calibrating fusion and confidence gates on the disjoint 25% split")
    pending_fusion_ids = [
        sample_id
        for sample_id in sorted(fusion_ids)
        if sample_id not in processed_fusion_ids
    ]
    initially_completed = len(processed_fusion_ids)
    fusion_started = monotonic()
    for session_index, query_id in enumerate(pending_fusion_ids, start=1):
        category = categories[query_id]
        query_gt = gt_internals[query_id]
        base = objective1_internals[query_id]
        query_exterior = objective1_exteriors[query_id]
        safe = enclosed_volume(query_exterior, args.margin)
        records = prepare_candidate_records(
            rankings[query_id],
            gt_internals,
            gt_exteriors,
            base,
            query_exterior,
            safe,
            args.resolution,
        )
        for record in records:
            record["predicted_quality"] = predict_quality(
                record["features"], rerankers[category]
            )
            candidate_prediction = query_exterior | (record["aligned_internal"] & safe)
            record["oracle_quality"] = exact_f1(
                query_gt, interior(candidate_prediction, args.margin)
            )
        selected = max(
            records,
            key=lambda record: (
                record["predicted_quality"],
                record["image_similarity"],
                -record["rank"],
            ),
        )
        oracle = max(records, key=lambda record: record["oracle_quality"])
        selected_quality = float(selected["predicted_quality"])
        selector_diagnostics.append({
            "split": "fusion_calibration",
            "sample_id": query_id,
            "category": category,
            "selected_id": selected["retrieved_id"],
            "selected_rank": selected["rank"],
            "predicted_internal_f1": selected_quality,
            "selected_actual_internal_f1": selected["oracle_quality"],
            "oracle20_internal_f1": oracle["oracle_quality"],
            "oracle20_rank": oracle["rank"],
        })
        support = support_counts([record["aligned_internal"] for record in records])
        baseline_metrics = internal_metrics(query_gt, base)
        baseline_metrics["core_fraction"] = component_core_fraction(base)
        baseline_by_category[category].append(baseline_metrics)

        for preset in COMPONENT_PRESETS:
            for budget_quantile in BUDGET_QUANTILES:
                category_budget = int(round(
                    category_budget_ratios[category][budget_quantile]
                    * len(query_exterior)
                ))
                for max_expansion in MAX_EXPANSIONS:
                    base_budget = (
                        int(round(max_expansion * len(base)))
                        if base
                        else category_budget
                    )
                    donor_budget = min(category_budget, base_budget)
                    transferred, _source, _diagnostics = transfer_supported_components(
                        gt_internals[str(selected["retrieved_id"])],
                        selected["alignment"],
                        safe,
                        support,
                        preset,
                        donor_budget,
                        args.resolution,
                    )
                    coverage = len(transferred) / len(base) if base else float(bool(transferred))
                    for base_min, base_core in BASE_PRESERVATION:
                        fused, _preserved, _fusion_diag = hybrid_fusion(
                            base,
                            transferred,
                            safe,
                            base_min,
                            base_core,
                            category_budget,
                        )
                        evaluated_fused = interior(
                            query_exterior | fused, args.margin
                        )
                        fused_metrics = internal_metrics(query_gt, evaluated_fused)
                        fused_metrics["core_fraction"] = component_core_fraction(
                            evaluated_fused
                        )
                        structural_key = (
                            category,
                            preset.name,
                            budget_quantile,
                            max_expansion,
                            base_min,
                            base_core,
                        )
                        for coverage_gate in COVERAGE_GATES:
                            for quality_gate in QUALITY_GATES:
                                use_fallback = (
                                    not transferred
                                    or coverage < coverage_gate
                                    or selected_quality < quality_gate
                                )
                                metrics = baseline_metrics if use_fallback else fused_metrics
                                key = structural_key + (coverage_gate, quality_gate)
                                accumulator = accumulators[key]
                                accumulator["sample_f1"] += float(metrics["internal_f1"])
                                accumulator["sample_f05"] += f05(
                                    float(metrics["internal_precision"]),
                                    float(metrics["internal_recall"]),
                                )
                                accumulator["precision"] += float(metrics["internal_precision"])
                                accumulator["recall"] += float(metrics["internal_recall"])
                                accumulator["predicted"] += int(metrics["pred_internal_voxels"])
                                accumulator["ground_truth"] += int(metrics["gt_internal_voxels"])
                                accumulator["core_fraction"] += float(metrics["core_fraction"])
                                accumulator["fallback"] += int(use_fallback)
                                counts[key] += 1
        processed_fusion_ids.add(query_id)
        save_checkpoint()
        query_index = initially_completed + session_index
        if query_index % 12 == 0 or session_index == len(pending_fusion_ids):
            elapsed = monotonic() - fusion_started
            remaining = (
                elapsed / session_index * (len(pending_fusion_ids) - session_index)
            )
            print(
                f"  prepared {query_index}/{len(fusion_ids)} fusion queries "
                f"({elapsed / 60:.1f} min elapsed, "
                f"~{remaining / 60:.1f} min remaining)"
            )

    summary_rows = []
    for key, accumulator in accumulators.items():
        (
            category,
            preset,
            budget_quantile,
            max_expansion,
            base_min,
            base_core,
            coverage_gate,
            quality_gate,
        ) = key
        count = counts[key]
        summary_rows.append({
            "category": category,
            "component_preset": preset,
            "budget_quantile": budget_quantile,
            "max_expansion": max_expansion,
            "base_min_component_voxels": base_min,
            "base_max_core_fraction": base_core,
            "minimum_transfer_coverage": coverage_gate,
            "minimum_predicted_quality": quality_gate,
            "internal_precision": accumulator["precision"] / count,
            "internal_recall": accumulator["recall"] / count,
            "internal_f1": accumulator["sample_f1"] / count,
            "internal_f05": accumulator["sample_f05"] / count,
            "micro_internal_ratio": (
                accumulator["predicted"] / accumulator["ground_truth"]
                if accumulator["ground_truth"]
                else 0.0
            ),
            "mean_core_fraction": accumulator["core_fraction"] / count,
            "fallback_fraction": accumulator["fallback"] / count,
            "matched_samples": count,
        })

    policy = {}
    for category in sorted(set(categories.values())):
        options = [row for row in summary_rows if row["category"] == category]
        selected = choose_conservative_policy(
            options,
            baseline_by_category[category],
            args.max_internal_ratio,
        )
        preset = next(
            item for item in COMPONENT_PRESETS
            if item.name == selected["component_preset"]
        )
        quantile = float(selected["budget_quantile"])
        policy[category] = {
            "top_k": args.top_k,
            "reranker": rerankers[category],
            "component_preset": component_preset_payload(preset),
            "budget_quantile": quantile,
            "max_internal_to_exterior_ratio": category_budget_ratios[category][quantile],
            "max_expansion": float(selected["max_expansion"]),
            "base_min_component_voxels": int(selected["base_min_component_voxels"]),
            "base_max_core_fraction": float(selected["base_max_core_fraction"]),
            "minimum_transfer_coverage": float(selected["minimum_transfer_coverage"]),
            "minimum_predicted_quality": float(selected["minimum_predicted_quality"]),
            "calibration_internal_precision": float(selected["internal_precision"]),
            "calibration_internal_recall": float(selected["internal_recall"]),
            "calibration_internal_f1": float(selected["internal_f1"]),
            "calibration_internal_f05": float(selected["internal_f05"]),
            "calibration_micro_internal_ratio": float(selected["micro_internal_ratio"]),
            "calibration_fallback_fraction": float(selected["fallback_fraction"]),
            "calibration_samples": int(selected["matched_samples"]),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "reranker_fit.csv", reranker_rows)
    write_csv(args.output_dir / "selector_calibration.csv", selector_diagnostics)
    write_csv(args.output_dir / "fusion_grid.csv", summary_rows)
    payload = {
        "method": "retrieval_v2_confidence_gated_hybrid",
        "view_index": args.view_index,
        "model": args.model,
        "resolution": args.resolution,
        "margin": args.margin,
        "split": {
            "rule": "sorted per category; index modulo 4 == 0 is fusion calibration",
            "reranker_pool_samples": len(reranker_pool_ids),
            "fusion_pool_samples": len(fusion_pool_ids),
            "reranker_fit_samples": len(reranker_ids),
            "fusion_calibration_samples": len(fusion_ids),
            "reranker_queries_per_category_cap": args.reranker_queries_per_category,
            "fusion_queries_per_category_cap": args.fusion_queries_per_category,
            "workers": args.workers,
        },
        "max_internal_ratio": args.max_internal_ratio,
        "categories": policy,
    }
    with (args.output_dir / "policy.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")

    print("\nFrozen retrieval-v2 policy")
    for category, selection in policy.items():
        print(
            f"  {category}: {selection['component_preset']['name']}, "
            f"coverage>={selection['minimum_transfer_coverage']:.2f}, "
            f"quality>={selection['minimum_predicted_quality']:.2f}, "
            f"F1={selection['calibration_internal_f1']:.4f}, "
            f"fallback={selection['calibration_fallback_fraction']:.1%}"
        )
    print(f"Wrote policy to {args.output_dir / 'policy.json'}")


if __name__ == "__main__":
    main()
