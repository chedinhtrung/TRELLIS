#!/usr/bin/env python3
"""Calibrate category-aware structural retrieval without held-out labels.

Cars retain the frozen retrieval-v2 policy.  Bus, cabinet, and file-cabinet
queries reuse the v2 DINO top-20 features but fit a query-centered ranker and
calibrate donor-derived structural fragments on the disjoint train-only fusion
split.  Prepared fusion queries are cached individually so policy sweeps and
failure recovery do not repeat geometry work.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import monotonic

import numpy as np

from calibrate_dino_consensus import load_train_embeddings
from calibrate_retrieval_v2 import (
    BUDGET_QUANTILES,
    balanced_cap,
    make_rankings,
    split_queries,
)
from coherent_retrieval import internal_metrics
from compare_internals import interior, read_voxels
from evaluate_dino_interior_transplant import enclosed_volume, read_metadata
from evaluate_retrieval_completion import write_csv
from retrieval_v2 import (
    STRUCTURAL_PRESETS,
    component_core_fraction,
    fit_query_centered_ridge,
    hybrid_fusion,
    predict_quality,
    prepare_candidate_records,
    safe_volume_for_category,
    structural_preset_payload,
    support_counts,
    transfer_structural_fragments,
)


DEFAULT_CATEGORIES = ("bus", "cabinet", "file_cabinet")
MAX_EXPANSIONS = (1.00, 1.25)
TRANSFER_MODES = ("fragments", "full")
FUSION_MODES = ("hybrid", "replace")
COVERAGE_GATES = (0.00, 0.25, 0.50, 0.75, 1.50)
SCORE_MARGIN_GATES = (0.00, 0.005, 0.010, 0.020)


def f05(precision: float, recall: float) -> float:
    denominator = 0.25 * precision + recall
    return 1.25 * precision * recall / denominator if denominator else 0.0


def read_v2_policy(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if payload.get("method") != "retrieval_v2_confidence_gated_hybrid":
        raise ValueError(f"Expected a retrieval-v2 policy: {path}")
    return payload


def read_reranker_rows(path: Path, categories: set[str]) -> list[dict]:
    with path.open(newline="") as file:
        rows = [row for row in csv.DictReader(file) if row["category"] in categories]
    if not rows:
        raise ValueError(f"No requested categories in {path}")
    return rows


def fit_centered_rerankers(rows: list[dict], categories: set[str], ridge: float) -> dict:
    models = {}
    for category in sorted(categories):
        category_rows = [row for row in rows if row["category"] == category]
        models[category] = fit_query_centered_ridge(
            category_rows,
            [float(row["target_internal_f1"]) for row in category_rows],
            [row["sample_id"] for row in category_rows],
            ridge,
        )
        print(
            f"  {category}: {models[category]['training_queries']} queries, "
            f"RMSE={models[category]['training_rmse']:.4f}, "
            f"corr={models[category]['training_correlation']:.3f}"
        )
    return models


def update_accumulator(accumulator: dict, metrics: dict, fallback: bool) -> None:
    accumulator["sample_f1"] += float(metrics["internal_f1"])
    accumulator["sample_f05"] += f05(
        float(metrics["internal_precision"]), float(metrics["internal_recall"])
    )
    accumulator["precision"] += float(metrics["internal_precision"])
    accumulator["recall"] += float(metrics["internal_recall"])
    accumulator["predicted"] += int(metrics["pred_internal_voxels"])
    accumulator["ground_truth"] += int(metrics["gt_internal_voxels"])
    accumulator["core_fraction"] += float(metrics["core_fraction"])
    accumulator["fallback"] += int(fallback)


def choose_structural_policy(
    options: list[dict], baseline_rows: list[dict], max_internal_ratio: float
) -> dict | None:
    """Choose a safe active policy; prefer changes within 0.002 of best F1."""
    baseline_precision = float(np.mean([
        float(row["internal_precision"]) for row in baseline_rows
    ]))
    baseline_core = float(np.mean([
        float(row["core_fraction"]) for row in baseline_rows
    ]))
    baseline_f1 = float(np.mean([
        float(row["internal_f1"]) for row in baseline_rows
    ]))
    predicted = sum(int(row["pred_internal_voxels"]) for row in baseline_rows)
    truth = sum(int(row["gt_internal_voxels"]) for row in baseline_rows)
    ratio_ceiling = max(max_internal_ratio, predicted / truth if truth else 0.0)
    epsilon = 1e-12
    valid = [
        row for row in options
        if float(row["micro_internal_ratio"]) <= ratio_ceiling + epsilon
        and float(row["internal_precision"]) >= baseline_precision - 0.01 - epsilon
        and float(row["mean_core_fraction"]) <= baseline_core + 0.02 + epsilon
    ]
    active = [row for row in valid if float(row["fallback_fraction"]) < 1.0 - epsilon]
    improving = [
        row for row in active
        if float(row["internal_f1"]) >= baseline_f1 - 0.002 - epsilon
    ]
    pool = improving
    if not pool:
        return None
    best_f1 = max(float(row["internal_f1"]) for row in pool)
    near_best = [
        row for row in pool if float(row["internal_f1"]) >= best_f1 - 0.002
    ]
    return max(
        near_best,
        key=lambda row: (
            -float(row["fallback_fraction"]),
            float(row["internal_f1"]),
            float(row["internal_f05"]),
            float(row["internal_precision"]),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate category-aware retrieval-v2.1 on train-only splits."
    )
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--objective1-voxels", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--v2-calibration-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES))
    parser.add_argument("--view-index", type=int, default=18)
    parser.add_argument("--model", default="dinov2_vitl14_reg")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--margin", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--max-internal-ratio", type=float, default=1.15)
    parser.add_argument("--fusion-queries-per-category", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    categories_to_fit = set(args.categories)
    if not categories_to_fit or not categories_to_fit <= set(DEFAULT_CATEGORIES):
        parser.error(f"--categories must be drawn from {DEFAULT_CATEGORIES}")
    if args.view_index != 18 or args.top_k < 2 or args.margin < 1:
        parser.error("v2.1 requires view 18, top-k >= 2, and a positive margin")
    if args.workers < 1 or args.fusion_queries_per_category < 1:
        parser.error("workers and fusion query count must be positive")

    v2_policy_path = args.v2_calibration_dir / "policy.json"
    reranker_rows_path = args.v2_calibration_dir / "reranker_fit.csv"
    for path in (v2_policy_path, reranker_rows_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed retrieval-v2 input: {path}")
    v2_payload = read_v2_policy(v2_policy_path)
    for name, expected in (
        ("view_index", args.view_index),
        ("resolution", args.resolution),
        ("margin", args.margin),
    ):
        if int(v2_payload.get(name, -1)) != expected:
            raise ValueError(
                f"Frozen v2 policy has {name}={v2_payload.get(name)!r}, "
                f"expected {expected}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "query_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    signature = {
        "version": 2,
        "categories": sorted(categories_to_fit),
        "view_index": args.view_index,
        "model": args.model,
        "resolution": args.resolution,
        "margin": args.margin,
        "top_k": args.top_k,
        "ridge": args.ridge,
        "fusion_queries_per_category": args.fusion_queries_per_category,
    }
    signature_path = args.output_dir / "cache_signature.json"
    if signature_path.is_file():
        existing = json.loads(signature_path.read_text())
        if existing != signature:
            raise ValueError(
                f"v2.1 cache configuration mismatch: {signature_path}; use a new output directory"
            )
    else:
        signature_path.write_text(json.dumps(signature, indent=2) + "\n")

    print("Fitting query-centered non-car rerankers from the completed v2 feature table")
    reranker_rows = read_reranker_rows(reranker_rows_path, categories_to_fit)
    rerankers = fit_centered_rerankers(reranker_rows, categories_to_fit, args.ridge)

    metadata = read_metadata(args.train_dir / "metadata.csv")
    embeddings = load_train_embeddings(
        args.embeddings, metadata, args.view_index, args.model
    )
    rankings = make_rankings(metadata, embeddings, args.top_k)
    category_by_id = {row["sha256"]: row["category"] for row in metadata}
    _reranker_pool, fusion_pool = split_queries(metadata)
    requested_pool = {
        sample_id for sample_id in fusion_pool
        if category_by_id[sample_id] in categories_to_fit
    }
    fusion_ids = balanced_cap(
        requested_pool, category_by_id, args.fusion_queries_per_category
    )
    pending_ids = [
        sample_id for sample_id in sorted(fusion_ids)
        if not (cache_dir / f"{sample_id}.pkl").is_file()
    ]
    print(
        f"Structural calibration queries: {len(fusion_ids)} total, "
        f"{len(pending_ids)} require geometry preparation"
    )

    if pending_ids:
        required_gt_ids = set(pending_ids)
        for query_id in pending_ids:
            required_gt_ids.update(
                str(row["retrieved_id"]) for row in rankings[query_id]
            )
        gt_internals = {}
        gt_exteriors = {}
        objective1_internals = {}
        objective1_exteriors = {}
        print(
            f"Loading GT for {len(required_gt_ids)} pending queries/donors and "
            f"Objective 1 for {len(pending_ids)} queries"
        )
        for index, sample_id in enumerate(sorted(required_gt_ids), start=1):
            gt_path = args.train_dir / "voxels" / f"{sample_id}.ply"
            gt = read_voxels(gt_path, args.resolution)
            gt_internals[sample_id] = interior(gt, args.margin)
            gt_exteriors[sample_id] = gt - gt_internals[sample_id]
            if sample_id in pending_ids:
                objective1_path = args.objective1_voxels / f"{sample_id}.ply"
                objective1 = read_voxels(objective1_path, args.resolution)
                objective1_internals[sample_id] = interior(objective1, args.margin)
                objective1_exteriors[sample_id] = (
                    objective1 - objective1_internals[sample_id]
                )
            if index % 100 == 0 or index == len(required_gt_ids):
                print(f"  loaded {index}/{len(required_gt_ids)}")

        def prepare_query(query_id: str) -> tuple[str, dict]:
            category = category_by_id[query_id]
            base = objective1_internals[query_id]
            exterior = objective1_exteriors[query_id]
            feature_safe = enclosed_volume(exterior, args.margin)
            transfer_safe = safe_volume_for_category(exterior, args.margin, category)
            records = prepare_candidate_records(
                rankings[query_id],
                gt_internals,
                gt_exteriors,
                base,
                exterior,
                feature_safe,
                args.resolution,
            )
            for record in records:
                record["predicted_quality"] = predict_quality(
                    record["features"], rerankers[category]
                )
            records.sort(
                key=lambda record: (
                    -record["predicted_quality"],
                    -record["image_similarity"],
                    record["rank"],
                )
            )
            return query_id, {
                "sample_id": query_id,
                "category": category,
                "query_gt": gt_internals[query_id],
                "query_gt_exterior": gt_exteriors[query_id],
                "base": base,
                "query_exterior": exterior,
                "safe": transfer_safe,
                "records": records,
            }

        started = monotonic()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for index, (query_id, payload) in enumerate(
                executor.map(prepare_query, pending_ids), start=1
            ):
                temporary = (cache_dir / f"{query_id}.pkl").with_suffix(".tmp")
                with temporary.open("wb") as file:
                    pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
                temporary.replace(cache_dir / f"{query_id}.pkl")
                if index % 6 == 0 or index == len(pending_ids):
                    elapsed = monotonic() - started
                    remaining = elapsed / index * (len(pending_ids) - index)
                    print(
                        f"  cached {index}/{len(pending_ids)} pending queries "
                        f"({elapsed / 60:.1f} min, ~{remaining / 60:.1f} min remaining)"
                    )

    payloads = []
    for sample_id in sorted(fusion_ids):
        with (cache_dir / f"{sample_id}.pkl").open("rb") as file:
            payloads.append(pickle.load(file))

    category_budget_ratios = {}
    for category in sorted(categories_to_fit):
        ratios = [
            len(payload["query_gt"]) / len(payload["query_gt_exterior"])
            for payload in payloads
            if payload["category"] == category and payload["query_gt_exterior"]
        ]
        category_budget_ratios[category] = {
            quantile: float(np.quantile(ratios, quantile))
            for quantile in BUDGET_QUANTILES
        }

    accumulators: dict[tuple, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    counts: dict[tuple, int] = defaultdict(int)
    baseline_by_category: dict[str, list[dict]] = defaultdict(list)
    selector_rows = []
    print("Sweeping cached structural-fragment and confidence policies")
    for payload in payloads:
        category = payload["category"]
        query_gt = payload["query_gt"]
        base = payload["base"]
        query_exterior = payload["query_exterior"]
        safe = payload["safe"]
        records = payload["records"]
        selected = records[0]
        score_margin = (
            float(selected["predicted_quality"])
            - float(records[1]["predicted_quality"])
        )
        support = support_counts([record["aligned_internal"] for record in records])
        baseline_metrics = internal_metrics(query_gt, base)
        baseline_metrics["core_fraction"] = component_core_fraction(base)
        baseline_by_category[category].append(baseline_metrics)
        full_prediction = query_exterior | (selected["aligned_internal"] & safe)
        full_metrics = internal_metrics(
            query_gt, interior(full_prediction, args.margin)
        )
        oracle_quality = max(
            internal_metrics(
                query_gt,
                interior(
                    query_exterior | (record["aligned_internal"] & safe),
                    args.margin,
                ),
            )["internal_f1"]
            for record in records
        )
        selector_rows.append({
            "sample_id": payload["sample_id"],
            "category": category,
            "selected_id": selected["retrieved_id"],
            "selected_rank": selected["rank"],
            "predicted_quality": selected["predicted_quality"],
            "score_margin": score_margin,
            "selected_full_internal_f1": full_metrics["internal_f1"],
            "objective1_internal_f1": baseline_metrics["internal_f1"],
            "oracle20_internal_f1": oracle_quality,
        })

        for preset in STRUCTURAL_PRESETS:
            for budget_quantile in BUDGET_QUANTILES:
                category_budget = int(round(
                    category_budget_ratios[category][budget_quantile]
                    * len(query_exterior)
                ))
                for max_expansion in MAX_EXPANSIONS:
                    base_budget = (
                        int(round(max_expansion * len(base)))
                        if base else category_budget
                    )
                    donor_budget = min(category_budget, base_budget)
                    fragment_transfer, _source, _fragment_diagnostics = transfer_structural_fragments(
                        selected["source_internal"],
                        selected["alignment"],
                        safe,
                        support,
                        preset,
                        donor_budget,
                        args.resolution,
                    )
                    usable_full = selected["aligned_internal"] & safe
                    for transfer_mode in TRANSFER_MODES:
                        if transfer_mode == "fragments":
                            transferred = fragment_transfer
                        else:
                            transferred = (
                                usable_full if len(usable_full) <= donor_budget else set()
                            )
                        denominator = min(len(usable_full), donor_budget)
                        coverage = (
                            len(transferred) / denominator
                            if denominator else float(bool(transferred))
                        )
                        for fusion_mode in FUSION_MODES:
                            if fusion_mode == "hybrid":
                                fused, _preserved, _diagnostics = hybrid_fusion(
                                    base, transferred, safe, 64, 0.25, category_budget
                                )
                            else:
                                fused = transferred | (base - safe)
                            evaluated = interior(query_exterior | fused, args.margin)
                            fused_metrics = internal_metrics(query_gt, evaluated)
                            fused_metrics["core_fraction"] = component_core_fraction(evaluated)
                            structural_key = (
                                category,
                                preset.name,
                                transfer_mode,
                                budget_quantile,
                                max_expansion,
                                fusion_mode,
                            )
                            for coverage_gate in COVERAGE_GATES:
                                for score_margin_gate in SCORE_MARGIN_GATES:
                                    fallback = (
                                        not transferred
                                        or coverage < coverage_gate
                                        or score_margin < score_margin_gate
                                    )
                                    metrics = baseline_metrics if fallback else fused_metrics
                                    key = structural_key + (
                                        coverage_gate, score_margin_gate
                                    )
                                    update_accumulator(accumulators[key], metrics, fallback)
                                    counts[key] += 1

    summary_rows = []
    for key, accumulator in accumulators.items():
        (
            category,
            preset,
            transfer_mode,
            budget_quantile,
            max_expansion,
            fusion_mode,
            coverage_gate,
            score_margin_gate,
        ) = key
        count = counts[key]
        summary_rows.append({
            "category": category,
            "structural_preset": preset,
            "transfer_mode": transfer_mode,
            "budget_quantile": budget_quantile,
            "max_expansion": max_expansion,
            "fusion_mode": fusion_mode,
            "minimum_transfer_coverage": coverage_gate,
            "minimum_score_margin": score_margin_gate,
            "internal_precision": accumulator["precision"] / count,
            "internal_recall": accumulator["recall"] / count,
            "internal_f1": accumulator["sample_f1"] / count,
            "internal_f05": accumulator["sample_f05"] / count,
            "micro_internal_ratio": (
                accumulator["predicted"] / accumulator["ground_truth"]
                if accumulator["ground_truth"] else 0.0
            ),
            "mean_core_fraction": accumulator["core_fraction"] / count,
            "fallback_fraction": accumulator["fallback"] / count,
            "matched_samples": count,
        })

    category_policies = {}
    for category in sorted(categories_to_fit):
        selected = choose_structural_policy(
            [row for row in summary_rows if row["category"] == category],
            baseline_by_category[category],
            args.max_internal_ratio,
        )
        if selected is None:
            print(
                f"  {category}: no safe improving structural policy; "
                "retaining frozen retrieval v2"
            )
            category_policies[category] = {
                "mode": "v2_hybrid",
                **v2_payload["categories"][category],
            }
            continue
        preset = next(
            item for item in STRUCTURAL_PRESETS
            if item.name == selected["structural_preset"]
        )
        quantile = float(selected["budget_quantile"])
        category_policies[category] = {
            "mode": "structural_fragments",
            "top_k": args.top_k,
            "reranker": rerankers[category],
            "structural_preset": structural_preset_payload(preset),
            "transfer_mode": selected["transfer_mode"],
            "budget_quantile": quantile,
            "max_internal_to_exterior_ratio": category_budget_ratios[category][quantile],
            "max_expansion": float(selected["max_expansion"]),
            "fusion_mode": selected["fusion_mode"],
            "minimum_transfer_coverage": float(
                selected["minimum_transfer_coverage"]
            ),
            "minimum_score_margin": float(selected["minimum_score_margin"]),
            "calibration_internal_precision": float(selected["internal_precision"]),
            "calibration_internal_recall": float(selected["internal_recall"]),
            "calibration_internal_f1": float(selected["internal_f1"]),
            "calibration_internal_f05": float(selected["internal_f05"]),
            "calibration_micro_internal_ratio": float(selected["micro_internal_ratio"]),
            "calibration_fallback_fraction": float(selected["fallback_fraction"]),
            "calibration_samples": int(selected["matched_samples"]),
        }

    merged_categories = {}
    for category, selection in v2_payload["categories"].items():
        if category in category_policies:
            merged_categories[category] = category_policies[category]
        else:
            merged_categories[category] = {"mode": "v2_hybrid", **selection}
    policy = {
        "method": "retrieval_v21_category_structural",
        "view_index": args.view_index,
        "model": args.model,
        "resolution": args.resolution,
        "margin": args.margin,
        "categories": merged_categories,
        "calibration": {
            "rule": "v2 reranker-fit split plus disjoint v2 fusion split; train only",
            "structural_categories": sorted(categories_to_fit),
            "fusion_queries": len(fusion_ids),
            "fusion_queries_per_category": args.fusion_queries_per_category,
            "cache_dir": str(cache_dir),
        },
    }
    write_csv(args.output_dir / "selector_calibration.csv", selector_rows)
    write_csv(args.output_dir / "structural_grid.csv", summary_rows)
    with (args.output_dir / "policy.json").open("w", encoding="utf-8") as file:
        json.dump(policy, file, indent=2)
        file.write("\n")

    print("\nFrozen retrieval-v2.1 policy")
    for category, selection in merged_categories.items():
        if selection["mode"] == "v2_hybrid":
            print(f"  {category}: unchanged retrieval v2")
        else:
            print(
                f"  {category}: {selection['structural_preset']['name']} / "
                f"{selection['transfer_mode']} / {selection['fusion_mode']}, "
                f"coverage>={selection['minimum_transfer_coverage']:.2f}, "
                f"margin>={selection['minimum_score_margin']:.3f}, "
                f"F1={selection['calibration_internal_f1']:.4f}, "
                f"fallback={selection['calibration_fallback_fraction']:.1%}"
            )
    print(f"Wrote policy to {args.output_dir / 'policy.json'}")


if __name__ == "__main__":
    main()
