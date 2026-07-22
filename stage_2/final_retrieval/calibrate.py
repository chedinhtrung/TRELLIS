#!/usr/bin/env python3
"""Train one category-blind selector over component and structural retrieval.

The expensive stage prepares donor/operator hypotheses once for train-only
queries and stores a durable cache per query.  Five query-level cross-fits then
calibrate one global no-regression gate.  No held-out test label is read.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import monotonic

import numpy as np

from .config import DONOR_FIT_QUERIES_PER_CATEGORY, operator_config
from .donor_models import EXPECTED_CATEGORIES, fit_donor_models
from .support import (
    balanced_cap,
    enclosed_volume,
    interior,
    internal_metrics,
    load_train_embeddings,
    make_train_rankings,
    read_metadata,
    read_voxels,
    split_train_queries,
    write_csv,
)
from .pipeline import (
    COMPONENT_ACTION,
    STRUCTURAL_ACTION,
    UNIFIED_FEATURE_NAMES,
    fit_selector_knn,
    generate_unified_candidates,
    predict_selector_distribution,
    shortlist_records,
)
from .geometry import (
    component_core_fraction,
    connected_components,
    exact_f1,
    prepare_candidate_records,
)


ACCEPTANCE_THRESHOLDS = (
    0.0, 0.0025, 0.005, 0.010, 0.020, 0.030, 0.050, 0.075,
    0.100, 0.125, 0.150, 0.200,
)
SELECTION_MARGIN_GATES = (0.0, 0.001, 0.0025, 0.005, 0.010)


def cache_path(cache_dir: Path, category: str, sample_id: str) -> Path:
    return cache_dir / f"{category}__{sample_id}.pkl"


def fold_assignments(
    query_ids: set[str], category_by_id: dict[str, str], folds: int
) -> dict[str, int]:
    """Deterministic category-stratified query folds."""
    by_category: dict[str, list[str]] = defaultdict(list)
    for sample_id in query_ids:
        by_category[category_by_id[sample_id]].append(sample_id)
    assignment = {}
    for category, sample_ids in sorted(by_category.items()):
        if len(sample_ids) < folds:
            raise ValueError(
                f"Category {category} has {len(sample_ids)} queries; need >= {folds} folds"
            )
        for index, sample_id in enumerate(sorted(sample_ids)):
            assignment[sample_id] = index % folds
    return assignment


def add_core_metric(metrics: dict, voxels: set) -> dict:
    return {
        **metrics,
        "volumetric_core_fraction": component_core_fraction(voxels),
    }


def donor_cache_path(cache_dir: Path, category: str, sample_id: str) -> Path:
    return cache_dir / f"{category}__{sample_id}.pkl"


def prepare_donor_training_rows(
    args: argparse.Namespace,
    donor_fit_queries: set[str],
    category_by_id: dict[str, str],
    rankings: dict[str, list[dict]],
) -> list[dict]:
    """Build the labeled top-K donor table directly from the training split."""
    cache_dir = args.output_dir / "donor_query_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        sample_id for sample_id in sorted(donor_fit_queries)
        if not donor_cache_path(
            cache_dir, category_by_id[sample_id], sample_id
        ).is_file()
    ]
    print(
        f"Donor fitting queries: {len(donor_fit_queries)} total; "
        f"{len(pending)} require geometry preparation"
    )
    if pending:
        required_ids = set(pending)
        for sample_id in pending:
            required_ids.update(
                str(row["retrieved_id"]) for row in rankings[sample_id]
            )
        gt_internals: dict[str, set] = {}
        gt_exteriors: dict[str, set] = {}
        objective1_internals: dict[str, set] = {}
        objective1_exteriors: dict[str, set] = {}
        print(
            f"Loading GT for {len(required_ids)} donor-fit queries/donors and "
            f"Objective 1 for {len(pending)} queries"
        )
        for index, sample_id in enumerate(sorted(required_ids), start=1):
            gt_path = args.train_dir / "voxels" / f"{sample_id}.ply"
            if not gt_path.is_file():
                raise FileNotFoundError(f"Missing training voxel PLY: {gt_path}")
            gt = read_voxels(gt_path, args.resolution)
            gt_internals[sample_id] = interior(gt, args.margin)
            gt_exteriors[sample_id] = gt - gt_internals[sample_id]
            if sample_id in pending:
                objective_path = args.objective1_voxels / f"{sample_id}.ply"
                if not objective_path.is_file():
                    raise FileNotFoundError(
                        f"Missing Objective-1 training voxel PLY: {objective_path}"
                    )
                objective = read_voxels(objective_path, args.resolution)
                objective1_internals[sample_id] = interior(objective, args.margin)
                objective1_exteriors[sample_id] = (
                    objective - objective1_internals[sample_id]
                )
            if index % 100 == 0 or index == len(required_ids):
                print(f"  loaded {index}/{len(required_ids)}")

        def prepare_query(sample_id: str) -> tuple[str, list[dict]]:
            base = objective1_internals[sample_id]
            exterior = objective1_exteriors[sample_id]
            safe = enclosed_volume(exterior, args.margin)
            records = prepare_candidate_records(
                rankings[sample_id],
                gt_internals,
                gt_exteriors,
                base,
                exterior,
                safe,
                args.resolution,
            )
            query_gt = gt_internals[sample_id]
            rows = []
            for record in records:
                candidate_prediction = exterior | (
                    record["aligned_internal"] & safe
                )
                label = exact_f1(
                    query_gt, interior(candidate_prediction, args.margin)
                )
                rows.append({
                    "split": "donor_fit",
                    "category": category_by_id[sample_id],
                    "sample_id": sample_id,
                    "candidate_id": str(record["retrieved_id"]),
                    "dino_rank": int(record["rank"]),
                    **record["features"],
                    "target_internal_f1": label,
                })
            return sample_id, rows

        started = monotonic()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for index, (sample_id, rows) in enumerate(
                executor.map(prepare_query, pending), start=1
            ):
                path = donor_cache_path(
                    cache_dir, category_by_id[sample_id], sample_id
                )
                temporary = path.with_suffix(".tmp")
                with temporary.open("wb") as file:
                    pickle.dump(rows, file, protocol=pickle.HIGHEST_PROTOCOL)
                temporary.replace(path)
                if index % 12 == 0 or index == len(pending):
                    elapsed = monotonic() - started
                    remaining = elapsed / index * (len(pending) - index)
                    print(
                        f"  cached {index}/{len(pending)} donor-fit queries "
                        f"({elapsed / 60:.1f} min, "
                        f"~{remaining / 60:.1f} min remaining)"
                    )

    rows = []
    for sample_id in sorted(donor_fit_queries):
        path = donor_cache_path(cache_dir, category_by_id[sample_id], sample_id)
        with path.open("rb") as file:
            rows.extend(pickle.load(file))
    write_csv(args.output_dir / "donor_features.csv", rows)
    return rows


def training_row(payload: dict, candidate: dict) -> dict:
    actual = candidate["actual_metrics"]
    baseline = payload["baseline_metrics"]
    return {
        "sample_id": payload["sample_id"],
        "category": payload["category"],
        "fold": payload["fold"],
        "action": candidate["action"],
        "donor_id": candidate["donor_id"],
        "dino_rank": candidate["rank"],
        "valid": int(candidate["valid"]),
        "filter_reason": candidate["filter_reason"] or "accepted_by_geometry",
        "target_internal_f1": actual["internal_f1"],
        "baseline_internal_f1": baseline["internal_f1"],
        "target_delta": actual["internal_f1"] - baseline["internal_f1"],
        **candidate["features"],
    }


def aggregate_gate(
    payloads: list[dict], threshold: float, margin_gate: float
) -> tuple[dict, list[dict]]:
    selected_rows = []
    chosen_metrics = []
    baseline_metrics = []
    category_deltas: dict[str, list[float]] = defaultdict(list)
    fallback_count = 0
    for payload in payloads:
        baseline = payload["baseline_metrics"]
        valid = [candidate for candidate in payload["candidates"] if candidate["valid"]]
        valid.sort(
            key=lambda candidate: (
                float(candidate["oof_prediction"]),
                float(candidate["features"]["image_similarity"]),
                -int(candidate["rank"]),
            ),
            reverse=True,
        )
        if valid:
            best = valid[0]
            selection_margin = (
                float(best["oof_prediction"] - valid[1]["oof_prediction"])
                if len(valid) > 1
                else float("inf")
            )
            fallback = (
                float(best["oof_prediction"]) < threshold
                or selection_margin < margin_gate
            )
        else:
            best = max(payload["candidates"], key=lambda row: row["oof_prediction"])
            selection_margin = 0.0
            fallback = True
        metrics = baseline if fallback else best["actual_metrics"]
        fallback_count += int(fallback)
        chosen_metrics.append(metrics)
        baseline_metrics.append(baseline)
        delta = float(metrics["internal_f1"] - baseline["internal_f1"])
        category_deltas[payload["category"]].append(delta)
        selected_rows.append({
            "sample_id": payload["sample_id"],
            "category": payload["category"],
            "fold": payload["fold"],
            "selected_action": best["action"],
            "selected_id": best["donor_id"],
            "selected_rank": best["rank"],
            "predicted_delta": best["oof_prediction"],
            "selection_margin": selection_margin,
            "used_objective1_fallback": int(fallback),
            "actual_internal_f1_delta": delta,
        })

    count = len(payloads)
    predicted = sum(int(row["pred_internal_voxels"]) for row in chosen_metrics)
    truth = sum(int(row["gt_internal_voxels"]) for row in chosen_metrics)
    summary = {
        "acceptance_threshold": threshold,
        "minimum_selection_margin": margin_gate,
        "internal_precision": float(np.mean([
            float(row["internal_precision"]) for row in chosen_metrics
        ])),
        "internal_recall": float(np.mean([
            float(row["internal_recall"]) for row in chosen_metrics
        ])),
        "internal_f1": float(np.mean([
            float(row["internal_f1"]) for row in chosen_metrics
        ])),
        "internal_f05": float(np.mean([
            float(row["internal_f05"]) for row in chosen_metrics
        ])),
        "mean_internal_f1_delta": float(np.mean([
            float(chosen["internal_f1"] - baseline["internal_f1"])
            for chosen, baseline in zip(chosen_metrics, baseline_metrics)
        ])),
        "micro_internal_ratio": predicted / truth if truth else 0.0,
        "mean_core_fraction": float(np.mean([
            float(row["volumetric_core_fraction"]) for row in chosen_metrics
        ])),
        "fallback_fraction": fallback_count / count,
        "matched_samples": count,
    }
    for category in sorted(category_deltas):
        summary[f"{category}_f1_delta"] = float(np.mean(category_deltas[category]))
    return summary, selected_rows


def select_global_gate(
    gate_rows: list[dict], baseline_rows: list[dict], max_internal_ratio: float,
    minimum_headline_delta: float,
) -> dict:
    baseline_precision = float(np.mean([
        float(row["internal_precision"]) for row in baseline_rows
    ]))
    baseline_core = float(np.mean([
        float(row["volumetric_core_fraction"]) for row in baseline_rows
    ]))
    predicted = sum(int(row["pred_internal_voxels"]) for row in baseline_rows)
    truth = sum(int(row["gt_internal_voxels"]) for row in baseline_rows)
    ratio_ceiling = max(max_internal_ratio, predicted / truth if truth else 0.0)
    epsilon = 1e-12
    valid = [
        row for row in gate_rows
        if row["fallback_fraction"] < 1.0 - epsilon
        and row["micro_internal_ratio"] <= ratio_ceiling + epsilon
        and row["internal_precision"] >= baseline_precision - 0.005 - epsilon
        and row["mean_core_fraction"] <= baseline_core + 0.01 + epsilon
        and row["mean_internal_f1_delta"] >= -epsilon
        and row.get("bus_f1_delta", -1.0) >= minimum_headline_delta - epsilon
        and row.get("car_f1_delta", -1.0) >= minimum_headline_delta - epsilon
    ]
    if not valid:
        best_bus = max(row.get("bus_f1_delta", -1.0) for row in gate_rows)
        best_car = max(row.get("car_f1_delta", -1.0) for row in gate_rows)
        raise RuntimeError(
            "Unified selector failed the fail-closed headline preflight: "
            f"best bus delta={best_bus:+.4f}, best car delta={best_car:+.4f}. "
            "The cached training work is safe; do not evaluate held-out data."
        )
    best_f1 = max(float(row["internal_f1"]) for row in valid)
    near_best = [row for row in valid if row["internal_f1"] >= best_f1 - 0.001]
    return max(
        near_best,
        key=lambda row: (
            row["fallback_fraction"],
            row["mean_internal_f1_delta"],
            row["internal_f05"],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one shared selector for all four retrieval categories."
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
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--selector-neighbors", type=int, default=8)
    parser.add_argument("--donor-ridge", type=float, default=1.0)
    parser.add_argument("--donor-shortlist", type=int, default=6)
    parser.add_argument(
        "--donor-fit-queries-per-category",
        type=int,
        default=DONOR_FIT_QUERIES_PER_CATEGORY,
    )
    parser.add_argument("--max-internal-ratio", type=float, default=1.15)
    parser.add_argument("--minimum-headline-delta", type=float, default=0.005)
    parser.add_argument(
        "--queries-per-category", type=int, default=0,
        help=(
            "Deterministic per-category selector cap; zero uses every shape "
            "not used to fit the donor rankers."
        ),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--uncertainty-weight", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.view_index != 18:
        parser.error("unified retrieval is intentionally trained on view 18")
    if args.top_k < 2 or args.folds < 2 or args.workers < 1:
        parser.error("top-k >= 2, folds >= 2, and workers >= 1 are required")
    if (
        args.queries_per_category < 0
        or args.selector_neighbors < 1
        or args.donor_ridge <= 0
        or not 1 <= args.donor_shortlist <= args.top_k
        or args.donor_fit_queries_per_category < 1
    ):
        parser.error("query cap and selector settings are invalid")
    if args.minimum_headline_delta < 0 or args.uncertainty_weight < 0:
        parser.error("headline delta and uncertainty weight cannot be negative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    signature_path = args.output_dir / "cache_signature.json"
    if signature_path.is_file():
        existing_signature = json.loads(signature_path.read_text())
        if int(existing_signature.get("version", -1)) != 1:
            raise ValueError(
                f"Legacy calibration cache detected at {signature_path}. "
                "Keep it as an artifact and train the self-contained pipeline "
                "in a new output directory."
            )
    cache_dir = args.output_dir / "query_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata = read_metadata(args.train_dir / "metadata.csv")
    category_by_id = {row["sha256"]: row["category"] for row in metadata}
    expected_categories = set(EXPECTED_CATEGORIES)
    if set(category_by_id.values()) != expected_categories:
        raise ValueError(
            f"Expected categories {sorted(expected_categories)}, got "
            f"{sorted(set(category_by_id.values()))}"
        )
    embeddings = load_train_embeddings(
        args.embeddings, metadata, args.view_index, args.model
    )
    rankings = make_train_rankings(metadata, embeddings, args.top_k)
    donor_pool, _ = split_train_queries(metadata)
    donor_fit_queries = balanced_cap(
        donor_pool,
        category_by_id,
        args.donor_fit_queries_per_category,
    )
    donor_rows = prepare_donor_training_rows(
        args, donor_fit_queries, category_by_id, rankings
    )
    donor_reranker, donor_experts = fit_donor_models(
        donor_rows, args.donor_ridge
    )
    operators = operator_config()
    print(
        "Shared donor reranker: "
        f"{donor_reranker['training_queries']} balanced queries, "
        f"RMSE={donor_reranker['training_rmse']:.4f}, "
        f"corr={donor_reranker['training_correlation']:.3f}"
    )
    # Keep selector cross-validation disjoint from the labeled queries used to
    # fit the shared and frozen expert donor rankers. Otherwise the selector
    # fold would be held out while its donor shortlist still benefited from
    # supervision for that same query.
    donor_fit_queries = set(donor_reranker["training_query_ids"])
    unknown_donor_fit_queries = donor_fit_queries - set(category_by_id)
    if unknown_donor_fit_queries:
        raise ValueError(
            "donor-ranker query IDs do not match the training metadata: "
            f"{sorted(unknown_donor_fit_queries)[:3]}"
        )
    if set(donor_reranker["training_query_counts"]) != expected_categories:
        raise ValueError("donor-ranker fit table does not contain all four categories")
    selector_pool = set(category_by_id) - donor_fit_queries
    query_ids = balanced_cap(
        selector_pool, category_by_id, args.queries_per_category
    )
    if query_ids & donor_fit_queries:
        raise AssertionError("selector queries overlap donor-ranker fit queries")
    selector_counts = Counter(category_by_id[sample_id] for sample_id in query_ids)
    if set(selector_counts) != expected_categories:
        raise ValueError("disjoint selector pool does not contain all four categories")
    if min(selector_counts.values()) < args.folds:
        raise ValueError("disjoint selector pool is too small for requested folds")
    print(
        "Disjoint selector queries: "
        + ", ".join(
            f"{category}={selector_counts[category]}"
            for category in sorted(expected_categories)
        )
    )
    folds = fold_assignments(query_ids, category_by_id, args.folds)
    signature = {
        "version": 1,
        "view_index": args.view_index,
        "model": args.model,
        "resolution": args.resolution,
        "margin": args.margin,
        "top_k": args.top_k,
        "donor_shortlist": args.donor_shortlist,
        "donor_reranker": donor_reranker,
        "folds": args.folds,
        "operators": operators,
        "donor_experts": donor_experts,
        "queries_per_category": args.queries_per_category,
        "query_ids": sorted(query_ids),
    }
    if signature_path.is_file():
        if json.loads(signature_path.read_text()) != signature:
            raise ValueError(
                f"Unified cache configuration mismatch: {signature_path}; "
                "use a new output directory"
            )
    else:
        signature_path.write_text(json.dumps(signature, indent=2) + "\n")

    pending = [
        sample_id for sample_id in sorted(query_ids)
        if not cache_path(cache_dir, category_by_id[sample_id], sample_id).is_file()
    ]
    print(
        f"Unified train queries: {len(query_ids)} total; "
        f"{len(pending)} require geometry preparation"
    )

    if pending:
        required_ids = set(pending)
        for sample_id in pending:
            required_ids.update(
                str(row["retrieved_id"]) for row in rankings[sample_id]
            )
        gt_internals = {}
        gt_exteriors = {}
        objective1_internals = {}
        objective1_exteriors = {}
        donor_components = {}
        print(
            f"Loading GT for {len(required_ids)} queries/donors and "
            f"Objective 1 for {len(pending)} pending queries"
        )
        for index, sample_id in enumerate(sorted(required_ids), start=1):
            gt_path = args.train_dir / "voxels" / f"{sample_id}.ply"
            if not gt_path.is_file():
                raise FileNotFoundError(f"Missing training voxel PLY: {gt_path}")
            gt = read_voxels(gt_path, args.resolution)
            gt_internals[sample_id] = interior(gt, args.margin)
            gt_exteriors[sample_id] = gt - gt_internals[sample_id]
            donor_components[sample_id] = connected_components(gt_internals[sample_id])
            if sample_id in pending:
                objective_path = args.objective1_voxels / f"{sample_id}.ply"
                if not objective_path.is_file():
                    raise FileNotFoundError(
                        f"Missing Objective-1 training voxel PLY: {objective_path}"
                    )
                objective = read_voxels(objective_path, args.resolution)
                objective1_internals[sample_id] = interior(objective, args.margin)
                objective1_exteriors[sample_id] = (
                    objective - objective1_internals[sample_id]
                )
            if index % 100 == 0 or index == len(required_ids):
                print(f"  loaded {index}/{len(required_ids)}")

        def prepare_query(sample_id: str) -> tuple[str, dict]:
            base = objective1_internals[sample_id]
            exterior = objective1_exteriors[sample_id]
            safe = enclosed_volume(exterior, args.margin)
            records = prepare_candidate_records(
                rankings[sample_id],
                gt_internals,
                gt_exteriors,
                base,
                exterior,
                safe,
                args.resolution,
            )
            hypotheses = shortlist_records(
                records, donor_reranker, args.donor_shortlist, donor_experts
            )
            candidates = generate_unified_candidates(
                records,
                gt_internals,
                base,
                exterior,
                safe,
                operators,
                args.resolution,
                args.margin,
                donor_components,
                hypotheses,
            )
            query_gt = gt_internals[sample_id]
            baseline = add_core_metric(internal_metrics(query_gt, base), base)
            compact_candidates = []
            for candidate in candidates:
                actual = add_core_metric(
                    internal_metrics(query_gt, candidate["evaluated_internal"]),
                    candidate["evaluated_internal"],
                )
                compact_candidates.append({
                    "action": candidate["action"],
                    "donor_id": candidate["donor_id"],
                    "rank": candidate["rank"],
                    "valid": candidate["valid"],
                    "filter_reason": candidate["filter_reason"],
                    "features": candidate["features"],
                    "actual_metrics": actual,
                })
            return sample_id, {
                "sample_id": sample_id,
                "category": category_by_id[sample_id],
                "fold": folds[sample_id],
                "baseline_metrics": baseline,
                "candidates": compact_candidates,
            }

        started = monotonic()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for index, (sample_id, payload) in enumerate(
                executor.map(prepare_query, pending), start=1
            ):
                path = cache_path(cache_dir, category_by_id[sample_id], sample_id)
                temporary = path.with_suffix(".tmp")
                with temporary.open("wb") as file:
                    pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
                temporary.replace(path)
                if index % 20 == 0 or index == len(pending):
                    elapsed = monotonic() - started
                    remaining = elapsed / index * (len(pending) - index)
                    print(
                        f"  cached {index}/{len(pending)} queries "
                        f"({elapsed / 60:.1f} min, ~{remaining / 60:.1f} min remaining)"
                    )

    payloads = []
    for sample_id in sorted(query_ids):
        path = cache_path(cache_dir, category_by_id[sample_id], sample_id)
        with path.open("rb") as file:
            payloads.append(pickle.load(file))

    rows = [
        training_row(payload, candidate)
        for payload in payloads
        for candidate in payload["candidates"]
    ]
    print(
        f"Fitting {args.folds} shared cross-fit KNN selectors on "
        f"{len(rows)} donor/operator rows"
    )
    crossfit_models = []
    for fold in range(args.folds):
        train_rows = [
            row for row in rows
            if int(row["fold"]) != fold and int(row["valid"])
        ]
        model = fit_selector_knn(
            train_rows,
            [float(row["target_delta"]) for row in train_rows],
            args.selector_neighbors,
        )
        model["held_out_fold"] = fold
        crossfit_models.append(model)
        print(
            f"  fold {fold}: {model['training_rows']} valid rows, "
            f"K={model['neighbors']}"
        )

    model_by_fold = {
        int(model["held_out_fold"]): model for model in crossfit_models
    }
    row_lookup = {
        (row["sample_id"], row["action"], row["donor_id"]): row for row in rows
    }
    for payload in payloads:
        model = model_by_fold[int(payload["fold"])]
        for candidate in payload["candidates"]:
            prediction, prediction_std = predict_selector_distribution(
                candidate["features"], model
            )
            conservative = prediction - args.uncertainty_weight * prediction_std
            candidate["oof_prediction"] = conservative
            row = row_lookup[
                (payload["sample_id"], candidate["action"], candidate["donor_id"])
            ]
            row["oof_predicted_delta"] = prediction
            row["oof_predicted_delta_std"] = prediction_std
            row["oof_conservative_score"] = conservative

    gate_rows = []
    selected_by_gate = {}
    for threshold in ACCEPTANCE_THRESHOLDS:
        for margin_gate in SELECTION_MARGIN_GATES:
            summary, selected = aggregate_gate(payloads, threshold, margin_gate)
            gate_rows.append(summary)
            selected_by_gate[(threshold, margin_gate)] = selected
    baseline_rows = [payload["baseline_metrics"] for payload in payloads]
    write_csv(args.output_dir / "candidate_training.csv", rows)
    write_csv(args.output_dir / "gate_grid.csv", gate_rows)
    selected_gate = select_global_gate(
        gate_rows,
        baseline_rows,
        args.max_internal_ratio,
        args.minimum_headline_delta,
    )
    selected_rows = selected_by_gate[(
        float(selected_gate["acceptance_threshold"]),
        float(selected_gate["minimum_selection_margin"]),
    )]
    write_csv(args.output_dir / "cross_validation.csv", selected_rows)

    valid_rows = [row for row in rows if int(row["valid"])]
    final_model = fit_selector_knn(
        valid_rows,
        [float(row["target_delta"]) for row in valid_rows],
        args.selector_neighbors,
    )

    action_counts = Counter(
        row["selected_action"]
        for row in selected_rows
        if not int(row["used_objective1_fallback"])
    )
    policy = {
        "method": "retrieval_unified_adaptive",
        "view_index": args.view_index,
        "model": args.model,
        "resolution": args.resolution,
        "margin": args.margin,
        "top_k": args.top_k,
        "category_feature_used": False,
        "same_category_gallery": True,
        "donor_reranker": donor_reranker,
        "donor_experts": donor_experts,
        "donor_shortlist": args.donor_shortlist,
        "operators": operators,
        "selector": {
            "models": [final_model],
            "acceptance_threshold": float(selected_gate["acceptance_threshold"]),
            "minimum_selection_margin": float(
                selected_gate["minimum_selection_margin"]
            ),
            "uncertainty_weight": args.uncertainty_weight,
            "global_gate": True,
            "feature_names": list(UNIFIED_FEATURE_NAMES),
            "neighbors": args.selector_neighbors,
        },
        "cross_validation": {
            **selected_gate,
            "folds": args.folds,
            "queries": len(payloads),
            "candidate_rows": len(rows),
            "minimum_headline_delta": args.minimum_headline_delta,
            "accepted_component_hybrid": action_counts[COMPONENT_ACTION],
            "accepted_structural_replace": action_counts[STRUCTURAL_ACTION],
            "rule": "category-stratified query folds; one shared selector and one global gate",
        },
        "training": {
            "train_dir": str(args.train_dir),
            "objective1_voxels": str(args.objective1_voxels),
            "embeddings": str(args.embeddings),
            "donor_query_cache": str(args.output_dir / "donor_query_cache"),
            "query_cache": str(cache_dir),
            "donor_fit_queries_per_category": (
                args.donor_fit_queries_per_category
            ),
            "queries_per_category_cap": args.queries_per_category,
            "workers": args.workers,
            "operator_configuration": "final_retrieval.config",
        },
    }
    with (args.output_dir / "policy.json").open("w", encoding="utf-8") as file:
        json.dump(policy, file, indent=2)
        file.write("\n")

    print("\nFrozen unified category-blind policy")
    print(
        f"  global gate: predicted delta>="
        f"{policy['selector']['acceptance_threshold']:.4f}, "
        f"selection margin>={policy['selector']['minimum_selection_margin']:.4f}"
    )
    print(
        f"  OOF F1={selected_gate['internal_f1']:.4f}, "
        f"delta={selected_gate['mean_internal_f1_delta']:+.4f}, "
        f"bus={selected_gate['bus_f1_delta']:+.4f}, "
        f"car={selected_gate['car_f1_delta']:+.4f}, "
        f"fallback={selected_gate['fallback_fraction']:.1%}"
    )
    print(
        f"  accepted actions: component={action_counts[COMPONENT_ACTION]}, "
        f"structural={action_counts[STRUCTURAL_ACTION]}"
    )
    print(f"Wrote policy to {args.output_dir / 'policy.json'}")


if __name__ == "__main__":
    main()
