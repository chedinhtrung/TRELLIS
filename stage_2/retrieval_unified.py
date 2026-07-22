#!/usr/bin/env python3
"""Category-blind adaptive retrieval over the proven car and bus operators.

Every query receives the same two donor-derived hypotheses:

* component_hybrid: the conservative component fusion that worked for cars;
* structural_replace: the coherent fragment replacement that worked for buses.

A single shared selector scores every donor/operator pair.  Category is never a
selector feature and there are no per-category gates or inference branches.
Objective 1 remains the fail-closed output whenever the shared gate is unsure.
"""

from __future__ import annotations

from copy import deepcopy
from math import log

import numpy as np

from retrieval_v2 import (
    component_core_fraction,
    component_preset_by_name,
    hybrid_fusion,
    predict_quality,
    structural_preset_by_name,
    support_counts,
    transfer_structural_fragments,
    transfer_supported_components,
)


COMPONENT_ACTION = "component_hybrid"
STRUCTURAL_ACTION = "structural_replace"
SELECTOR_BASIS = "standardized_linear_square_structural_interactions_v1"
KNN_SELECTOR = "category_blind_weighted_knn_v1"
_KNN_RUNTIME_ARRAYS: dict[int, tuple[np.ndarray, ...]] = {}


UNIFIED_FEATURE_NAMES = (
    "image_similarity",
    "exterior_tolerant_f1",
    "objective1_internal_tolerant_f1",
    "candidate_medoid_f1",
    "safe_retained_fraction",
    "log_internal_size_ratio",
    "dino_rank_fraction",
    "shared_donor_score",
    "shared_donor_rank_fraction",
    "expert_0_score",
    "expert_1_score",
    "expert_2_score",
    "expert_3_score",
    "expert_0_rank_fraction",
    "expert_1_rank_fraction",
    "expert_2_rank_fraction",
    "expert_3_rank_fraction",
    "is_structural",
    "alignment_strength",
    "alignment_scale_deviation",
    "query_long_mid_ratio",
    "query_mid_short_ratio",
    "safe_to_exterior_ratio",
    "base_internal_to_exterior_ratio",
    "base_core_fraction",
    "transfer_coverage",
    "budget_usage",
    "transferred_to_base_ratio",
    "fused_to_base_ratio",
    "fused_to_exterior_ratio",
    "removed_base_fraction",
    "added_base_fraction",
    "fused_core_fraction",
    "log_kept_transfer_components",
    "rejected_transfer_fraction",
)


def extract_champion_operators(v2_payload: dict, v21_payload: dict) -> dict:
    """Turn the successful class policies into two global operator recipes.

    The recipes are sourced from the frozen car and bus policies once, then
    applied to every query regardless of category.  Their old rerankers and
    gates are deliberately discarded; the unified selector replaces them.
    """
    if v2_payload.get("method") != "retrieval_v2_confidence_gated_hybrid":
        raise ValueError("component champion requires a frozen retrieval-v2 policy")
    if v21_payload.get("method") != "retrieval_v21_category_structural":
        raise ValueError("structural champion requires a frozen retrieval-v2.1 policy")
    component_source = v2_payload.get("categories", {}).get("car")
    structural_source = v21_payload.get("categories", {}).get("bus")
    if not component_source or not structural_source:
        raise ValueError("frozen policies do not contain the car and bus champions")
    if structural_source.get("mode") != "structural_fragments":
        raise ValueError("the frozen bus champion is not structural retrieval")

    component = {
        "action": COMPONENT_ACTION,
        "component_preset": deepcopy(component_source["component_preset"]),
        "max_internal_to_exterior_ratio": float(
            component_source["max_internal_to_exterior_ratio"]
        ),
        "max_expansion": float(component_source["max_expansion"]),
        "base_min_component_voxels": int(
            component_source["base_min_component_voxels"]
        ),
        "base_max_core_fraction": float(
            component_source["base_max_core_fraction"]
        ),
    }
    structural = {
        "action": STRUCTURAL_ACTION,
        "structural_preset": deepcopy(structural_source["structural_preset"]),
        "transfer_mode": str(structural_source.get("transfer_mode", "fragments")),
        "fusion_mode": str(structural_source.get("fusion_mode", "replace")),
        "max_internal_to_exterior_ratio": float(
            structural_source["max_internal_to_exterior_ratio"]
        ),
        "max_expansion": float(structural_source["max_expansion"]),
        "base_min_component_voxels": int(
            structural_source.get("base_min_component_voxels", 64)
        ),
        "base_max_core_fraction": float(
            structural_source.get("base_max_core_fraction", 0.25)
        ),
    }
    if structural["transfer_mode"] != "fragments":
        raise ValueError("unified structural champion must transfer donor fragments")
    if structural["fusion_mode"] not in {"replace", "hybrid"}:
        raise ValueError("unsupported structural champion fusion mode")
    return {COMPONENT_ACTION: component, STRUCTURAL_ACTION: structural}


def extract_donor_experts(v21_payload: dict) -> list[dict]:
    """Return the four frozen donor scorers as universally applied experts."""
    if v21_payload.get("method") != "retrieval_v21_category_structural":
        raise ValueError("donor experts require a frozen retrieval-v2.1 policy")
    categories = v21_payload.get("categories", {})
    if set(categories) != {"bus", "cabinet", "car", "file_cabinet"}:
        raise ValueError("frozen v2.1 policy does not contain four donor experts")
    experts = []
    for category in sorted(categories):
        model = categories[category].get("reranker")
        if not model:
            raise ValueError(f"missing frozen donor expert for {category}")
        experts.append({"source": category, "model": deepcopy(model)})
    return experts


def _extent_features(exterior: set[tuple[int, int, int]]) -> tuple[float, float]:
    if not exterior:
        return 1.0, 1.0
    points = np.asarray(list(exterior), dtype=np.int32)
    spans = np.sort(points.max(axis=0) - points.min(axis=0) + 1)[::-1].astype(float)
    return float(spans[0] / max(spans[1], 1.0)), float(
        spans[1] / max(spans[2], 1.0)
    )


def shortlist_records(
    records: list[dict], reranker: dict, count: int,
    experts: list[dict] | None = None,
) -> list[dict]:
    """Use one shared donor model plus deployable diversity anchors."""
    if count < 1 or count > len(records):
        raise ValueError("donor shortlist count must lie inside the ranking")
    for record in records:
        record["shared_donor_score"] = predict_quality(record["features"], reranker)
    donor_order = sorted(
        records,
        key=lambda row: (
            -float(row["shared_donor_score"]),
            -float(row["image_similarity"]),
            int(row["rank"]),
        ),
    )
    for index, record in enumerate(donor_order, start=1):
        record["shared_donor_rank"] = index

    experts = experts or []
    if len(experts) not in {0, 4}:
        raise ValueError("unified retrieval expects zero or four donor experts")
    expert_anchors = []
    for expert_index, expert in enumerate(experts):
        expert_order = sorted(
            records,
            key=lambda row: (
                -predict_quality(row["features"], expert["model"]),
                -float(row["image_similarity"]),
                int(row["rank"]),
            ),
        )
        for rank, record in enumerate(expert_order, start=1):
            record.setdefault("expert_scores", {})[expert_index] = predict_quality(
                record["features"], expert["model"]
            )
            record.setdefault("expert_ranks", {})[expert_index] = rank
        expert_anchors.append(expert_order[0])
    if not experts:
        for record in records:
            record["expert_scores"] = {index: 0.0 for index in range(4)}
            record["expert_ranks"] = {
                index: int(record["rank"]) for index in range(4)
            }

    anchors = [
        donor_order[0],
        max(records, key=lambda row: (row["image_similarity"], -row["rank"])),
        max(records, key=lambda row: (
            row["features"]["exterior_tolerant_f1"], -row["rank"]
        )),
        max(records, key=lambda row: (
            row["features"]["candidate_medoid_f1"], -row["rank"]
        )),
        max(records, key=lambda row: (
            row["features"]["objective1_internal_tolerant_f1"], -row["rank"]
        )),
    ]
    selected = []
    seen = set()
    for record in expert_anchors + anchors + donor_order:
        donor_id = str(record["retrieved_id"])
        if donor_id in seen:
            continue
        selected.append(record)
        seen.add(donor_id)
        if len(selected) == count:
            break
    return selected


def _fusion_diagnostics(
    base: set[tuple[int, int, int]],
    preserved: set[tuple[int, int, int]],
    fused: set[tuple[int, int, int]],
    safe: set[tuple[int, int, int]],
    budget: int,
) -> dict[str, int]:
    return {
        "protected_objective1_voxels": len(base - safe),
        "unmatched_objective1_voxels": 0,
        "eligible_objective1_components": 0,
        "preserved_objective1_voxels": len(preserved),
        "preserved_objective1_components": 0,
        "rejected_objective1_budget_components": 0,
        "fused_internal_voxels": len(fused),
        "total_internal_budget": budget,
    }


def _hard_filter_reason(
    transferred: set[tuple[int, int, int]],
    base: set[tuple[int, int, int]],
    fused: set[tuple[int, int, int]],
    base_core_fraction: float,
    fused_core_fraction: float,
) -> str:
    reasons = []
    if not transferred:
        reasons.append("empty_transfer")
    if base and len(fused) / len(base) > 1.60:
        reasons.append("excess_expansion")
    if (
        fused_core_fraction > max(0.20, base_core_fraction + 0.04)
    ):
        reasons.append("dense_candidate")
    return "+".join(reasons)


def generate_unified_candidates(
    records: list[dict],
    candidate_internals: dict[str, set[tuple[int, int, int]]],
    objective1_internal: set[tuple[int, int, int]],
    objective1_exterior: set[tuple[int, int, int]],
    safe_volume: set[tuple[int, int, int]],
    operators: dict,
    resolution: int,
    margin: int,
    donor_components: dict[str, list[set[tuple[int, int, int]]]] | None = None,
    hypothesis_records: list[dict] | None = None,
) -> list[dict]:
    """Generate both global operator hypotheses for every retrieved donor."""
    # Keep selector-only imports lightweight; PLY-backed compare_internals
    # pulls in utils3d, which is needed only during actual geometry generation.
    from compare_internals import interior

    if set(operators) != {COMPONENT_ACTION, STRUCTURAL_ACTION}:
        raise ValueError("unified retrieval requires exactly the two champion operators")
    if not records:
        raise ValueError("unified retrieval requires at least one donor record")

    support = support_counts([record["aligned_internal"] for record in records])
    base_core = component_core_fraction(objective1_internal)
    long_mid, mid_short = _extent_features(objective1_exterior)
    query_common = {
        "query_long_mid_ratio": long_mid,
        "query_mid_short_ratio": mid_short,
        "safe_to_exterior_ratio": len(safe_volume)
        / max(len(objective1_exterior), 1),
        "base_internal_to_exterior_ratio": len(objective1_internal)
        / max(len(objective1_exterior), 1),
        "base_core_fraction": base_core,
    }
    candidates = []
    top_k = len(records)
    hypothesis_records = hypothesis_records or records
    for record in hypothesis_records:
        donor_id = str(record["retrieved_id"])
        components = donor_components.get(donor_id) if donor_components else None
        scale = np.asarray(record["alignment"].scale, dtype=np.float64)
        record_common = {
            **record["features"],
            **query_common,
            "dino_rank_fraction": float(record["rank"]) / top_k,
            "shared_donor_score": float(record.get("shared_donor_score", 0.0)),
            "shared_donor_rank_fraction": float(
                record.get("shared_donor_rank", record["rank"])
            ) / top_k,
            **{
                f"expert_{index}_score": float(record["expert_scores"][index])
                for index in range(4)
            },
            **{
                f"expert_{index}_rank_fraction": float(
                    record["expert_ranks"][index]
                ) / top_k
                for index in range(4)
            },
            "alignment_strength": float(record["alignment"].strength),
            "alignment_scale_deviation": float(np.mean(np.abs(scale - 1.0))),
        }

        for action in (COMPONENT_ACTION, STRUCTURAL_ACTION):
            recipe = operators[action]
            category_budget = int(round(
                float(recipe["max_internal_to_exterior_ratio"])
                * len(objective1_exterior)
            ))
            base_budget = (
                int(round(float(recipe["max_expansion"]) * len(objective1_internal)))
                if objective1_internal
                else category_budget
            )
            donor_budget = min(category_budget, base_budget)

            if action == COMPONENT_ACTION:
                preset = component_preset_by_name(recipe["component_preset"]["name"])
                transferred, accepted_source, transfer_diag = transfer_supported_components(
                    candidate_internals[donor_id],
                    record["alignment"],
                    safe_volume,
                    support,
                    preset,
                    donor_budget,
                    resolution,
                    components,
                )
                coverage_denominator = len(objective1_internal)
                fused, preserved, fusion_diag = hybrid_fusion(
                    objective1_internal,
                    transferred,
                    safe_volume,
                    int(recipe["base_min_component_voxels"]),
                    float(recipe["base_max_core_fraction"]),
                    category_budget,
                )
            else:
                preset = structural_preset_by_name(
                    recipe["structural_preset"]["name"]
                )
                transferred, accepted_source, transfer_diag = transfer_structural_fragments(
                    candidate_internals[donor_id],
                    record["alignment"],
                    safe_volume,
                    support,
                    preset,
                    donor_budget,
                    resolution,
                    components,
                )
                usable = record["aligned_internal"] & safe_volume
                coverage_denominator = min(len(usable), donor_budget)
                if recipe["fusion_mode"] == "hybrid":
                    fused, preserved, fusion_diag = hybrid_fusion(
                        objective1_internal,
                        transferred,
                        safe_volume,
                        int(recipe["base_min_component_voxels"]),
                        float(recipe["base_max_core_fraction"]),
                        category_budget,
                    )
                else:
                    preserved = objective1_internal - safe_volume
                    fused = transferred | preserved
                    fusion_diag = _fusion_diagnostics(
                        objective1_internal,
                        preserved,
                        fused,
                        safe_volume,
                        category_budget,
                    )

            prediction = objective1_exterior | fused
            evaluated_internal = interior(prediction, margin)
            fused_core = component_core_fraction(evaluated_internal)
            coverage = (
                len(transferred) / coverage_denominator
                if coverage_denominator
                else float(bool(transferred))
            )
            removed = objective1_internal - evaluated_internal
            added = evaluated_internal - objective1_internal
            features = {
                **record_common,
                "is_structural": float(action == STRUCTURAL_ACTION),
                "transfer_coverage": float(coverage),
                "budget_usage": len(transferred) / max(donor_budget, 1),
                "transferred_to_base_ratio": len(transferred)
                / max(len(objective1_internal), 1),
                "fused_to_base_ratio": len(evaluated_internal)
                / max(len(objective1_internal), 1),
                "fused_to_exterior_ratio": len(evaluated_internal)
                / max(len(objective1_exterior), 1),
                "removed_base_fraction": len(removed)
                / max(len(objective1_internal), 1),
                "added_base_fraction": len(added)
                / max(len(objective1_internal), 1),
                "fused_core_fraction": fused_core,
                "log_kept_transfer_components": log(
                    float(transfer_diag.get("kept_components", 0)) + 1.0
                ),
                "rejected_transfer_fraction": sum(
                    int(transfer_diag.get(name, 0))
                    for name in (
                        "rejected_small_components",
                        "rejected_clipped_components",
                        "rejected_fragmented_components",
                        "rejected_dense_components",
                        "rejected_unsupported_components",
                        "rejected_budget_components",
                    )
                ) / max(int(transfer_diag.get("donor_components", 0)), 1),
            }
            missing = set(UNIFIED_FEATURE_NAMES) - set(features)
            if missing:
                raise AssertionError(f"missing unified features: {sorted(missing)}")
            filter_reason = _hard_filter_reason(
                transferred,
                objective1_internal,
                evaluated_internal,
                base_core,
                fused_core,
            )
            transfer_diag.setdefault(
                "usable_donor_voxels", len(record["aligned_internal"] & safe_volume)
            )
            transfer_diag.setdefault("rejected_dense_components", 0)
            candidates.append({
                "action": action,
                "donor_id": donor_id,
                "rank": int(record["rank"]),
                "image_similarity": float(record["image_similarity"]),
                "alignment": record["alignment"],
                "features": features,
                "prediction": prediction,
                "evaluated_internal": evaluated_internal,
                "transferred": transferred,
                "accepted_source": accepted_source,
                "preserved": preserved,
                "transfer_coverage": float(coverage),
                "transfer_coverage_denominator": int(coverage_denominator),
                "donor_budget": int(donor_budget),
                "valid": not filter_reason,
                "filter_reason": filter_reason,
                "transfer_diagnostics": transfer_diag,
                "fusion_diagnostics": fusion_diag,
            })
    return candidates


def _raw_matrix(feature_rows: list[dict[str, float]]) -> np.ndarray:
    if not feature_rows:
        raise ValueError("selector requires non-empty feature rows")
    return np.asarray(
        [[float(row[name]) for name in UNIFIED_FEATURE_NAMES] for row in feature_rows],
        dtype=np.float64,
    )


def _design_matrix(
    raw: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    normalized = np.clip((raw - mean) / scale, -6.0, 6.0)
    structural = raw[:, UNIFIED_FEATURE_NAMES.index("is_structural") :][:, :1]
    return np.concatenate(
        (normalized, normalized * normalized, normalized * structural), axis=1
    )


def fit_selector_ridge(
    feature_rows: list[dict[str, float]],
    labels: list[float],
    weights: list[float] | None = None,
    ridge: float = 10.0,
) -> dict:
    """Fit a compact nonlinear shared scorer with no category input."""
    if len(feature_rows) != len(labels) or not feature_rows:
        raise ValueError("selector fitting requires equally sized non-empty inputs")
    if ridge <= 0:
        raise ValueError("selector ridge must be positive")
    raw = _raw_matrix(feature_rows)
    target = np.asarray(labels, dtype=np.float64)
    sample_weights = (
        np.ones(len(target), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    if sample_weights.shape != target.shape or np.any(sample_weights <= 0):
        raise ValueError("selector weights must be positive and match labels")
    sample_weights /= sample_weights.sum()
    mean = np.average(raw, axis=0, weights=sample_weights)
    variance = np.average((raw - mean) ** 2, axis=0, weights=sample_weights)
    scale = np.sqrt(variance)
    scale[scale < 1e-8] = 1.0
    design = _design_matrix(raw, mean, scale)
    design_mean = np.average(design, axis=0, weights=sample_weights)
    centered = design - design_mean
    target_mean = float(np.average(target, weights=sample_weights))
    centered_target = target - target_mean
    root_weight = np.sqrt(sample_weights * len(sample_weights))
    weighted_design = centered * root_weight[:, None]
    weighted_target = centered_target * root_weight
    regularizer = ridge * np.eye(weighted_design.shape[1], dtype=np.float64)
    coefficient = np.linalg.solve(
        weighted_design.T @ weighted_design + regularizer,
        weighted_design.T @ weighted_target,
    )
    prediction = target_mean + centered @ coefficient
    correlation = (
        float(np.corrcoef(prediction, target)[0, 1])
        if np.std(prediction) > 1e-12 and np.std(target) > 1e-12
        else 0.0
    )
    return {
        "feature_names": list(UNIFIED_FEATURE_NAMES),
        "basis": SELECTOR_BASIS,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "design_mean": design_mean.tolist(),
        "coefficient": coefficient.tolist(),
        "intercept": target_mean,
        "ridge": float(ridge),
        "training_rows": len(feature_rows),
        "training_rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "training_correlation": correlation,
    }


def predict_selector(features: dict[str, float], model: dict) -> float:
    if model.get("model_type", "ridge") == KNN_SELECTOR:
        return predict_selector_distribution(features, model)[0]
    if tuple(model.get("feature_names", ())) != UNIFIED_FEATURE_NAMES:
        raise ValueError("unified selector feature schema mismatch")
    if model.get("basis") != SELECTOR_BASIS:
        raise ValueError("unified selector basis mismatch")
    raw = _raw_matrix([features])
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    design = _design_matrix(raw, mean, scale)[0]
    design_mean = np.asarray(model["design_mean"], dtype=np.float64)
    coefficient = np.asarray(model["coefficient"], dtype=np.float64)
    return float(model["intercept"] + (design - design_mean) @ coefficient)


def fit_selector_knn(
    feature_rows: list[dict[str, float]],
    labels: list[float],
    neighbors: int = 8,
    distance_floor: float = 0.05,
) -> dict:
    """Store a small nonlinear category-blind neighborhood regressor.

    Retrieval safety is highly local: similar aspect ratios, transfer coverage,
    and candidate density behave alike.  KNN captures that boundary without a
    heavyweight dependency or a high-capacity neural network.
    """
    if len(feature_rows) != len(labels) or not feature_rows:
        raise ValueError("KNN selector requires equally sized non-empty inputs")
    if neighbors < 1 or distance_floor <= 0:
        raise ValueError("KNN neighbors and distance floor must be positive")
    raw = _raw_matrix(feature_rows)
    mean = raw.mean(axis=0)
    scale = raw.std(axis=0)
    scale[scale < 1e-8] = 1.0
    feature_weight = np.ones(len(UNIFIED_FEATURE_NAMES), dtype=np.float64)
    for name in ("is_structural", "query_long_mid_ratio", "query_mid_short_ratio"):
        feature_weight[UNIFIED_FEATURE_NAMES.index(name)] = 2.0
    transformed = np.clip((raw - mean) / scale, -5.0, 5.0) * feature_weight
    return {
        "model_type": KNN_SELECTOR,
        "feature_names": list(UNIFIED_FEATURE_NAMES),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "feature_weight": feature_weight.tolist(),
        "training_features": transformed.astype(np.float32).tolist(),
        "training_labels": [float(value) for value in labels],
        "neighbors": min(int(neighbors), len(labels)),
        "distance_floor": float(distance_floor),
        "training_rows": len(labels),
    }


def predict_selector_distribution(
    features: dict[str, float], model: dict
) -> tuple[float, float]:
    model_type = model.get("model_type", "ridge")
    if model_type != KNN_SELECTOR:
        return predict_selector(features, model), 0.0
    if tuple(model.get("feature_names", ())) != UNIFIED_FEATURE_NAMES:
        raise ValueError("unified KNN selector feature schema mismatch")
    arrays = _KNN_RUNTIME_ARRAYS.get(id(model))
    if arrays is None:
        arrays = (
            np.asarray(model["mean"], dtype=np.float64),
            np.asarray(model["scale"], dtype=np.float64),
            np.asarray(model["feature_weight"], dtype=np.float64),
            np.asarray(model["training_features"], dtype=np.float32),
            np.asarray(model["training_labels"], dtype=np.float64),
        )
        _KNN_RUNTIME_ARRAYS[id(model)] = arrays
    mean, scale, feature_weight, training, labels = arrays
    query = np.asarray(
        [float(features[name]) for name in UNIFIED_FEATURE_NAMES], dtype=np.float64
    )
    query = np.clip((query - mean) / scale, -5.0, 5.0) * feature_weight
    distances = np.mean((training - query) ** 2, axis=1)
    neighbors = min(int(model["neighbors"]), len(labels))
    indices = np.argpartition(distances, neighbors - 1)[:neighbors]
    weights = 1.0 / (distances[indices] + float(model["distance_floor"]))
    prediction = float(np.sum(weights * labels[indices]) / np.sum(weights))
    variance = float(
        np.sum(weights * (labels[indices] - prediction) ** 2) / np.sum(weights)
    )
    return prediction, float(np.sqrt(max(variance, 0.0)))


def predict_selector_ensemble(
    features: dict[str, float], models: list[dict]
) -> tuple[float, float]:
    if not models:
        raise ValueError("unified selector ensemble is empty")
    distributions = [
        predict_selector_distribution(features, model) for model in models
    ]
    means = np.asarray([item[0] for item in distributions], dtype=np.float64)
    variances = np.asarray([item[1] ** 2 for item in distributions], dtype=np.float64)
    combined_mean = float(means.mean())
    combined_variance = float(
        np.mean(variances + means * means) - combined_mean * combined_mean
    )
    return combined_mean, float(np.sqrt(max(combined_variance, 0.0)))


def select_unified_candidate(
    candidates: list[dict],
    models: list[dict],
    acceptance_threshold: float,
    minimum_selection_margin: float,
    uncertainty_weight: float,
) -> tuple[dict, bool, str, float]:
    """Apply one global lower-confidence gate to all categories."""
    if not candidates:
        raise ValueError("cannot select from an empty candidate list")
    if uncertainty_weight < 0:
        raise ValueError("uncertainty weight cannot be negative")
    for candidate in candidates:
        mean, std = predict_selector_ensemble(candidate["features"], models)
        candidate["predicted_delta"] = mean
        candidate["predicted_delta_std"] = std
        candidate["conservative_score"] = mean - uncertainty_weight * std
    valid = [candidate for candidate in candidates if candidate["valid"]]
    ordering = lambda candidate: (
        float(candidate["conservative_score"]),
        float(candidate["predicted_delta"]),
        float(candidate["image_similarity"]),
        -int(candidate["rank"]),
    )
    if not valid:
        best = max(candidates, key=ordering)
        return best, True, "no_valid_candidate", 0.0
    ordered = sorted(valid, key=ordering, reverse=True)
    best = ordered[0]
    selection_margin = (
        float(best["conservative_score"] - ordered[1]["conservative_score"])
        if len(ordered) > 1
        else float("inf")
    )
    reasons = []
    if float(best["conservative_score"]) < acceptance_threshold:
        reasons.append("low_predicted_improvement")
    if selection_margin < minimum_selection_margin:
        reasons.append("low_selection_margin")
    return (
        best,
        bool(reasons),
        "+".join(reasons) if reasons else "accepted",
        selection_margin,
    )
