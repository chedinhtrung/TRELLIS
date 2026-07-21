#!/usr/bin/env python3
"""Core geometry for retrieval-v2.

The module deliberately keeps retrieval conservative:

* one real donor is selected from the DINO top-K list;
* the donor is fitted with a small, smooth axis-aligned transform;
* complete donor components are accepted or rejected using neighbor support;
* Objective-1 interior components that are not explained by the donor survive;
* a calibrated confidence gate falls back to Objective 1 when transfer is weak.

No voxelwise voting is allowed to create new geometry.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass
from math import log

import numpy as np


Voxel = tuple[int, int, int]


NEIGHBORS_26 = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
)
AXIS_NEIGHBORS = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)
_OFFSET_CACHE: dict[int, np.ndarray] = {}


@dataclass(frozen=True)
class Alignment:
    """A small affine fit in continuous voxel coordinates."""

    scale: tuple[float, float, float]
    source_center: tuple[float, float, float]
    mapped_center: tuple[float, float, float]
    strength: float
    exterior_tolerant_f1: float


@dataclass(frozen=True)
class ComponentPreset:
    name: str
    min_component_voxels: int
    min_retained_fraction: float
    min_other_support: int
    min_supported_fraction: float


COMPONENT_PRESETS = (
    ComponentPreset("detail", 12, 0.65, 1, 0.10),
    ComponentPreset("balanced", 24, 0.70, 1, 0.20),
    ComponentPreset("strict", 32, 0.75, 2, 0.25),
)


FEATURE_NAMES = (
    "image_similarity",
    "exterior_tolerant_f1",
    "objective1_internal_tolerant_f1",
    "candidate_medoid_f1",
    "safe_retained_fraction",
    "log_internal_size_ratio",
)


def connected_components(voxels: set[Voxel]) -> list[set[Voxel]]:
    remaining = set(voxels)
    components: list[set[Voxel]] = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        component = {seed}
        queue = deque([seed])
        while queue:
            x, y, z = queue.popleft()
            for dx, dy, dz in NEIGHBORS_26:
                neighbor = (x + dx, y + dy, z + dz)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    components.sort(key=lambda component: (-len(component), min(component)))
    return components


def exact_f1(left: set[Voxel], right: set[Voxel]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    precision = overlap / len(right)
    recall = overlap / len(left)
    return 2.0 * precision * recall / (precision + recall) if overlap else 0.0


def _points(voxels: set[Voxel]) -> np.ndarray:
    if not voxels:
        return np.empty((0, 3), dtype=np.float64)
    # Geometry below is order-independent. Avoiding an O(n log n) Python sort
    # matters because each top-20 query performs many voxel comparisons.
    return np.asarray(list(voxels), dtype=np.float64)


def _tolerant_membership(
    source: set[Voxel], target: set[Voxel], tolerance: float
) -> tuple[list[Voxel], np.ndarray]:
    """Match integer voxels with an inclusive L-infinity radius.

    A dense local lookup is substantially faster here than constructing a
    SciPy KD-tree for every small 64^3 voxel comparison.  Every call site uses
    integer voxel coordinates and an integer tolerance, so this is also an
    exact implementation of the intended matching rule.
    """
    if tolerance < 0 or tolerance != int(tolerance):
        raise ValueError("voxel tolerance must be a non-negative integer")
    ordered = list(source)
    if not ordered or not target:
        return ordered, np.zeros(len(ordered), dtype=bool)

    radius = int(tolerance)
    source_points = np.asarray(ordered, dtype=np.int32)
    target_points = np.asarray(list(target), dtype=np.int32)
    minimum = np.minimum(
        source_points.min(axis=0), target_points.min(axis=0) - radius
    )
    maximum = np.maximum(
        source_points.max(axis=0), target_points.max(axis=0) + radius
    )
    shape = tuple((maximum - minimum + 1).tolist())
    lookup = np.zeros(shape, dtype=bool)

    offsets = _OFFSET_CACHE.get(radius)
    if offsets is None:
        offsets = np.asarray(
            [
                (dx, dy, dz)
                for dx in range(-radius, radius + 1)
                for dy in range(-radius, radius + 1)
                for dz in range(-radius, radius + 1)
            ],
            dtype=np.int32,
        )
        _OFFSET_CACHE[radius] = offsets
    expanded = (
        target_points[:, None, :] + offsets[None, :, :] - minimum
    ).reshape(-1, 3)
    lookup[tuple(expanded.T)] = True
    shifted_source = source_points - minimum
    return ordered, lookup[tuple(shifted_source.T)]


def tolerant_overlap_fraction(
    source: set[Voxel], target: set[Voxel], tolerance: float = 1.0
) -> float:
    """Fraction of source points within L-infinity tolerance of target."""
    if not source:
        return 1.0 if not target else 0.0
    if not target:
        return 0.0
    _ordered, matched = _tolerant_membership(source, target, tolerance)
    return float(np.mean(matched))


def tolerant_f1(
    truth: set[Voxel], prediction: set[Voxel], tolerance: float = 1.0
) -> float:
    if not truth and not prediction:
        return 1.0
    precision = tolerant_overlap_fraction(prediction, truth, tolerance)
    recall = tolerant_overlap_fraction(truth, prediction, tolerance)
    return (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def alignment_payload(alignment: Alignment) -> dict:
    return asdict(alignment)


def alignment_from_payload(payload: dict) -> Alignment:
    return Alignment(
        scale=tuple(float(value) for value in payload["scale"]),
        source_center=tuple(float(value) for value in payload["source_center"]),
        mapped_center=tuple(float(value) for value in payload["mapped_center"]),
        strength=float(payload["strength"]),
        exterior_tolerant_f1=float(payload["exterior_tolerant_f1"]),
    )


def transform_points(points: np.ndarray, alignment: Alignment) -> np.ndarray:
    scale = np.asarray(alignment.scale, dtype=np.float64)
    source_center = np.asarray(alignment.source_center, dtype=np.float64)
    mapped_center = np.asarray(alignment.mapped_center, dtype=np.float64)
    return (np.asarray(points, dtype=np.float64) - source_center) * scale + mapped_center


def align_voxels(
    voxels: set[Voxel], alignment: Alignment, resolution: int
) -> set[Voxel]:
    if not voxels:
        return set()
    mapped = np.rint(transform_points(_points(voxels), alignment)).astype(np.int32)
    valid = np.all((mapped >= 0) & (mapped < resolution), axis=1)
    return {tuple(row) for row in mapped[valid].tolist()}


def find_alignment(
    source_exterior: set[Voxel],
    target_exterior: set[Voxel],
    resolution: int,
    strengths: tuple[float, ...] = (0.0, 0.35, 0.70),
    scale_limit: float = 0.12,
    tolerance: float = 1.0,
) -> Alignment:
    """Choose a conservative robust-extent fit by tolerant surface F1.

    Strength zero is identity.  Larger values move smoothly toward a robust
    10--90 percentile fit, with each axis scale capped to avoid the destructive
    full-AABB warps used by earlier experiments.
    """
    if not source_exterior or not target_exterior:
        return Alignment((1.0, 1.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0, 0.0)
    if not strengths or any(value < 0.0 or value > 1.0 for value in strengths):
        raise ValueError("alignment strengths must lie in [0, 1]")

    source = _points(source_exterior)
    target = _points(target_exterior)
    source_low, source_high = np.quantile(source, (0.10, 0.90), axis=0)
    target_low, target_high = np.quantile(target, (0.10, 0.90), axis=0)
    source_span = np.maximum(source_high - source_low, 1.0)
    target_span = np.maximum(target_high - target_low, 1.0)
    full_scale = np.clip(
        target_span / source_span, 1.0 - scale_limit, 1.0 + scale_limit
    )
    source_center = 0.5 * (source_low + source_high)
    target_center = 0.5 * (target_low + target_high)

    scored: list[tuple[float, float, Alignment]] = []
    for strength in strengths:
        scale = 1.0 + strength * (full_scale - 1.0)
        mapped_center = source_center + strength * (target_center - source_center)
        alignment = Alignment(
            tuple(float(value) for value in scale),
            tuple(float(value) for value in source_center),
            tuple(float(value) for value in mapped_center),
            float(strength),
            0.0,
        )
        aligned = align_voxels(source_exterior, alignment, resolution)
        score = tolerant_f1(target_exterior, aligned, tolerance)
        alignment = Alignment(
            alignment.scale,
            alignment.source_center,
            alignment.mapped_center,
            alignment.strength,
            score,
        )
        scored.append((score, -strength, alignment))
    return max(scored)[2]


def medoid_scores(candidate_internals: list[set[Voxel]]) -> list[float]:
    """Exact pairwise agreement; used only as a coherent-neighborhood cue."""
    count = len(candidate_internals)
    if count == 1:
        return [1.0]
    totals = np.zeros(count, dtype=np.float64)
    for left in range(count):
        for right in range(left + 1, count):
            score = exact_f1(candidate_internals[left], candidate_internals[right])
            totals[left] += score
            totals[right] += score
    return (totals / (count - 1)).tolist()


def candidate_features(
    image_similarity: float,
    exterior_tolerant_f1: float,
    objective1_internal: set[Voxel],
    candidate_internal: set[Voxel],
    medoid_f1: float,
    safe_volume: set[Voxel],
) -> dict[str, float]:
    safe_internal = candidate_internal & safe_volume
    retained = len(safe_internal) / len(candidate_internal) if candidate_internal else 0.0
    return {
        "image_similarity": float(image_similarity),
        "exterior_tolerant_f1": float(exterior_tolerant_f1),
        "objective1_internal_tolerant_f1": tolerant_f1(
            objective1_internal, safe_internal, 1.0
        ),
        "candidate_medoid_f1": float(medoid_f1),
        "safe_retained_fraction": retained,
        "log_internal_size_ratio": log(
            (len(safe_internal) + 1.0) / (len(objective1_internal) + 1.0)
        ),
    }


def prepare_candidate_records(
    ranking: list[dict],
    candidate_internals: dict[str, set[Voxel]],
    candidate_exteriors: dict[str, set[Voxel]],
    objective1_internal: set[Voxel],
    objective1_exterior: set[Voxel],
    safe_volume: set[Voxel],
    resolution: int,
) -> list[dict]:
    """Align a ranked neighborhood and compute all deployable reranker cues."""
    records = []
    for row in ranking:
        candidate_id = str(row["retrieved_id"])
        alignment = find_alignment(
            candidate_exteriors[candidate_id], objective1_exterior, resolution
        )
        records.append({
            **row,
            "alignment": alignment,
            "aligned_exterior": align_voxels(
                candidate_exteriors[candidate_id], alignment, resolution
            ),
            "aligned_internal": align_voxels(
                candidate_internals[candidate_id], alignment, resolution
            ),
        })
    medoids = medoid_scores([
        record["aligned_internal"] & safe_volume for record in records
    ])
    for record, medoid in zip(records, medoids):
        record["features"] = candidate_features(
            float(record["image_similarity"]),
            float(record["alignment"].exterior_tolerant_f1),
            objective1_internal,
            record["aligned_internal"],
            medoid,
            safe_volume,
        )
    return records


def fit_ridge(
    feature_rows: list[dict[str, float]], labels: list[float], ridge: float = 1.0
) -> dict:
    if len(feature_rows) != len(labels) or not feature_rows:
        raise ValueError("ridge fitting requires equally sized non-empty rows and labels")
    matrix = np.asarray(
        [[row[name] for name in FEATURE_NAMES] for row in feature_rows],
        dtype=np.float64,
    )
    target = np.asarray(labels, dtype=np.float64)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = (matrix - mean) / scale
    centered_target = target - target.mean()
    regularizer = ridge * np.eye(normalized.shape[1], dtype=np.float64)
    coefficient = np.linalg.solve(
        normalized.T @ normalized + regularizer,
        normalized.T @ centered_target,
    )
    prediction = target.mean() + normalized @ coefficient
    correlation = (
        float(np.corrcoef(prediction, target)[0, 1])
        if np.std(prediction) > 1e-12 and np.std(target) > 1e-12
        else 0.0
    )
    return {
        "feature_names": list(FEATURE_NAMES),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficient": coefficient.tolist(),
        "intercept": float(target.mean()),
        "ridge": float(ridge),
        "training_rows": len(feature_rows),
        "training_rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "training_correlation": correlation,
    }


def predict_quality(features: dict[str, float], model: dict) -> float:
    if tuple(model["feature_names"]) != FEATURE_NAMES:
        raise ValueError("reranker feature schema mismatch")
    values = np.asarray([features[name] for name in FEATURE_NAMES], dtype=np.float64)
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    coefficient = np.asarray(model["coefficient"], dtype=np.float64)
    prediction = float(model["intercept"] + ((values - mean) / scale) @ coefficient)
    return float(np.clip(prediction, 0.0, 1.0))


def support_counts(candidate_internals: list[set[Voxel]]) -> Counter[Voxel]:
    counts: Counter[Voxel] = Counter()
    for voxels in candidate_internals:
        counts.update(voxels)
    return counts


def component_core_fraction(component: set[Voxel]) -> float:
    if not component:
        return 0.0
    core = sum(
        all((x + dx, y + dy, z + dz) in component for dx, dy, dz in AXIS_NEIGHBORS)
        for x, y, z in component
    )
    return core / len(component)


def transfer_supported_components(
    donor_internal: set[Voxel],
    alignment: Alignment,
    safe_volume: set[Voxel],
    all_support_counts: Counter[Voxel],
    preset: ComponentPreset,
    voxel_budget: int,
    resolution: int,
) -> tuple[set[Voxel], set[Voxel], dict[str, float | int]]:
    """Transfer complete aligned donor components and retain provenance."""
    candidates = []
    rejected_small = rejected_clipped = rejected_fragmented = rejected_support = 0
    donor_components = connected_components(donor_internal)
    for original_component in donor_components:
        aligned_whole = align_voxels(original_component, alignment, resolution)
        clipped = aligned_whole & safe_volume
        if len(clipped) < preset.min_component_voxels:
            rejected_small += 1
            continue
        retained = len(clipped) / len(aligned_whole) if aligned_whole else 0.0
        if retained < preset.min_retained_fraction:
            rejected_clipped += 1
            continue
        parts = connected_components(clipped)
        dominant = parts[0]
        if len(dominant) / len(clipped) < 0.90:
            rejected_fragmented += 1
            continue
        supported = sum(
            max(0, all_support_counts.get(voxel, 0) - int(voxel in aligned_whole))
            >= preset.min_other_support
            for voxel in dominant
        )
        supported_fraction = supported / len(dominant)
        if supported_fraction < preset.min_supported_fraction:
            rejected_support += 1
            continue
        candidates.append(
            (supported_fraction, retained, dominant, original_component)
        )

    candidates.sort(key=lambda item: (-len(item[2]), -item[0], -item[1], min(item[2])))
    transferred: set[Voxel] = set()
    accepted_source: set[Voxel] = set()
    rejected_budget = 0
    kept = 0
    for _support, _retained, component, original_component in candidates:
        addition = component - transferred
        if len(transferred) + len(addition) > voxel_budget:
            rejected_budget += 1
            continue
        transferred.update(component)
        accepted_source.update(original_component)
        kept += 1

    return transferred, accepted_source, {
        "donor_internal_voxels": len(donor_internal),
        "donor_components": len(donor_components),
        "eligible_components": len(candidates),
        "kept_components": kept,
        "transferred_voxels": len(transferred),
        "voxel_budget": voxel_budget,
        "budget_usage": len(transferred) / voxel_budget if voxel_budget else 0.0,
        "rejected_small_components": rejected_small,
        "rejected_clipped_components": rejected_clipped,
        "rejected_fragmented_components": rejected_fragmented,
        "rejected_unsupported_components": rejected_support,
        "rejected_budget_components": rejected_budget,
    }


def unmatched_voxels(
    source: set[Voxel], reference: set[Voxel], tolerance: float = 1.0
) -> set[Voxel]:
    if not source or not reference:
        return set(source)
    ordered, matched = _tolerant_membership(source, reference, tolerance)
    return {voxel for voxel, is_matched in zip(ordered, matched) if not is_matched}


def hybrid_fusion(
    objective1_internal: set[Voxel],
    donor_internal: set[Voxel],
    safe_volume: set[Voxel],
    base_min_component_voxels: int,
    base_max_core_fraction: float,
    total_budget: int,
) -> tuple[set[Voxel], set[Voxel], dict[str, float | int]]:
    """Replace donor-matched regions but preserve credible unmatched base surfaces."""
    donor = donor_internal & safe_volume
    # The safety mask governs donor insertion, not geometry that Objective 1
    # already produced. Existing base voxels outside the mask are protected.
    protected = objective1_internal - safe_volume
    preserved_candidates = []
    unmatched_count = 0
    for component in connected_components(objective1_internal):
        unmatched = unmatched_voxels(component & safe_volume, donor, 1.0)
        unmatched_count += len(unmatched)
        # Judge credibility before donor subtraction. Otherwise a valid large
        # surface can split into small fragments and be incorrectly discarded.
        if (
            unmatched
            and len(component) >= base_min_component_voxels
            and component_core_fraction(component) <= base_max_core_fraction
        ):
            preserved_candidates.append(unmatched)
    preserved_candidates.sort(key=lambda component: (-len(component), min(component)))
    fused = set(donor) | protected
    preserved: set[Voxel] = set(protected)
    effective_budget = max(total_budget, len(fused))
    rejected_budget = 0
    for component in preserved_candidates:
        addition = component - fused
        if len(fused) + len(addition) > effective_budget:
            rejected_budget += 1
            continue
        fused.update(component)
        preserved.update(component)
    return fused, preserved, {
        "protected_objective1_voxels": len(protected),
        "unmatched_objective1_voxels": unmatched_count,
        "eligible_objective1_components": len(preserved_candidates),
        "preserved_objective1_voxels": len(preserved),
        "preserved_objective1_components": sum(
            component <= preserved for component in preserved_candidates
        ),
        "rejected_objective1_budget_components": rejected_budget,
        "fused_internal_voxels": len(fused),
        "total_internal_budget": effective_budget,
    }


def component_preset_by_name(name: str) -> ComponentPreset:
    for preset in COMPONENT_PRESETS:
        if preset.name == name:
            return preset
    raise ValueError(f"unknown component preset: {name}")


def component_preset_payload(preset: ComponentPreset) -> dict:
    return asdict(preset)
