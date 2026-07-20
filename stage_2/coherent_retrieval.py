#!/usr/bin/env python3
"""Small, deterministic building blocks for coherent interior retrieval.

The important rule in this module is that retrieval votes may accept or reject
an entire donor component, but they never create geometry.  Every transferred
voxel therefore comes from one real training shape instead of a voxelwise union
of several incompatible shapes.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass


Voxel = tuple[int, int, int]


NEIGHBORS_26 = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
)


@dataclass(frozen=True)
class TransferPreset:
    """Conservative component-level transfer settings."""

    name: str
    min_component_voxels: int
    min_retained_fraction: float
    min_other_support: int
    min_supported_fraction: float
    budget_quantile: float


# These are deliberately few and interpretable.  Calibration chooses among
# them on training shapes only; it does not tune against the held-out test set.
TRANSFER_PRESETS = (
    TransferPreset("donor", 24, 0.75, 0, 0.00, 0.75),
    TransferPreset("detail", 12, 0.65, 1, 0.10, 0.90),
    TransferPreset("balanced", 16, 0.70, 1, 0.20, 0.75),
    TransferPreset("precise", 24, 0.75, 2, 0.25, 0.75),
    TransferPreset("strict", 32, 0.80, 3, 0.35, 0.50),
)


def preset_by_name(name: str) -> TransferPreset:
    for preset in TRANSFER_PRESETS:
        if preset.name == name:
            return preset
    raise ValueError(
        f"Unknown transfer preset {name!r}; expected one of "
        f"{[preset.name for preset in TRANSFER_PRESETS]}"
    )


def connected_components(voxels: set[Voxel]) -> list[set[Voxel]]:
    """Return deterministic 26-connected components, largest first.

    Voxelized diagonal triangle surfaces are disconnected under 6-connectivity,
    so 26-connectivity is the appropriate definition for mesh-derived voxels.
    """
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


def intersection_over_union(left: set[Voxel], right: set[Voxel]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def choose_donor(
    ranking: list[dict],
    target_exterior: set[Voxel],
    candidate_exteriors: dict[str, set[Voxel]],
    selection_k: int,
    exterior_weight: float,
) -> tuple[dict, float]:
    """Choose one donor with image and exterior evidence only."""
    if selection_k < 1 or selection_k > len(ranking):
        raise ValueError(
            f"selection_k={selection_k} is invalid for ranking of length {len(ranking)}"
        )
    scored = []
    for row in ranking[:selection_k]:
        candidate_id = str(row["retrieved_id"])
        exterior_iou = intersection_over_union(
            target_exterior, candidate_exteriors[candidate_id]
        )
        score = float(row["image_similarity"]) + exterior_weight * exterior_iou
        scored.append(
            (
                score,
                float(row["image_similarity"]),
                -int(row["rank"]),
                row,
                exterior_iou,
            )
        )
    _score, _similarity, _negative_rank, selected, exterior_iou = max(scored)
    return selected, exterior_iou


def build_support_counts(
    ranking: list[dict],
    candidate_internals: dict[str, set[Voxel]],
    support_k: int,
) -> Counter[Voxel]:
    if support_k < 1 or support_k > len(ranking):
        raise ValueError(
            f"support_k={support_k} is invalid for ranking of length {len(ranking)}"
        )
    counts: Counter[Voxel] = Counter()
    for row in ranking[:support_k]:
        counts.update(candidate_internals[str(row["retrieved_id"])])
    return counts


def prepare_clipped_components(
    donor_components: list[set[Voxel]], safe_volume: set[Voxel]
) -> tuple[list[tuple[set[Voxel], float]], int]:
    """Clip and connectivity-check donor components once for a target shape."""
    prepared = []
    rejected_fragmented = 0
    for component in donor_components:
        clipped = component & safe_volume
        if not clipped:
            prepared.append((set(), 0.0))
            continue
        clipped_parts = connected_components(clipped)
        dominant = clipped_parts[0]
        if len(dominant) / len(clipped) < 0.90:
            rejected_fragmented += 1
            continue
        prepared.append((dominant, len(dominant) / len(component)))
    return prepared, rejected_fragmented


def transfer_components(
    donor_internal: set[Voxel],
    donor_components: list[set[Voxel]],
    safe_volume: set[Voxel],
    support_counts: Counter[Voxel],
    preset: TransferPreset,
    voxel_budget: int,
    prepared_components: tuple[list[tuple[set[Voxel], float]], int] | None = None,
) -> tuple[set[Voxel], dict[str, float | int]]:
    """Transfer whole, supported donor components within a conservative budget.

    Clipping is permitted only when most of the original component survives.
    Components are accepted atomically after clipping; no vote-generated or
    morphologically generated voxel is ever inserted.
    """
    if voxel_budget < 0:
        raise ValueError("voxel_budget cannot be negative")

    if prepared_components is None:
        prepared_components = prepare_clipped_components(
            donor_components, safe_volume
        )
    clipped_components, rejected_fragmented = prepared_components

    accepted = []
    rejected_small = 0
    rejected_clipped = 0
    rejected_support = 0

    for clipped, retained_fraction in clipped_components:
        if len(clipped) < preset.min_component_voxels:
            rejected_small += 1
            continue
        if retained_fraction < preset.min_retained_fraction:
            rejected_clipped += 1
            continue

        if preset.min_other_support:
            supported = sum(
                max(0, support_counts.get(voxel, 0) - int(voxel in donor_internal))
                >= preset.min_other_support
                for voxel in clipped
            )
            supported_fraction = supported / len(clipped)
        else:
            supported_fraction = 1.0
        if supported_fraction < preset.min_supported_fraction:
            rejected_support += 1
            continue
        accepted.append((supported_fraction, retained_fraction, clipped))

    # Prefer large structural surfaces when the learned budget is tight.  Vote
    # support is a gate above, not an excuse to fill the object with many small
    # high-confidence fragments.
    accepted.sort(
        key=lambda item: (-len(item[2]), -item[0], -item[1], min(item[2]))
    )
    transferred: set[Voxel] = set()
    kept_components = 0
    rejected_budget = 0
    for _supported_fraction, _retained_fraction, component in accepted:
        if len(transferred) + len(component) > voxel_budget:
            rejected_budget += 1
            continue
        transferred.update(component)
        kept_components += 1

    diagnostics: dict[str, float | int] = {
        "donor_internal_voxels": len(donor_internal),
        "donor_components": len(donor_components),
        "eligible_components": len(accepted),
        "kept_components": kept_components,
        "transferred_voxels": len(transferred),
        "voxel_budget": voxel_budget,
        "budget_usage": len(transferred) / voxel_budget if voxel_budget else 0.0,
        "rejected_small_components": rejected_small,
        "rejected_clipped_components": rejected_clipped,
        "rejected_fragmented_components": rejected_fragmented,
        "rejected_unsupported_components": rejected_support,
        "rejected_budget_components": rejected_budget,
    }
    return transferred, diagnostics


def internal_metrics(
    ground_truth: set[Voxel], prediction: set[Voxel]
) -> dict[str, float | int]:
    true_positive = len(ground_truth & prediction)
    precision = (
        true_positive / len(prediction)
        if prediction
        else (1.0 if not ground_truth else 0.0)
    )
    recall = (
        true_positive / len(ground_truth)
        if ground_truth
        else (1.0 if not prediction else 0.0)
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    beta_squared = 0.25
    f05 = (
        (1 + beta_squared) * precision * recall
        / (beta_squared * precision + recall)
        if beta_squared * precision + recall
        else 0.0
    )
    ratio = (
        len(prediction) / len(ground_truth)
        if ground_truth
        else (1.0 if not prediction else 0.0)
    )
    return {
        "internal_precision": precision,
        "internal_recall": recall,
        "internal_f1": f1,
        "internal_f05": f05,
        "gt_internal_voxels": len(ground_truth),
        "pred_internal_voxels": len(prediction),
        "pred_to_gt_internal_ratio": ratio,
    }


def component_metrics(
    voxels: set[Voxel], small_threshold: int = 24
) -> dict[str, float | int]:
    components = connected_components(voxels)
    small = [component for component in components if len(component) < small_threshold]
    small_voxels = sum(map(len, small))
    axis_neighbors = (
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    )
    volumetric_core_voxels = sum(
        all(
            (voxel[0] + dx, voxel[1] + dy, voxel[2] + dz) in voxels
            for dx, dy, dz in axis_neighbors
        )
        for voxel in voxels
    )
    return {
        "internal_components_26": len(components),
        "small_internal_components_26": len(small),
        "small_internal_component_voxels": small_voxels,
        "small_internal_component_fraction": (
            small_voxels / len(voxels) if voxels else 0.0
        ),
        "volumetric_core_voxels": volumetric_core_voxels,
        "volumetric_core_fraction": (
            volumetric_core_voxels / len(voxels) if voxels else 0.0
        ),
    }


def preset_payload(preset: TransferPreset) -> dict:
    return asdict(preset)
