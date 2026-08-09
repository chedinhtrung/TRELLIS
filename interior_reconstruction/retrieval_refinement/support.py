"""Final retrieval I/O, metrics, DINO ranking, and smooth-mesh utilities."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from .geometry import Alignment, connected_components, transform_points


Voxel = tuple[int, int, int]

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


def read_voxels(path: Path, resolution: int) -> set[Voxel]:
    import utils3d

    points = np.asarray(utils3d.io.read_ply(str(path))[0], dtype=np.float32)
    if points.size == 0:
        return set()
    points = points.reshape(-1, 3)
    voxels = np.floor((points + 0.5) * resolution).astype(np.int32)
    voxels = np.clip(voxels, 0, resolution - 1)
    return {tuple(voxel) for voxel in voxels.tolist()}


def interior(voxels: set[Voxel], margin: int = 1) -> set[Voxel]:
    if margin < 1:
        raise ValueError("margin must be at least 1")
    if not voxels:
        return set()
    internal = set(voxels)
    for axis in range(3):
        groups: dict[tuple[int, int], list[Voxel]] = {}
        other_axes = [index for index in range(3) if index != axis]
        for voxel in voxels:
            key = (voxel[other_axes[0]], voxel[other_axes[1]])
            groups.setdefault(key, []).append(voxel)
        for group in groups.values():
            minimum = min(voxel[axis] for voxel in group)
            maximum = max(voxel[axis] for voxel in group)
            for voxel in group:
                if voxel[axis] - minimum < margin or maximum - voxel[axis] < margin:
                    internal.discard(voxel)
    return internal


def internal_metrics(ground_truth: set[Voxel], prediction: set[Voxel]) -> dict:
    true_positive = len(ground_truth & prediction)
    precision = (
        true_positive / len(prediction)
        if prediction else (1.0 if not ground_truth else 0.0)
    )
    recall = (
        true_positive / len(ground_truth)
        if ground_truth else (1.0 if not prediction else 0.0)
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f05 = (
        1.25 * precision * recall / (0.25 * precision + recall)
        if 0.25 * precision + recall else 0.0
    )
    return {
        "internal_precision": precision,
        "internal_recall": recall,
        "internal_f1": f1,
        "internal_f05": f05,
        "gt_internal_voxels": len(ground_truth),
        "pred_internal_voxels": len(prediction),
        "pred_to_gt_internal_ratio": (
            len(prediction) / len(ground_truth)
            if ground_truth else (1.0 if not prediction else 0.0)
        ),
    }


def component_metrics(voxels: set[Voxel], small_threshold: int = 24) -> dict:
    components = connected_components(voxels)
    small = [component for component in components if len(component) < small_threshold]
    small_voxels = sum(map(len, small))
    axis_neighbors = (
        (1, 0, 0), (-1, 0, 0), (0, 1, 0),
        (0, -1, 0), (0, 0, 1), (0, 0, -1),
    )
    core_voxels = sum(
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
        "volumetric_core_voxels": core_voxels,
        "volumetric_core_fraction": core_voxels / len(voxels) if voxels else 0.0,
    }


def _safe_ratio(numerator: int, denominator: int, both_empty: bool) -> float:
    if denominator:
        return numerator / denominator
    return 1.0 if both_empty else 0.0


def score_voxels(gt: set[Voxel], pred: set[Voxel], margin: int) -> dict:
    gt_internal = interior(gt, margin)
    pred_internal = interior(pred, margin)
    true_positive = len(gt_internal & pred_internal)
    precision = _safe_ratio(true_positive, len(pred_internal), not gt_internal)
    recall = _safe_ratio(true_positive, len(gt_internal), not pred_internal)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = gt | pred
    gt_exterior = gt - gt_internal
    pred_exterior = pred - pred_internal
    exterior_union = gt_exterior | pred_exterior
    metrics = {
        "voxel_iou": len(gt & pred) / len(union) if union else 1.0,
        "exterior_iou": (
            len(gt_exterior & pred_exterior) / len(exterior_union)
            if exterior_union else 1.0
        ),
        "internal_precision": precision,
        "internal_recall": recall,
        "internal_f1": f1,
        "gt_voxels": len(gt),
        "pred_voxels": len(pred),
        "gt_internal_voxels": len(gt_internal),
        "pred_internal_voxels": len(pred_internal),
    }
    metrics["pred_to_gt_voxel_ratio"] = _safe_ratio(
        len(pred), len(gt), not pred
    )
    metrics["pred_to_gt_internal_ratio"] = _safe_ratio(
        len(pred_internal), len(gt_internal), not pred_internal
    )
    return metrics


def add_derived_metrics(metrics: dict, predicted_internal: set[Voxel]) -> dict:
    precision = float(metrics["internal_precision"])
    recall = float(metrics["internal_recall"])
    metrics["internal_f05"] = (
        1.25 * precision * recall / (0.25 * precision + recall)
        if 0.25 * precision + recall else 0.0
    )
    metrics.update(component_metrics(predicted_internal))
    return metrics


def read_metadata(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing metadata: {path}")
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    missing = {"sha256", "category"} - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    clean_rows = []
    seen = set()
    for row in rows:
        sample_id = row["sha256"].strip()
        category = row["category"].strip()
        if not sample_id or not category:
            raise ValueError(f"Empty sha256 or category in {path}")
        if sample_id in seen:
            raise ValueError(f"Duplicate sha256 in {path}: {sample_id}")
        seen.add(sample_id)
        clean_rows.append({"sha256": sample_id, "category": category})
    return clean_rows


def read_selected_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Expected non-empty unique IDs in {path}")
    return set(ids)


def enclosed_volume(surface: set[Voxel], margin: int) -> set[Voxel]:
    if not surface:
        return set()
    axis_volumes = []
    for axis in range(3):
        other_axes = [index for index in range(3) if index != axis]
        groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for voxel in surface:
            groups[(voxel[other_axes[0]], voxel[other_axes[1]])].append(voxel[axis])
        axis_volume: set[Voxel] = set()
        for key, positions in groups.items():
            start = min(positions) + margin
            stop = max(positions) - margin
            for position in range(start, stop + 1):
                voxel = [0, 0, 0]
                voxel[axis] = position
                voxel[other_axes[0]] = key[0]
                voxel[other_axes[1]] = key[1]
                axis_volume.add(tuple(voxel))
        axis_volumes.append(axis_volume)
    return axis_volumes[0] & axis_volumes[1] & axis_volumes[2]


def write_voxels(path: Path, voxels: set[Voxel], resolution: int) -> None:
    import utils3d

    path.parent.mkdir(parents=True, exist_ok=True)
    if voxels:
        coordinates = np.asarray(sorted(voxels), dtype=np.float32)
        points = ((coordinates + 0.5) / resolution - 0.5).astype(np.float32)
    else:
        points = np.zeros((0, 3), dtype=np.float32)
    utils3d.io.write_ply(str(path), points)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0])
        seen = set(fieldnames)
        for row in rows[1:]:
            for name in row:
                if name not in seen:
                    fieldnames.append(name)
                    seen.add(name)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_train_embeddings(
    path: Path, rows: list[dict[str, str]], view_index: int, model_name: str
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing DINO embedding cache: {path}")
    with np.load(path, allow_pickle=False) as data:
        sample_ids = data["sample_ids"].astype(str).tolist()
        categories = data["categories"].astype(str).tolist()
        cached_view = int(data["view_index"].item())
        cached_model = str(data["model_name"].item())
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    if sample_ids != [row["sha256"] for row in rows]:
        raise ValueError(f"Embedding order does not match metadata: {path}")
    if categories != [row["category"] for row in rows]:
        raise ValueError(f"Embedding categories do not match metadata: {path}")
    if cached_view != view_index or cached_model != model_name:
        raise ValueError("DINO embedding configuration mismatch")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(rows):
        raise ValueError(f"Invalid embedding shape: {embeddings.shape}")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-3):
        raise ValueError("Expected L2-normalized DINO embeddings")
    return embeddings


def make_train_rankings(
    metadata: list[dict[str, str]], embeddings: np.ndarray, top_k: int
) -> dict[str, list[dict]]:
    category_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        category_indices[row["category"]].append(index)
    rankings = {}
    for category, indices in sorted(category_indices.items()):
        if len(indices) <= top_k:
            raise ValueError(f"Category {category} has too few shapes for top-{top_k}")
        similarities = embeddings[indices] @ embeddings[indices].T
        np.fill_diagonal(similarities, -np.inf)
        orders = np.argsort(-similarities, axis=1, kind="stable")[:, :top_k]
        for local_query, global_query in enumerate(indices):
            query_id = metadata[global_query]["sha256"]
            rankings[query_id] = [
                {
                    "rank": rank,
                    "category": category,
                    "retrieved_id": metadata[indices[int(local_candidate)]]["sha256"],
                    "image_similarity": float(
                        similarities[local_query, int(local_candidate)]
                    ),
                }
                for rank, local_candidate in enumerate(orders[local_query], start=1)
            ]
    return rankings


def split_train_queries(metadata: list[dict[str, str]]) -> tuple[set[str], set[str]]:
    by_category: dict[str, list[str]] = defaultdict(list)
    for row in metadata:
        by_category[row["category"]].append(row["sha256"])
    donor_fit, remaining = set(), set()
    for sample_ids in by_category.values():
        for index, sample_id in enumerate(sorted(sample_ids)):
            (remaining if index % 4 == 0 else donor_fit).add(sample_id)
    return donor_fit, remaining


def balanced_cap(
    sample_ids: set[str], categories: dict[str, str], per_category: int
) -> set[str]:
    if per_category == 0:
        return set(sample_ids)
    by_category: dict[str, list[str]] = defaultdict(list)
    for sample_id in sample_ids:
        by_category[categories[sample_id]].append(sample_id)
    return {
        sample_id
        for ids in by_category.values()
        for sample_id in sorted(ids)[:per_category]
    }


def read_rankings(
    path: Path, query_mode: str, selected_ids: set[str] | None, maximum_k: int
) -> dict[str, list[dict]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing DINO rankings: {path}")
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    required = {
        "query_mode", "sample_id", "category", "rank",
        "retrieved_id", "image_cosine_similarity",
    }
    if not rows or required - set(rows[0]):
        raise ValueError(f"Invalid DINO ranking table: {path}")
    rankings: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        sample_id = row["sample_id"].strip()
        rank = int(row["rank"])
        if (
            row["query_mode"] != query_mode
            or rank > maximum_k
            or (selected_ids is not None and sample_id not in selected_ids)
        ):
            continue
        rankings[sample_id].append({
            "rank": rank,
            "category": row["category"].strip(),
            "retrieved_id": row["retrieved_id"].strip(),
            "image_similarity": float(row["image_cosine_similarity"]),
        })
    for sample_id, ranking in rankings.items():
        ranking.sort(key=lambda row: int(row["rank"]))
        if [row["rank"] for row in ranking] != list(range(1, maximum_k + 1)):
            raise ValueError(f"Incomplete top-{maximum_k} ranking for {sample_id}")
        donor_ids = [row["retrieved_id"] for row in ranking]
        if len(donor_ids) != len(set(donor_ids)):
            raise ValueError(f"Duplicate donors in ranking for {sample_id}")
    if not rankings:
        raise ValueError(f"No {query_mode} rankings in {path}")
    return dict(rankings)


def aggregate(rows: list[dict], group_names: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[name] for name in group_names)].append(row)
    return [
        {
            **dict(zip(group_names, key)),
            **{
                metric: float(np.mean([float(row[metric]) for row in group]))
                for metric in BASE_METRICS
            },
            "matched_samples": len(group),
        }
        for key, group in sorted(groups.items())
    ]


def paired_summary(rows: list[dict], baseline_method: str, margin: int) -> list[dict]:
    by_method: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if int(row["margin"]) == margin:
            by_method[row["method"]][row["sample_id"]] = float(row["internal_f1"])
    baseline = by_method[baseline_method]
    output = []
    rng = np.random.default_rng(12345)
    for method in sorted(by_method):
        sample_ids = sorted(baseline.keys() & by_method[method].keys())
        deltas = np.asarray(
            [by_method[method][sample_id] - baseline[sample_id] for sample_id in sample_ids]
        )
        if method == baseline_method:
            low = high = 0.0
        else:
            indices = rng.integers(0, len(deltas), size=(10000, len(deltas)))
            low, high = np.quantile(deltas[indices].mean(axis=1), [0.025, 0.975])
        output.append({
            "method": method,
            "margin": margin,
            "mean_internal_f1_delta": float(deltas.mean()),
            "median_internal_f1_delta": float(np.median(deltas)),
            "wins": int(np.count_nonzero(deltas > 1e-12)),
            "losses": int(np.count_nonzero(deltas < -1e-12)),
            "ties": int(np.count_nonzero(np.abs(deltas) <= 1e-12)),
            "bootstrap_95_low": float(low),
            "bootstrap_95_high": float(high),
            "matched_samples": len(sample_ids),
        })
    return output


def choose_gallery(
    rows: list[dict], categories: list[str], count_per_category: int,
    margin: int, final_method: str,
) -> list[dict]:
    primary: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        if int(row["margin"]) == margin:
            primary[row["method"]][row["sample_id"]] = row
    selected = []
    for category in categories:
        candidates = []
        for sample_id, final in primary[final_method].items():
            if final["category"] != category:
                continue
            baseline = primary["objective1"][sample_id]
            quality_delta = (
                0.70 * (float(final["internal_f05"]) - float(baseline["internal_f05"]))
                + 0.30 * (float(final["internal_f1"]) - float(baseline["internal_f1"]))
                - 0.25 * max(0.0, float(final["pred_to_gt_internal_ratio"]) - 1.0)
            )
            candidates.append({
                "sample_id": sample_id,
                "category": category,
                "semantic_quality_delta": quality_delta,
                "internal_f1_delta": float(final["internal_f1"]) - float(baseline["internal_f1"]),
                "objective1_internal_f1": float(baseline["internal_f1"]),
                "retrieval_internal_f1": float(final["internal_f1"]),
                "retrieval_internal_precision": float(final["internal_precision"]),
                "retrieval_internal_recall": float(final["internal_recall"]),
                "retrieval_internal_ratio": float(final["pred_to_gt_internal_ratio"]),
                "used_objective1_fallback": int(final["used_objective1_fallback"]),
            })
        candidates.sort(
            key=lambda row: (
                -row["used_objective1_fallback"],
                row["semantic_quality_delta"],
                row["retrieval_internal_precision"],
            ),
            reverse=True,
        )
        selected.extend(candidates[:count_per_category])
    return selected


def _load_mesh(path: Path):
    import trimesh
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError(f"Expected non-empty triangle mesh: {path}")
    return loaded


def _allowed_mask(voxels: set[Voxel], resolution: int) -> np.ndarray:
    mask = np.zeros((resolution, resolution, resolution), dtype=bool)
    if voxels:
        coordinates = np.asarray(sorted(voxels), dtype=np.int32)
        mask[tuple(coordinates.T)] = True
    padded = np.pad(mask, 1)
    dilated = np.zeros_like(mask)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                dilated |= padded[
                    1 + dx : 1 + dx + resolution,
                    1 + dy : 1 + dy + resolution,
                    1 + dz : 1 + dz + resolution,
                ]
    return dilated


def _face_mask(mesh, allowed: set[Voxel], resolution: int) -> np.ndarray:
    centers = np.asarray(mesh.triangles_center, dtype=np.float64)[:, None, :]
    vertices = np.asarray(mesh.vertices, dtype=np.float64)[np.asarray(mesh.faces)]
    probes = np.concatenate((centers, vertices), axis=1)
    coordinates = np.floor((probes + 0.5) * resolution).astype(np.int32)
    valid = np.all((coordinates >= 0) & (coordinates < resolution), axis=2)
    keep = np.zeros(len(coordinates), dtype=bool)
    allowed_mask = _allowed_mask(allowed, resolution)
    for probe in range(coordinates.shape[1]):
        valid_probe = valid[:, probe]
        valid_coordinates = coordinates[valid_probe, probe]
        keep[valid_probe] |= allowed_mask[tuple(valid_coordinates.T)]
    return keep


def _submesh(mesh, face_mask: np.ndarray):
    import trimesh
    faces = np.asarray(mesh.faces)[face_mask]
    if len(faces) == 0:
        return trimesh.Trimesh(
            vertices=np.empty((0, 3)),
            faces=np.empty((0, 3), dtype=np.int64),
            process=False,
        )
    used, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices)[used],
        faces=inverse.reshape(-1, 3),
        process=False,
    )


def export_smooth_hybrid_mesh(
    objective1_mesh_path: Path, donor_mesh_path: Path, output_path: Path,
    objective1_kept_voxels: set[Voxel], accepted_donor_source: set[Voxel],
    transferred_donor_voxels: set[Voxel], alignment: Alignment, resolution: int,
) -> None:
    import trimesh
    base = _load_mesh(objective1_mesh_path)
    base_part = _submesh(base, _face_mask(base, objective1_kept_voxels, resolution))
    donor = _load_mesh(donor_mesh_path)
    donor_source = _submesh(donor, _face_mask(donor, accepted_donor_source, resolution))
    if len(donor_source.faces):
        continuous = (np.asarray(donor_source.vertices) + 0.5) * resolution - 0.5
        donor_source.vertices = (transform_points(continuous, alignment) + 0.5) / resolution - 0.5
        donor_part = _submesh(
            donor_source,
            _face_mask(donor_source, transferred_donor_voxels, resolution),
        )
    else:
        donor_part = donor_source
    if len(donor_part.faces) == 0:
        raise ValueError(f"No donor triangles matched transfer for {output_path.stem}")
    pieces = [mesh for mesh in (base_part, donor_part) if len(mesh.faces)]
    if not pieces:
        raise ValueError(f"Mesh transfer removed every face for {output_path.stem}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.util.concatenate(pieces).export(output_path)
