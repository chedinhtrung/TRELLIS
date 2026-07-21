#!/usr/bin/env python3
"""Apply a frozen retrieval-v2 or category-aware v2.1 policy."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from coherent_retrieval import component_metrics
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
from retrieval_v2 import (
    Alignment,
    align_voxels,
    alignment_payload,
    component_preset_by_name,
    hybrid_fusion,
    predict_quality,
    prepare_candidate_records,
    safe_volume_for_category,
    structural_preset_by_name,
    support_counts,
    transfer_supported_components,
    transfer_structural_fragments,
    transform_points,
)


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
    path: Path, view_index: int, resolution: int, margin: int
) -> tuple[str, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing retrieval-v2 policy: {path}")
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    method = payload.get("method")
    if method not in {
        "retrieval_v2_confidence_gated_hybrid",
        "retrieval_v21_category_structural",
    }:
        raise ValueError(f"Unsupported policy method: {method!r}")
    for name, expected in (
        ("view_index", view_index),
        ("resolution", resolution),
        ("margin", margin),
    ):
        if int(payload.get(name, -1)) != expected:
            raise ValueError(f"Policy {name}={payload.get(name)!r}, expected {expected}")
    if not payload.get("categories"):
        raise ValueError("Policy contains no category selections")
    return str(method), payload["categories"]


def add_derived_metrics(metrics: dict, predicted_internal: set) -> dict:
    precision = float(metrics["internal_precision"])
    recall = float(metrics["internal_recall"])
    metrics["internal_f05"] = (
        1.25 * precision * recall / (0.25 * precision + recall)
        if 0.25 * precision + recall
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
    rows: list[dict],
    categories: list[str],
    count_per_category: int,
    margin: int,
    final_method: str,
) -> list[dict]:
    primary = defaultdict(dict)
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
                "internal_f1_delta": (
                    float(final["internal_f1"]) - float(baseline["internal_f1"])
                ),
                "objective1_internal_f1": float(baseline["internal_f1"]),
                "retrieval_internal_f1": float(final["internal_f1"]),
                "retrieval_internal_precision": float(final["internal_precision"]),
                "retrieval_internal_recall": float(final["internal_recall"]),
                "retrieval_internal_ratio": float(
                    final["pred_to_gt_internal_ratio"]
                ),
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
    try:
        import trimesh
    except ImportError as error:  # pragma: no cover - TRELLIS installs trimesh.
        raise ImportError("Smooth retrieval meshes require trimesh") from error
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError(f"Expected a non-empty triangle mesh: {path}")
    return loaded


def _allowed_mask(voxels: set, resolution: int) -> np.ndarray:
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


def _face_mask(mesh, allowed: set, resolution: int) -> np.ndarray:
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
            vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=np.int64), process=False
        )
    used, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    vertices = np.asarray(mesh.vertices)[used]
    return trimesh.Trimesh(
        vertices=vertices,
        faces=inverse.reshape(-1, 3),
        process=False,
    )


def export_smooth_hybrid_mesh(
    objective1_mesh_path: Path,
    donor_mesh_path: Path,
    output_path: Path,
    objective1_kept_voxels: set,
    accepted_donor_source: set,
    transferred_donor_voxels: set,
    alignment: Alignment,
    resolution: int,
) -> None:
    """Append accepted high-resolution donor triangles to the decoded base mesh."""
    import trimesh

    base = _load_mesh(objective1_mesh_path)
    base_part = _submesh(base, _face_mask(base, objective1_kept_voxels, resolution))

    donor = _load_mesh(donor_mesh_path)
    donor_source_part = _submesh(
        donor, _face_mask(donor, accepted_donor_source, resolution)
    )
    if len(donor_source_part.faces):
        continuous = (np.asarray(donor_source_part.vertices) + 0.5) * resolution - 0.5
        mapped = transform_points(continuous, alignment)
        donor_source_part.vertices = (mapped + 0.5) / resolution - 0.5
        donor_part = _submesh(
            donor_source_part,
            _face_mask(donor_source_part, transferred_donor_voxels, resolution),
        )
    else:
        donor_part = donor_source_part

    if len(donor_part.faces) == 0:
        raise ValueError(
            f"No donor triangles matched the accepted transfer for {output_path.stem}"
        )

    pieces = [mesh for mesh in (base_part, donor_part) if len(mesh.faces)]
    if not pieces:
        raise ValueError(f"Mesh transfer removed every face for {output_path.stem}")
    combined = trimesh.util.concatenate(pieces)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.export(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval-v2 on held-out view-18 shapes.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--objective1-voxels", type=Path, required=True)
    parser.add_argument("--objective1-meshes", type=Path)
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
    parser.add_argument("--save-smooth-meshes", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Allow an interrupted output directory to be recomputed and overwritten.",
    )
    args = parser.parse_args()

    if args.view_index != 18:
        raise ValueError("retrieval-v2 is intentionally evaluated with view 18")
    if args.transplant_margin not in args.margins:
        raise ValueError("--transplant-margin must be included in --margins")
    if (
        args.output_dir.exists()
        and any(args.output_dir.iterdir())
        and not args.resume
    ):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}. Use a new directory."
        )
    if args.save_smooth_meshes and args.objective1_meshes is None:
        raise ValueError("--save-smooth-meshes requires --objective1-meshes")

    policy_method, policy = read_policy(
        args.policy, args.view_index, args.resolution, args.transplant_margin
    )
    final_method = (
        "retrieval_v21"
        if policy_method == "retrieval_v21_category_structural"
        else "retrieval_v2"
    )
    maximum_k = max(int(selection["top_k"]) for selection in policy.values())
    selected_ids = read_selected_ids(args.ids_file)
    rankings = read_rankings(
        args.rankings, f"dino_view{args.view_index:03d}", selected_ids, maximum_k
    )
    train_dir = args.dataset_root / "train"
    test_dir = args.dataset_root / "test"
    train_rows = read_metadata(train_dir / "metadata.csv")
    test_rows = read_metadata(test_dir / "metadata.csv")
    if selected_ids is not None:
        test_rows = [row for row in test_rows if row["sha256"] in selected_ids]
    categories = {row["sha256"]: row["category"] for row in train_rows}
    if {row["category"] for row in test_rows} != set(policy):
        raise ValueError("Policy categories do not match held-out categories")
    if set(rankings) != {row["sha256"] for row in test_rows}:
        raise ValueError("Ranking IDs do not exactly match selected held-out IDs")

    candidate_internals = {}
    candidate_exteriors = {}

    def load_candidate(candidate_id: str) -> None:
        if candidate_id in candidate_internals:
            return
        path = train_dir / "voxels" / f"{candidate_id}.ply"
        if not path.is_file():
            raise FileNotFoundError(f"Missing training voxel PLY: {path}")
        voxels = read_voxels(path, args.resolution)
        candidate_internals[candidate_id] = interior(voxels, args.transplant_margin)
        candidate_exteriors[candidate_id] = voxels - candidate_internals[candidate_id]

    voxel_output = args.output_dir / "predictions" / final_method / "voxels"
    mesh_output = args.output_dir / "predictions" / final_method / "mesh"
    gt_voxel_reference = args.output_dir / "references" / "ground_truth_voxels"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gt_voxel_reference.mkdir(parents=True, exist_ok=True)
    per_sample_rows = []
    applied_rows = []

    for index, test_row in enumerate(test_rows, start=1):
        sample_id = test_row["sha256"]
        category = test_row["category"]
        selection = policy[category]
        ranking = rankings[sample_id][: int(selection["top_k"])]
        for row in ranking:
            candidate_id = str(row["retrieved_id"])
            if categories.get(candidate_id) != category:
                raise ValueError(f"Candidate category mismatch: {sample_id} -> {candidate_id}")
            load_candidate(candidate_id)

        gt_path = test_dir / "voxels" / f"{sample_id}.ply"
        objective1_path = args.objective1_voxels / f"{sample_id}.ply"
        for path in (gt_path, objective1_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing held-out voxel PLY: {path}")
        gt = read_voxels(gt_path, args.resolution)
        objective1 = read_voxels(objective1_path, args.resolution)
        objective1_internal = interior(objective1, args.transplant_margin)
        objective1_exterior = objective1 - objective1_internal
        mode = str(selection.get("mode", "v2_hybrid"))
        feature_safe = enclosed_volume(objective1_exterior, args.transplant_margin)
        safe = (
            safe_volume_for_category(
                objective1_exterior, args.transplant_margin, category
            )
            if mode == "structural_fragments"
            else feature_safe
        )

        records = prepare_candidate_records(
            ranking,
            candidate_internals,
            candidate_exteriors,
            objective1_internal,
            objective1_exterior,
            feature_safe,
            args.resolution,
        )
        for record in records:
            record["predicted_quality"] = predict_quality(
                record["features"], selection["reranker"]
            )
        selected = max(
            records,
            key=lambda record: (
                record["predicted_quality"],
                record["image_similarity"],
                -record["rank"],
            ),
        )
        ordered_records = sorted(
            records,
            key=lambda record: (
                -record["predicted_quality"],
                -record["image_similarity"],
                record["rank"],
            ),
        )
        score_margin = (
            float(ordered_records[0]["predicted_quality"])
            - float(ordered_records[1]["predicted_quality"])
        )
        donor_id = str(selected["retrieved_id"])
        support = support_counts([record["aligned_internal"] for record in records])
        category_budget = int(round(
            float(selection["max_internal_to_exterior_ratio"])
            * len(objective1_exterior)
        ))
        base_budget = (
            int(round(float(selection["max_expansion"]) * len(objective1_internal)))
            if objective1_internal
            else category_budget
        )
        donor_budget = min(category_budget, base_budget)
        if mode == "v2_hybrid":
            preset = component_preset_by_name(selection["component_preset"]["name"])
            transferred, accepted_source, transfer_diag = transfer_supported_components(
                candidate_internals[donor_id],
                selected["alignment"],
                safe,
                support,
                preset,
                donor_budget,
                args.resolution,
            )
            coverage_denominator = len(objective1_internal)
        elif mode == "structural_fragments":
            preset = structural_preset_by_name(
                selection["structural_preset"]["name"]
            )
            if selection.get("transfer_mode", "fragments") == "fragments":
                transferred, accepted_source, transfer_diag = transfer_structural_fragments(
                    candidate_internals[donor_id],
                    selected["alignment"],
                    safe,
                    support,
                    preset,
                    donor_budget,
                    args.resolution,
                )
            elif selection["transfer_mode"] == "full":
                usable_full = selected["aligned_internal"] & safe
                transferred = usable_full if len(usable_full) <= donor_budget else set()
                accepted_source = (
                    set(candidate_internals[donor_id]) if transferred else set()
                )
                transfer_diag = {
                    "donor_internal_voxels": len(candidate_internals[donor_id]),
                    "donor_components": 1,
                    "eligible_components": int(bool(transferred)),
                    "kept_components": int(bool(transferred)),
                    "transferred_voxels": len(transferred),
                    "usable_donor_voxels": len(usable_full),
                    "voxel_budget": donor_budget,
                    "budget_usage": (
                        len(transferred) / donor_budget if donor_budget else 0.0
                    ),
                    "rejected_small_components": 0,
                    "rejected_clipped_components": 0,
                    "rejected_fragmented_components": 0,
                    "rejected_dense_components": 0,
                    "rejected_unsupported_components": 0,
                    "rejected_budget_components": int(
                        bool(usable_full) and not transferred
                    ),
                }
            else:
                raise ValueError(
                    f"Unsupported structural transfer mode: "
                    f"{selection.get('transfer_mode')!r}"
                )
            coverage_denominator = min(
                len(selected["aligned_internal"] & safe), donor_budget
            )
        else:
            raise ValueError(f"Unsupported category policy mode: {mode}")
        transfer_diag.setdefault(
            "usable_donor_voxels", len(selected["aligned_internal"] & safe)
        )
        transfer_diag.setdefault("rejected_dense_components", 0)
        coverage = (
            len(transferred) / coverage_denominator
            if coverage_denominator
            else float(bool(transferred))
        )
        gate_reasons = []
        if not transferred:
            gate_reasons.append("empty_transfer")
        if coverage < float(selection["minimum_transfer_coverage"]):
            gate_reasons.append("low_coverage")
        if mode == "v2_hybrid" and float(selected["predicted_quality"]) < float(
            selection["minimum_predicted_quality"]
        ):
            gate_reasons.append("low_predicted_quality")
        if mode == "structural_fragments" and score_margin < float(
            selection["minimum_score_margin"]
        ):
            gate_reasons.append("low_score_margin")
        use_fallback = bool(gate_reasons)

        if use_fallback:
            fused_internal = objective1_internal
            preserved = objective1_internal
            fusion_diag = {
                "protected_objective1_voxels": len(objective1_internal - safe),
                "unmatched_objective1_voxels": len(objective1_internal),
                "eligible_objective1_components": 0,
                "preserved_objective1_voxels": len(objective1_internal),
                "preserved_objective1_components": 0,
                "rejected_objective1_budget_components": 0,
                "fused_internal_voxels": len(objective1_internal),
                "total_internal_budget": category_budget,
            }
            prediction = objective1
            accepted_source = set()
            transferred_for_mesh = set()
        else:
            if mode == "v2_hybrid" or selection.get("fusion_mode") == "hybrid":
                base_minimum = int(selection.get("base_min_component_voxels", 64))
                base_core = float(selection.get("base_max_core_fraction", 0.25))
                fused_internal, preserved, fusion_diag = hybrid_fusion(
                    objective1_internal,
                    transferred,
                    safe,
                    base_minimum,
                    base_core,
                    category_budget,
                )
            elif selection.get("fusion_mode") == "replace":
                preserved = objective1_internal - safe
                fused_internal = transferred | preserved
                fusion_diag = {
                    "protected_objective1_voxels": len(preserved),
                    "unmatched_objective1_voxels": 0,
                    "eligible_objective1_components": 0,
                    "preserved_objective1_voxels": len(preserved),
                    "preserved_objective1_components": 0,
                    "rejected_objective1_budget_components": 0,
                    "fused_internal_voxels": len(fused_internal),
                    "total_internal_budget": category_budget,
                }
            else:
                raise ValueError(
                    f"Unsupported structural fusion mode: {selection.get('fusion_mode')!r}"
                )
            prediction = objective1_exterior | fused_internal
            transferred_for_mesh = transferred

        if not objective1_exterior <= prediction:
            raise AssertionError(f"Objective-1 shell was not preserved: {sample_id}")
        if transferred_for_mesh - safe:
            raise AssertionError(f"Transferred donor escaped the safe volume: {sample_id}")

        alignment = alignment_payload(selected["alignment"])
        common = {
            "sample_id": sample_id,
            "category": category,
            "selected_id": donor_id,
            "selected_rank": int(selected["rank"]),
            "image_similarity": float(selected["image_similarity"]),
            "predicted_donor_quality": float(selected["predicted_quality"]),
            "selected_exterior_tolerant_f1": float(selected["alignment"].exterior_tolerant_f1),
            "alignment_strength": float(selected["alignment"].strength),
            "alignment_scale_x": alignment["scale"][0],
            "alignment_scale_y": alignment["scale"][1],
            "alignment_scale_z": alignment["scale"][2],
            "policy_mode": mode,
            "selection_score_margin": score_margin,
            "safe_volume_voxels": len(safe),
            "transfer_coverage": coverage,
            "transfer_coverage_denominator": coverage_denominator,
            "objective1_internal_voxels": len(objective1_internal),
            "objective1_exterior_voxels": len(objective1_exterior),
            "removed_objective1_internal_voxels": len(objective1_internal - prediction),
            "added_vs_objective1_voxels": len(prediction - objective1),
            "used_objective1_fallback": int(use_fallback),
            "gate_reason": "+".join(gate_reasons) if gate_reasons else "accepted",
            **transfer_diag,
            **fusion_diag,
        }
        for method, variant in (("objective1", objective1), (final_method, prediction)):
            for margin in sorted(args.margins):
                predicted_internal = interior(variant, margin)
                metrics = add_derived_metrics(
                    score_voxels(gt, variant, margin), predicted_internal
                )
                metrics["objective1_shell_preservation"] = (
                    len(objective1_exterior & variant) / len(objective1_exterior)
                    if objective1_exterior
                    else 1.0
                )
                per_sample_rows.append({
                    "method": method,
                    "margin": margin,
                    **common,
                    **metrics,
                })

        write_voxels(voxel_output / f"{sample_id}.ply", prediction, args.resolution)
        shutil.copyfile(gt_path, gt_voxel_reference / f"{sample_id}.ply")
        if args.save_smooth_meshes:
            objective1_mesh = args.objective1_meshes / f"{sample_id}.ply"
            if not objective1_mesh.is_file():
                raise FileNotFoundError(f"Missing Objective-1 mesh: {objective1_mesh}")
            if use_fallback:
                mesh_output.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(objective1_mesh, mesh_output / f"{sample_id}.ply")
            else:
                donor_mesh = train_dir / "renders" / donor_id / "mesh.ply"
                if not donor_mesh.is_file():
                    raise FileNotFoundError(f"Missing normalized donor mesh: {donor_mesh}")
                export_smooth_hybrid_mesh(
                    objective1_mesh,
                    donor_mesh,
                    mesh_output / f"{sample_id}.ply",
                    objective1_exterior | preserved,
                    accepted_source,
                    transferred_for_mesh,
                    selected["alignment"],
                    args.resolution,
                )

        applied_rows.append({key: value for key, value in common.items() if not isinstance(value, (set, dict, list))})
        if index % 25 == 0 or index == len(test_rows):
            print(f"Processed {index}/{len(test_rows)}")

    summary_rows = aggregate(per_sample_rows, ("method", "margin"))
    category_rows = aggregate(per_sample_rows, ("method", "category", "margin"))
    paired_rows = paired_summary(per_sample_rows, "objective1", args.transplant_margin)
    write_csv(args.output_dir / "per_sample.csv", per_sample_rows)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_csv(args.output_dir / "category_summary.csv", category_rows)
    write_csv(args.output_dir / "paired_summary.csv", paired_rows)
    write_csv(args.output_dir / "applied_policy.csv", applied_rows)
    gallery = choose_gallery(
        per_sample_rows,
        args.gallery_categories,
        args.gallery_per_category,
        args.transplant_margin,
        final_method,
    )
    if gallery:
        write_csv(args.output_dir / "visualization_manifest.csv", gallery)
        gt_mesh_reference = args.output_dir / "references" / "ground_truth_mesh"
        gt_mesh_reference.mkdir(parents=True, exist_ok=True)
        for row in gallery:
            sample_id = row["sample_id"]
            source = test_dir / "renders" / sample_id / "mesh.ply"
            if not source.is_file():
                raise FileNotFoundError(f"Missing ground-truth display mesh: {source}")
            shutil.copyfile(source, gt_mesh_reference / f"{sample_id}.ply")
    shutil.copyfile(args.policy, args.output_dir / "policy.json")
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump({
            "dataset_root": str(args.dataset_root),
            "objective1_voxels": str(args.objective1_voxels),
            "objective1_meshes": str(args.objective1_meshes) if args.objective1_meshes else None,
            "rankings": str(args.rankings),
            "policy": str(args.policy),
            "policy_method": policy_method,
            "view_index": args.view_index,
            "resolution": args.resolution,
            "transplant_margin": args.transplant_margin,
            "margins": sorted(args.margins),
            "save_smooth_meshes": args.save_smooth_meshes,
            "resume": args.resume,
        }, file, indent=2)
        file.write("\n")

    print(f"\nHeld-out {final_method} results")
    paired = {row["method"]: row for row in paired_rows}
    for row in summary_rows:
        if int(row["margin"]) != args.transplant_margin:
            continue
        comparison = paired[row["method"]]
        print(
            f"  {row['method']}: P/R/F1/F0.5="
            f"{row['internal_precision']:.4f}/{row['internal_recall']:.4f}/"
            f"{row['internal_f1']:.4f}/{row['internal_f05']:.4f}, "
            f"ratio={row['pred_to_gt_internal_ratio']:.3f}, "
            f"components={row['internal_components_26']:.1f}, "
            f"F1 delta={comparison['mean_internal_f1_delta']:+.4f}, "
            f"W/L/T={comparison['wins']}/{comparison['losses']}/{comparison['ties']}"
        )
    print(f"Wrote {final_method} results to {args.output_dir}")


if __name__ == "__main__":
    main()
