#!/usr/bin/env python3
"""Apply the frozen category-blind adaptive retrieval policy."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

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
from evaluate_retrieval_v2 import (
    add_derived_metrics,
    aggregate,
    choose_gallery,
    export_smooth_hybrid_mesh,
)
from retrieval_unified import (
    generate_unified_candidates,
    select_unified_candidate,
    shortlist_records,
)
from retrieval_v2 import alignment_payload, connected_components, prepare_candidate_records


FINAL_METHOD = "retrieval_unified"


def read_unified_policy(
    path: Path, view_index: int, resolution: int, margin: int
) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing unified retrieval policy: {path}")
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if payload.get("method") != "retrieval_unified_adaptive":
        raise ValueError(f"Unsupported unified policy method: {payload.get('method')!r}")
    for name, expected in (
        ("view_index", view_index),
        ("resolution", resolution),
        ("margin", margin),
    ):
        if int(payload.get(name, -1)) != expected:
            raise ValueError(f"Policy {name}={payload.get(name)!r}, expected {expected}")
    if payload.get("category_feature_used") is not False:
        raise ValueError("unified policy must explicitly disable category features")
    if (
        not payload.get("operators")
        or not payload.get("selector", {}).get("models")
        or not payload.get("donor_reranker")
        or len(payload.get("donor_experts", [])) != 4
        or int(payload.get("donor_shortlist", 0)) < 1
    ):
        raise ValueError(
            "unified policy is missing operators, donor ranker, or selector models"
        )
    return payload


def fallback_fusion_diagnostics(base: set, safe: set, budget: int) -> dict:
    return {
        "protected_objective1_voxels": len(base - safe),
        "unmatched_objective1_voxels": len(base),
        "eligible_objective1_components": 0,
        "preserved_objective1_voxels": len(base),
        "preserved_objective1_components": 0,
        "rejected_objective1_budget_components": 0,
        "fused_internal_voxels": len(base),
        "total_internal_budget": budget,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one shared retrieval selector on held-out view-18 shapes."
    )
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
    parser.add_argument(
        "--gallery-categories",
        nargs="+",
        default=["bus", "cabinet", "car", "file_cabinet"],
    )
    parser.add_argument("--gallery-per-category", type=int, default=15)
    parser.add_argument("--save-smooth-meshes", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.view_index != 18:
        parser.error("unified retrieval is intentionally evaluated with view 18")
    if args.transplant_margin not in args.margins:
        parser.error("--transplant-margin must be included in --margins")
    if args.save_smooth_meshes and args.objective1_meshes is None:
        parser.error("--save-smooth-meshes requires --objective1-meshes")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}. Use --resume."
        )

    policy = read_unified_policy(
        args.policy, args.view_index, args.resolution, args.transplant_margin
    )
    top_k = int(policy["top_k"])
    selected_ids = read_selected_ids(args.ids_file)
    rankings = read_rankings(
        args.rankings, f"dino_view{args.view_index:03d}", selected_ids, top_k
    )
    train_dir = args.dataset_root / "train"
    test_dir = args.dataset_root / "test"
    train_rows = read_metadata(train_dir / "metadata.csv")
    test_rows = read_metadata(test_dir / "metadata.csv")
    if selected_ids is not None:
        test_rows = [row for row in test_rows if row["sha256"] in selected_ids]
    if set(rankings) != {row["sha256"] for row in test_rows}:
        raise ValueError("Ranking IDs do not exactly match selected held-out IDs")
    train_categories = {row["sha256"]: row["category"] for row in train_rows}

    candidate_internals = {}
    candidate_exteriors = {}
    donor_components = {}

    def load_candidate(candidate_id: str) -> None:
        if candidate_id in candidate_internals:
            return
        path = train_dir / "voxels" / f"{candidate_id}.ply"
        if not path.is_file():
            raise FileNotFoundError(f"Missing training voxel PLY: {path}")
        voxels = read_voxels(path, args.resolution)
        candidate_internals[candidate_id] = interior(
            voxels, args.transplant_margin
        )
        candidate_exteriors[candidate_id] = (
            voxels - candidate_internals[candidate_id]
        )
        donor_components[candidate_id] = connected_components(
            candidate_internals[candidate_id]
        )

    voxel_output = args.output_dir / "predictions" / FINAL_METHOD / "voxels"
    mesh_output = args.output_dir / "predictions" / FINAL_METHOD / "mesh"
    gt_voxel_reference = args.output_dir / "references" / "ground_truth_voxels"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gt_voxel_reference.mkdir(parents=True, exist_ok=True)
    per_sample_rows = []
    applied_rows = []
    selector = policy["selector"]

    for index, test_row in enumerate(test_rows, start=1):
        sample_id = test_row["sha256"]
        category = test_row["category"]
        ranking = rankings[sample_id]
        for row in ranking:
            donor_id = str(row["retrieved_id"])
            if train_categories.get(donor_id) != category:
                raise ValueError(
                    f"Same-category gallery violation: {sample_id} -> {donor_id}"
                )
            load_candidate(donor_id)

        gt_path = test_dir / "voxels" / f"{sample_id}.ply"
        objective_path = args.objective1_voxels / f"{sample_id}.ply"
        for path in (gt_path, objective_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing held-out voxel PLY: {path}")
        objective1 = read_voxels(objective_path, args.resolution)
        objective1_internal = interior(objective1, args.transplant_margin)
        objective1_exterior = objective1 - objective1_internal
        safe = enclosed_volume(objective1_exterior, args.transplant_margin)
        records = prepare_candidate_records(
            ranking,
            candidate_internals,
            candidate_exteriors,
            objective1_internal,
            objective1_exterior,
            safe,
            args.resolution,
        )
        hypotheses = shortlist_records(
            records,
            policy["donor_reranker"],
            int(policy["donor_shortlist"]),
            policy["donor_experts"],
        )
        candidates = generate_unified_candidates(
            records,
            candidate_internals,
            objective1_internal,
            objective1_exterior,
            safe,
            policy["operators"],
            args.resolution,
            args.transplant_margin,
            donor_components,
            hypotheses,
        )
        selected, use_fallback, gate_reason, selection_margin = (
            select_unified_candidate(
                candidates,
                selector["models"],
                float(selector["acceptance_threshold"]),
                float(selector["minimum_selection_margin"]),
                float(selector["uncertainty_weight"]),
            )
        )

        if use_fallback:
            prediction = objective1
            preserved = objective1_internal
            accepted_source = set()
            transferred_for_mesh = set()
            fusion_diag = fallback_fusion_diagnostics(
                objective1_internal, safe, int(selected["donor_budget"])
            )
        else:
            prediction = selected["prediction"]
            preserved = selected["preserved"]
            accepted_source = selected["accepted_source"]
            transferred_for_mesh = selected["transferred"]
            fusion_diag = selected["fusion_diagnostics"]
        if not objective1_exterior <= prediction:
            raise AssertionError(f"Objective-1 shell was not preserved: {sample_id}")
        if transferred_for_mesh - safe:
            raise AssertionError(f"Transferred donor escaped safe volume: {sample_id}")

        # Load held-out ground truth only after retrieval and gating are frozen
        # for this sample. It is used exclusively for reporting below.
        gt = read_voxels(gt_path, args.resolution)
        alignment = alignment_payload(selected["alignment"])
        transfer_diag = selected["transfer_diagnostics"]
        common = {
            "sample_id": sample_id,
            "category": category,
            "selected_id": selected["donor_id"],
            "selected_rank": selected["rank"],
            "image_similarity": selected["image_similarity"],
            "predicted_donor_quality": selected["predicted_delta"],
            "predicted_delta_std": selected["predicted_delta_std"],
            "conservative_score": selected["conservative_score"],
            "selected_exterior_tolerant_f1": selected["features"][
                "exterior_tolerant_f1"
            ],
            "alignment_strength": float(selected["alignment"].strength),
            "alignment_scale_x": alignment["scale"][0],
            "alignment_scale_y": alignment["scale"][1],
            "alignment_scale_z": alignment["scale"][2],
            "policy_mode": selected["action"],
            "selection_score_margin": selection_margin,
            "safe_volume_voxels": len(safe),
            "transfer_coverage": selected["transfer_coverage"],
            "transfer_coverage_denominator": selected[
                "transfer_coverage_denominator"
            ],
            "objective1_internal_voxels": len(objective1_internal),
            "objective1_exterior_voxels": len(objective1_exterior),
            "removed_objective1_internal_voxels": len(
                objective1_internal - prediction
            ),
            "added_vs_objective1_voxels": len(prediction - objective1),
            "used_objective1_fallback": int(use_fallback),
            "gate_reason": gate_reason,
            "candidate_count": len(candidates),
            "valid_candidate_count": sum(int(row["valid"]) for row in candidates),
            "candidate_filter_reason": selected["filter_reason"] or "passed",
            **transfer_diag,
            **fusion_diag,
        }
        for method, variant in (
            ("objective1", objective1),
            (FINAL_METHOD, prediction),
        ):
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
                donor_mesh = (
                    train_dir / "renders" / selected["donor_id"] / "mesh.ply"
                )
                if not donor_mesh.is_file():
                    raise FileNotFoundError(
                        f"Missing normalized donor mesh: {donor_mesh}"
                    )
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

        applied_rows.append(common)
        if index % 25 == 0 or index == len(test_rows):
            print(f"Processed {index}/{len(test_rows)}")

    summary_rows = aggregate(per_sample_rows, ("method", "margin"))
    category_rows = aggregate(per_sample_rows, ("method", "category", "margin"))
    paired_rows = paired_summary(
        per_sample_rows, "objective1", args.transplant_margin
    )
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
        FINAL_METHOD,
    )
    if gallery:
        write_csv(args.output_dir / "visualization_manifest.csv", gallery)
        gt_mesh_reference = args.output_dir / "references" / "ground_truth_mesh"
        gt_mesh_reference.mkdir(parents=True, exist_ok=True)
        for row in gallery:
            sample_id = row["sample_id"]
            source = test_dir / "renders" / sample_id / "mesh.ply"
            if not source.is_file():
                raise FileNotFoundError(f"Missing GT display mesh: {source}")
            shutil.copyfile(source, gt_mesh_reference / f"{sample_id}.ply")
    shutil.copyfile(args.policy, args.output_dir / "policy.json")
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump({
            "dataset_root": str(args.dataset_root),
            "objective1_voxels": str(args.objective1_voxels),
            "objective1_meshes": (
                str(args.objective1_meshes) if args.objective1_meshes else None
            ),
            "rankings": str(args.rankings),
            "policy": str(args.policy),
            "policy_method": policy["method"],
            "view_index": args.view_index,
            "resolution": args.resolution,
            "transplant_margin": args.transplant_margin,
            "margins": sorted(args.margins),
            "save_smooth_meshes": args.save_smooth_meshes,
            "resume": args.resume,
        }, file, indent=2)
        file.write("\n")

    print(f"\nHeld-out {FINAL_METHOD} results")
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
            f"F1 delta={comparison['mean_internal_f1_delta']:+.4f}, "
            f"W/L/T={comparison['wins']}/{comparison['losses']}/{comparison['ties']}"
        )
    print(f"Wrote {FINAL_METHOD} results to {args.output_dir}")


if __name__ == "__main__":
    main()
