#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from stage_2.cache_ss_coordinate_endpoints import (
    as_bool,
    conditioning_image_path,
    file_sha256,
    validate_latent,
    write_json,
)
from stage_2.compare_internals import read_voxels, score_pair


def configure_model(flow_model, model_cfg: dict) -> None:
    from trellis.modules.lora import apply_lora

    lora_cfg = model_cfg["lora"]
    apply_lora(
        flow_model,
        rank=lora_cfg["rank"],
        alpha=lora_cfg["alpha"],
        dropout=lora_cfg["dropout"],
        target_patterns=lora_cfg["target_patterns"],
    )
    categories = model_cfg.get("categories")
    if categories is not None:
        flow_model.enable_category_conditioning(categories)
    head_cfg = model_cfg["coordinate_head"]
    flow_model.enable_coordinate_head(
        hidden_channels=head_cfg.get("hidden_channels", 64),
        output_resolution=head_cfg.get("output_resolution", 64),
        residual_scale=head_cfg.get("residual_scale", 20.0),
        base_logit_clip=head_cfg.get("base_logit_clip", 10.0),
    )


def load_adapter(flow_model, checkpoint: Path) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected = {
        key
        for key in flow_model.state_dict()
        if ".lora_down" in key
        or ".lora_up" in key
        or key.startswith("category_embedding.")
        or key.startswith("coordinate_head.")
    }
    if set(state) != expected:
        raise RuntimeError(
            f"Adapter checkpoint mismatch for {checkpoint}: "
            f"missing={sorted(expected - set(state))}, unexpected={sorted(set(state) - expected)}"
        )
    incompatible = flow_model.load_state_dict(state, strict=False)
    missing = sorted(expected.intersection(incompatible.missing_keys))
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Failed to load {checkpoint}: missing={missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )


def checkpoint_sort_key(path: Path) -> tuple[int, str]:
    if "step" in path.stem:
        return int(path.stem.rsplit("step", 1)[1]), path.name
    return 10**12, path.name


def mean_row(rows: list[dict], checkpoint: str, threshold: float, margin: int) -> dict:
    metrics = (
        "voxel_iou",
        "exterior_iou",
        "internal_precision",
        "internal_recall",
        "internal_f1",
        "gt_voxels",
        "pred_voxels",
        "gt_internal_voxels",
        "pred_internal_voxels",
    )
    return {
        "checkpoint": checkpoint,
        "threshold": threshold,
        "margin": margin,
        **{metric: float(np.mean([row[metric] for row in rows])) for metric in metrics},
        "num_samples": len(rows),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a coordinate-head checkpoint and logit threshold on validation data."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pipeline", default="microsoft/TRELLIS-image-large")
    parser.add_argument("--latent-name", default="o1_generated_view18_seed42")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[-8, -4, -2, -1, -0.5, 0, 0.5, 1, 2, 4, 8],
    )
    parser.add_argument("--margins", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--selection-margin", type=int, default=2)
    parser.add_argument("--max-exterior-drop", type=float, default=0.01)
    parser.add_argument("--min-internal-gain", type=float, default=0.0)
    parser.add_argument("--visual-count", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if len(args.thresholds) != len(set(args.thresholds)):
        parser.error("--thresholds must not contain duplicates")
    if 0 not in args.thresholds:
        parser.error("--thresholds must include 0 for the original Objective-1 reference")
    if any(margin < 1 for margin in args.margins):
        parser.error("--margins must be positive")
    if args.selection_margin not in args.margins:
        parser.error("--selection-margin must be included in --margins")
    if args.max_exterior_drop < 0:
        parser.error("--max-exterior-drop must be non-negative")
    if args.min_internal_gain < 0:
        parser.error("--min-internal-gain must be non-negative")
    if args.visual_count <= 0:
        parser.error("--visual-count must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    output_dir = args.output_dir or args.model_dir / "validation"
    config_path = args.model_dir / "config.json"
    cache_dir = args.dataset_dir / "ss_latents" / args.latent_name
    manifest_path = cache_dir / "cache_config.json"
    complete_path = cache_dir / "cache_complete.json"
    metadata_path = args.dataset_dir / "metadata.csv"
    for path in (config_path, manifest_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.limit is None and not complete_path.is_file():
        raise FileNotFoundError(
            f"Full validation requires a complete endpoint cache: {complete_path}"
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("pipeline") != args.pipeline:
        raise ValueError("Validation pipeline does not match the endpoint cache")
    model_cfg = config["models"]["denoiser"]
    dataset_cfg = config["dataset"]["args"]
    head_cfg = model_cfg.get("coordinate_head", {})
    if not head_cfg.get("freeze_backbone", False):
        raise ValueError("Validation requires a run with coordinate_head.freeze_backbone=true")
    if dataset_cfg.get("latent_model") != args.latent_name:
        raise ValueError("Training config latent_model does not match --latent-name")
    if dataset_cfg.get("view_index") != manifest.get("view_index"):
        raise ValueError("Training view_index does not match the endpoint cache")
    if not dataset_cfg.get("pipeline_preprocessing", False):
        raise ValueError("Training must use pipeline_preprocessing=true")

    init_checkpoint = Path(model_cfg["init_lora_ckpt"])
    if not init_checkpoint.is_absolute():
        init_checkpoint = REPO_ROOT / init_checkpoint
    if file_sha256(init_checkpoint) != manifest.get("ss_checkpoint_sha256"):
        raise ValueError("Endpoint cache was not generated by the Objective-1 checkpoint in config.json")

    candidates = sorted(
        list((args.model_dir / "ckpts").glob("denoiser_lora_step*.pt"))
        + list((args.model_dir / "ckpts").glob("denoiser_lora_final.pt")),
        key=checkpoint_sort_key,
    )
    unique_candidates = []
    seen_checkpoint_hashes = set()
    for checkpoint in candidates:
        checkpoint_hash = file_sha256(checkpoint)
        if checkpoint_hash not in seen_checkpoint_hashes:
            unique_candidates.append(checkpoint)
            seen_checkpoint_hashes.add(checkpoint_hash)
    candidates = unique_candidates
    if not candidates:
        raise FileNotFoundError(f"No coordinate-head adapter checkpoints under {args.model_dir / 'ckpts'}")

    metadata = pd.read_csv(metadata_path)
    cache_column = f"ss_latent_{args.latent_name}"
    if cache_column not in metadata:
        raise ValueError(f"Metadata is missing cache column {cache_column}")
    rows = metadata[as_bool(metadata[cache_column]) & as_bool(metadata["voxelized"])]
    ids = rows["sha256"].astype(str).tolist()
    if args.limit is not None:
        ids = ids[: args.limit]
    if not ids:
        raise ValueError("Validation cache contains no eligible objects")
    categories = dict(zip(metadata["sha256"].astype(str), metadata["category"].astype(str)))

    from trellis.pipelines import TrellisImageTo3DPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("Coordinate-head validation requires a CUDA GPU")
    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.pipeline)
    pipeline.to(torch.device("cuda"))
    current_sampler = json.loads(json.dumps(pipeline.sparse_structure_sampler_params))
    if current_sampler != manifest.get("sampler_params"):
        raise ValueError("Current sparse-structure sampler settings do not match the endpoint cache")
    flow_model = pipeline.models["sparse_structure_flow_model"]
    configure_model(flow_model, model_cfg)
    flow_model.eval()
    decoder = pipeline.models["sparse_structure_decoder"].eval()
    category_names = model_cfg.get("categories")
    expected_shape = (
        flow_model.in_channels,
        flow_model.resolution,
        flow_model.resolution,
        flow_model.resolution,
    )

    ground_truth = {
        sample_id: read_voxels(args.dataset_dir / "voxels" / f"{sample_id}.ply", 64)
        for sample_id in ids
    }
    summary_rows = []
    per_sample_rows = []
    baseline_rows = {
        (threshold, margin): []
        for threshold in args.thresholds
        for margin in args.margins
    }

    for candidate_index, checkpoint in enumerate(candidates):
        load_adapter(flow_model, checkpoint)
        candidate_rows = {
            (threshold, margin): []
            for threshold in args.thresholds
            for margin in args.margins
        }

        for sample_id in tqdm(ids, desc=f"Validating {checkpoint.name}"):
            latent_path = cache_dir / f"{sample_id}.npz"
            validate_latent(
                latent_path,
                expected_shape,
                int(manifest["view_index"]),
                int(manifest["seed"]),
            )
            with np.load(latent_path, allow_pickle=False) as data:
                latent = torch.from_numpy(np.asarray(data["mean"])).unsqueeze(0).cuda().float()
            image_path = conditioning_image_path(
                args.dataset_dir, sample_id, int(manifest["view_index"])
            )
            with Image.open(image_path) as image, torch.inference_mode():
                image = pipeline.preprocess_image(image)
                category = [categories[sample_id]] if category_names is not None else None
                cond = pipeline.get_cond([image], category=category)
                base_logits = decoder(latent)
                _, total_logits = flow_model(
                    latent,
                    torch.zeros(1, device=latent.device, dtype=torch.float32),
                    cond["cond"],
                    category=category,
                    return_coordinate_head=True,
                    base_logits=base_logits,
                )
                base_array = base_logits[0, 0].float().cpu().numpy()
                total_array = total_logits[0, 0].float().cpu().numpy()

            gt = ground_truth[sample_id]
            if candidate_index == 0:
                for threshold in args.thresholds:
                    base_coords = {
                        tuple(coord) for coord in np.argwhere(base_array > threshold).tolist()
                    }
                    for margin in args.margins:
                        row = {
                            "checkpoint": "objective1",
                            "threshold": threshold,
                            "sample_id": sample_id,
                            "category": categories[sample_id],
                            "margin": margin,
                            **score_pair(gt, base_coords, margin),
                        }
                        baseline_rows[(threshold, margin)].append(row)
                        per_sample_rows.append(row)

            for threshold in args.thresholds:
                pred = {tuple(coord) for coord in np.argwhere(total_array > threshold).tolist()}
                for margin in args.margins:
                    row = {
                        "checkpoint": checkpoint.name,
                        "threshold": threshold,
                        "sample_id": sample_id,
                        "category": categories[sample_id],
                        "margin": margin,
                        **score_pair(gt, pred, margin),
                    }
                    candidate_rows[(threshold, margin)].append(row)
                    per_sample_rows.append(row)

        for threshold in args.thresholds:
            for margin in args.margins:
                summary_rows.append(
                    mean_row(candidate_rows[(threshold, margin)], checkpoint.name, threshold, margin)
                )

    baseline_summary = [
        mean_row(baseline_rows[(threshold, margin)], "objective1", threshold, margin)
        for threshold in args.thresholds
        for margin in args.margins
    ]
    summary_rows = baseline_summary + summary_rows
    original_baseline = next(
        row
        for row in baseline_summary
        if row["threshold"] == 0 and row["margin"] == args.selection_margin
    )
    minimum_exterior_iou = original_baseline["exterior_iou"] - args.max_exterior_drop
    calibrated_baselines = [
        row
        for row in baseline_summary
        if row["margin"] == args.selection_margin
        and row["exterior_iou"] >= minimum_exterior_iou
    ]
    calibrated_baseline = max(
        calibrated_baselines,
        key=lambda row: (row["internal_f1"], row["voxel_iou"], row["exterior_iou"]),
    )
    eligible = [
        row
        for row in summary_rows
        if row["checkpoint"] != "objective1"
        and row["margin"] == args.selection_margin
        and row["exterior_iou"] >= minimum_exterior_iou
        and row["internal_f1"]
        > calibrated_baseline["internal_f1"] + args.min_internal_gain
    ]
    if not eligible:
        write_csv(output_dir / "summary.csv", summary_rows)
        write_csv(output_dir / "per_sample.csv", per_sample_rows)
        raise RuntimeError(
            "No checkpoint/threshold improves validation internal F1 while satisfying the "
            "exterior-IoU constraint. "
            "The head should not be sent to the test set."
        )
    best = max(
        eligible,
        key=lambda row: (row["internal_f1"], row["voxel_iou"], row["exterior_iou"]),
    )
    best_source = args.model_dir / "ckpts" / best["checkpoint"]
    best_checkpoint = args.model_dir / "ckpts" / "denoiser_lora_best.pt"
    shutil.copy2(best_source, best_checkpoint)
    (args.model_dir / "coordinate_threshold.txt").write_text(
        f"{best['threshold']}\n", encoding="utf-8"
    )
    (args.model_dir / "objective1_threshold.txt").write_text(
        f"{calibrated_baseline['threshold']}\n", encoding="utf-8"
    )

    best_sample_rows = {
        row["sample_id"]: row
        for row in per_sample_rows
        if row["checkpoint"] == best["checkpoint"]
        and row["threshold"] == best["threshold"]
        and row["margin"] == args.selection_margin
    }
    baseline_by_sample = {
        row["sample_id"]: row
        for row in per_sample_rows
        if row["checkpoint"] == "objective1"
        and row["threshold"] == calibrated_baseline["threshold"]
        and row["margin"] == args.selection_margin
    }
    ranked_ids = sorted(
        ids,
        key=lambda sample_id: (
            best_sample_rows[sample_id]["internal_f1"]
            - baseline_by_sample[sample_id]["internal_f1"]
        ),
        reverse=True,
    )
    visual_ids = ranked_ids[: min(args.visual_count, len(ranked_ids))]
    (output_dir / "best_visual_ids.txt").parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "best_visual_ids.txt").write_text("\n".join(visual_ids) + "\n", encoding="utf-8")

    selection = {
        **best,
        "source_checkpoint": str(best_source),
        "best_checkpoint": str(best_checkpoint),
        "objective1_threshold": calibrated_baseline["threshold"],
        "objective1_exterior_iou": calibrated_baseline["exterior_iou"],
        "objective1_internal_f1": calibrated_baseline["internal_f1"],
        "internal_f1_gain": best["internal_f1"] - calibrated_baseline["internal_f1"],
        "original_objective1_exterior_iou": original_baseline["exterior_iou"],
        "max_exterior_drop": args.max_exterior_drop,
        "min_internal_gain": args.min_internal_gain,
        "selection_margin": args.selection_margin,
        "cache_provenance": manifest,
    }
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "per_sample.csv", per_sample_rows)
    write_json(output_dir / "best.json", selection)
    write_json(args.model_dir / "best.json", selection)

    print(json.dumps(selection, indent=2))
    print(f"Copied selected adapter to {best_checkpoint}")
    print(f"Wrote selected threshold to {args.model_dir / 'coordinate_threshold.txt'}")
    print(f"Wrote Objective-1 threshold to {args.model_dir / 'objective1_threshold.txt'}")
    print(f"Validation samples to inspect: {output_dir / 'best_visual_ids.txt'}")


if __name__ == "__main__":
    main()
