#!/usr/bin/env python3
"""Retrieve complete training shapes using DINOv2 image similarity.

The gallery and query use the same indexed conditioning view. Embeddings are
cached, candidates are restricted to the known object category, and geometry is
evaluated with the same voxel metrics as the Day 1 exterior-retrieval baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from evaluate_retrieval_completion import (
    aggregate,
    load_split,
    read_ids,
    score_pair,
    write_csv,
)


IMAGE_SIZE = 518
IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def read_metadata(split_dir: Path, selected_ids: set[str] | None = None) -> list[dict[str, str]]:
    metadata_path = split_dir / "metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    with metadata_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No rows found in {metadata_path}")
    required = {"sha256", "category"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{metadata_path} is missing columns: {sorted(missing)}")

    if selected_ids is not None:
        known_ids = {row["sha256"].strip() for row in rows}
        unknown = sorted(selected_ids - known_ids)
        if unknown:
            raise ValueError(f"IDs missing from {metadata_path}: {unknown[:5]}")
        rows = [row for row in rows if row["sha256"].strip() in selected_ids]

    seen = set()
    clean_rows = []
    for row in rows:
        sample_id = row["sha256"].strip()
        category = row["category"].strip()
        if not sample_id or not category:
            raise ValueError(f"Empty sha256 or category in {metadata_path}")
        if sample_id in seen:
            raise ValueError(f"Duplicate sha256 in {metadata_path}: {sample_id}")
        seen.add(sample_id)
        clean_rows.append({"sha256": sample_id, "category": category})
    return clean_rows


def conditioning_image_path(split_dir: Path, sample_id: str, view_index: int) -> Path:
    render_dir = split_dir / "renders_cond" / sample_id
    transforms_path = render_dir / "transforms.json"
    if not transforms_path.is_file():
        raise FileNotFoundError(f"Missing render metadata: {transforms_path}")
    with transforms_path.open(encoding="utf-8") as file:
        frames = json.load(file).get("frames", [])
    if view_index >= len(frames):
        raise ValueError(
            f"View {view_index} is unavailable for {sample_id}; found {len(frames)} views"
        )
    image_path = render_dir / frames[view_index]["file_path"]
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing conditioning image: {image_path}")
    return image_path


def preprocess_image(path: Path) -> torch.Tensor:
    """Apply the same crop, alpha compositing, and normalization as TRELLIS."""
    with Image.open(path) as source:
        image = source.convert("RGBA")
    image_array = np.asarray(image)
    foreground = np.argwhere(image_array[:, :, 3] > 0.8 * 255)
    if foreground.size == 0:
        raise ValueError(f"Conditioning image has an empty alpha mask: {path}")

    left = foreground[:, 1].min()
    top = foreground[:, 0].min()
    right = foreground[:, 1].max()
    bottom = foreground[:, 0].max()
    center = ((left + right) / 2, (top + bottom) / 2)
    size = int(max(right - left, bottom - top) * 1.2)
    if size <= 0:
        raise ValueError(f"Conditioning image foreground is too small: {path}")

    crop = (
        center[0] - size // 2,
        center[1] - size // 2,
        center[0] + size // 2,
        center[1] + size // 2,
    )
    image = image.crop(crop).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    image_array = np.asarray(image).astype(np.float32) / 255.0
    rgb = image_array[:, :, :3] * image_array[:, :, 3:4]
    tensor = torch.from_numpy(rgb).permute(2, 0, 1)
    return (tensor - IMAGE_MEAN) / IMAGE_STD


def cache_path(output_dir: Path, split: str, view_index: int, model_name: str) -> Path:
    safe_model_name = model_name.replace("/", "_")
    return output_dir / "embeddings" / (
        f"{split}_view{view_index:03d}_{safe_model_name}.npz"
    )


def load_embedding_cache(
    path: Path,
    rows: list[dict[str, str]],
    view_index: int,
    model_name: str,
) -> np.ndarray | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        sample_ids = data["sample_ids"].astype(str).tolist()
        categories = data["categories"].astype(str).tolist()
        cached_view = int(data["view_index"].item())
        cached_model = str(data["model_name"].item())
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)

    expected_ids = [row["sha256"] for row in rows]
    expected_categories = [row["category"] for row in rows]
    if (
        sample_ids != expected_ids
        or categories != expected_categories
        or cached_view != view_index
        or cached_model != model_name
    ):
        raise ValueError(f"Embedding cache does not match this run: {path}")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(rows):
        raise ValueError(f"Invalid embedding shape in {path}: {embeddings.shape}")
    print(f"Reusing {len(rows)} cached embeddings from {path}")
    return embeddings


def save_embedding_cache(
    path: Path,
    rows: list[dict[str, str]],
    embeddings: np.ndarray,
    view_index: int,
    model_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sample_ids=np.asarray([row["sha256"] for row in rows]),
        categories=np.asarray([row["category"] for row in rows]),
        embeddings=embeddings.astype(np.float32),
        view_index=np.asarray(view_index),
        model_name=np.asarray(model_name),
    )
    print(f"Cached embeddings at {path}")


def load_dino(model_name: str, device: torch.device) -> torch.nn.Module:
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    print(f"Loading {model_name} on {device}")
    model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)
    model.eval().to(device)
    return model


def extract_embeddings(
    model: torch.nn.Module,
    rows: list[dict[str, str]],
    image_paths: dict[str, Path],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    outputs = []
    use_autocast = device.type == "cuda"
    print(f"Extracting {len(rows)} image embeddings")
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        batch = torch.stack(
            [preprocess_image(image_paths[row["sha256"]]) for row in batch_rows]
        ).to(device)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_autocast,
        ):
            features = model(batch, is_training=True)
            if "x_norm_clstoken" in features:
                embeddings = features["x_norm_clstoken"]
            else:
                embeddings = F.layer_norm(
                    features["x_prenorm"][:, 0],
                    features["x_prenorm"].shape[-1:],
                )
            embeddings = F.normalize(embeddings.float(), dim=-1)
        outputs.append(embeddings.cpu().numpy())
        completed = min(start + batch_size, len(rows))
        if completed % 100 == 0 or completed == len(rows):
            print(f"  embedded {completed}/{len(rows)}")
    return np.concatenate(outputs, axis=0).astype(np.float32)


def copy_visualization(
    output_dir: Path,
    mode_name: str,
    test_sample,
    query_image: Path,
    top1_sample,
    top1_image: Path,
    oracle_sample,
    oracle_image: Path,
) -> None:
    sample_dir = output_dir / "visualizations" / mode_name / test_sample.sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(test_sample.voxel_path, sample_dir / "ground_truth.ply")
    shutil.copyfile(query_image, sample_dir / "query.png")
    shutil.copyfile(top1_sample.voxel_path, sample_dir / f"top1__{top1_sample.sample_id}.ply")
    shutil.copyfile(top1_image, sample_dir / f"top1__{top1_sample.sample_id}.png")
    shutil.copyfile(
        oracle_sample.voxel_path,
        sample_dir / f"oracle__{oracle_sample.sample_id}.ply",
    )
    shutil.copyfile(oracle_image, sample_dir / f"oracle__{oracle_sample.sample_id}.png")


def evaluate(
    train_samples,
    query_samples,
    train_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    train_image_paths: dict[str, Path],
    query_image_paths: dict[str, Path],
    margins: list[int],
    oracle_margin: int,
    k_values: list[int],
    view_index: int,
    output_dir: Path,
    visualizations_per_category: int,
) -> tuple[list[dict], list[dict]]:
    mode_name = f"dino_view{view_index:03d}"
    maximum_k = max(k_values)
    train_embedding_by_id = {
        sample.sample_id: embedding
        for sample, embedding in zip(train_samples, train_embeddings)
    }
    query_embedding_by_id = {
        sample.sample_id: embedding
        for sample, embedding in zip(query_samples, query_embeddings)
    }
    train_by_category = defaultdict(list)
    for sample in train_samples:
        train_by_category[sample.category].append(sample)
    for samples in train_by_category.values():
        samples.sort(key=lambda sample: sample.sample_id)

    selection_rows = []
    ranking_rows = []
    visualized = defaultdict(int)
    print(f"Retrieving with {mode_name}")
    for index, test_sample in enumerate(query_samples, start=1):
        candidates = train_by_category.get(test_sample.category, [])
        if maximum_k > len(candidates):
            raise ValueError(
                f"Requested K={maximum_k}, but category {test_sample.category} has "
                f"only {len(candidates)} training shapes"
            )
        candidate_embeddings = np.stack(
            [train_embedding_by_id[candidate.sample_id] for candidate in candidates]
        )
        similarities = candidate_embeddings @ query_embedding_by_id[test_sample.sample_id]
        order = np.argsort(-similarities, kind="stable")[:maximum_k]
        ranking = [(candidates[i], float(similarities[i])) for i in order]
        oracle_scores = [
            float(score_pair(test_sample, candidate, oracle_margin)["internal_f1"])
            for candidate, _similarity in ranking
        ]

        for rank, ((candidate, similarity), oracle_score) in enumerate(
            zip(ranking, oracle_scores), start=1
        ):
            ranking_rows.append({
                "query_mode": mode_name,
                "sample_id": test_sample.sample_id,
                "category": test_sample.category,
                "rank": rank,
                "retrieved_id": candidate.sample_id,
                "image_cosine_similarity": similarity,
                f"internal_f1_margin_{oracle_margin}": oracle_score,
            })

        selections = [("top1", 1, 0)]
        for k in k_values:
            if k > 1:
                best_index = max(
                    range(k),
                    key=lambda candidate_index: (
                        oracle_scores[candidate_index],
                        ranking[candidate_index][1],
                        ranking[candidate_index][0].sample_id,
                    ),
                )
                selections.append(("oracle", k, best_index))

        for selection, k, candidate_index in selections:
            candidate, similarity = ranking[candidate_index]
            for margin in margins:
                selection_rows.append({
                    "query_mode": mode_name,
                    "selection": selection,
                    "k": k,
                    "sample_id": test_sample.sample_id,
                    "category": test_sample.category,
                    "retrieved_id": candidate.sample_id,
                    "retrieved_rank": candidate_index + 1,
                    "image_cosine_similarity": similarity,
                    "margin": margin,
                    **score_pair(test_sample, candidate, margin),
                })

        if visualized[test_sample.category] < visualizations_per_category:
            oracle_index = max(
                range(maximum_k),
                key=lambda candidate_index: (
                    oracle_scores[candidate_index],
                    ranking[candidate_index][1],
                    ranking[candidate_index][0].sample_id,
                ),
            )
            top1_sample = ranking[0][0]
            oracle_sample = ranking[oracle_index][0]
            copy_visualization(
                output_dir,
                mode_name,
                test_sample,
                query_image_paths[test_sample.sample_id],
                top1_sample,
                train_image_paths[top1_sample.sample_id],
                oracle_sample,
                train_image_paths[oracle_sample.sample_id],
            )
            visualized[test_sample.category] += 1

        if index % 25 == 0 or index == len(query_samples):
            print(f"  retrieved {index}/{len(query_samples)}")

    return selection_rows, ranking_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate category-restricted DINOv2 image retrieval."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--query-split", choices=("val", "test"), default="test")
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--view-index", type=int, default=18)
    parser.add_argument("--model", default="dinov2_vitl14_reg")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--margins", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--oracle-margin", type=int, default=2)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 20])
    parser.add_argument("--visualizations-per-category", type=int, default=3)
    parser.add_argument("--force-embeddings", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.view_index < 0:
        raise ValueError("--view-index cannot be negative")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.resolution < 1:
        raise ValueError("--resolution must be positive")
    if any(margin < 1 for margin in args.margins):
        raise ValueError("--margins must contain positive integers")
    if len(args.margins) != len(set(args.margins)):
        raise ValueError("--margins must not contain duplicates")
    if args.oracle_margin not in args.margins:
        raise ValueError("--oracle-margin must be included in --margins")
    if any(k < 1 for k in args.k) or len(args.k) != len(set(args.k)):
        raise ValueError("--k must contain unique positive integers")
    if args.visualizations_per_category < 0:
        raise ValueError("--visualizations-per-category cannot be negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu")

    selected_ids = read_ids(args.ids_file)
    train_dir = args.dataset_root / "train"
    query_dir = args.dataset_root / args.query_split
    train_rows = read_metadata(train_dir)
    query_rows = read_metadata(query_dir, selected_ids)
    train_image_paths = {
        row["sha256"]: conditioning_image_path(
            train_dir, row["sha256"], args.view_index
        )
        for row in train_rows
    }
    query_image_paths = {
        row["sha256"]: conditioning_image_path(
            query_dir, row["sha256"], args.view_index
        )
        for row in query_rows
    }

    train_cache_path = cache_path(
        args.output_dir, "train", args.view_index, args.model
    )
    query_cache_path = cache_path(
        args.output_dir, args.query_split, args.view_index, args.model
    )
    train_embeddings = None if args.force_embeddings else load_embedding_cache(
        train_cache_path, train_rows, args.view_index, args.model
    )
    query_embeddings = None if args.force_embeddings else load_embedding_cache(
        query_cache_path, query_rows, args.view_index, args.model
    )

    if train_embeddings is None or query_embeddings is None:
        device = torch.device(args.device)
        model = load_dino(args.model, device)
        if train_embeddings is None:
            train_embeddings = extract_embeddings(
                model,
                train_rows,
                train_image_paths,
                args.batch_size,
                device,
            )
            save_embedding_cache(
                train_cache_path,
                train_rows,
                train_embeddings,
                args.view_index,
                args.model,
            )
        if query_embeddings is None:
            query_embeddings = extract_embeddings(
                model,
                query_rows,
                query_image_paths,
                args.batch_size,
                device,
            )
            save_embedding_cache(
                query_cache_path,
                query_rows,
                query_embeddings,
                args.view_index,
                args.model,
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    margins = sorted(args.margins)
    train_samples = load_split(
        train_dir,
        args.resolution,
        margins,
    )
    query_samples = load_split(
        query_dir,
        args.resolution,
        margins,
        selected_ids,
    )
    if [sample.sample_id for sample in train_samples] != [
        row["sha256"] for row in train_rows
    ]:
        raise ValueError("Training embedding and voxel orders do not match")
    if [sample.sample_id for sample in query_samples] != [
        row["sha256"] for row in query_rows
    ]:
        raise ValueError("Query embedding and voxel orders do not match")
    overlap = sorted(
        {sample.sample_id for sample in train_samples}
        & {sample.sample_id for sample in query_samples}
    )
    if overlap:
        raise ValueError(f"Training/query ID leakage detected: {overlap[:5]}")

    selection_rows, ranking_rows = evaluate(
        train_samples=train_samples,
        query_samples=query_samples,
        train_embeddings=train_embeddings,
        query_embeddings=query_embeddings,
        train_image_paths=train_image_paths,
        query_image_paths=query_image_paths,
        margins=margins,
        oracle_margin=args.oracle_margin,
        k_values=sorted(args.k),
        view_index=args.view_index,
        output_dir=args.output_dir,
        visualizations_per_category=args.visualizations_per_category,
    )
    summary_rows = aggregate(
        selection_rows, ("query_mode", "selection", "k", "margin")
    )
    category_summary_rows = aggregate(
        selection_rows, ("query_mode", "selection", "k", "category", "margin")
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_sample.csv", selection_rows)
    write_csv(args.output_dir / "rankings.csv", ranking_rows)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_csv(args.output_dir / "category_summary.csv", category_summary_rows)
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "dataset_root": str(args.dataset_root),
                "query_split": args.query_split,
                "view_index": args.view_index,
                "model": args.model,
                "resolution": args.resolution,
                "margins": margins,
                "oracle_margin": args.oracle_margin,
                "k": sorted(args.k),
            },
            file,
            indent=2,
        )
        file.write("\n")

    print("\nPrimary margin results")
    for row in summary_rows:
        if row["margin"] != args.oracle_margin:
            continue
        print(
            f"  {row['query_mode']} {row['selection']}@{row['k']}: "
            f"IoU={row['voxel_iou']:.4f}, exterior IoU={row['exterior_iou']:.4f}, "
            f"internal F1={row['internal_f1']:.4f}"
        )
    print(f"\nWrote DINO retrieval results to {args.output_dir}")


if __name__ == "__main__":
    main()
