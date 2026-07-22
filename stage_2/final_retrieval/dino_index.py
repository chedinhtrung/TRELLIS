#!/usr/bin/env python3
"""Build the view-18 DINOv2 train embedding cache and held-out top-K index."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_functional
from PIL import Image

from .config import MODEL_NAME, TOP_K, VIEW_INDEX
from .support import read_metadata, read_selected_ids, write_csv


IMAGE_SIZE = 518
IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def conditioning_image_path(
    split_dir: Path, sample_id: str, view_index: int
) -> Path:
    render_dir = split_dir / "renders_cond" / sample_id
    transforms_path = render_dir / "transforms.json"
    if not transforms_path.is_file():
        raise FileNotFoundError(f"Missing render metadata: {transforms_path}")
    with transforms_path.open(encoding="utf-8") as file:
        frames = json.load(file).get("frames", [])
    if view_index >= len(frames):
        raise ValueError(
            f"View {view_index} unavailable for {sample_id}; found {len(frames)} views"
        )
    image_path = render_dir / frames[view_index]["file_path"]
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing conditioning image: {image_path}")
    return image_path


def preprocess_image(path: Path) -> torch.Tensor:
    """Match the crop, alpha compositing, and normalization used by TRELLIS."""
    with Image.open(path) as source:
        image = source.convert("RGBA")
    array = np.asarray(image)
    foreground = np.argwhere(array[:, :, 3] > 0.8 * 255)
    if foreground.size == 0:
        raise ValueError(f"Empty alpha mask: {path}")
    left, right = foreground[:, 1].min(), foreground[:, 1].max()
    top, bottom = foreground[:, 0].min(), foreground[:, 0].max()
    center = ((left + right) / 2, (top + bottom) / 2)
    size = int(max(right - left, bottom - top) * 1.2)
    if size <= 0:
        raise ValueError(f"Foreground is too small: {path}")
    crop = (
        center[0] - size // 2,
        center[1] - size // 2,
        center[0] + size // 2,
        center[1] + size // 2,
    )
    image = image.crop(crop).resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS
    )
    array = np.asarray(image).astype(np.float32) / 255.0
    rgb = array[:, :, :3] * array[:, :, 3:4]
    tensor = torch.from_numpy(rgb).permute(2, 0, 1)
    return (tensor - IMAGE_MEAN) / IMAGE_STD


def embedding_path(
    output_dir: Path, split: str, view_index: int, model_name: str
) -> Path:
    safe_model = model_name.replace("/", "_")
    return output_dir / "embeddings" / (
        f"{split}_view{view_index:03d}_{safe_model}.npz"
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
    if (
        sample_ids != [row["sha256"] for row in rows]
        or categories != [row["category"] for row in rows]
        or cached_view != view_index
        or cached_model != model_name
        or embeddings.ndim != 2
        or embeddings.shape[0] != len(rows)
    ):
        raise ValueError(f"Embedding cache does not match this run: {path}")
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


def extract_embeddings(
    model: torch.nn.Module,
    rows: list[dict[str, str]],
    image_paths: dict[str, Path],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    outputs = []
    print(f"Extracting {len(rows)} image embeddings")
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        batch = torch.stack([
            preprocess_image(image_paths[row["sha256"]]) for row in batch_rows
        ]).to(device)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            features = model(batch, is_training=True)
            if "x_norm_clstoken" in features:
                vectors = features["x_norm_clstoken"]
            else:
                vectors = torch_functional.layer_norm(
                    features["x_prenorm"][:, 0],
                    features["x_prenorm"].shape[-1:],
                )
            vectors = torch_functional.normalize(vectors.float(), dim=-1)
        outputs.append(vectors.cpu().numpy())
        completed = min(start + batch_size, len(rows))
        if completed % 100 == 0 or completed == len(rows):
            print(f"  embedded {completed}/{len(rows)}")
    return np.concatenate(outputs, axis=0).astype(np.float32)


def load_dino(model_name: str, device: torch.device) -> torch.nn.Module:
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    print(f"Loading {model_name} on {device}")
    model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)
    return model.eval().to(device)


def make_test_rankings(
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    top_k: int,
    view_index: int,
) -> list[dict]:
    train_vectors = {
        row["sha256"]: vector
        for row, vector in zip(train_rows, train_embeddings)
    }
    test_vectors = {
        row["sha256"]: vector
        for row, vector in zip(test_rows, test_embeddings)
    }
    gallery: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in train_rows:
        gallery[row["category"]].append(row)
    for rows in gallery.values():
        rows.sort(key=lambda row: row["sha256"])

    output = []
    query_mode = f"dino_view{view_index:03d}"
    for index, query in enumerate(test_rows, start=1):
        candidates = gallery[query["category"]]
        if len(candidates) < top_k:
            raise ValueError(
                f"Category {query['category']} has fewer than {top_k} donors"
            )
        matrix = np.stack([train_vectors[row["sha256"]] for row in candidates])
        similarities = matrix @ test_vectors[query["sha256"]]
        order = np.argsort(-similarities, kind="stable")[:top_k]
        for rank, candidate_index in enumerate(order, start=1):
            candidate = candidates[int(candidate_index)]
            output.append({
                "query_mode": query_mode,
                "sample_id": query["sha256"],
                "category": query["category"],
                "rank": rank,
                "retrieved_id": candidate["sha256"],
                "image_cosine_similarity": float(similarities[candidate_index]),
            })
        if index % 25 == 0 or index == len(test_rows):
            print(f"  retrieved {index}/{len(test_rows)}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the self-contained view-18 DINO retrieval index."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--view-index", type=int, default=VIEW_INDEX)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--force-embeddings", action="store_true")
    args = parser.parse_args()
    if args.view_index != VIEW_INDEX:
        parser.error(f"the final method requires view {VIEW_INDEX}")
    if args.top_k < 1 or args.batch_size < 1:
        parser.error("top-k and batch size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    train_dir = args.dataset_root / "train"
    test_dir = args.dataset_root / "test"
    train_rows = read_metadata(train_dir / "metadata.csv")
    test_rows = read_metadata(test_dir / "metadata.csv")
    selected_ids = read_selected_ids(args.ids_file)
    if selected_ids is not None:
        known = {row["sha256"] for row in test_rows}
        if selected_ids - known:
            raise ValueError("selected IDs are not present in test metadata")
        test_rows = [row for row in test_rows if row["sha256"] in selected_ids]
    train_images = {
        row["sha256"]: conditioning_image_path(
            train_dir, row["sha256"], args.view_index
        )
        for row in train_rows
    }
    test_images = {
        row["sha256"]: conditioning_image_path(
            test_dir, row["sha256"], args.view_index
        )
        for row in test_rows
    }
    train_path = embedding_path(
        args.output_dir, "train", args.view_index, args.model
    )
    test_path = embedding_path(
        args.output_dir, "test", args.view_index, args.model
    )
    train_embeddings = None if args.force_embeddings else load_embedding_cache(
        train_path, train_rows, args.view_index, args.model
    )
    test_embeddings = None if args.force_embeddings else load_embedding_cache(
        test_path, test_rows, args.view_index, args.model
    )
    if train_embeddings is None or test_embeddings is None:
        device = torch.device(args.device)
        model = load_dino(args.model, device)
        if train_embeddings is None:
            train_embeddings = extract_embeddings(
                model, train_rows, train_images, args.batch_size, device
            )
            save_embedding_cache(
                train_path, train_rows, train_embeddings, args.view_index, args.model
            )
        if test_embeddings is None:
            test_embeddings = extract_embeddings(
                model, test_rows, test_images, args.batch_size, device
            )
            save_embedding_cache(
                test_path, test_rows, test_embeddings, args.view_index, args.model
            )

    ranking_rows = make_test_rankings(
        train_rows,
        test_rows,
        train_embeddings,
        test_embeddings,
        args.top_k,
        args.view_index,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "rankings.csv", ranking_rows)
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump({
            "dataset_root": str(args.dataset_root),
            "view_index": args.view_index,
            "model": args.model,
            "top_k": args.top_k,
            "held_out_labels_used": False,
        }, file, indent=2)
        file.write("\n")
    print(f"Wrote DINO retrieval index to {args.output_dir}")


if __name__ == "__main__":
    main()
