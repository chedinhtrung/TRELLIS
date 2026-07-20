#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from stage_2.export_ss_flow_voxels import load_ss_lora


def as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def conditioning_image_path(dataset_dir: Path, sample_id: str, view_index: int) -> Path:
    render_dir = dataset_dir / "renders_cond" / sample_id
    with (render_dir / "transforms.json").open("r", encoding="utf-8") as file:
        frames = json.load(file)["frames"]
    if view_index >= len(frames):
        raise ValueError(
            f"View {view_index} is unavailable for {sample_id}; found {len(frames)} views"
        )
    return render_dir / frames[view_index]["file_path"]


def validate_latent(
    path: Path,
    expected_shape: tuple[int, ...],
    view_index: int | None = None,
    seed: int | None = None,
) -> None:
    with np.load(path, allow_pickle=False) as data:
        if "mean" not in data:
            raise ValueError(f"Cached endpoint has no 'mean' array: {path}")
        latent = np.asarray(data["mean"])
        if view_index is not None:
            if "view_index" not in data or int(np.asarray(data["view_index"]).item()) != view_index:
                raise ValueError(f"Cached endpoint has the wrong view_index: {path}")
        if seed is not None:
            if "seed" not in data or int(np.asarray(data["seed"]).item()) != seed:
                raise ValueError(f"Cached endpoint has the wrong seed: {path}")
    if latent.shape != expected_shape:
        raise ValueError(f"Expected cached endpoint shape {expected_shape}, got {latent.shape}: {path}")
    if not np.issubdtype(latent.dtype, np.floating) or not np.isfinite(latent).all():
        raise ValueError(f"Cached endpoint must contain finite floating-point values: {path}")


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache Objective-1 generated sparse-structure endpoint latents."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--ss-lora-ckpt", type=Path, required=True)
    parser.add_argument("--pipeline", default="microsoft/TRELLIS-image-large")
    parser.add_argument("--latent-name", default="o1_generated_view0_seed42")
    parser.add_argument("--view-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.latent_name):
        parser.error("--latent-name may contain only letters, numbers, '.', '_' and '-'")
    if args.view_index < 0 or args.seed < 0:
        parser.error("--view-index and --seed must be non-negative")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    metadata_path = args.dataset_dir / "metadata.csv"
    for required in (metadata_path, args.ss_lora_ckpt):
        if not required.is_file():
            raise FileNotFoundError(required)

    metadata = pd.read_csv(metadata_path)
    for column in ("sha256", "category", "cond_rendered", "voxelized"):
        if column not in metadata:
            raise ValueError(f"Metadata is missing required column: {column}")
    eligible = metadata[as_bool(metadata["cond_rendered"]) & as_bool(metadata["voxelized"])]
    ids = eligible["sha256"].astype(str).tolist()
    if args.limit is not None:
        ids = ids[: args.limit]
    if not ids:
        raise ValueError("No conditioning-rendered, voxelized objects were found")
    categories = dict(zip(metadata["sha256"].astype(str), metadata["category"].astype(str)))

    from trellis.pipelines import TrellisImageTo3DPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("Endpoint caching requires a CUDA GPU")
    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.pipeline)
    pipeline.to(torch.device("cuda"))
    category_names = load_ss_lora(pipeline, args.ss_lora_ckpt)

    resolved_sampler = dict(pipeline.sparse_structure_sampler_params)

    latent_dir = args.dataset_dir / "ss_latents" / args.latent_name
    latent_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = latent_dir / "cache_config.json"
    manifest = json.loads(json.dumps({
        "version": 1,
        "pipeline": args.pipeline,
        "ss_checkpoint_sha256": file_sha256(args.ss_lora_ckpt),
        "view_index": args.view_index,
        "seed": args.seed,
        "sampler_params": resolved_sampler,
    }))
    if manifest_path.exists():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if current != manifest:
            raise RuntimeError(
                f"Cache settings do not match {manifest_path}. Use a new --latent-name."
            )
    elif any(latent_dir.glob("*.npz")):
        raise RuntimeError(f"Cached latents exist without a manifest under {latent_dir}")
    else:
        write_json(manifest_path, manifest)

    flow_model = pipeline.models["sparse_structure_flow_model"]
    expected_shape = (
        flow_model.in_channels,
        flow_model.resolution,
        flow_model.resolution,
        flow_model.resolution,
    )

    for sample_id in tqdm(ids, desc=f"Caching {args.dataset_dir.name} SS endpoints"):
        output_path = latent_dir / f"{sample_id}.npz"
        if output_path.exists():
            if not args.skip_existing:
                raise FileExistsError(
                    f"Cache file already exists: {output_path}. Pass --skip-existing to resume."
                )
            validate_latent(output_path, expected_shape, args.view_index, args.seed)
            continue

        image_path = conditioning_image_path(args.dataset_dir, sample_id, args.view_index)
        if not image_path.is_file():
            raise FileNotFoundError(f"Conditioning render not found: {image_path}")
        voxel_path = args.dataset_dir / "voxels" / f"{sample_id}.ply"
        if not voxel_path.is_file():
            raise FileNotFoundError(f"Ground-truth voxels not found: {voxel_path}")
        with Image.open(image_path) as image, torch.inference_mode():
            torch.manual_seed(args.seed)
            image = pipeline.preprocess_image(image)
            category = [categories[sample_id]] if category_names is not None else None
            cond = pipeline.get_cond([image], category=category)
            latent = pipeline.sample_sparse_structure_latent(
                cond,
                num_samples=1,
            )[0].float().cpu().numpy()

        temporary = output_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            mean=latent,
            view_index=np.int32(args.view_index),
            seed=np.int64(args.seed),
        )
        temporary.replace(output_path)
        validate_latent(output_path, expected_shape, args.view_index, args.seed)

    missing = [sample_id for sample_id in ids if not (latent_dir / f"{sample_id}.npz").is_file()]
    if missing:
        raise RuntimeError(f"Cache is incomplete; missing {len(missing)} endpoints")

    column = f"ss_latent_{args.latent_name}"
    metadata[column] = metadata["sha256"].astype(str).map(
        lambda sample_id: (latent_dir / f"{sample_id}.npz").is_file()
    )
    temporary_metadata = metadata_path.with_suffix(".csv.tmp")
    metadata.to_csv(temporary_metadata, index=False)
    temporary_metadata.replace(metadata_path)

    if args.limit is None:
        write_json(
            latent_dir / "cache_complete.json",
            {
                "num_endpoints": len(ids),
                "view_index": args.view_index,
                "seed": args.seed,
                "ss_checkpoint_sha256": manifest["ss_checkpoint_sha256"],
            },
        )

    print(f"Validated {len(ids)} cached endpoints in {latent_dir}")
    print(f"Updated metadata column: {column}")


if __name__ == "__main__":
    main()
