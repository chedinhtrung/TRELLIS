#!/usr/bin/env python3
"""Prepare a 16-train/4-validation decoder pilot from the 20 target pilot IDs."""

import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np


CATEGORIES = ("car", "bus", "cabinet", "file_cabinet")
DEFAULT_LATENT_NAME = "dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16_internal_v1"


def read_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(ids) != 20:
        raise ValueError(f"Expected exactly 20 pilot IDs in {path}, found {len(ids)}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate pilot IDs found in {path}")
    return ids


def read_metadata(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    required = {"sha256", "category", "aesthetic_score", "num_voxels"}
    missing = sorted(required - set(fieldnames))
    if missing:
        raise ValueError(f"{path} is missing columns required by Slat2RenderGeo: {missing}")
    return rows, fieldnames


def validate_latent(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Internal SLAT target not found: {path}")
    with np.load(path) as data:
        if "coords" not in data or "feats" not in data:
            raise ValueError(f"Internal SLAT target must contain coords and feats: {path}")
        coords = np.asarray(data["coords"])
        feats = np.asarray(data["feats"])
    if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) == 0:
        raise ValueError(f"Invalid coordinates in {path}: {coords.shape}")
    if feats.ndim != 2 or len(feats) != len(coords):
        raise ValueError(f"Invalid features in {path}: {feats.shape}")
    if not np.isfinite(feats).all():
        raise ValueError(f"Non-finite features in {path}")


def validate_renders(render_dir: Path) -> None:
    mesh_path = render_dir / "mesh.ply"
    transforms_path = render_dir / "transforms.json"
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Training mesh not found: {mesh_path}")
    if not transforms_path.is_file():
        raise FileNotFoundError(f"Render metadata not found: {transforms_path}")
    with transforms_path.open(encoding="utf-8") as file:
        transforms = json.load(file)
    frames = transforms.get("frames", [])
    if not frames:
        raise ValueError(f"No render frames listed in {transforms_path}")
    for frame in frames:
        file_path = frame.get("file_path")
        if not file_path or not (render_dir / file_path).is_file():
            raise FileNotFoundError(f"Render frame not found: {render_dir / str(file_path)}")


def validate_conditioning_renders(render_dir: Path) -> None:
    transforms_path = render_dir / "transforms.json"
    if not transforms_path.is_file():
        raise FileNotFoundError(f"Conditioning-render metadata not found: {transforms_path}")
    with transforms_path.open(encoding="utf-8") as file:
        frames = json.load(file).get("frames", [])
    if not frames:
        raise ValueError(f"No conditioning renders listed in {transforms_path}")
    for frame in frames:
        file_path = frame.get("file_path")
        if not file_path or not (render_dir / file_path).is_file():
            raise FileNotFoundError(
                f"Conditioning render not found: {render_dir / str(file_path)}"
            )


def ensure_symlink(path: Path, target: Path) -> None:
    target = target.resolve()
    if path.is_symlink():
        if path.resolve() != target:
            raise FileExistsError(f"Refusing to replace existing symlink {path} -> {path.readlink()}")
        return
    if path.exists():
        raise FileExistsError(f"Refusing to replace existing path: {path}")
    path.symlink_to(target, target_is_directory=True)


def write_generated_file(path: Path, content: str) -> None:
    """Permit identical reruns without overwriting a different existing file."""
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Refusing to overwrite existing generated file: {path}")
        return
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--pilot_ids", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--latent_name", default=DEFAULT_LATENT_NAME)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    metadata_rows, fieldnames = read_metadata(data_dir / "metadata.csv")
    rows_by_id = {str(row["sha256"]).strip(): row for row in metadata_rows}
    if len(rows_by_id) != len(metadata_rows):
        raise ValueError(f"Duplicate sha256 values found in {data_dir / 'metadata.csv'}")

    pilot_ids = read_ids(args.pilot_ids)
    missing_metadata = [sample_id for sample_id in pilot_ids if sample_id not in rows_by_id]
    if missing_metadata:
        raise ValueError(f"Pilot IDs missing from metadata: {missing_metadata}")

    by_category = {category: [] for category in CATEGORIES}
    for sample_id in pilot_ids:
        category = str(rows_by_id[sample_id]["category"]).strip()
        if category not in by_category:
            raise ValueError(f"Unexpected category for {sample_id}: {category}")
        by_category[category].append(sample_id)
        validate_latent(data_dir / "latents" / args.latent_name / f"{sample_id}.npz")
        validate_renders(data_dir / "renders" / sample_id)
        validate_conditioning_renders(data_dir / "renders_cond" / sample_id)

    wrong_counts = {category: len(ids) for category, ids in by_category.items() if len(ids) != 5}
    if wrong_counts:
        raise ValueError(f"Expected 5 pilot IDs per category, got: {wrong_counts}")

    train_ids = [sample_id for category in CATEGORIES for sample_id in by_category[category][:4]]
    val_ids = [by_category[category][4] for category in CATEGORIES]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_root = args.output_dir / "train"
    training_root.mkdir(parents=True, exist_ok=True)
    ensure_symlink(training_root / "renders", data_dir / "renders")
    ensure_symlink(training_root / "renders_cond", data_dir / "renders_cond")
    ensure_symlink(training_root / "latents", data_dir / "latents")

    latent_column = f"latent_{args.latent_name}"
    if latent_column not in fieldnames:
        fieldnames.append(latent_column)
    train_rows = []
    for sample_id in train_ids:
        row = dict(rows_by_id[sample_id])
        row[latent_column] = "True"
        train_rows.append(row)

    metadata_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(metadata_buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(train_rows)
    write_generated_file(training_root / "metadata.csv", metadata_buffer.getvalue())

    write_generated_file(args.output_dir / "train_ids.txt", "\n".join(train_ids) + "\n")
    write_generated_file(args.output_dir / "val_ids.txt", "\n".join(val_ids) + "\n")

    print(f"Training IDs: {len(train_ids)} ({args.output_dir / 'train_ids.txt'})")
    print(f"Validation IDs: {len(val_ids)} ({args.output_dir / 'val_ids.txt'})")
    print(f"Training root: {training_root}")


if __name__ == "__main__":
    main()
