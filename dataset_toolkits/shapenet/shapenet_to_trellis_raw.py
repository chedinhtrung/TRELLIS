from __future__ import annotations

import argparse
import csv
import shutil
import os
import math
from pathlib import Path

import pandas as pd
from tqdm import tqdm

"""
    Copy (actually make symlinks for efficiency) a subset from ShapeNet to create a new dataset with the same shape 
    that TRELLIS expects.
    
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a minimal TRELLIS-style dataset from ShapeNet using symlinks to raw OBJ files."
    )
    parser.add_argument(
        "--shapenet-root",
        type=Path,
        required=True,
        help="Root directory of the ShapeNet dataset.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        required=True,
        help="List of ShapeNet category folder names to include.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        required=True,
        help="Maximum number of objects per category.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Output directory for the TRELLIS-style dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, clear the existing output directory contents before rebuilding.",
    )
    parser.add_argument(
        "--train",
        type=float,
        default=0.8,
        help="Fraction of selected objects to write under <outdir>/train.",
    )
    parser.add_argument(
        "--val",
        type=float,
        default=0.1,
        help="Fraction of selected objects to write under <outdir>/val.",
    )
    parser.add_argument(
        "--test",
        type=float,
        default=0.1,
        help="Fraction of selected objects to write under <outdir>/test.",
    )
    return parser.parse_args()


def stable_sample_id(category: str, object_id: str) -> str:
    return f"{category}__{object_id}"


def pick_obj_file(models_dir: Path) -> Path | None:
    preferred = models_dir / "model_normalized.obj"
    if preferred.exists():
        return preferred
    obj_files = sorted(models_dir.glob("*.obj"))
    return obj_files[0] if obj_files else None


def make_symlink(target: Path, link_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(target)


def validate_split_ratios(train: float, val: float, test: float) -> None:
    ratios = {"train": train, "val": val, "test": test}
    for name, ratio in ratios.items():
        if ratio < 0:
            raise ValueError(f"--{name} must be non-negative, got {ratio}")
    total = train + val + test
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"--train + --val + --test must equal 1.0, got {total}")


def collect_samples(shapenet_root: Path, categories: list[str], limit: int) -> list[dict]:
    samples: list[dict] = []
    for category in tqdm(categories, desc="Categories"):
        category_dir = shapenet_root / category
        if not category_dir.exists():
            raise FileNotFoundError(f"Category directory not found: {category_dir}")

        selected = 0
        for object_dir in tqdm(sorted(category_dir.iterdir()), desc=f"{category}", leave=False):
            if selected >= limit:
                break
            if not object_dir.is_dir():
                continue

            models_dir = object_dir / "models"
            obj_path = pick_obj_file(models_dir)
            if obj_path is None:
                continue

            object_id = object_dir.name
            sample_id = stable_sample_id(category, object_id)
            samples.append(
                {
                    "sha256": sample_id,
                    "file_identifier": sample_id,
                    "category": category,
                    "object_id": object_id,
                    "object_dir": object_dir,
                    "obj_name": obj_path.name,
                }
            )
            selected += 1

    return samples


def split_samples(samples: list[dict], train: float, val: float) -> dict[str, list[dict]]:
    total = len(samples)
    train_count = int(total * train)
    val_count = int(total * val)
    return {
        "train": samples[:train_count],
        "val": samples[train_count:train_count + val_count],
        "test": samples[train_count + val_count:],
    }


def write_split(split_dir: Path, split_name: str, samples: list[dict], overwrite: bool) -> None:
    if overwrite and split_dir.exists():
        shutil.rmtree(split_dir)

    raw_dir = split_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for sample in samples:
        link_dir = raw_dir / sample["sha256"]
        rel_target = os.path.relpath(sample["object_dir"], start=raw_dir)
        make_symlink(Path(rel_target), link_dir)

        local_path = link_dir / "models" / sample["obj_name"]
        rows.append(
            {
                "sha256": sample["sha256"],
                "file_identifier": sample["file_identifier"],
                "local_path": str(local_path.relative_to(split_dir)),
                "category": sample["category"],
                "object_id": sample["object_id"],
                "split": split_name,
                "source_obj": str(local_path.relative_to(split_dir)),
            }
        )

    metadata = pd.DataFrame(rows)
    metadata.to_csv(split_dir / "metadata.csv", index=False, quoting=csv.QUOTE_MINIMAL)


def main() -> None:
    args = parse_args()
    validate_split_ratios(args.train, args.val, args.test)

    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for split_name in ("train", "val", "test"):
            split_dir = args.outdir / split_name
            if split_dir.exists():
                shutil.rmtree(split_dir)

    samples = collect_samples(args.shapenet_root, args.categories, args.limit)
    if not samples:
        raise RuntimeError("No ShapeNet objects were selected.")

    splits = split_samples(samples, args.train, args.val)
    for split_name, split_samples_ in splits.items():
        write_split(args.outdir / split_name, split_name, split_samples_, args.overwrite)

    print(f"Selected {len(samples)} ShapeNet objects")
    for split_name, split_samples_ in splits.items():
        split_dir = args.outdir / split_name
        print(f"Wrote {len(split_samples_)} rows to {split_dir / 'metadata.csv'}")
        print(f"Created {len(split_samples_)} symlinks under {split_dir / 'raw'}")


if __name__ == "__main__":
    main()
