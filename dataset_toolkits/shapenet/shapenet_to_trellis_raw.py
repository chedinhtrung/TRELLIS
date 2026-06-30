from __future__ import annotations

import argparse
import csv
import shutil
import os
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


def collect_rows(shapenet_root: Path, categories: list[str], limit: int, outdir: Path) -> list[dict]:
    raw_dir = outdir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
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
            link_dir = raw_dir / sample_id
            rel_target = os.path.relpath(object_dir, start=raw_dir)
            make_symlink(Path(rel_target), link_dir)

            rows.append(
                {
                    "sha256": sample_id,
                    "file_identifier": sample_id,
                    "local_path": str((link_dir / "models" / obj_path.name).relative_to(outdir)),
                    "category": category,
                    "object_id": object_id,
                    "source_obj": str((link_dir / "models" / obj_path.name).relative_to(outdir)),
                }
            )
            selected += 1

    return rows


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.outdir / "raw"
    if args.overwrite:
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
        metadata_path = args.outdir / "metadata.csv"
        if metadata_path.exists():
            metadata_path.unlink()
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(args.shapenet_root, args.categories, args.limit, args.outdir)
    if not rows:
        raise RuntimeError("No ShapeNet objects were selected.")

    metadata = pd.DataFrame(rows)
    metadata.to_csv(args.outdir / "metadata.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"Wrote {len(metadata)} rows to {args.outdir / 'metadata.csv'}")
    print(f"Created {len(rows)} symlinks under {args.outdir / 'raw'}")


if __name__ == "__main__":
    main()
