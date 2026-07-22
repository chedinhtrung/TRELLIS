#!/usr/bin/env python3
"""Select a deterministic category-balanced pilot, or every metadata row."""

import argparse
import csv
from pathlib import Path


CATEGORIES = ("car", "bus", "cabinet", "file_cabinet")
BASE_LATENT_COLUMN = "latent_dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16"


def is_true(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=5)
    parser.add_argument("--max-num-voxels", type=int, default=32768)
    args = parser.parse_args()

    if args.per_category < 0:
        parser.error("--per-category cannot be negative")
    if args.max_num_voxels <= 0:
        parser.error("--max-num-voxels must be positive")

    with args.metadata.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if args.per_category == 0:
        ids = [str(row.get("sha256", "")).strip() for row in rows]
        if not ids or any(not sample_id for sample_id in ids):
            raise ValueError("Metadata contains no samples or an empty sha256")
        if len(ids) != len(set(ids)):
            raise ValueError("Metadata IDs are not unique")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(ids) + "\n", encoding="utf-8")
        print(f"Wrote all {len(ids)} metadata IDs to {args.output}")
        return

    selected: dict[str, list[str]] = {category: [] for category in CATEGORIES}
    for row in rows:
        category = str(row.get("category", "")).strip()
        sample_id = str(row.get("sha256", "")).strip()
        try:
            num_voxels = int(float(row.get("num_voxels", 0)))
        except (TypeError, ValueError):
            continue
        ready = is_true(row.get("voxelized")) and is_true(row.get(BASE_LATENT_COLUMN))
        if (
            category in selected
            and sample_id
            and ready
            and num_voxels <= args.max_num_voxels
            and len(selected[category]) < args.per_category
        ):
            selected[category].append(sample_id)

    missing = {
        category: args.per_category - len(ids)
        for category, ids in selected.items()
        if len(ids) < args.per_category
    }
    if missing:
        raise ValueError(f"Not enough samples in metadata: {missing}")

    ids = [sample_id for category in CATEGORIES for sample_id in selected[category]]
    if len(ids) != len(set(ids)):
        raise ValueError("Selected IDs are not unique")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(ids) + "\n", encoding="utf-8")
    print(f"Wrote {len(ids)} IDs ({args.per_category} per category) to {args.output}")


if __name__ == "__main__":
    main()
