from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import (
    CATEGORIES,
    DEFAULT_DATASET_DIR,
    DEFAULT_SHAPENET_ROOT,
    ensure_dir,
    read_score_rows,
    stable_id,
    write_metadata,
    write_splits,
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line options for building the small Stage 1 subset.

    The defaults point to the local ShapeNet download and to the converted
    ShapeNetInternals_small dataset folder.  --per-category controls
    how many objects are selected from each ShapeNet category, and
    --overwrite allows regenerating an existing metadata file.
    """
    parser = argparse.ArgumentParser(description="Prepare a small ShapeNet-internals TRELLIS metadata subset.")
    parser.add_argument("--shapenet-root", type=Path, default=DEFAULT_SHAPENET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--per-category", type=int, default=3, help="Objects to select per category.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def select_category(shapenet_root: Path, category: str, per_category: int) -> list[dict]:
    """
    Select the highest-scoring usable objects from one ShapeNet category.
    """
    category_root = shapenet_root / category
    score_rows = read_score_rows(shapenet_root, category)
    candidates = []
    for object_dir in sorted(category_root.iterdir()):
        # Each ShapeNet object is stored as ShapeNet/<category>/<object_id>/.
        if not object_dir.is_dir():
            continue
        object_id = object_dir.name
        model_dir = object_dir / "models"
        obj_path = model_dir / "model_normalized.obj"
        surface_path = model_dir / "model_normalized.surface.binvox"
        solid_path = model_dir / "model_normalized.solid.binvox"
        if not obj_path.exists() or not surface_path.exists():
            continue
        score = score_rows.get(object_id, {})
        # Missing score values are treated as zero so unscored objects can
        # still participate, but scored internal-rich objects sort first.
        complexity = float(score.get("complexity_score", 0.0) or 0.0)
        inner_face_count = float(score.get("inner_face_count", 0.0) or 0.0)
        inner_edge_count = float(score.get("inner_edge_count", 0.0) or 0.0)
        inner_face_ratio = float(score.get("inner_face_ratio", 0.0) or 0.0)
        inner_edge_ratio = float(score.get("inner_edge_ratio", 0.0) or 0.0)
        candidates.append({
            "sha256": stable_id(category, object_id),
            "file_identifier": object_id,
            "local_path": str(obj_path),
            "category": category,
            "shapenet_id": object_id,
            "source_obj": str(obj_path),
            "surface_binvox": str(surface_path),
            "solid_binvox": str(solid_path) if solid_path.exists() else "",
            "has_solid_binvox": bool(solid_path.exists()),
            "captions": "[]",
            "aesthetic_score": 5.0,
            "inner_face_count": inner_face_count,
            "inner_edge_count": inner_edge_count,
            "inner_face_ratio": inner_face_ratio,
            "inner_edge_ratio": inner_edge_ratio,
            "complexity_score": complexity,
            "rendered": False,
            "voxelized": False,
            "num_voxels": 0,
            "cond_rendered": False,
            "feature_dinov2_vitl14_reg": False,
            "ss_latent_ss_enc_conv3d_16l8_fp16": False,
            "latent_dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16": False,
        })
    # Sort by internal-geometry evidence first, then by id for deterministic
    # tie-breaking.  The reverse sort puts the strongest candidates first.
    candidates.sort(key=lambda row: (row["complexity_score"], row["inner_face_count"], row["sha256"]), reverse=True)
    return candidates[:per_category]


def main() -> None:
    """
    Create metadata, split files, and a selected-id list for Stage 1.
    """
    args = parse_args()
    metadata_path = args.output_dir / "metadata.csv"
    if metadata_path.exists() and not args.overwrite:
        print(f"metadata.csv already exists, skipping: {metadata_path}")
        return

    ensure_dir(args.output_dir)
    rows = []
    for category in CATEGORIES:
        # Keep category coverage balanced by taking the same number of objects
        # from each category rather than simply choosing the global top scores.
        selected = select_category(args.shapenet_root, category, args.per_category)
        if len(selected) < args.per_category:
            print(f"Warning: selected only {len(selected)} objects for category {category}")
        rows.extend(selected)

    if not rows:
        raise RuntimeError(f"No ShapeNet objects selected from {args.shapenet_root}")

    ids = [row["sha256"] for row in rows]
    write_splits(args.output_dir, ids)
    split_lookup = {}
    # Re-read the split files so the split column matches exactly what was
    # written to disk and what TRELLIS preprocessing will consume.
    for split_name in ("train", "val", "test"):
        split_file = args.output_dir / "splits" / f"{split_name}.txt"
        for sample_id in split_file.read_text(encoding="utf-8").splitlines():
            split_lookup[sample_id] = split_name
    for row in rows:
        row["split"] = split_lookup.get(row["sha256"], "train")

    metadata = pd.DataFrame(rows)
    write_metadata(args.output_dir, metadata)
    (args.output_dir / "selected_ids.txt").write_text("".join(f"{sample_id}\n" for sample_id in ids), encoding="utf-8")

    print(f"Wrote {len(metadata)} samples to {metadata_path}")
    print(metadata[["sha256", "category", "shapenet_id", "split", "complexity_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
