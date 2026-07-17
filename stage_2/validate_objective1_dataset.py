from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "test")
EXPECTED_CATEGORIES = {"car", "bus", "file_cabinet", "cabinet"}
EXPECTED_NUM_VIEWS = 40
SS_LATENT_MODEL = "ss_enc_conv3d_16l8_fp16"
SLAT_LATENT_MODEL = "dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16"
SS_LATENT_COLUMN = f"ss_latent_{SS_LATENT_MODEL}"
SLAT_LATENT_COLUMN = f"latent_{SLAT_LATENT_MODEL}"
REQUIRED_COLUMNS = {
    "sha256",
    "split",
    "category",
    "rendered",
    "cond_rendered",
    "voxelized",
    SS_LATENT_COLUMN,
    SLAT_LATENT_COLUMN,
}
TRUE_VALUES = {"1", "true", "yes"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a full non-cutout ShapeNet dataset for Objective 1."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    return parser.parse_args()


def is_true(value: Any) -> bool:
    return str(value).strip().lower() in TRUE_VALUES


def nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def read_metadata(path: Path, errors: list[str]) -> tuple[list[dict[str, str]], set[str]]:
    if not nonempty_file(path):
        errors.append(f"missing or empty metadata: {path}")
        return [], set()

    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            rows = list(reader)
    except Exception as exc:
        errors.append(f"cannot read metadata {path}: {exc}")
        return [], set()

    missing_columns = sorted(REQUIRED_COLUMNS - columns)
    if missing_columns:
        errors.append(f"{path}: missing columns {missing_columns}")
    return rows, columns


def read_transforms(path: Path, sample_label: str, errors: list[str]) -> list[dict[str, Any]]:
    if not nonempty_file(path):
        errors.append(f"{sample_label}: missing or empty {path}")
        return []

    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        errors.append(f"{sample_label}: cannot read {path}: {exc}")
        return []

    frames = payload.get("frames")
    if not isinstance(frames, list):
        errors.append(f"{sample_label}: {path} has no frames list")
        return []
    return frames


def validate_render_folder(
    folder: Path,
    sample_label: str,
    errors: list[str],
    *,
    require_mesh: bool,
) -> None:
    transforms_path = folder / "transforms.json"
    frames = read_transforms(transforms_path, sample_label, errors)

    numeric_pngs = (
        sorted(path for path in folder.glob("*.png") if path.stem.isdigit())
        if folder.is_dir()
        else []
    )
    numeric_png_names = {path.name for path in numeric_pngs}
    expected_png_names = {f"{index:03d}.png" for index in range(EXPECTED_NUM_VIEWS)}
    if numeric_png_names != expected_png_names:
        errors.append(
            f"{sample_label}: numeric PNG set in {folder} is incomplete or unexpected; "
            f"missing={sorted(expected_png_names - numeric_png_names)[:10]}, "
            f"extra={sorted(numeric_png_names - expected_png_names)[:10]}"
        )
    for image_path in numeric_pngs:
        if not nonempty_file(image_path):
            errors.append(f"{sample_label}: empty image {image_path}")

    if len(frames) != EXPECTED_NUM_VIEWS:
        errors.append(
            f"{sample_label}: expected {EXPECTED_NUM_VIEWS} frames in {transforms_path}, "
            f"found {len(frames)}"
        )

    referenced_png_names: list[str] = []
    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            errors.append(f"{sample_label}: frame {frame_index} is not an object")
            continue
        if is_true(frame.get("cutout", False)) or "cutout_axis" in frame:
            errors.append(f"{sample_label}: cutout frame found at index {frame_index}")

        file_path = frame.get("file_path")
        if not isinstance(file_path, str):
            errors.append(f"{sample_label}: frame {frame_index} has no file_path")
            continue
        relative_path = Path(file_path)
        if (
            relative_path.parent != Path(".")
            or relative_path.suffix.lower() != ".png"
            or not relative_path.stem.isdigit()
        ):
            errors.append(
                f"{sample_label}: frame {frame_index} does not reference a numeric PNG: {file_path}"
            )
            continue
        referenced_png_names.append(relative_path.name)
        if not nonempty_file(folder / relative_path):
            errors.append(f"{sample_label}: missing frame image {folder / relative_path}")

    if set(referenced_png_names) != expected_png_names or len(referenced_png_names) != EXPECTED_NUM_VIEWS:
        errors.append(
            f"{sample_label}: frame references do not match 000.png through "
            f"{EXPECTED_NUM_VIEWS - 1:03d}.png exactly once"
        )

    if require_mesh and not nonempty_file(folder / "mesh.ply"):
        errors.append(f"{sample_label}: missing or empty {folder / 'mesh.ply'}")


def validate_sample(
    split_dir: Path,
    split: str,
    row: dict[str, str],
    row_number: int,
    errors: list[str],
) -> str | None:
    sample_id = str(row.get("sha256", "")).strip()
    sample_label = f"{split} row {row_number} ({sample_id or 'missing id'})"
    if not sample_id:
        errors.append(f"{sample_label}: empty sha256")
        return None

    if str(row.get("split", "")).strip() != split:
        errors.append(
            f"{sample_label}: metadata split is {row.get('split')!r}, expected {split!r}"
        )

    category = str(row.get("category", "")).strip()
    if category not in EXPECTED_CATEGORIES:
        errors.append(f"{sample_label}: unexpected category {category!r}")

    for flag in ("rendered", "cond_rendered", "voxelized", SS_LATENT_COLUMN, SLAT_LATENT_COLUMN):
        if not is_true(row.get(flag, False)):
            errors.append(f"{sample_label}: metadata flag {flag} is not true")

    validate_render_folder(
        split_dir / "renders" / sample_id,
        f"{sample_label} ordinary render",
        errors,
        require_mesh=True,
    )
    validate_render_folder(
        split_dir / "renders_cond" / sample_id,
        f"{sample_label} conditioning render",
        errors,
        require_mesh=False,
    )

    required_files = (
        split_dir / "voxels" / f"{sample_id}.ply",
        split_dir / "ss_latents" / SS_LATENT_MODEL / f"{sample_id}.npz",
        split_dir / "latents" / SLAT_LATENT_MODEL / f"{sample_id}.npz",
    )
    for path in required_files:
        if not nonempty_file(path):
            errors.append(f"{sample_label}: missing or empty {path}")

    return sample_id


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    errors: list[str] = []
    ids_by_split: dict[str, set[str]] = {}

    print(f"Validating Objective 1 dataset: {dataset_root}")
    for split in SPLITS:
        split_dir = dataset_root / split
        rows, _columns = read_metadata(split_dir / "metadata.csv", errors)
        ids: list[str] = []
        categories: Counter[str] = Counter()

        for row_number, row in enumerate(rows, start=2):
            category = str(row.get("category", "")).strip()
            if category:
                categories[category] += 1
            sample_id = validate_sample(split_dir, split, row, row_number, errors)
            if sample_id is not None:
                ids.append(sample_id)

        duplicate_ids = sorted(sample_id for sample_id, count in Counter(ids).items() if count > 1)
        if duplicate_ids:
            errors.append(f"{split}: duplicate sha256 values: {duplicate_ids[:10]}")

        present_categories = set(categories)
        if present_categories != EXPECTED_CATEGORIES:
            errors.append(
                f"{split}: category set is {sorted(present_categories)}, "
                f"expected {sorted(EXPECTED_CATEGORIES)}"
            )

        ids_by_split[split] = set(ids)
        category_summary = ", ".join(
            f"{category}={categories.get(category, 0)}"
            for category in sorted(EXPECTED_CATEGORIES)
        )
        print(f"  {split}: rows={len(rows)}, unique_ids={len(ids_by_split[split])}, {category_summary}")

    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            overlap = sorted(ids_by_split.get(left, set()) & ids_by_split.get(right, set()))
            if overlap:
                errors.append(
                    f"split leakage between {left} and {right}: {len(overlap)} IDs; "
                    f"examples={overlap[:10]}"
                )

    if errors:
        print(f"\nFAILED: found {len(errors)} validation error(s).")
        for error in errors[:100]:
            print(f"  - {error}")
        if len(errors) > 100:
            print(f"  ... {len(errors) - 100} additional error(s) omitted")
        raise SystemExit(1)

    print("\nPASSED: train/val/test are disjoint and all required Objective 1 artifacts are complete.")


if __name__ == "__main__":
    main()
