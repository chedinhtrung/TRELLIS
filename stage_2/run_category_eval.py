#!/usr/bin/env python3
import argparse
import csv
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def read_metadata(path: Path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def select_ids(rows, samples_per_category):
    selected = []
    counts = {}
    for row in rows:
        category = row["category"]
        if counts.get(category, 0) < samples_per_category:
            selected.append(row["sha256"])
            counts[category] = counts.get(category, 0) + 1
    if any(count < samples_per_category for count in counts.values()):
        raise ValueError("The test set does not contain enough samples in every category")
    return selected


def run(command):
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Run the category-conditioning diversity evaluation.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-ss-lora-ckpt", type=Path, required=True)
    parser.add_argument("--baseline-slat-lora-ckpt", type=Path, required=True)
    parser.add_argument("--category-ss-lora-ckpt", type=Path, required=True)
    parser.add_argument("--category-slat-lora-ckpt", type=Path, required=True)
    parser.add_argument("--decoder-lora-ckpt", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--samples-per-category", type=int, default=5)
    parser.add_argument("--view-index", type=int, default=0)
    args = parser.parse_args()

    if args.samples_per_category <= 0:
        parser.error("--samples-per-category must be positive")
    if len(args.seeds) < 2 or len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must contain at least two unique values")
    if args.view_index < 0:
        parser.error("--view-index must be non-negative")
    checkpoints = [
        args.baseline_ss_lora_ckpt,
        args.baseline_slat_lora_ckpt,
        args.category_ss_lora_ckpt,
        args.category_slat_lora_ckpt,
        args.decoder_lora_ckpt,
    ]
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    metadata_path = args.dataset_dir / "metadata.csv"
    ids = select_ids(read_metadata(metadata_path), args.samples_per_category)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ids_file = args.output_dir / "selected_ids.txt"
    ids_file.write_text("\n".join(ids) + "\n")

    methods = [
        ("no_category", args.baseline_ss_lora_ckpt, args.baseline_slat_lora_ckpt),
        ("category", args.category_ss_lora_ckpt, args.category_slat_lora_ckpt),
    ]
    for method, ss_checkpoint, slat_checkpoint in methods:
        for seed in args.seeds:
            run([
                sys.executable,
                str(REPO_ROOT / "stage_2/export_full_pipeline_voxels.py"),
                "--dataset-dir", str(args.dataset_dir),
                "--output-dir", str(args.output_dir / "predictions" / method / f"seed_{seed}"),
                "--ids-file", str(ids_file),
                "--ss-lora-ckpt", str(ss_checkpoint),
                "--slat-lora-ckpt", str(slat_checkpoint),
                "--decoder-lora-ckpt", str(args.decoder_lora_ckpt),
                "--view-index", str(args.view_index),
                "--seed", str(seed),
                "--skip-existing",
            ])

    run([
        sys.executable,
        str(REPO_ROOT / "stage_2/compare_internal_diversity.py"),
        "--pred-root", str(args.output_dir / "predictions"),
        "--metadata", str(metadata_path),
        "--ids-file", str(ids_file),
        "--methods", "no_category", "category",
        "--seeds", *[str(seed) for seed in args.seeds],
        "--output", str(args.output_dir / "metrics/summary.csv"),
        "--per-sample-output", str(args.output_dir / "metrics/per_sample.csv"),
    ])


if __name__ == "__main__":
    main()
