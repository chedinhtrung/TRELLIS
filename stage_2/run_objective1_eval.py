#!/usr/bin/env python3
import argparse
import csv
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_METHOD = "base_flows__base_decoder"
ALL_METHODS = [
    BASE_METHOD,
    "lora_flows__base_decoder",
    "base_flows__lora_decoder",
    "lora_flows__lora_decoder",
]


def read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No samples found in {path}")
    return rows


def select_ids(rows: list[dict[str, str]], samples_per_category: int) -> list[str]:
    if samples_per_category <= 0:
        return [row["sha256"] for row in rows]

    counts = {}
    selected = []
    for row in rows:
        category = row["category"]
        if counts.get(category, 0) < samples_per_category:
            selected.append(row["sha256"])
            counts[category] = counts.get(category, 0) + 1
    return selected


def run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the controlled Objective 1 evaluation.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ss-lora-ckpt", type=Path)
    parser.add_argument("--slat-lora-ckpt", type=Path)
    parser.add_argument("--decoder-lora-ckpt", type=Path)
    parser.add_argument("--view-index", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--samples-per-category", type=int, default=0)
    args = parser.parse_args()

    if args.view_index < 0:
        parser.error("--view-index must be non-negative")
    if args.samples_per_category < 0:
        parser.error("--samples-per-category must be non-negative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")

    checkpoints = [args.ss_lora_ckpt, args.slat_lora_ckpt, args.decoder_lora_ckpt]
    if any(checkpoints) and not all(checkpoints):
        parser.error("Provide all three LoRA checkpoints, or none for the base-model evaluation")
    for checkpoint in checkpoints:
        if checkpoint is not None and not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    metadata_path = args.dataset_dir / "metadata.csv"
    rows = read_metadata(metadata_path)
    ids = select_ids(rows, args.samples_per_category)
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample IDs in selected metadata rows")

    image_name = f"{args.view_index:03d}.png"
    for sample_id in ids:
        image_path = args.dataset_dir / "renders_cond" / sample_id / image_name
        voxel_path = args.dataset_dir / "voxels" / f"{sample_id}.ply"
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing conditioning image: {image_path}")
        if not voxel_path.is_file():
            raise FileNotFoundError(f"Missing ground-truth voxels: {voxel_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ids_file = args.output_dir / "selected_ids.txt"
    ids_file.write_text("\n".join(ids) + "\n")
    methods = ALL_METHODS if all(checkpoints) else [BASE_METHOD]

    for seed in args.seeds:
        for method in methods:
            command = [
                sys.executable,
                str(REPO_ROOT / "stage_2/export_full_pipeline_voxels.py"),
                "--dataset-dir", str(args.dataset_dir),
                "--output-dir", str(args.output_dir / "predictions" / method / f"seed_{seed}"),
                "--ids-file", str(ids_file),
                "--view-index", str(args.view_index),
                "--seed", str(seed),
            ]
            if method in {"lora_flows__base_decoder", "lora_flows__lora_decoder"}:
                command += [
                    "--ss-lora-ckpt", str(args.ss_lora_ckpt),
                    "--slat-lora-ckpt", str(args.slat_lora_ckpt),
                ]
            if method in {"base_flows__lora_decoder", "lora_flows__lora_decoder"}:
                command += ["--decoder-lora-ckpt", str(args.decoder_lora_ckpt)]
            run(command)

    run([
        sys.executable,
        str(REPO_ROOT / "stage_2/compare_internals.py"),
        "--gt-voxels", str(args.dataset_dir / "voxels"),
        "--pred-root", str(args.output_dir / "predictions"),
        "--metadata", str(metadata_path),
        "--ids-file", str(ids_file),
        "--methods", *methods,
        "--output", str(args.output_dir / "metrics/summary.csv"),
        "--per-sample-output", str(args.output_dir / "metrics/per_sample.csv"),
    ])


if __name__ == "__main__":
    main()
