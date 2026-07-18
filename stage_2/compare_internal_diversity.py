#!/usr/bin/env python3
import argparse
import csv
import itertools
from pathlib import Path

import numpy as np

from compare_internals import interior, read_voxels


def jaccard(a, b):
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def read_ids(path: Path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def read_categories(path: Path):
    with path.open(newline="") as file:
        return {row["sha256"]: row["category"] for row in csv.DictReader(file)}


def main():
    parser = argparse.ArgumentParser(description="Measure internal diversity across random seeds.")
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--ids-file", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-sample-output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=64)
    args = parser.parse_args()

    ids = read_ids(args.ids_file)
    categories = read_categories(args.metadata)
    per_sample = []

    for method in args.methods:
        if args.seeds is None:
            seed_dirs = sorted(path for path in (args.pred_root / method).glob("seed_*") if path.is_dir())
        else:
            seed_dirs = [args.pred_root / method / f"seed_{seed}" for seed in args.seeds]
            missing_dirs = [path for path in seed_dirs if not path.is_dir()]
            if missing_dirs:
                raise FileNotFoundError(f"Missing seed directories: {missing_dirs}")
        if len(seed_dirs) < 2:
            raise ValueError(f"Method {method} needs at least two seed directories")

        for sample_id in ids:
            voxel_sets = []
            for seed_dir in seed_dirs:
                path = seed_dir / "voxels" / f"{sample_id}.ply"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing prediction: {path}")
                voxels = read_voxels(path, args.resolution)
                internal_voxels = interior(voxels)
                voxel_sets.append((internal_voxels, voxels - internal_voxels))

            pairs = list(itertools.combinations(voxel_sets, 2))
            internal_iou = np.mean([jaccard(a[0], b[0]) for a, b in pairs])
            exterior_iou = np.mean([jaccard(a[1], b[1]) for a, b in pairs])
            per_sample.append({
                "method": method,
                "sample_id": sample_id,
                "category": categories[sample_id],
                "num_seeds": len(seed_dirs),
                "internal_diversity": 1.0 - float(internal_iou),
                "exterior_consistency": float(exterior_iou),
            })

    summary = []
    for method in args.methods:
        method_rows = [row for row in per_sample if row["method"] == method]
        for category in ["all"] + sorted({row["category"] for row in method_rows}):
            rows = method_rows if category == "all" else [row for row in method_rows if row["category"] == category]
            summary.append({
                "method": method,
                "category": category,
                "num_samples": len(rows),
                "num_seeds": rows[0]["num_seeds"],
                "internal_diversity": float(np.mean([row["internal_diversity"] for row in rows])),
                "exterior_consistency": float(np.mean([row["exterior_consistency"] for row in rows])),
            })

    args.per_sample_output.parent.mkdir(parents=True, exist_ok=True)
    with args.per_sample_output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(per_sample[0]))
        writer.writeheader()
        writer.writerows(per_sample)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    for row in summary:
        if row["category"] == "all":
            print(
                f"{row['method']}: internal diversity={row['internal_diversity']:.4f}, "
                f"exterior consistency={row['exterior_consistency']:.4f}"
            )


if __name__ == "__main__":
    main()
