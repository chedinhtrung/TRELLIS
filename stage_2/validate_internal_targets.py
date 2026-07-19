#!/usr/bin/env python3
"""Validate the 20-object pilot of interior-aware SLAT targets."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


BASE_LATENT_NAME = "dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16"
OUTPUT_LATENT_NAME = f"{BASE_LATENT_NAME}_internal_v1"


def read_categories(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as file:
        return {
            str(row["sha256"]).strip(): str(row["category"]).strip()
            for row in csv.DictReader(file)
        }


def write_coverage_ply(
    path: Path,
    coords: np.ndarray,
    route: np.ndarray,
    view_count: np.ndarray,
) -> None:
    """Write a colored point cloud for quick manual coverage inspection."""
    points = (np.asarray(coords, dtype=np.float32) + 0.5) / 64.0 - 0.5
    colors = np.full((len(points), 3), 160, dtype=np.uint8)
    colors[route & (view_count == 0)] = (220, 40, 40)
    colors[route & (view_count == 1)] = (255, 165, 0)
    colors[route & (view_count >= 2)] = (40, 200, 80)
    with path.open("w", encoding="utf-8") as file:
        file.write("ply\nformat ascii 1.0\n")
        file.write(f"element vertex {len(points)}\n")
        file.write("property float x\nproperty float y\nproperty float z\n")
        file.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        file.write("end_header\n")
        for point, color in zip(points, colors):
            file.write(
                f"{point[0]:.7f} {point[1]:.7f} {point[2]:.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def read_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not ids:
        raise ValueError(f"No instance IDs found in {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate instance IDs found in {path}")
    return ids


def interior_mask(coords: np.ndarray, margin: int) -> np.ndarray:
    """Match compare_internals.py: behind both extrema along every axis line."""
    coords = np.asarray(coords, dtype=np.int32)
    mask = np.ones(len(coords), dtype=bool)
    for axis in range(3):
        other = [index for index in range(3) if index != axis]
        groups: dict[tuple[int, int], list[int]] = {}
        for index, coord in enumerate(coords):
            key = (int(coord[other[0]]), int(coord[other[1]]))
            groups.setdefault(key, []).append(index)
        for indices in groups.values():
            values = coords[indices, axis]
            keep = (values - values.min() >= margin) & (values.max() - values >= margin)
            mask[np.asarray(indices)] &= keep
    return mask


def load_npz(path: Path, required: tuple[str, ...]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"{path} is missing arrays: {missing}")
        return {name: np.asarray(data[name]) for name in required}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--route_margin", type=int, default=2)
    parser.add_argument("--expected_count", type=int, default=20, help="Use 0 to disable this check")
    parser.add_argument("--min_coverage", type=float, default=0.90)
    parser.add_argument("--min_two_view_coverage", type=float, default=0.70)
    parser.add_argument("--min_category_coverage", type=float, default=0.80)
    parser.add_argument("--outside_atol", type=float, default=0.0)
    parser.add_argument("--base_latent_name", default=BASE_LATENT_NAME)
    parser.add_argument("--output_latent_name", default=OUTPUT_LATENT_NAME)
    args = parser.parse_args()

    if args.route_margin < 1:
        parser.error("--route_margin must be at least 1")
    if args.expected_count < 0:
        parser.error("--expected_count must be non-negative")
    if not all(
        0 <= value <= 1
        for value in (
            args.min_coverage,
            args.min_two_view_coverage,
            args.min_category_coverage,
        )
    ):
        parser.error("coverage thresholds must be between 0 and 1")
    if args.outside_atol < 0:
        parser.error("--outside_atol must be non-negative")

    ids = read_ids(args.instances)
    base_dir = args.data_dir / "latents" / args.base_latent_name
    target_dir = args.data_dir / "latents" / args.output_latent_name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = args.output_dir / "coverage_voxels"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    categories = read_categories(args.data_dir / "metadata.csv")

    rows = []
    invariant_failures = []
    total_route = 0
    total_once = 0
    total_twice = 0
    total_changed = 0
    total_inside_l2 = 0.0
    category_totals: dict[str, list[int]] = {}

    if args.expected_count and len(ids) != args.expected_count:
        invariant_failures.append(
            f"Expected {args.expected_count} pilot IDs but received {len(ids)}"
        )

    for sample_id in ids:
        row = {
            "sample_id": sample_id,
            "category": categories.get(sample_id, ""),
            "status": "pass",
            "num_voxels": 0,
            "route_voxels": 0,
            "covered_once": 0,
            "covered_twice": 0,
            "coverage": 0.0,
            "two_view_coverage": 0.0,
            "max_outside_delta": 0.0,
            "max_inside_delta": 0.0,
            "mean_inside_l2": 0.0,
            "changed_inside_voxels": 0,
            "error": "",
        }
        try:
            base = load_npz(base_dir / f"{sample_id}.npz", ("coords", "feats"))
            target = load_npz(
                target_dir / f"{sample_id}.npz",
                ("coords", "feats", "route_mask", "cut_view_count"),
            )
            old_coords = np.asarray(base["coords"])
            old_feats = np.asarray(base["feats"], dtype=np.float32)
            new_coords = np.asarray(target["coords"])
            new_feats = np.asarray(target["feats"], dtype=np.float32)
            saved_route = np.asarray(target["route_mask"]).reshape(-1).astype(bool)
            view_count = np.asarray(target["cut_view_count"]).reshape(-1)

            if not np.array_equal(old_coords, new_coords):
                raise ValueError("base and target coordinates are not exactly aligned")
            if old_feats.shape != new_feats.shape or len(new_feats) != len(new_coords):
                raise ValueError(f"latent shape mismatch: base={old_feats.shape}, target={new_feats.shape}")
            if len(saved_route) != len(new_coords) or len(view_count) != len(new_coords):
                raise ValueError("route_mask or cut_view_count has the wrong length")
            if not np.isfinite(old_feats).all() or not np.isfinite(new_feats).all():
                raise ValueError("latent contains non-finite values")
            keys = [tuple(map(int, coord)) for coord in new_coords]
            if len(keys) != len(set(keys)):
                raise ValueError("target contains duplicate coordinates")

            recomputed_route = interior_mask(new_coords, args.route_margin)
            if not np.array_equal(saved_route, recomputed_route):
                raise ValueError("saved route_mask does not match the recomputed geometry route")

            delta = np.abs(new_feats - old_feats)
            outside_delta = delta[~saved_route]
            inside_delta = delta[saved_route]
            max_outside = float(outside_delta.max()) if outside_delta.size else 0.0
            max_inside = float(inside_delta.max()) if inside_delta.size else 0.0
            if max_outside > args.outside_atol:
                raise ValueError(
                    f"outside-route latent changed by {max_outside:.8g} (limit {args.outside_atol:.8g})"
                )

            route_count = int(saved_route.sum())
            if route_count == 0:
                raise ValueError("object has no routed voxels at this margin")
            once = int(((view_count >= 1) & saved_route).sum())
            twice = int(((view_count >= 2) & saved_route).sum())
            coverage = once / route_count if route_count else 0.0
            two_view_coverage = twice / route_count if route_count else 0.0
            inside_l2 = np.linalg.norm(new_feats[saved_route] - old_feats[saved_route], axis=1)
            mean_inside_l2 = float(inside_l2.mean())
            changed_inside = int((inside_l2 > 1e-6).sum())
            if once > 0 and changed_inside == 0:
                raise ValueError("covered routed voxels produced no SLAT target change")

            row.update(
                num_voxels=len(new_coords),
                route_voxels=route_count,
                covered_once=once,
                covered_twice=twice,
                coverage=coverage,
                two_view_coverage=two_view_coverage,
                max_outside_delta=max_outside,
                max_inside_delta=max_inside,
                mean_inside_l2=mean_inside_l2,
                changed_inside_voxels=changed_inside,
            )
            total_route += route_count
            total_once += once
            total_twice += twice
            total_changed += changed_inside
            total_inside_l2 += float(inside_l2.sum())
            category_counts = category_totals.setdefault(row["category"], [0, 0])
            category_counts[0] += route_count
            category_counts[1] += once
            write_coverage_ply(
                diagnostics_dir / f"{sample_id}.ply",
                new_coords,
                saved_route,
                view_count,
            )
        except Exception as error:
            row["status"] = "fail"
            row["error"] = str(error)
            invariant_failures.append(f"{sample_id}: {error}")
        rows.append(row)

    pooled_coverage = total_once / total_route if total_route else 0.0
    pooled_two_view = total_twice / total_route if total_route else 0.0
    coverage_failures = []
    if total_route == 0:
        coverage_failures.append("The pilot contains no margin-routed voxels")
    if pooled_coverage < args.min_coverage:
        coverage_failures.append(
            f"One-view coverage {pooled_coverage:.4f} is below {args.min_coverage:.4f}"
        )
    if pooled_two_view < args.min_two_view_coverage:
        coverage_failures.append(
            f"Two-view coverage {pooled_two_view:.4f} is below {args.min_two_view_coverage:.4f}"
        )
    category_coverage = {
        category: covered / routed if routed else 0.0
        for category, (routed, covered) in category_totals.items()
    }
    for category, coverage in category_coverage.items():
        if not category:
            coverage_failures.append("At least one pilot ID is missing a category")
        elif coverage < args.min_category_coverage:
            coverage_failures.append(
                f"{category} one-view coverage {coverage:.4f} is below "
                f"{args.min_category_coverage:.4f}"
            )

    base_stats = base_dir / "stats.json"
    target_stats = target_dir / "stats.json"
    stats_match = None
    stats_preserved = False
    if base_stats.is_file():
        if target_stats.is_file():
            with base_stats.open() as file:
                old_stats = json.load(file)
            with target_stats.open() as file:
                new_stats = json.load(file)
            stats_match = old_stats == new_stats
            stats_preserved = stats_match
        if not stats_preserved:
            invariant_failures.append("Target stats.json is missing or differs from base stats.json")
    else:
        manifest_path = target_dir / "target_info.json"
        if manifest_path.is_file():
            with manifest_path.open() as file:
                manifest = json.load(file)
            stats_preserved = (
                manifest.get("base_latent_name") == args.base_latent_name
                and manifest.get("stats_source") == str(base_stats)
                and manifest.get("stats_copied") is False
            )
        if not stats_preserved:
            invariant_failures.append("Base stats are absent and their source was not recorded")

    summary = {
        "passed": not invariant_failures and not coverage_failures,
        "num_instances": len(ids),
        "expected_count": args.expected_count,
        "route_margin": args.route_margin,
        "routed_voxels": total_route,
        "covered_once": total_once,
        "covered_twice": total_twice,
        "changed_inside_voxels": total_changed,
        "changed_inside_fraction": total_changed / total_route if total_route else 0.0,
        "mean_inside_l2": total_inside_l2 / total_route if total_route else 0.0,
        "coverage": pooled_coverage,
        "two_view_coverage": pooled_two_view,
        "min_coverage": args.min_coverage,
        "min_two_view_coverage": args.min_two_view_coverage,
        "min_category_coverage": args.min_category_coverage,
        "category_coverage": category_coverage,
        "stats_match": stats_match,
        "stats_preserved": stats_preserved,
        "invariant_failures": invariant_failures,
        "coverage_failures": coverage_failures,
    }

    csv_path = args.output_dir / "per_sample.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = args.output_dir / "summary.json"
    with json_path.open("w") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
