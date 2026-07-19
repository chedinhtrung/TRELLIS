#!/usr/bin/env python3
"""Evaluate decoded meshes against the deterministic internal cutaway renders."""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import nvdiffrast.torch as dr


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dataset_toolkits.render_internal_kiui import (  # noqa: E402
    AXES,
    CUT_FRACTIONS,
    NUM_VIEWS,
    RECIPE_VERSION,
    _load_canonical_mesh,
    _render_one,
    clip_mesh,
)


METRICS = (
    "mask_iou",
    "depth_mae_gt_mask",
    "depth_mae_intersection",
    "smooth_l1_depth",
)


def read_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not ids:
        raise ValueError(f"No sample IDs found in {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate sample IDs found in {path}")
    return ids


def parse_methods(values: list[str]) -> dict[str, Path]:
    methods: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --method {value!r}; expected NAME=MESH_DIR")
        name, directory = value.split("=", 1)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError(
                f"Invalid method name {name!r}; use only letters, numbers, '.', '_' or '-'"
            )
        if name in methods:
            raise ValueError(f"Duplicate method name: {name}")
        path = Path(directory).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"Mesh directory not found for {name}: {path}")
        methods[name] = path
    return methods


def validate_manifest(folder: Path) -> dict:
    manifest_path = folder / "transforms.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)

    if manifest.get("recipe_version") != RECIPE_VERSION:
        raise ValueError(
            f"{manifest_path}: expected recipe {RECIPE_VERSION!r}, "
            f"got {manifest.get('recipe_version')!r}"
        )
    resolution = manifest.get("resolution")
    frames = manifest.get("frames")
    if not isinstance(resolution, int) or resolution <= 0:
        raise ValueError(f"{manifest_path}: invalid resolution {resolution!r}")
    if not isinstance(frames, list) or len(frames) != NUM_VIEWS:
        raise ValueError(
            f"{manifest_path}: expected {NUM_VIEWS} frames, got "
            f"{len(frames) if isinstance(frames, list) else 'invalid'}"
        )

    required = {
        "file_path",
        "depth_path",
        "camera_angle_x",
        "transform_matrix",
        "axis",
        "cut_fraction",
        "plane_normal",
        "plane_offset",
        "keep",
    }
    expected_views = {
        (axis, fraction) for axis, _ in AXES for fraction in CUT_FRACTIONS
    }
    found_views = set()
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"{manifest_path}: frame {index} is not an object")
        missing = sorted(required - set(frame))
        if missing:
            raise ValueError(f"{manifest_path}: frame {index} missing {missing}")
        if frame["keep"] != "le":
            raise ValueError(f"{manifest_path}: frame {index} has unsupported keep={frame['keep']!r}")
        view_key = (frame["axis"], float(frame["cut_fraction"]))
        if view_key in found_views:
            raise ValueError(f"{manifest_path}: duplicate view {view_key}")
        found_views.add(view_key)

        image_path = folder / frame["file_path"]
        depth_path = folder / frame["depth_path"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if not depth_path.is_file():
            raise FileNotFoundError(depth_path)

        pose = np.asarray(frame["transform_matrix"], dtype=np.float32)
        normal = np.asarray(frame["plane_normal"], dtype=np.float32)
        values = np.asarray(
            [frame["camera_angle_x"], frame["plane_offset"]], dtype=np.float32
        )
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError(f"{manifest_path}: frame {index} has an invalid camera matrix")
        if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-8:
            raise ValueError(f"{manifest_path}: frame {index} has an invalid plane normal")
        if not np.isfinite(values).all() or not 0 < float(frame["camera_angle_x"]) < np.pi:
            raise ValueError(f"{manifest_path}: frame {index} has invalid scalar metadata")

        with Image.open(image_path) as image:
            if image.size != (resolution, resolution) or "A" not in image.getbands():
                raise ValueError(
                    f"{image_path}: expected {resolution}x{resolution} image with alpha"
                )
        depth = np.load(depth_path, allow_pickle=False)
        if depth.shape != (resolution, resolution) or not np.isfinite(depth).all():
            raise ValueError(f"{depth_path}: invalid depth array {depth.shape}")
    if found_views != expected_views:
        raise ValueError(f"{manifest_path}: axis/cut-fraction combinations do not match the recipe")
    return manifest


def smooth_l1(diff: np.ndarray, beta: float) -> float:
    values = np.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    return float(values.mean())


def score_view(
    gt_rgba: np.ndarray,
    gt_depth: np.ndarray,
    pred_rgba: np.ndarray,
    pred_depth: np.ndarray,
    beta: float,
) -> dict[str, float | int]:
    if gt_rgba.ndim != 3 or gt_rgba.shape[-1] != 4:
        raise ValueError(f"Invalid ground-truth RGBA shape: {gt_rgba.shape}")
    if pred_rgba.shape != gt_rgba.shape:
        raise ValueError(
            f"Predicted/ground-truth RGBA shapes differ: {pred_rgba.shape} vs {gt_rgba.shape}"
        )
    if gt_depth.shape != gt_rgba.shape[:2] or pred_depth.shape != gt_depth.shape:
        raise ValueError(
            f"Depth/RGBA shapes differ: gt={gt_depth.shape}, pred={pred_depth.shape}, "
            f"rgba={gt_rgba.shape}"
        )
    if not np.isfinite(gt_depth).all() or not np.isfinite(pred_depth).all():
        raise ValueError("Ground-truth or predicted depth contains non-finite values")

    gt_mask = gt_rgba[..., 3] > 127
    pred_mask = pred_rgba[..., 3] > 127
    if not gt_mask.any():
        raise ValueError("Ground-truth cutaway has an empty foreground mask")

    intersection = gt_mask & pred_mask
    union = gt_mask | pred_mask
    mask_iou = float(intersection.sum() / union.sum()) if union.any() else 1.0
    absolute_depth_error = np.abs(
        pred_depth.astype(np.float32) - gt_depth.astype(np.float32)
    )
    gt_depth_error = absolute_depth_error[gt_mask]
    intersection_depth = (
        float(absolute_depth_error[intersection].mean())
        if intersection.any()
        else float("nan")
    )
    return {
        "gt_mask_pixels": int(gt_mask.sum()),
        "pred_mask_pixels": int(pred_mask.sum()),
        "intersection_pixels": int(intersection.sum()),
        "valid_intersection_depth": int(intersection.any()),
        "mask_iou": mask_iou,
        "depth_mae_gt_mask": float(gt_depth_error.mean()),
        "depth_mae_intersection": intersection_depth,
        "smooth_l1_depth": smooth_l1(
            np.where(gt_mask, absolute_depth_error, 0.0), beta
        ),
    }


def mean_metric(rows: list[dict], name: str) -> float:
    values = np.asarray([row[name] for row in rows], dtype=np.float64)
    finite = np.isfinite(values)
    return float(values[finite].mean()) if finite.any() else float("nan")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--ids_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        metavar="NAME=MESH_DIR",
        help="Named directory containing <sample_id>.ply files; repeat for each method",
    )
    parser.add_argument(
        "--smooth_l1_beta",
        type=float,
        default=1.0 / 512.0,
        help="Smooth-L1 beta; defaults to 1/(2*256), matching the mesh decoder trainer",
    )
    args = parser.parse_args()

    if args.smooth_l1_beta <= 0:
        parser.error("--smooth_l1_beta must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the cutaway renderer")

    ids = read_ids(args.ids_file)
    methods = parse_methods(args.method)
    teacher_root = args.data_dir / "renders_internal_v1"
    if not teacher_root.is_dir():
        raise FileNotFoundError(f"Internal render directory not found: {teacher_root}")

    manifests: dict[str, dict] = {}
    for sample_id in ids:
        manifests[sample_id] = validate_manifest(teacher_root / sample_id)
        for method, mesh_dir in methods.items():
            mesh_path = mesh_dir / f"{sample_id}.ply"
            if not mesh_path.is_file():
                raise FileNotFoundError(f"Missing {method} prediction: {mesh_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    render_root = args.output_dir / "predicted_cutaways"
    device = torch.device("cuda")
    context = dr.RasterizeCudaContext(device=device)
    per_view_rows: list[dict] = []

    total = len(methods) * len(ids)
    progress = tqdm(total=total, desc="Evaluating predicted cutaways")
    for method, mesh_dir in methods.items():
        for sample_id in ids:
            manifest = manifests[sample_id]
            resolution = int(manifest["resolution"])
            output_folder = render_root / method / sample_id
            output_folder.mkdir(parents=True, exist_ok=True)
            mesh = _load_canonical_mesh(mesh_dir / f"{sample_id}.ply", device)
            mesh.vc = None
            mesh.vt = None
            mesh.ft = None
            mesh.albedo = None

            for view_index, frame in enumerate(manifest["frames"]):
                teacher_folder = teacher_root / sample_id
                with Image.open(teacher_folder / frame["file_path"]) as image:
                    gt_rgba = np.asarray(image.convert("RGBA"))
                gt_depth = np.load(teacher_folder / frame["depth_path"], allow_pickle=False)
                plane_normal = torch.as_tensor(
                    frame["plane_normal"], dtype=torch.float32, device=device
                )
                plane_offset = float(frame["plane_offset"])
                retained_vertices = mesh.v @ plane_normal - plane_offset <= 1e-7
                has_retained_face = torch.any(retained_vertices[mesh.f.long()])
                del retained_vertices
                if has_retained_face:
                    clipped = clip_mesh(mesh, plane_normal, plane_offset)
                    pose = np.asarray(frame["transform_matrix"], dtype=np.float32)
                    fov_degrees = float(np.rad2deg(float(frame["camera_angle_x"])))
                    pred_rgba, pred_depth = _render_one(
                        clipped, pose, fov_degrees, resolution, context
                    )
                    del clipped
                else:
                    pred_rgba = np.zeros((resolution, resolution, 4), dtype=np.uint8)
                    pred_depth = np.zeros((resolution, resolution), dtype=np.float16)

                Image.fromarray(pred_rgba, mode="RGBA").save(
                    output_folder / Path(frame["file_path"]).name
                )
                metrics = score_view(
                    gt_rgba,
                    gt_depth,
                    pred_rgba,
                    pred_depth,
                    args.smooth_l1_beta,
                )
                per_view_rows.append(
                    {
                        "method": method,
                        "sample_id": sample_id,
                        "view": view_index,
                        "axis": frame.get("axis", ""),
                        "cut_fraction": frame.get("cut_fraction", ""),
                        **metrics,
                    }
                )
            del mesh
            torch.cuda.empty_cache()
            progress.update(1)
    progress.close()

    per_sample_rows: list[dict] = []
    for method in methods:
        for sample_id in ids:
            rows = [
                row
                for row in per_view_rows
                if row["method"] == method and row["sample_id"] == sample_id
            ]
            if len(rows) != NUM_VIEWS:
                raise RuntimeError(
                    f"Expected {NUM_VIEWS} evaluated views for {method}/{sample_id}, got {len(rows)}"
                )
            per_sample_rows.append(
                {
                    "method": method,
                    "sample_id": sample_id,
                    "num_views": len(rows),
                    "valid_intersection_depth_views": sum(
                        row["valid_intersection_depth"] for row in rows
                    ),
                    **{metric: mean_metric(rows, metric) for metric in METRICS},
                }
            )

    summary_rows: list[dict] = []
    for method in methods:
        rows = [row for row in per_sample_rows if row["method"] == method]
        summary_rows.append(
            {
                "method": method,
                "num_samples": len(rows),
                "num_views": len(rows) * NUM_VIEWS,
                "valid_intersection_depth_views": sum(
                    row["valid_intersection_depth_views"] for row in rows
                ),
                **{metric: mean_metric(rows, metric) for metric in METRICS},
            }
        )

    write_csv(args.output_dir / "per_view.csv", per_view_rows)
    write_csv(args.output_dir / "per_sample.csv", per_sample_rows)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    print(f"Wrote cutaway renders to {render_root}")
    print(f"Wrote metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
