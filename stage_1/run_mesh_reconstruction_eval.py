from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import trimesh
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from common import (
    DEFAULT_DATASET_DIR,
    REPO_ROOT,
    SLAT_LATENT_MODEL,
    artifact_exists,
    ensure_dir,
    positions_to_grid,
    python_cmd,
    read_metadata,
    read_ply_points,
    run_command,
)

sys.path.insert(0, str(REPO_ROOT))

import trellis.models as models
import trellis.modules.sparse as sp


DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "shapenet_mesh_reconstruction"
DEFAULT_SLAT_ENCODER = "microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16"
DEFAULT_MESH_DECODER = "microsoft/TRELLIS-image-large/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16"
BINVOX_GRID_TO_OBJ_AXIS = (0, 2, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TRELLIS SLAT latent -> mesh decoder reconstruction on ShapeNet internals.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of objects; 0 means all available objects.")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--slat-latent-model", default=SLAT_LATENT_MODEL)
    parser.add_argument("--slat-encoder", default=DEFAULT_SLAT_ENCODER)
    parser.add_argument("--mesh-decoder", default=DEFAULT_MESH_DECODER)
    parser.add_argument("--skip-ensure-latents", action="store_true", help="Do not call dataset_toolkits/encode_latent.py before decoding.")
    parser.add_argument("--overwrite-latents", action="store_true", help="Delete selected cached SLAT latents before re-encoding.")
    parser.add_argument("--mesh-sample-points", type=int, default=50000)
    parser.add_argument("--voxel-sample-points", type=int, default=500000)
    parser.add_argument("--visual-sample-points", type=int, default=100000)
    parser.add_argument("--fscore-threshold", type=float, default=0.01, help="Distance threshold in ShapeNet OBJ units.")
    parser.add_argument("--overwrite-meshes", action="store_true")
    return parser.parse_args()


def read_binvox_header(path: Path) -> dict[str, object]:
    with path.open("rb") as fp:
        header = fp.readline().decode("ascii", errors="replace").strip()
        if not header.startswith("#binvox"):
            raise ValueError(f"Not a binvox file: {path}")
        dim_line = fp.readline().decode("ascii", errors="replace").strip().split()
        translate_line = fp.readline().decode("ascii", errors="replace").strip().split()
        scale_line = fp.readline().decode("ascii", errors="replace").strip().split()
    if dim_line[0] != "dim" or translate_line[0] != "translate" or scale_line[0] != "scale":
        raise ValueError(f"Unexpected binvox header in {path}")
    return {
        "dims": tuple(int(v) for v in dim_line[1:4]),
        "translate": np.asarray([float(v) for v in translate_line[1:4]], dtype=np.float32),
        "scale": float(scale_line[1]),
    }


def grid_to_obj(points_grid: np.ndarray, header: dict[str, object]) -> np.ndarray:
    points = points_grid[:, BINVOX_GRID_TO_OBJ_AXIS] if points_grid.ndim == 2 else points_grid
    return (points + 0.5) * float(header["scale"]) + np.asarray(header["translate"], dtype=np.float32)


def obj_to_grid(points_obj: np.ndarray, header: dict[str, object]) -> np.ndarray:
    points = (points_obj - np.asarray(header["translate"], dtype=np.float32)) / float(header["scale"]) - 0.5
    if points.ndim == 2:
        inverse = np.argsort(BINVOX_GRID_TO_OBJ_AXIS)
        points = points[:, inverse]
    return points


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
        raise ValueError(f"Empty mesh: {path}")
    return mesh


def make_mesh(vertices: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )


def export_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    ensure_dir(path.parent)
    make_mesh(vertices, faces).export(path)


def sample_mesh_points(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    if count <= 0:
        return np.asarray(mesh.vertices, dtype=np.float32)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        points = mesh.sample(count)
    finally:
        np.random.set_state(state)
    if points.shape[0] == 0:
        points = np.asarray(mesh.vertices, dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def nearest_metrics(gt_points: np.ndarray, pred_points: np.ndarray, threshold: float) -> dict[str, float]:
    if gt_points.shape[0] == 0 or pred_points.shape[0] == 0:
        return {
            "chamfer_l1": np.nan,
            "chamfer_l2": np.nan,
            "fscore": np.nan,
            "mesh_precision": np.nan,
            "mesh_recall": np.nan,
        }
    gt_tree = cKDTree(gt_points)
    pred_tree = cKDTree(pred_points)
    pred_to_gt = gt_tree.query(pred_points, k=1, workers=-1)[0]
    gt_to_pred = pred_tree.query(gt_points, k=1, workers=-1)[0]
    precision = float((pred_to_gt <= threshold).mean())
    recall = float((gt_to_pred <= threshold).mean())
    fscore = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "chamfer_l1": float(pred_to_gt.mean() + gt_to_pred.mean()),
        "chamfer_l2": float(np.square(pred_to_gt).mean() + np.square(gt_to_pred).mean()),
        "fscore": float(fscore),
        "mesh_precision": precision,
        "mesh_recall": recall,
    }


def points_to_grid(points_grid: np.ndarray, resolution: int) -> tuple[np.ndarray, float]:
    grid = np.zeros((resolution, resolution, resolution), dtype=bool)
    if points_grid.size == 0:
        return grid, 0.0
    coords_float = (points_grid + 0.5) * resolution
    out_of_bounds = np.logical_or(coords_float < 0, coords_float >= resolution).any(axis=1)
    coords = np.floor(coords_float).astype(np.int64)
    coords = np.clip(coords, 0, resolution - 1)
    grid[coords[:, 0], coords[:, 1], coords[:, 2]] = True
    return grid, float(out_of_bounds.mean())


def voxel_metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    tp = int(np.logical_and(gt, pred).sum())
    fp = int(np.logical_and(~gt, pred).sum())
    fn = int(np.logical_and(gt, ~pred).sum())
    union = int(np.logical_or(gt, pred).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "voxel_iou": tp / union if union > 0 else 0.0,
        "voxel_precision": precision,
        "voxel_recall": recall,
        "voxel_f1": f1,
        "voxel_tp": tp,
        "voxel_fp": fp,
        "voxel_fn": fn,
        "gt_surface_voxels": int(gt.sum()),
        "pred_surface_voxels": int(pred.sum()),
    }


def gt_external_mask(gt: np.ndarray) -> np.ndarray:
    external = np.zeros_like(gt, dtype=bool)
    for axis in range(3):
        other_axes = [i for i in range(3) if i != axis]
        for a in range(gt.shape[other_axes[0]]):
            for b in range(gt.shape[other_axes[1]]):
                slc = [slice(None), slice(None), slice(None)]
                slc[other_axes[0]] = a
                slc[other_axes[1]] = b
                line = gt[tuple(slc)]
                hits = np.flatnonzero(line)
                if hits.size == 0:
                    continue
                first = [slice(None), slice(None), slice(None)]
                first[axis] = hits[0]
                first[other_axes[0]] = a
                first[other_axes[1]] = b
                last = [slice(None), slice(None), slice(None)]
                last[axis] = hits[-1]
                last[other_axes[0]] = a
                last[other_axes[1]] = b
                external[tuple(first)] = True
                external[tuple(last)] = True
    return external


def internal_external_recalls(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    external = gt_external_mask(gt)
    internal = gt & ~external
    ext_count = int(external.sum())
    int_count = int(internal.sum())
    return {
        "gt_external_voxels": ext_count,
        "gt_internal_candidate_voxels": int_count,
        "external_recall": float((pred & external).sum() / ext_count) if ext_count else np.nan,
        "internal_candidate_recall": float((pred & internal).sum() / int_count) if int_count else np.nan,
    }


def binary_panel(slice_: np.ndarray, scale: int = 6) -> Image.Image:
    image = Image.fromarray((slice_.astype(np.uint8) * 255), mode="L").convert("RGB")
    return image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)


def overlay_panel(gt: np.ndarray, pred: np.ndarray, scale: int = 6) -> Image.Image:
    panel = np.zeros((*gt.shape, 3), dtype=np.uint8)
    panel[gt & pred] = [0, 220, 80]
    panel[gt & ~pred] = [40, 130, 255]
    panel[~gt & pred] = [240, 70, 60]
    image = Image.fromarray(panel, mode="RGB")
    return image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)


def add_label(image: Image.Image, label: str) -> Image.Image:
    out = Image.new("RGB", (image.width, image.height + 18), "white")
    out.paste(image, (0, 18))
    draw = ImageDraw.Draw(out)
    draw.text((4, 2), label, fill="black")
    return out


def occupied_bbox_center(grid: np.ndarray) -> tuple[int, int, int]:
    coords = np.argwhere(grid)
    if coords.size == 0:
        return tuple(v // 2 for v in grid.shape)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return tuple(((mins + maxs) // 2).astype(int))


def voxel_cross_section_image(gt: np.ndarray, pred: np.ndarray) -> Image.Image:
    x, y, z = occupied_bbox_center(gt)
    planes = [
        (f"xy @ z={z}", gt[:, :, z], pred[:, :, z]),
        (f"xz @ y={y}", gt[:, y, :], pred[:, y, :]),
        (f"yz @ x={x}", gt[x, :, :], pred[x, :, :]),
    ]
    rows = []
    for name, gt_slice, pred_slice in planes:
        panels = [
            add_label(binary_panel(gt_slice), f"GT vox {name}"),
            add_label(binary_panel(pred_slice), f"Recon mesh vox {name}"),
            add_label(overlay_panel(gt_slice, pred_slice), "green=TP blue=miss red=extra"),
        ]
        row = Image.new("RGB", (sum(p.width for p in panels), max(p.height for p in panels)), "white")
        xoff = 0
        for panel in panels:
            row.paste(panel, (xoff, 0))
            xoff += panel.width
        rows.append(row)
    out = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows)), "white")
    yoff = 0
    for row in rows:
        out.paste(row, (0, yoff))
        yoff += row.height
    return out


def point_panel(points: np.ndarray, axis: int, plane: float, thickness: float, scale: int = 6) -> Image.Image:
    other = [i for i in range(3) if i != axis]
    panel = np.zeros((64, 64), dtype=np.uint8)
    if points.shape[0] > 0:
        mask = np.abs(points[:, axis] - plane) <= thickness
        pts = points[mask][:, other]
        coords = np.floor((pts + 0.5) * 64).astype(np.int64)
        valid = np.logical_and(coords >= 0, coords < 64).all(axis=1)
        coords = coords[valid]
        panel[coords[:, 0], coords[:, 1]] = 255
    image = Image.fromarray(panel, mode="L").convert("RGB")
    return image.resize((64 * scale, 64 * scale), Image.Resampling.NEAREST)


def point_overlay_panel(gt_points: np.ndarray, pred_points: np.ndarray, axis: int, plane: float, thickness: float, scale: int = 6) -> Image.Image:
    other = [i for i in range(3) if i != axis]
    panel = np.zeros((64, 64, 3), dtype=np.uint8)
    for points, color in ((gt_points, np.array([40, 130, 255], dtype=np.uint8)), (pred_points, np.array([240, 70, 60], dtype=np.uint8))):
        if points.shape[0] == 0:
            continue
        mask = np.abs(points[:, axis] - plane) <= thickness
        pts = points[mask][:, other]
        coords = np.floor((pts + 0.5) * 64).astype(np.int64)
        valid = np.logical_and(coords >= 0, coords < 64).all(axis=1)
        coords = coords[valid]
        existing = panel[coords[:, 0], coords[:, 1]].sum(axis=1) > 0
        panel[coords[:, 0], coords[:, 1]] = color
        if existing.size:
            both = coords[existing]
            panel[both[:, 0], both[:, 1]] = [0, 220, 80]
    image = Image.fromarray(panel, mode="RGB")
    return image.resize((64 * scale, 64 * scale), Image.Resampling.NEAREST)


def mesh_slab_image(gt_points_grid: np.ndarray, pred_points_grid: np.ndarray, gt_grid: np.ndarray, resolution: int) -> Image.Image:
    center = occupied_bbox_center(gt_grid)
    planes = [
        ("x", 0, (center[0] + 0.5) / resolution - 0.5),
        ("y", 1, (center[1] + 0.5) / resolution - 0.5),
        ("z", 2, (center[2] + 0.5) / resolution - 0.5),
    ]
    thickness = 1.5 / resolution
    rows = []
    for name, axis, plane in planes:
        panels = [
            add_label(point_panel(gt_points_grid, axis, plane, thickness), f"GT mesh slab {name}={plane:.3f}"),
            add_label(point_panel(pred_points_grid, axis, plane, thickness), f"Recon mesh slab {name}={plane:.3f}"),
            add_label(point_overlay_panel(gt_points_grid, pred_points_grid, axis, plane, thickness), "blue=GT red=recon green=both"),
        ]
        row = Image.new("RGB", (sum(p.width for p in panels), max(p.height for p in panels)), "white")
        xoff = 0
        for panel in panels:
            row.paste(panel, (xoff, 0))
            xoff += panel.width
        rows.append(row)
    out = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows)), "white")
    yoff = 0
    for row in rows:
        out.paste(row, (0, yoff))
        yoff += row.height
    return out


def load_slat(sample_id: str, dataset_dir: Path, latent_model: str, device: torch.device) -> sp.SparseTensor:
    path = dataset_dir / "latents" / latent_model / f"{sample_id}.npz"
    data = np.load(path)
    coords = torch.cat([
        torch.zeros(data["coords"].shape[0], 1, dtype=torch.int32),
        torch.from_numpy(data["coords"]).int(),
    ], dim=1).to(device)
    feats = torch.from_numpy(data["feats"]).float().to(device)
    return sp.SparseTensor(coords=coords, feats=feats)


def select_metadata(dataset_dir: Path, latent_model: str, limit: int) -> pd.DataFrame:
    metadata = read_metadata(dataset_dir)
    latent_col = f"latent_{latent_model}"
    required = metadata["voxelized"] == True
    if latent_col in metadata.columns:
        required &= metadata[latent_col] == True
    metadata = metadata[required].copy()
    if limit:
        metadata = metadata.head(limit)
    if metadata.empty:
        raise RuntimeError("No voxelized samples with SLAT latent metadata are available.")
    return metadata


def ensure_latents(args: argparse.Namespace, metadata: pd.DataFrame) -> None:
    if args.skip_ensure_latents:
        return
    instances_path = args.output_dir / "instances.txt"
    instances_path.write_text("".join(f"{sid}\n" for sid in metadata["sha256"]), encoding="utf-8")
    latent_dir = args.dataset_dir / "latents" / args.slat_latent_model
    if args.overwrite_latents:
        for sample_id in metadata["sha256"]:
            path = latent_dir / f"{sample_id}.npz"
            if path.exists():
                path.unlink()
    env = os.environ.copy()
    env["SPCONV_ALGO"] = env.get("SPCONV_ALGO", "native")
    run_command(
        python_cmd(
            REPO_ROOT / "dataset_toolkits" / "encode_latent.py",
            "--output_dir", str(args.dataset_dir),
            "--instances", str(instances_path),
            "--enc_pretrained", args.slat_encoder,
        ),
        env=env,
        log_path=args.output_dir / "logs" / "mesh_reconstruction_eval.log",
    )


def write_report(
    path: Path,
    args: argparse.Namespace,
    metadata: pd.DataFrame | None,
    metrics_df: pd.DataFrame | None,
    error: str | None = None,
) -> None:
    lines = [
        "# ShapeNet Mesh Reconstruction Feasibility Report",
        "",
        f"- Dataset: `{args.dataset_dir}`",
        f"- Results: `{args.output_dir}`",
        "- Reconstruction path tested: `ShapeNet renders/features -> ElasticSLatEncoder -> SLAT latent -> SLatMeshDecoder -> SparseFeatures2Mesh/FlexiCubes -> MeshExtractResult`",
        "- Raw decoder meshes are saved before TRELLIS GLB postprocessing, because postprocessing can remove invisible/internal faces.",
        "",
        "## Components",
        "",
        f"- SLAT encoder checkpoint: `{args.slat_encoder}`",
        "- Encoder class: `ElasticSLatEncoder` via `dataset_toolkits/encode_latent.py`.",
        f"- Mesh decoder checkpoint: `{args.mesh_decoder}`",
        "- Decoder class: `SLatMeshDecoder` loaded through `trellis.models.from_pretrained`.",
        "- Inference function: `TrellisImageTo3DPipeline.decode_slat` normally calls `self.models['slat_decoder_mesh'](slat)`; this wrapper calls the same decoder directly.",
        "- Mesh extraction: `SLatMeshDecoder.forward` -> `SparseSubdivideBlock3d` x2 -> `SparseFeatures2Mesh` -> `FlexiCubes` -> `MeshExtractResult`.",
        "",
        "## Commands Executed",
        "",
        "```bash",
    ]
    if not args.skip_ensure_latents:
        lines.append(
            f"/workspace/venv/bin/python dataset_toolkits/encode_latent.py --output_dir {args.dataset_dir} --instances {args.output_dir / 'instances.txt'} --enc_pretrained {args.slat_encoder}"
        )
    eval_cmd = [
        "/workspace/venv/bin/python",
        "stage_1/run_mesh_reconstruction_eval.py",
        "--dataset-dir", str(args.dataset_dir),
        "--output-dir", str(args.output_dir),
    ]
    if args.overwrite_latents:
        eval_cmd.append("--overwrite-latents")
    if args.overwrite_meshes:
        eval_cmd.append("--overwrite-meshes")
    if args.skip_ensure_latents:
        eval_cmd.append("--skip-ensure-latents")
    if args.limit:
        eval_cmd.extend(["--limit", str(args.limit)])
    lines.append(" ".join(eval_cmd))
    lines.extend(["```", ""])
    if metadata is not None:
        lines.extend([
            "## Subset",
            "",
            f"- Total evaluated rows: {len(metadata)}",
        ])
        if "category" in metadata.columns:
            for category, count in metadata["category"].value_counts().sort_index().items():
                lines.append(f"- `{category}`: {count}")
        lines.append("")
    if error:
        lines.extend(["## Error / Blocker", "", "```text", error, "```", ""])
    if metrics_df is not None and not metrics_df.empty:
        numeric = metrics_df.select_dtypes(include=[np.number])
        means = numeric.mean(numeric_only=True)
        lines.extend([
            "## Quantitative Metrics",
            "",
            f"- Successful mesh decodes: `{int(metrics_df['decode_success'].sum())} / {len(metrics_df)}`",
            f"- Mean pred vertices: `{means.get('pred_vertices', np.nan):.1f}`",
            f"- Mean pred faces: `{means.get('pred_faces', np.nan):.1f}`",
            f"- Mean Chamfer L1: `{means.get('chamfer_l1', np.nan):.6f}`",
            f"- Mean F-score @ {args.fscore_threshold}: `{means.get('fscore', np.nan):.4f}`",
            f"- Mean surface-voxel IoU: `{means.get('voxel_iou', np.nan):.4f}`",
            f"- Mean surface-voxel precision: `{means.get('voxel_precision', np.nan):.4f}`",
            f"- Mean surface-voxel recall: `{means.get('voxel_recall', np.nan):.4f}`",
            f"- Mean surface-voxel F1: `{means.get('voxel_f1', np.nan):.4f}`",
            f"- Mean external recall: `{means.get('external_recall', np.nan):.4f}`",
            f"- Mean internal-candidate recall: `{means.get('internal_candidate_recall', np.nan):.4f}`",
            "",
            "Per-object metrics are in `metrics.csv`.",
            "",
            "## Failure Cases",
            "",
            "No mesh decode failures occurred in this run." if int(metrics_df["decode_success"].sum()) == len(metrics_df) else "Some mesh decodes failed; see `metrics.csv` for rows with `decode_success = False`.",
            "",
            "## Visual Examples",
            "",
            "- Raw decoder-grid meshes: `meshes_grid/`",
            "- ShapeNet OBJ-frame reconstructed meshes: `meshes_obj_frame/`",
            "- GT surface voxel vs reconstructed mesh surface voxel cross-sections: `cross_sections_voxel/`",
            "- GT mesh vs reconstructed mesh thin-slab cross-sections: `cross_sections_mesh/`",
            "",
            "## Interpretation",
            "",
        ])
        internal_recall = means.get("internal_candidate_recall", np.nan)
        voxel_f1 = means.get("voxel_f1", np.nan)
        if np.isfinite(internal_recall) and np.isfinite(voxel_f1) and internal_recall >= 0.7 and voxel_f1 >= 0.5:
            lines.extend([
                "Case A: the pretrained mesh decoder appears to preserve internal-candidate geometry reasonably well on this small subset.",
                "",
                "Recommendation: proceed with LoRA fine-tuning of `SparseStructureFlowModel` and `ElasticSLatFlowModel` without modifying the decoder, while keeping mesh/internal validation in the training loop.",
            ])
        else:
            lines.extend([
                "Case B: the pretrained mesh decoder appears to lose substantial target surface geometry and/or internal-candidate geometry on this small subset.",
                "",
                "Recommendation: decoder/VAE fine-tuning with internal-aware supervision is needed before LoRA on the flow generators is likely to learn internals meaningfully.",
            ])
        lines.extend([
            "",
            "## Notes And Risks",
            "",
            "- Internal/external separation is approximate: external voxels are the first/last occupied GT surface voxels along the six cardinal grid directions; the rest are reported as internal candidates.",
            "- Mesh Chamfer/F-score are sampled-surface metrics and may hide internal failures when exterior area dominates.",
        "- Surface-voxel metrics compare the decoded mesh surface voxelization against `model_normalized.surface.binvox` converted to 64^3.",
            f"- The Stage 1 binvox conversion preserves the binvox index frame; for mesh metrics, decoded grid-frame vertices are transformed back to ShapeNet OBJ coordinates using each binvox header's `translate` and `scale` with grid-to-OBJ axis order `{BINVOX_GRID_TO_OBJ_AXIS}`.",
        ])
    else:
        lines.extend([
            "## Interpretation",
            "",
            "No metrics were produced. Resolve the blocker before deciding whether LoRA is enough.",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_sample(
    row: pd.Series,
    args: argparse.Namespace,
    decoder,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    sample_id = row["sha256"]
    category = row.get("category", "")
    shapenet_id = row.get("shapenet_id", "")
    source_obj = Path(str(row["source_obj"]))
    surface_binvox = Path(str(row["surface_binvox"]))
    header = read_binvox_header(surface_binvox)

    gt_mesh = load_mesh(source_obj)
    gt_vertices_obj = np.asarray(gt_mesh.vertices, dtype=np.float32)
    gt_faces = np.asarray(gt_mesh.faces, dtype=np.int64)

    grid_mesh_path = args.output_dir / "meshes_grid" / f"{sample_id}.ply"
    obj_mesh_path = args.output_dir / "meshes_obj_frame" / f"{sample_id}.ply"
    if grid_mesh_path.exists() and obj_mesh_path.exists() and not args.overwrite_meshes:
        pred_mesh_grid = load_mesh(grid_mesh_path)
        pred_vertices_grid = np.asarray(pred_mesh_grid.vertices, dtype=np.float32)
        pred_faces = np.asarray(pred_mesh_grid.faces, dtype=np.int64)
    else:
        slat = load_slat(sample_id, args.dataset_dir, args.slat_latent_model, device)
        with torch.no_grad():
            decoded = decoder(slat)[0]
        if not decoded.success:
            raise RuntimeError(f"Mesh decoder produced an empty mesh for {sample_id}")
        pred_vertices_grid = decoded.vertices.detach().cpu().numpy().astype(np.float32)
        pred_faces = decoded.faces.detach().cpu().numpy().astype(np.int64)
        export_mesh(grid_mesh_path, pred_vertices_grid, pred_faces)
        torch.cuda.empty_cache()

    pred_vertices_obj = grid_to_obj(pred_vertices_grid, header)
    export_mesh(obj_mesh_path, pred_vertices_obj, pred_faces)

    gt_points_obj = sample_mesh_points(gt_mesh, args.mesh_sample_points, seed)
    pred_points_obj = sample_mesh_points(make_mesh(pred_vertices_obj, pred_faces), args.mesh_sample_points, seed + 1)
    mesh_metrics = nearest_metrics(gt_points_obj, pred_points_obj, args.fscore_threshold)

    gt_points_grid = obj_to_grid(gt_points_obj, header)
    pred_points_grid = obj_to_grid(pred_points_obj, header)

    gt_voxel_path = args.dataset_dir / "voxels" / f"{sample_id}.ply"
    gt_grid = positions_to_grid(read_ply_points(gt_voxel_path), args.resolution)
    pred_voxel_points = sample_mesh_points(make_mesh(pred_vertices_grid, pred_faces), args.voxel_sample_points, seed + 2)
    pred_voxel_points = np.concatenate([pred_voxel_points, pred_vertices_grid], axis=0)
    pred_grid, pred_oob = points_to_grid(pred_voxel_points, args.resolution)
    vox_metrics = voxel_metrics(gt_grid, pred_grid)
    int_ext = internal_external_recalls(gt_grid, pred_grid)

    visual_gt_points = sample_mesh_points(gt_mesh, args.visual_sample_points, seed + 3)
    visual_pred_points = sample_mesh_points(make_mesh(pred_vertices_grid, pred_faces), args.visual_sample_points, seed + 4)
    visual_gt_grid = obj_to_grid(visual_gt_points, header)
    voxel_cross_section_image(gt_grid, pred_grid).save(args.output_dir / "cross_sections_voxel" / f"{sample_id}.png")
    mesh_slab_image(visual_gt_grid, visual_pred_points, gt_grid, args.resolution).save(args.output_dir / "cross_sections_mesh" / f"{sample_id}.png")

    obj_bounds = np.concatenate([gt_vertices_obj.min(axis=0), gt_vertices_obj.max(axis=0)])
    pred_bounds = np.concatenate([pred_vertices_obj.min(axis=0), pred_vertices_obj.max(axis=0)])
    return {
        "sha256": sample_id,
        "category": category,
        "shapenet_id": shapenet_id,
        "decode_success": True,
        "gt_vertices": int(gt_vertices_obj.shape[0]),
        "gt_faces": int(gt_faces.shape[0]),
        "pred_vertices": int(pred_vertices_grid.shape[0]),
        "pred_faces": int(pred_faces.shape[0]),
        "pred_grid_oob_point_fraction": pred_oob,
        **{f"gt_bound_{i}": float(v) for i, v in enumerate(obj_bounds)},
        **{f"pred_bound_{i}": float(v) for i, v in enumerate(pred_bounds)},
        **mesh_metrics,
        **vox_metrics,
        **int_ext,
    }


def main() -> None:
    args = parse_args()
    os.environ.setdefault("SPCONV_ALGO", "native")
    ensure_dir(args.output_dir)
    for folder in ["meshes_grid", "meshes_obj_frame", "cross_sections_voxel", "cross_sections_mesh", "logs"]:
        ensure_dir(args.output_dir / folder)

    metadata = None
    metrics_df = None
    try:
        metadata = select_metadata(args.dataset_dir, args.slat_latent_model, args.limit)
        ensure_latents(args, metadata)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("The TRELLIS mesh decoder path requires CUDA in this script.")
        decoder = models.from_pretrained(args.mesh_decoder).eval().to(device)

        rows = []
        for i, (_, row) in enumerate(metadata.iterrows()):
            sample_id = row["sha256"]
            print(f"[{i + 1}/{len(metadata)}] Decoding/evaluating {sample_id}", flush=True)
            try:
                rows.append(evaluate_sample(row, args, decoder, device, seed=1000 + i * 17))
            except Exception as exc:
                print(f"Error evaluating {sample_id}: {exc}", flush=True)
                rows.append({
                    "sha256": sample_id,
                    "category": row.get("category", ""),
                    "shapenet_id": row.get("shapenet_id", ""),
                    "decode_success": False,
                    "error": str(exc),
                })

        metrics_df = pd.DataFrame(rows)
        metrics_df.to_csv(args.output_dir / "metrics.csv", index=False)
        write_report(args.output_dir / "report.md", args, metadata, metrics_df)
        print(metrics_df.to_string(index=False))
        print(f"Wrote report: {args.output_dir / 'report.md'}")
    except Exception:
        error = traceback.format_exc()
        print(error)
        write_report(args.output_dir / "report.md", args, metadata, metrics_df, error=error)
        raise


if __name__ == "__main__":
    main()
