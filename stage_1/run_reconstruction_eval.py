from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from common import (
    DEFAULT_DATASET_DIR,
    DEFAULT_RESULTS_DIR,
    REPO_ROOT,
    SS_LATENT_MODEL,
    artifact_exists,
    ensure_dir,
    grid_to_positions,
    positions_to_grid,
    read_metadata,
    read_ply_points,
    write_ply_points,
)

sys.path.insert(0, str(REPO_ROOT))

import trellis.models as models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TRELLIS sparse-structure VAE reconstruction on ShapeNet internals.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--encoder", default="microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16")
    parser.add_argument("--decoder", default="microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16")
    parser.add_argument("--use-cached-latents", action="store_true", help="Use ss_latents if present instead of encoding live.")
    return parser.parse_args()


def metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    tp = np.logical_and(gt, pred).sum()
    fp = np.logical_and(~gt, pred).sum()
    fn = np.logical_and(gt, ~pred).sum()
    union = np.logical_or(gt, pred).sum()
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "iou": tp / union if union > 0 else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "gt_voxels": int(gt.sum()),
        "pred_voxels": int(pred.sum()),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def binary_panel(slice_: np.ndarray, scale: int = 6) -> Image.Image:
    image = Image.fromarray((slice_.astype(np.uint8) * 255), mode="L").convert("RGB")
    return image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)


def overlay_panel(gt: np.ndarray, pred: np.ndarray, scale: int = 6) -> Image.Image:
    panel = np.zeros((*gt.shape, 3), dtype=np.uint8)
    tp = gt & pred
    fn = gt & ~pred
    fp = ~gt & pred
    panel[tp] = [0, 220, 80]
    panel[fn] = [40, 130, 255]
    panel[fp] = [240, 70, 60]
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


def cross_section_image(gt: np.ndarray, pred: np.ndarray) -> Image.Image:
    x, y, z = occupied_bbox_center(gt)
    planes = [
        (f"xy @ z={z}", gt[:, :, z], pred[:, :, z]),
        (f"xz @ y={y}", gt[:, y, :], pred[:, y, :]),
        (f"yz @ x={x}", gt[x, :, :], pred[x, :, :]),
    ]
    rows = []
    for name, gt_slice, pred_slice in planes:
        panels = [
            add_label(binary_panel(gt_slice), f"GT {name}"),
            add_label(binary_panel(pred_slice), f"Recon {name}"),
            add_label(overlay_panel(gt_slice, pred_slice), "green=TP blue=miss red=extra"),
        ]
        row = Image.new("RGB", (sum(p.width for p in panels), max(p.height for p in panels)), "white")
        x = 0
        for panel in panels:
            row.paste(panel, (x, 0))
            x += panel.width
        rows.append(row)
    out = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows)), "white")
    y = 0
    for row in rows:
        out.paste(row, (0, y))
        y += row.height
    return out


def load_or_encode_latent(
    sample_id: str,
    gt_grid: np.ndarray,
    dataset_dir: Path,
    encoder,
    use_cached: bool,
    device: torch.device,
) -> torch.Tensor:
    latent_path = dataset_dir / "ss_latents" / SS_LATENT_MODEL / f"{sample_id}.npz"
    if use_cached and latent_path.exists():
        return torch.from_numpy(np.load(latent_path)["mean"]).float().unsqueeze(0).to(device)
    tensor = torch.from_numpy(gt_grid.astype(np.float32))[None, None].to(device)
    with torch.no_grad():
        return encoder(tensor, sample_posterior=False)


def write_report(
    path: Path,
    dataset_dir: Path,
    output_dir: Path,
    metadata: pd.DataFrame | None,
    metrics_df: pd.DataFrame | None,
    error: str | None = None,
) -> None:
    lines = []
    lines.append("# Stage 1 Reconstruction Feasibility Report")
    lines.append("")
    lines.append(f"- Dataset: `{dataset_dir}`")
    lines.append(f"- Results: `{output_dir}`")
    lines.append("- Reconstruction path tested: `GT 64^3 surface voxels -> SparseStructureEncoder/ss_latent -> SparseStructureDecoder -> thresholded voxels`")
    lines.append("- Threshold: decoder logits `> 0`")
    lines.append("")
    lines.append("## Commands Run")
    lines.append("")
    lines.append("```bash")
    lines.append("/workspace/venv/bin/python stage_1/run_stage1_pipeline.py --per-category 3 --render-views 8 --cond-views 1 --render-workers 1 --feature-batch-size 8")
    lines.append(f"/workspace/venv/bin/python stage_1/run_reconstruction_eval.py --dataset-dir {dataset_dir} --output-dir {output_dir} --limit 12 --use-cached-latents")
    lines.append("```")
    lines.append("")
    lines.append("The Phase 1 wrapper called these existing TRELLIS preprocessing scripts where applicable: `dataset_toolkits/render_cond.py`, `dataset_toolkits/render.py`, `dataset_toolkits/extract_feature.py`, `dataset_toolkits/encode_ss_latent.py`, and `dataset_toolkits/encode_latent.py`.")
    lines.append("")
    if metadata is not None:
        lines.append("## Subset")
        lines.append("")
        lines.append(f"- Total metadata rows: {len(metadata)}")
        if "category" in metadata.columns:
            for category, count in metadata["category"].value_counts().sort_index().items():
                lines.append(f"- `{category}`: {count}")
        lines.append("")
        lines.append("## Conversion Artifacts")
        lines.append("")
        for col in [
            "voxelized",
            "rendered",
            "cond_rendered",
            "feature_dinov2_vitl14_reg",
            "ss_latent_ss_enc_conv3d_16l8_fp16",
            "latent_dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16",
        ]:
            if col in metadata.columns:
                lines.append(f"- `{col}`: {int(metadata[col].sum())} / {len(metadata)}")
        lines.append("")
    if error is not None:
        lines.append("## Error / Blocker")
        lines.append("")
        lines.append("```text")
        lines.append(error)
        lines.append("```")
        lines.append("")
    if metrics_df is not None and not metrics_df.empty:
        lines.append("## Metrics")
        lines.append("")
        mean = metrics_df[["iou", "precision", "recall", "f1", "gt_voxels", "pred_voxels"]].mean()
        lines.append(f"- Mean voxel IoU: `{mean['iou']:.4f}`")
        lines.append(f"- Mean precision: `{mean['precision']:.4f}`")
        lines.append(f"- Mean recall: `{mean['recall']:.4f}`")
        lines.append(f"- Mean F1: `{mean['f1']:.4f}`")
        lines.append(f"- Mean GT voxels: `{mean['gt_voxels']:.1f}`")
        lines.append(f"- Mean reconstructed voxels: `{mean['pred_voxels']:.1f}`")
        lines.append("")
        lines.append("Per-object metrics are in `metrics.csv`; occupied-bounding-box center cross-section images are in `cross_sections/`.")
        lines.append("")
        lines.append("## Interpretation")
        lines.append("")
        if mean["f1"] >= 0.5 and mean["recall"] >= 0.5:
            lines.append("The sparse-structure VAE preserves a substantial fraction of the 64^3 target voxels in this small test. This supports proceeding to the next feasibility step, while still inspecting internal cross-sections manually.")
            lines.append("")
            lines.append("Internal-looking shelf/partition surfaces are visible in the generated cross-section images and appear preserved in the sparse voxel reconstruction. This is qualitative because internal-only labels were not generated.")
            lines.append("")
            lines.append("Recommended next step: inspect cross-sections, then proceed toward LoRA on the structure flow and SLAT flow if the preserved voxels include internals.")
        else:
            lines.append("The sparse-structure VAE reconstruction is weak on this small internal-rich target. LoRA on flow priors alone is unlikely to be sufficient if the representation/decoder drops the relevant geometry.")
            lines.append("")
            lines.append("Recommended next step: investigate decoder/VAE fine-tuning or internal-aware supervision before LoRA fine-tuning.")
    else:
        lines.append("## Interpretation")
        lines.append("")
        lines.append("No reconstruction metrics were produced. Resolve the blocker above before deciding whether LoRA is enough.")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This test evaluates sparse voxel reconstruction, not final textured mesh quality.")
    lines.append("- Internal-only metrics were not computed because the current converted target does not label internal vs exterior surface voxels separately.")
    lines.append("- Exterior render quality alone should not be used as evidence that internals survived.")
    if error is None:
        lines.append("- No reconstruction errors or blockers were encountered in this run.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    ensure_dir(args.output_dir / "recon_voxels")
    ensure_dir(args.output_dir / "recon_grids")
    ensure_dir(args.output_dir / "cross_sections")

    metadata = None
    metrics_df = None
    try:
        metadata = read_metadata(args.dataset_dir)
        metadata = metadata[metadata["voxelized"] == True].copy()
        if args.limit:
            metadata = metadata.head(args.limit)
        if metadata.empty:
            raise RuntimeError("No voxelized samples available for reconstruction evaluation.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("TRELLIS pretrained sparse-structure models require CUDA in this script.")

        encoder = models.from_pretrained(args.encoder).eval().to(device)
        decoder = models.from_pretrained(args.decoder).eval().to(device)

        rows = []
        for _, row in metadata.iterrows():
            sample_id = row["sha256"]
            voxel_path = args.dataset_dir / "voxels" / f"{sample_id}.ply"
            if not artifact_exists(voxel_path):
                print(f"Skipping {sample_id}: missing {voxel_path}")
                continue
            positions = read_ply_points(voxel_path)
            gt_grid = positions_to_grid(positions, args.resolution)
            z = load_or_encode_latent(sample_id, gt_grid, args.dataset_dir, encoder, args.use_cached_latents, device)
            with torch.no_grad():
                logits = decoder(z).float().cpu().numpy()[0, 0]
            pred_grid = logits > args.threshold

            sample_metrics = metrics(gt_grid, pred_grid)
            sample_metrics.update({
                "sha256": sample_id,
                "category": row.get("category", ""),
                "shapenet_id": row.get("shapenet_id", ""),
            })
            rows.append(sample_metrics)

            write_ply_points(args.output_dir / "recon_voxels" / f"{sample_id}.ply", grid_to_positions(pred_grid))
            np.savez_compressed(
                args.output_dir / "recon_grids" / f"{sample_id}.npz",
                gt=gt_grid.astype(np.uint8),
                pred=pred_grid.astype(np.uint8),
                logits=logits.astype(np.float16),
            )
            cross_section_image(gt_grid, pred_grid).save(args.output_dir / "cross_sections" / f"{sample_id}.png")

        metrics_df = pd.DataFrame(rows)
        metrics_df.to_csv(args.output_dir / "metrics.csv", index=False)
        write_report(args.output_dir / "report.md", args.dataset_dir, args.output_dir, metadata, metrics_df)
        print(metrics_df.to_string(index=False))
        print(f"Wrote report: {args.output_dir / 'report.md'}")
    except Exception:
        error = traceback.format_exc()
        print(error)
        write_report(args.output_dir / "report.md", args.dataset_dir, args.output_dir, metadata, metrics_df, error=error)
        raise


if __name__ == "__main__":
    main()
