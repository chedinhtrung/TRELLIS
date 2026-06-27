from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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
    """Parse options for sparse-structure reconstruction evaluation."""
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
    """Compute voxel overlap metrics between target and reconstruction."""
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


def load_or_encode_latent(
    sample_id: str,
    gt_grid: np.ndarray,
    dataset_dir: Path,
    encoder,
    use_cached: bool,
    device: torch.device,
) -> torch.Tensor:
    """Load a cached SS latent when requested, otherwise encode the GT grid."""
    latent_path = dataset_dir / "ss_latents" / SS_LATENT_MODEL / f"{sample_id}.npz"
    if use_cached and latent_path.exists():
        return torch.from_numpy(np.load(latent_path)["mean"]).float().unsqueeze(0).to(device)

    tensor = torch.from_numpy(gt_grid.astype(np.float32))[None, None].to(device)
    with torch.no_grad():
        return encoder(tensor, sample_posterior=False)


def main() -> None:
    """Run sparse-structure reconstruction evaluation and save metrics."""
    args = parse_args()
    ensure_dir(args.output_dir)
    ensure_dir(args.output_dir / "recon_voxels")

    # Work only on samples whose converted 64^3 voxel target is available.
    metadata = read_metadata(args.dataset_dir)
    metadata = metadata[metadata["voxelized"] == True].copy()
    if args.limit:
        metadata = metadata.head(args.limit)
    if metadata.empty:
        raise RuntimeError("No voxelized samples available for reconstruction evaluation.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("TRELLIS pretrained sparse-structure models require CUDA in this script.")

    # Load the pretrained sparse-structure VAE components.
    encoder = models.from_pretrained(args.encoder).eval().to(device)
    decoder = models.from_pretrained(args.decoder).eval().to(device)

    rows = []
    for _, row in metadata.iterrows():
        sample_id = row["sha256"]
        voxel_path = args.dataset_dir / "voxels" / f"{sample_id}.ply"
        if not artifact_exists(voxel_path):
            print(f"Skipping {sample_id}: missing {voxel_path}")
            continue

        # Convert the sparse voxel PLY into the dense grid expected by the SS encoder.
        positions = read_ply_points(voxel_path)
        gt_grid = positions_to_grid(positions, args.resolution)

        latent = load_or_encode_latent(sample_id, gt_grid, args.dataset_dir, encoder, args.use_cached_latents, device)
        with torch.no_grad():
            logits = decoder(latent).float().cpu().numpy()[0, 0]

        # Threshold decoder logits into a binary occupancy grid, then compare
        # that grid against the original target grid.
        pred_grid = logits > args.threshold
        sample_metrics = metrics(gt_grid, pred_grid)
        sample_metrics.update({
            "sha256": sample_id,
            "category": row.get("category", ""),
            "shapenet_id": row.get("shapenet_id", ""),
        })
        rows.append(sample_metrics)

        # Save only the reconstructed sparse voxel PLY for this sample.
        write_ply_points(args.output_dir / "recon_voxels" / f"{sample_id}.ply", grid_to_positions(pred_grid))

    # The metrics CSV is the only summary artifact produced by this script.
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(args.output_dir / "metrics.csv", index=False)
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
