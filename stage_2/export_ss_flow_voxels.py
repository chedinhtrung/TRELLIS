#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import utils3d
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def read_ids(metadata_path: Path) -> list[str]:
    with metadata_path.open(newline="") as f:
        return [row["sha256"] for row in csv.DictReader(f)]


def load_ss_lora(pipeline, ckpt_path: Path) -> None:
    from trellis.modules.lora import apply_lora

    model = pipeline.models["sparse_structure_flow_model"]
    apply_lora(model, rank=8, alpha=8.0, dropout=0.0, target_patterns=["blocks."])
    state = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [key for key in unexpected if "lora_" in key]
    missing = [key for key in missing if "lora_" in key]
    if missing or unexpected:
        raise RuntimeError(f"LoRA checkpoint mismatch. missing={missing}, unexpected={unexpected}")


def coords_to_points(coords: torch.Tensor, resolution: int) -> np.ndarray:
    coords = coords.detach().cpu()
    coords = coords[coords[:, 0] == 0, 1:]
    return ((coords.float() + 0.5) / resolution - 0.5).numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TRELLIS ss_flow sparse-structure samples as voxel PLYs.")
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / "datasets/ShapeNetInternals_small")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/shapenet_internals_predictions/ss_flow_voxels")
    parser.add_argument("--pipeline", default="microsoft/TRELLIS-image-large")
    parser.add_argument("--lora-ckpt", type=Path, default=REPO_ROOT / "results/shapenet_internals_lora/ss_flow/ckpts/denoiser_lora_step0002000.pt")
    parser.add_argument("--no-lora", action="store_true", help="Export the base pretrained ss_flow model.")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--cfg-strength", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    ids = read_ids(args.dataset_dir / "metadata.csv")
    if args.limit is not None:
        ids = ids[:args.limit]

    from trellis.pipelines import TrellisImageTo3DPipeline

    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.pipeline)
    pipeline.to(torch.device(args.device))

    if not args.no_lora:
        load_ss_lora(pipeline, args.lora_ckpt)

    sampler_params = {}
    if args.steps is not None:
        sampler_params["steps"] = args.steps
    if args.cfg_strength is not None:
        sampler_params["cfg_strength"] = args.cfg_strength

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for sample_id in tqdm(ids, desc="Exporting ss_flow voxels"):
        out_path = args.output_dir / f"{sample_id}.ply"
        if args.skip_existing and out_path.exists():
            continue

        image_path = args.dataset_dir / "renders_cond" / sample_id / "000.png"
        if not image_path.exists():
            print(f"Skipping missing render: {image_path}")
            continue

        with Image.open(image_path) as image, torch.inference_mode():
            torch.manual_seed(args.seed)
            cond = pipeline.get_cond([image])
            coords = pipeline.sample_sparse_structure(cond, num_samples=1, sampler_params=sampler_params)

        points = coords_to_points(coords, args.resolution)
        utils3d.io.write_ply(out_path, points)

    print(f"Wrote voxel PLYs to {args.output_dir}")


if __name__ == "__main__":
    main()
