#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import utils3d
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def read_ids(metadata_path: Path) -> list[str]:
    with metadata_path.open(newline="") as f:
        return [row["sha256"] for row in csv.DictReader(f)]


def load_lora(model, ckpt_path: Path) -> None:
    from trellis.modules.lora import apply_lora

    apply_lora(model, rank=8, alpha=8.0, dropout=0.0, target_patterns=["blocks."])
    state = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [key for key in missing if "lora_" in key]
    unexpected = [key for key in unexpected if "lora_" in key]
    if missing or unexpected:
        raise RuntimeError(f"LoRA checkpoint mismatch. missing={missing}, unexpected={unexpected}")


def mesh_to_voxel_points(mesh, resolution: int) -> np.ndarray:
    vertices = np.clip(mesh.vertices.detach().cpu().numpy(), -0.5 + 1e-6, 0.5 - 1e-6)
    faces = mesh.faces.detach().cpu().numpy()

    o3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(faces),
    )
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        o3d_mesh,
        voxel_size=1 / resolution,
        min_bound=(-0.5, -0.5, -0.5),
        max_bound=(0.5, 0.5, 0.5),
    )
    coords = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()], dtype=np.float32)
    if len(coords) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return ((coords + 0.5) / resolution - 0.5).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export full TRELLIS mesh outputs as voxelized PLYs.")
    parser.add_argument("--mode", choices=["base", "lora"], required=True)
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / "datasets/ShapeNetInternals_small")
    parser.add_argument("--pred-root", type=Path, default=REPO_ROOT / "results/shapenet_internals_lora/predictions")
    parser.add_argument("--pipeline", default="microsoft/TRELLIS-image-large")
    parser.add_argument("--ss-lora-ckpt", type=Path, default=REPO_ROOT / "results/shapenet_internals_lora/ss_flow/ckpts/denoiser_lora_step0002000.pt")
    parser.add_argument("--slat-lora-ckpt", type=Path, default=REPO_ROOT / "results/shapenet_internals_lora/slat_flow/ckpts/denoiser_lora_step0002000.pt")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    output_dir = args.pred_root / ("base_ss_slat_voxelized" if args.mode == "base" else "lora_ss_slat_voxelized")
    output_dir.mkdir(parents=True, exist_ok=True)

    ids = read_ids(args.dataset_dir / "metadata.csv")
    if args.limit is not None:
        ids = ids[:args.limit]

    from trellis.pipelines import TrellisImageTo3DPipeline

    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.pipeline)
    pipeline.to(torch.device(args.device))

    if args.mode == "lora":
        load_lora(pipeline.models["sparse_structure_flow_model"], args.ss_lora_ckpt)
        load_lora(pipeline.models["slat_flow_model"], args.slat_lora_ckpt)

    for sample_id in tqdm(ids, desc=f"Exporting {args.mode} full-pipeline voxels"):
        out_path = output_dir / f"{sample_id}.ply"
        if args.skip_existing and out_path.exists():
            continue

        image_path = args.dataset_dir / "renders_cond" / sample_id / "000.png"
        if not image_path.exists():
            print(f"Skipping missing render: {image_path}")
            continue

        with Image.open(image_path) as image, torch.inference_mode():
            torch.manual_seed(args.seed)
            cond = pipeline.get_cond([image])
            coords = pipeline.sample_sparse_structure(cond, num_samples=1)
            slat = pipeline.sample_slat(cond, coords)
            mesh = pipeline.decode_slat(slat, formats=["mesh"])["mesh"][0]

        points = mesh_to_voxel_points(mesh, args.resolution)
        utils3d.io.write_ply(out_path, points)

    print(f"Wrote voxel PLYs to {output_dir}")


if __name__ == "__main__":
    main()
