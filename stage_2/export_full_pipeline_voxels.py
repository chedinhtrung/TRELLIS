import argparse
import csv
import json
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


def read_categories(metadata_path: Path) -> dict[str, str]:
    with metadata_path.open(newline="") as f:
        return {row["sha256"]: row["category"] for row in csv.DictReader(f)}


def read_ids_file(path: Path, dataset_ids: list[str]) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    unknown = sorted(set(ids) - set(dataset_ids))
    if unknown:
        raise ValueError(f"IDs file contains samples absent from dataset metadata: {unknown}")
    return ids


def read_gt_coords(path: Path, resolution: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"GT SLAT latent not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        if "coords" not in data:
            raise ValueError(f"GT SLAT latent has no 'coords' array: {path}")
        coords = np.asarray(data["coords"])

    if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) == 0:
        raise ValueError(f"Expected non-empty GT coordinates with shape [N, 3], got {coords.shape}: {path}")
    if not np.issubdtype(coords.dtype, np.integer):
        raise ValueError(f"GT coordinates must have an integer dtype, got {coords.dtype}: {path}")

    coords = coords.astype(np.int32, copy=False)
    if coords.min() < 0 or coords.max() >= resolution:
        raise ValueError(f"GT coordinates fall outside [0, {resolution - 1}]: {path}")
    if len(np.unique(coords, axis=0)) != len(coords):
        raise ValueError(f"GT coordinates contain duplicates: {path}")
    return coords


def add_batch_column(coords: np.ndarray, device: torch.device) -> torch.Tensor:
    batched = np.concatenate(
        [np.zeros((len(coords), 1), dtype=np.int32), coords],
        axis=1,
    )
    return torch.from_numpy(batched).to(device=device).contiguous()


def coordinate_noise(
    coords: torch.Tensor,
    channels: int,
    resolution: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    if coords.ndim != 2 or coords.shape[1] != 4 or torch.any(coords[:, 0] != 0):
        raise ValueError("Coordinate-controlled SLAT noise requires one sample with [N, 4] coordinates")
    xyz = coords[:, 1:].long()
    if torch.any(xyz < 0) or torch.any(xyz >= resolution):
        raise ValueError(f"SLAT coordinates fall outside [0, {resolution - 1}]")

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    field = torch.randn(
        resolution,
        resolution,
        resolution,
        channels,
        generator=generator,
        device=device,
    )
    return field[xyz[:, 0], xyz[:, 1], xyz[:, 2]]


def _load_model_cfg_from_run(ckpt_path: Path, model_key: str) -> dict:
    """Load and validate LoRA hyperparameters from the checkpoint's run config."""
    config_path = ckpt_path.parents[1] / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"LoRA run config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    model_cfg = cfg.get("models", {}).get(model_key, {})
    lora_cfg = model_cfg.get("lora")
    if not isinstance(lora_cfg, dict):
        raise ValueError(f"Missing models.{model_key}.lora configuration in {config_path}")
    required = {"rank", "alpha", "dropout", "target_patterns"}
    missing = sorted(required - set(lora_cfg))
    if missing:
        raise ValueError(f"Incomplete models.{model_key}.lora configuration in {config_path}: missing {missing}")
    return model_cfg


def load_lora(model, ckpt_path: Path, *, model_key: str = "denoiser"):
    from trellis.modules.lora import apply_lora

    model_cfg = _load_model_cfg_from_run(ckpt_path, model_key)
    lora_cfg = model_cfg["lora"]
    apply_lora(
        model,
        rank=lora_cfg["rank"],
        alpha=lora_cfg["alpha"],
        dropout=lora_cfg["dropout"],
        target_patterns=lora_cfg["target_patterns"],
    )
    categories = model_cfg.get("categories")
    if categories is not None:
        model.enable_category_conditioning(categories)
    interior_expert_cfg = model_cfg.get("interior_expert")
    if interior_expert_cfg is not None:
        if not hasattr(model, "enable_interior_expert"):
            raise ValueError(f"{model.__class__.__name__} does not support an interior expert")
        model.enable_interior_expert(
            hidden_channels=interior_expert_cfg.get("hidden_channels", 256),
            margin=interior_expert_cfg.get("margin", 2),
        )

    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    expected = {
        key
        for key in model.state_dict()
        if ".lora_down" in key
        or ".lora_up" in key
        or key.startswith("category_embedding.")
        or key.startswith("interior_expert.")
    }
    if set(state) != expected:
        raise RuntimeError(
            f"Adapter checkpoint mismatch. missing={sorted(expected - set(state))}, "
            f"unexpected={sorted(set(state) - expected)}"
        )
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [key for key in missing if key in expected]
    if missing or unexpected:
        raise RuntimeError(f"Adapter checkpoint mismatch. missing={missing}, unexpected={unexpected}")
    model.eval()
    return categories


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
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / "datasets/ShapeNetInternals_small")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/shapenet_internals_lora/predictions/full_pipeline")
    parser.add_argument("--pipeline", default="microsoft/TRELLIS-image-large")
    parser.add_argument("--ss-lora-ckpt", type=Path, default=None, help="Optional LoRA checkpoint for sparse_structure_flow_model")
    parser.add_argument("--slat-lora-ckpt", type=Path, default=None, help="Optional LoRA checkpoint for slat_flow_model")
    parser.add_argument("--decoder-lora-ckpt", type=Path, default=None, help="Optional LoRA checkpoint for slat_decoder_mesh")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ids-file", type=Path, default=None, help="Optional text file containing one sample ID per line")
    parser.add_argument("--view-index", type=int, default=0, help="Numeric renders_cond view index to use")
    parser.add_argument(
        "--gt-coords-latent-name",
        default=None,
        metavar="NAME",
        help=(
            "Bypass sparse-structure sampling and load GT coordinates from "
            "latents/NAME/<sample_id>.npz. Only coordinates are loaded; SLAT features are still generated."
        ),
    )
    parser.add_argument(
        "--slat-seed",
        type=int,
        default=None,
        help="Optional seed for coordinate-indexed SLAT noise used in controlled comparisons",
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    method_dir = args.output_dir
    mesh_dir = method_dir / "mesh"
    voxel_dir = method_dir / "voxels"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    voxel_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = args.dataset_dir / "metadata.csv"
    ids = read_ids(metadata_path)
    categories = read_categories(metadata_path)
    if args.ids_file is not None:
        ids = read_ids_file(args.ids_file, ids)
    if args.limit is not None:
        ids = ids[:args.limit]
    if args.view_index < 0:
        raise ValueError("--view-index must be non-negative")
    if args.seed < 0 or (args.slat_seed is not None and args.slat_seed < 0):
        raise ValueError("--seed and --slat-seed must be non-negative")

    image_paths = {
        sample_id: args.dataset_dir / "renders_cond" / sample_id / f"{args.view_index:03d}.png"
        for sample_id in ids
    }
    for image_path in image_paths.values():
        if not image_path.is_file():
            raise FileNotFoundError(f"Conditioning render not found: {image_path}")

    gt_coords = None
    if args.gt_coords_latent_name is not None:
        latent_dir = args.dataset_dir / "latents" / args.gt_coords_latent_name
        gt_coords = {
            sample_id: read_gt_coords(latent_dir / f"{sample_id}.npz", args.resolution)
            for sample_id in ids
        }
        print(f"Using GT coordinates from {latent_dir}; sparse-structure sampling is disabled")

    for checkpoint in (args.ss_lora_ckpt, args.slat_lora_ckpt, args.decoder_lora_ckpt):
        if checkpoint is not None and not checkpoint.is_file():
            raise FileNotFoundError(f"LoRA checkpoint not found: {checkpoint}")

    from trellis.pipelines import TrellisImageTo3DPipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.pipeline)
    pipeline.to(device)

    use_flow_lora = args.ss_lora_ckpt is not None and args.slat_lora_ckpt is not None
    use_decoder_lora = args.decoder_lora_ckpt is not None

    if (args.ss_lora_ckpt is None) != (args.slat_lora_ckpt is None):
        raise ValueError("Provide both --ss-lora-ckpt and --slat-lora-ckpt, or neither")

    if not use_flow_lora and not use_decoder_lora:
        print("Running base model (no LoRA checkpoints provided)")

    category_names = None
    if use_flow_lora:
        print(f"Applying flow LoRA checkpoints: ss={args.ss_lora_ckpt}, slat={args.slat_lora_ckpt}")
        ss_categories = load_lora(pipeline.models["sparse_structure_flow_model"], args.ss_lora_ckpt, model_key="denoiser")
        slat_categories = load_lora(pipeline.models["slat_flow_model"], args.slat_lora_ckpt, model_key="denoiser")
        if ss_categories != slat_categories:
            raise ValueError("SS-flow and SLAT-flow category configurations do not match")
        category_names = ss_categories

    if use_decoder_lora:
        print(f"Applying decoder LoRA checkpoint: decoder={args.decoder_lora_ckpt}")
        load_lora(pipeline.models["slat_decoder_mesh"], args.decoder_lora_ckpt, model_key="decoder")

    for sample_id in tqdm(ids, desc="Exporting full-pipeline voxels"):
        mesh_out_path = mesh_dir / f"{sample_id}.ply"
        voxel_out_path = voxel_dir / f"{sample_id}.ply"
        if args.skip_existing and mesh_out_path.exists() and voxel_out_path.exists():
            continue

        with Image.open(image_paths[sample_id]) as image, torch.inference_mode():
            torch.manual_seed(args.seed)
            image = pipeline.preprocess_image(image)
            category = [categories[sample_id]] if category_names is not None else None
            cond = pipeline.get_cond([image], category=category)
            if gt_coords is None:
                coords = pipeline.sample_sparse_structure(cond, num_samples=1)
            else:
                coords = add_batch_column(gt_coords[sample_id], device)
            slat_noise = None
            if args.slat_seed is not None:
                flow_model = pipeline.models["slat_flow_model"]
                slat_noise = coordinate_noise(
                    coords,
                    flow_model.in_channels,
                    flow_model.resolution,
                    args.slat_seed,
                    device,
                )
            slat = pipeline.sample_slat(cond, coords, noise_feats=slat_noise)
            mesh = pipeline.decode_slat(slat, formats=["mesh"])["mesh"][0]

        utils3d.io.write_ply(mesh_out_path, mesh.vertices.detach().cpu().numpy(), mesh.faces.detach().cpu().numpy())
        points = mesh_to_voxel_points(mesh, args.resolution)
        utils3d.io.write_ply(voxel_out_path, points)

    print(f"Wrote mesh PLYs to {mesh_dir}")
    print(f"Wrote voxel PLYs to {voxel_dir}")


if __name__ == "__main__":
    main()
