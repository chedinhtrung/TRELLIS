import argparse
import csv
import json
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


def read_categories(metadata_path: Path) -> dict[str, str]:
    with metadata_path.open(newline="") as f:
        return {row["sha256"]: row["category"] for row in csv.DictReader(f)}


def read_ids_file(path: Path, dataset_ids: list[str]) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    unknown = sorted(set(ids) - set(dataset_ids))
    if unknown:
        raise ValueError(f"IDs file contains samples absent from dataset metadata: {unknown}")
    return ids


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


def load_ss_lora(pipeline, ckpt_path: Path):
    from trellis.modules.lora import apply_lora

    model = pipeline.models["sparse_structure_flow_model"]
    model_cfg = _load_model_cfg_from_run(ckpt_path, "denoiser")
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
    state = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [key for key in unexpected if "lora_" in key or key.startswith("category_embedding.")]
    missing = [key for key in missing if "lora_" in key or key.startswith("category_embedding.")]
    if missing or unexpected:
        raise RuntimeError(f"LoRA checkpoint mismatch. missing={missing}, unexpected={unexpected}")
    model.eval()
    return categories


def coords_to_points(coords: torch.Tensor, resolution: int) -> np.ndarray:
    coords = coords.detach().cpu()
    coords = coords[coords[:, 0] == 0, 1:]
    return ((coords.float() + 0.5) / resolution - 0.5).numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TRELLIS ss_flow sparse-structure samples as voxel PLYs.")
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / "datasets/ShapeNetInternals_small")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/shapenet_internals_predictions/ss_flow_voxels")
    parser.add_argument("--pipeline", default="microsoft/TRELLIS-image-large")
    parser.add_argument("--lora-ckpt", type=Path, default=None, help="Optional LoRA checkpoint for sparse_structure_flow_model")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--cfg-strength", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ids-file", type=Path, default=None, help="Optional text file containing one sample ID per line")
    parser.add_argument("--view-index", type=int, default=0, help="Numeric renders_cond view index to use")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    metadata_path = args.dataset_dir / "metadata.csv"
    ids = read_ids(metadata_path)
    categories = read_categories(metadata_path)
    if args.ids_file is not None:
        ids = read_ids_file(args.ids_file, ids)
    if args.limit is not None:
        ids = ids[:args.limit]
    if args.view_index < 0:
        raise ValueError("--view-index must be non-negative")

    from trellis.pipelines import TrellisImageTo3DPipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.pipeline)
    pipeline.to(device)

    category_names = None
    if args.lora_ckpt is not None:
        print(f"Running with LoRA checkpoint: {args.lora_ckpt}")
        category_names = load_ss_lora(pipeline, args.lora_ckpt)
    else:
        print("Running base model (no LoRA checkpoint provided)")

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

        image_path = args.dataset_dir / "renders_cond" / sample_id / f"{args.view_index:03d}.png"
        if not image_path.exists():
            raise FileNotFoundError(f"Conditioning render not found: {image_path}")

        with Image.open(image_path) as image, torch.inference_mode():
            torch.manual_seed(args.seed)
            image = pipeline.preprocess_image(image)
            category = [categories[sample_id]] if category_names is not None else None
            cond = pipeline.get_cond([image], category=category)
            coords = pipeline.sample_sparse_structure(cond, num_samples=1, sampler_params=sampler_params)

        points = coords_to_points(coords, args.resolution)
        utils3d.io.write_ply(out_path, points)

    print(f"Wrote voxel PLYs to {args.output_dir}")


if __name__ == "__main__":
    main()
