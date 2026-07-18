#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "dataset_toolkits"))

from render_kiui import (  # noqa: E402
    _build_axis_cutout_views,
    _load_mesh_compat,
    _render_cutout_views,
    _render_views,
)


METHODS = [
    ("base_flows__base_decoder", "Base flows + base decoder"),
    ("lora_flows__base_decoder", "LoRA flows + base decoder"),
    ("base_flows__lora_decoder", "Base flows + LoRA decoder"),
    ("lora_flows__lora_decoder", "LoRA flows + LoRA decoder"),
]
VIEW_LABELS = ["Input", "Exterior", "Cut +X", "Cut +Y", "Cut +Z"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render qualitative comparison sheets for the four Objective 1 methods."
    )
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--view-index", type=int, default=0)
    parser.add_argument("--samples-per-category", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--ssaa", type=float, default=1.0)
    args = parser.parse_args()

    if args.view_index < 0:
        parser.error("--view-index must be non-negative")
    if args.samples_per_category <= 0:
        parser.error("--samples-per-category must be positive")
    if args.resolution <= 0:
        parser.error("--resolution must be positive")
    if args.ssaa < 1:
        parser.error("--ssaa must be at least 1")
    if not torch.cuda.is_available():
        parser.error("CUDA is required by the nvdiffrast renderer")
    return args


def select_sample_ids(pred_root: Path, seed: int, samples_per_category: int):
    method_sets = []
    for method, _ in METHODS:
        mesh_dir = pred_root / method / f"seed_{seed}" / "mesh"
        if not mesh_dir.is_dir():
            raise FileNotFoundError(f"Missing mesh directory: {mesh_dir}")
        method_sets.append({path.stem for path in mesh_dir.glob("*.ply")})

    common_ids = sorted(set.intersection(*method_sets))
    if not common_ids:
        raise ValueError("No mesh IDs are shared by all four methods")

    selected = []
    counts = {}
    for sample_id in common_ids:
        category = sample_id.split("__", 1)[0]
        if counts.get(category, 0) < samples_per_category:
            selected.append((category, sample_id))
            counts[category] = counts.get(category, 0) + 1

    short = [category for category, count in counts.items() if count < samples_per_category]
    if short:
        raise ValueError(
            f"Not enough shared meshes for categories: {', '.join(short)}"
        )
    return selected


def composite_rgba(image: Image.Image, resolution: int):
    image = image.convert("RGBA")
    background = Image.new("RGB", image.size, "white")
    background.paste(image.convert("RGB"), mask=image.getchannel("A"))
    return background.resize((resolution, resolution), Image.Resampling.LANCZOS)


def tensor_to_image(rgba: torch.Tensor):
    array = (rgba.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    return composite_rgba(Image.fromarray(array, mode="RGBA"), array.shape[0])


def render_mesh(mesh_path: Path, log_path: Path, resolution: int, ssaa: float):
    mesh = _load_mesh_compat(str(mesh_path), str(log_path), mesh_path.stem)
    exterior_view = [{
        "yaw": np.deg2rad(45.0),
        "pitch": np.deg2rad(25.0),
        "radius": 2.0,
        "fov": np.deg2rad(40.0),
    }]
    axis_views = _build_axis_cutout_views(6)
    cutout_views = [axis_views[index] for index in (0, 2, 4)]

    exterior, _ = _render_views(mesh, exterior_view, resolution, ssaa=ssaa)
    cutouts, _ = _render_cutout_views(mesh, cutout_views, resolution, ssaa=ssaa)
    return [tensor_to_image(exterior[0])] + [tensor_to_image(view) for view in cutouts]


def make_sheet(input_image: Image.Image, method_images, resolution: int):
    label_width = 220
    header_height = 28
    sheet = Image.new(
        "RGB",
        (label_width + len(VIEW_LABELS) * resolution, header_height + len(METHODS) * resolution),
        "white",
    )
    draw = ImageDraw.Draw(sheet)

    for column, label in enumerate(VIEW_LABELS):
        draw.text((label_width + column * resolution + 5, 7), label, fill="black")

    for row, ((_, method_label), images) in enumerate(zip(METHODS, method_images)):
        y = header_height + row * resolution
        draw.text((5, y + 8), method_label, fill="black")
        for column, image in enumerate([input_image] + images):
            sheet.paste(image, (label_width + column * resolution, y))
    return sheet


def main():
    args = parse_args()
    selected = select_sample_ids(args.pred_root, args.seed, args.samples_per_category)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "render.log"
    image_name = f"{args.view_index:03d}.png"

    for category, sample_id in tqdm(selected, desc="Comparison sheets"):
        input_path = args.dataset_dir / "renders_cond" / sample_id / image_name
        if not input_path.is_file():
            raise FileNotFoundError(f"Missing conditioning image: {input_path}")
        with Image.open(input_path) as image:
            input_image = composite_rgba(image, args.resolution)

        method_images = []
        for method, _ in METHODS:
            mesh_path = args.pred_root / method / f"seed_{args.seed}" / "mesh" / f"{sample_id}.ply"
            method_images.append(render_mesh(mesh_path, log_path, args.resolution, args.ssaa))

        category_dir = args.output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        make_sheet(input_image, method_images, args.resolution).save(category_dir / f"{sample_id}.png")

    (args.output_dir / "selected_ids.txt").write_text(
        "\n".join(sample_id for _, sample_id in selected) + "\n"
    )
    print(f"Wrote {len(selected)} comparison sheets to {args.output_dir}")


if __name__ == "__main__":
    main()
