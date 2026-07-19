"""Extract visibility-checked DINO features from internal cutaway renders.

The cutaway renderer writes RGB, camera depth, camera transforms, and the
cutting plane for every view.  A voxel contributes to a view only when it is
inside the retained half-space and agrees with the rendered depth buffer.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import utils3d
from PIL import Image, ImageDraw
from torchvision import transforms
from tqdm import tqdm


torch.set_grad_enabled(False)

VOXEL_RESOLUTION = 64
IMAGE_SIZE = 518
PATCH_SIZE = 14
RENDER_RECIPE = "internal_cutaway_v1"
EXPECTED_VIEWS = 18


def _read_instances(value):
    if value is None:
        return None
    if os.path.isfile(value):
        with open(value, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_image(path, normalize):
    image = Image.open(path).convert("RGBA").resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS
    )
    image = np.asarray(image, dtype=np.float32) / 255.0
    image = image[..., :3] * image[..., 3:4]
    image = torch.from_numpy(image).permute(2, 0, 1)
    return normalize(image)


def _load_depth(path):
    loaded = np.load(path)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        key = "depth" if "depth" in loaded.files else loaded.files[0]
        depth = loaded[key]
        loaded.close()
    else:
        depth = loaded
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"Depth map must be HxW, got {depth.shape}: {path}")
    return torch.from_numpy(depth)


def _camera(frame):
    c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
    # Renderer stores Blender/OpenGL c2w.  utils3d.project_cv expects OpenCV.
    c2w[:3, 1:3] *= -1
    extrinsics = torch.inverse(c2w)
    fov = torch.tensor(frame["camera_angle_x"], dtype=torch.float32)
    intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
    return extrinsics, intrinsics


def _halfspace_mask(positions, frame):
    normal = torch.tensor(
        frame["plane_normal"], dtype=positions.dtype, device=positions.device
    )
    offset = float(frame["plane_offset"])
    signed = positions @ normal
    keep = frame.get("keep", "le")
    if keep == "le":
        return signed <= offset + 1e-6
    if keep == "ge":
        return signed >= offset - 1e-6
    raise ValueError(f"Unsupported cutting-plane rule: {keep}")


def _depth_mask(uv, point_depth, depth_map, tolerance):
    """Test depth agreement against the closest valid value in a 3x3 window."""
    height, width = depth_map.shape
    px = torch.round(uv[:, 0] * (width - 1)).long()
    py = torch.round(uv[:, 1] * (height - 1)).long()

    best_difference = torch.full_like(point_depth, float("inf"))
    foreground = torch.zeros_like(point_depth, dtype=torch.bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            x = (px + dx).clamp(0, width - 1)
            y = (py + dy).clamp(0, height - 1)
            rendered_depth = depth_map[y, x]
            valid_depth = torch.isfinite(rendered_depth) & (rendered_depth > 0)
            difference = torch.abs(rendered_depth - point_depth)
            best_difference = torch.minimum(
                best_difference,
                torch.where(valid_depth, difference, torch.full_like(difference, float("inf"))),
            )
            foreground |= valid_depth
    return foreground & (best_difference <= tolerance)


def _save_projection_overlay(image_path, output_path, uv, visible):
    """Draw accepted voxel projections in green for the pilot visual check."""
    image = Image.open(image_path).convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS
    )
    points = uv[visible].detach().cpu().numpy()
    if len(points) > 2000:
        points = points[np.linspace(0, len(points) - 1, 2000, dtype=np.int64)]
    draw = ImageDraw.Draw(image)
    for u, v in points:
        x = int(round(float(u) * (IMAGE_SIZE - 1)))
        y = int(round(float(v) * (IMAGE_SIZE - 1)))
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(0, 255, 0))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)


def _extract_one(
    sha256,
    data_dir,
    render_name,
    feature_name,
    model,
    normalize,
    batch_size,
    tolerance,
    diagnostics_dir,
):
    render_dir = os.path.join(data_dir, render_name, sha256)
    with open(os.path.join(render_dir, "transforms.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("recipe_version") != RENDER_RECIPE:
        raise ValueError(
            f"Expected render recipe {RENDER_RECIPE}, got {manifest.get('recipe_version')}"
        )
    frames = manifest.get("frames", [])
    if len(frames) != EXPECTED_VIEWS:
        raise ValueError(f"Expected {EXPECTED_VIEWS} internal views, got {len(frames)}")
    required_fields = {
        "file_path", "depth_path", "transform_matrix", "camera_angle_x",
        "plane_normal", "plane_offset", "keep",
    }
    for frame in frames:
        missing = required_fields - set(frame)
        if missing:
            raise ValueError(f"Internal render frame is missing fields: {sorted(missing)}")

    positions = utils3d.io.read_ply(
        os.path.join(data_dir, "voxels", f"{sha256}.ply")
    )[0]
    positions = torch.from_numpy(positions).float().cuda()
    indices = ((positions + 0.5) * VOXEL_RESOLUTION).long()
    if not torch.all((indices >= 0) & (indices < VOXEL_RESOLUTION)):
        raise ValueError("Voxel coordinates outside the 64^3 grid")

    feature_sum = None
    view_count = torch.zeros(positions.shape[0], dtype=torch.int32, device="cuda")
    patch_grid_size = IMAGE_SIZE // PATCH_SIZE

    for start in range(0, len(frames), batch_size):
        batch_frames = frames[start : start + batch_size]
        images = []
        depths = []
        extrinsics = []
        intrinsics = []
        for frame in batch_frames:
            images.append(_load_image(os.path.join(render_dir, frame["file_path"]), normalize))
            depths.append(_load_depth(os.path.join(render_dir, frame["depth_path"])))
            ext, intr = _camera(frame)
            extrinsics.append(ext)
            intrinsics.append(intr)

        images = torch.stack(images).cuda()
        extrinsics = torch.stack(extrinsics).cuda()
        intrinsics = torch.stack(intrinsics).cuda()
        output = model(images, is_training=True)
        tokens = output["x_prenorm"][:, model.num_register_tokens + 1 :]
        feature_dim = tokens.shape[-1]
        expected_tokens = patch_grid_size * patch_grid_size
        if tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"Expected {expected_tokens} DINO patches, got {tokens.shape[1]}"
            )
        patchtokens = tokens.permute(0, 2, 1).reshape(
            len(batch_frames), feature_dim, patch_grid_size, patch_grid_size
        )
        if feature_sum is None:
            feature_sum = torch.zeros(
                positions.shape[0], feature_dim, dtype=torch.float32, device="cuda"
            )

        projected_uv, projected_depth = utils3d.torch.project_cv(
            positions, extrinsics, intrinsics
        )
        projected_depth = projected_depth.squeeze(-1)

        for local_index, frame in enumerate(batch_frames):
            uv = projected_uv[local_index]
            point_depth = projected_depth[local_index]
            in_frame = (
                (uv[:, 0] >= 0)
                & (uv[:, 0] <= 1)
                & (uv[:, 1] >= 0)
                & (uv[:, 1] <= 1)
                & torch.isfinite(point_depth)
                & (point_depth > 0)
            )
            halfspace = _halfspace_mask(positions, frame)
            depth = depths[local_index].to(device=positions.device)
            visible = in_frame & halfspace & _depth_mask(
                uv, point_depth, depth, tolerance
            )
            if diagnostics_dir and start == 0 and local_index == 0:
                _save_projection_overlay(
                    os.path.join(render_dir, frame["file_path"]),
                    os.path.join(diagnostics_dir, sha256, "000_projection.png"),
                    uv,
                    visible,
                )
            if not torch.any(visible):
                continue

            visible_indices = torch.where(visible)[0]
            grid = uv[visible_indices].mul(2).sub(1).reshape(1, -1, 1, 2)
            sampled = F.grid_sample(
                patchtokens[local_index : local_index + 1],
                grid,
                mode="bilinear",
                align_corners=False,
            )[0, :, :, 0].transpose(0, 1).float()
            feature_sum[visible_indices] += sampled
            view_count[visible_indices] += 1

    if feature_sum is None:
        raise RuntimeError("DINO produced no features")
    covered = view_count > 0
    features = torch.zeros_like(feature_sum)
    features[covered] = feature_sum[covered] / view_count[covered, None].float()
    coverage = covered.float().mean().item()

    save_dir = os.path.join(data_dir, "features", feature_name)
    os.makedirs(save_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(save_dir, f"{sha256}.npz"),
        indices=indices.cpu().numpy().astype(np.uint8),
        patchtokens=features.cpu().numpy().astype(np.float16),
        view_count=view_count.cpu().numpy().astype(np.uint16),
        coverage=np.float32(coverage),
        depth_tolerance_voxels=np.float32(tolerance * VOXEL_RESOLUTION),
        render_recipe=np.asarray(RENDER_RECIPE),
    )
    return coverage, int(covered.sum().item()), int(covered.numel())


def main():
    parser = argparse.ArgumentParser(
        description="Extract visibility-checked DINO features from cutaway renders"
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--instances", default=None, help="Text file or comma-separated IDs")
    parser.add_argument("--render_name", default="renders_internal_v1")
    parser.add_argument("--feature_name", default="dinov2_vitl14_reg_internal_v1")
    parser.add_argument("--model", default="dinov2_vitl14_reg")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--depth_tolerance_voxels", type=float, default=1.5)
    parser.add_argument("--diagnostics_dir", default=None)
    parser.add_argument("--override", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    if args.depth_tolerance_voxels <= 0:
        parser.error("--depth_tolerance_voxels must be positive")

    metadata_path = os.path.join(args.data_dir, "metadata.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.csv not found: {metadata_path}")
    metadata = pd.read_csv(metadata_path)
    instances = _read_instances(args.instances)
    if instances is not None:
        metadata_ids = set(metadata["sha256"].astype(str))
        missing = sorted(set(instances) - metadata_ids)
        if missing:
            raise ValueError(f"Instances not found in metadata.csv: {missing}")
        metadata = metadata[metadata["sha256"].astype(str).isin(instances)]

    sha256s = metadata["sha256"].astype(str).tolist()
    if not sha256s:
        raise ValueError("No instances selected")
    feature_dir = os.path.join(args.data_dir, "features", args.feature_name)
    os.makedirs(feature_dir, exist_ok=True)

    model = torch.hub.load("facebookresearch/dinov2", args.model)
    model.eval().cuda()
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    tolerance = args.depth_tolerance_voxels / VOXEL_RESOLUTION

    records = []
    for sha256 in tqdm(sha256s, desc="Extracting internal features"):
        save_path = os.path.join(feature_dir, f"{sha256}.npz")
        overlay_path = (
            os.path.join(args.diagnostics_dir, sha256, "000_projection.png")
            if args.diagnostics_dir
            else None
        )
        diagnostics_missing = overlay_path is not None and not os.path.exists(overlay_path)
        if os.path.exists(save_path) and not args.override and not diagnostics_missing:
            with np.load(save_path) as saved:
                coverage = float(saved["coverage"])
                covered = int(np.count_nonzero(saved["view_count"]))
                total = int(saved["view_count"].shape[0])
            records.append(
                {"sha256": sha256, "success": True, "coverage": coverage,
                 "covered_voxels": covered, "total_voxels": total}
            )
            continue
        try:
            coverage, covered, total = _extract_one(
                sha256=sha256,
                data_dir=args.data_dir,
                render_name=args.render_name,
                feature_name=args.feature_name,
                model=model,
                normalize=normalize,
                batch_size=args.batch_size,
                tolerance=tolerance,
                diagnostics_dir=args.diagnostics_dir,
            )
            records.append(
                {"sha256": sha256, "success": True, "coverage": coverage,
                 "covered_voxels": covered, "total_voxels": total}
            )
        except Exception as error:
            print(f"\nError extracting {sha256}: {error}")
            records.append(
                {"sha256": sha256, "success": False, "coverage": 0.0,
                 "covered_voxels": 0, "total_voxels": 0, "error": str(error)}
            )

    report_path = os.path.join(args.data_dir, "internal_feature_v1_report.csv")
    pd.DataFrame.from_records(records).to_csv(report_path, index=False)
    successful = [record for record in records if record["success"]]
    if successful:
        mean_coverage = np.mean([record["coverage"] for record in successful])
        print(f"Saved {len(successful)}/{len(records)} objects; mean coverage={mean_coverage:.4f}")
    failed = [record for record in records if not record["success"]]
    if failed:
        raise RuntimeError(
            f"Internal feature extraction failed for {len(failed)}/{len(records)} objects; "
            f"see {report_path}"
        )


if __name__ == "__main__":
    main()
