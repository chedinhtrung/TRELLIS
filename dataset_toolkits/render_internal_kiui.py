#!/usr/bin/env python3
"""Render deterministic cutaway views used to build interior SLAT targets.

The input mesh is the canonical, normalized mesh already produced by
``render_kiui.py``.  This script never changes the ordinary render directory.
"""

import argparse
import csv
import json
import os
import shutil
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image

import nvdiffrast.torch as dr
from kiui.cam import get_perspective
from kiui.mesh import Mesh


RECIPE_VERSION = "internal_cutaway_v1"
CUT_FRACTIONS = (0.25, 0.50, 0.75)
AXES = (
    ("+x", (1.0, 0.0, 0.0)),
    ("-x", (-1.0, 0.0, 0.0)),
    ("+y", (0.0, 1.0, 0.0)),
    ("-y", (0.0, -1.0, 0.0)),
    ("+z", (0.0, 0.0, 1.0)),
    ("-z", (0.0, 0.0, -1.0)),
)
NUM_VIEWS = len(AXES) * len(CUT_FRACTIONS)


def _camera_pose(camera_position: np.ndarray) -> np.ndarray:
    """Return an OpenGL camera-to-world matrix (-Z forward, +Y up)."""
    target = np.zeros(3, dtype=np.float32)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    fallback_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    forward = target - camera_position
    forward /= np.linalg.norm(forward)
    camera_z = -forward
    camera_y = world_up - np.dot(world_up, camera_z) * camera_z
    if np.linalg.norm(camera_y) < 1e-6:
        camera_y = fallback_up - np.dot(fallback_up, camera_z) * camera_z
    camera_y /= np.linalg.norm(camera_y)
    camera_x = np.cross(camera_y, camera_z)
    camera_x /= np.linalg.norm(camera_x)

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.stack([camera_x, camera_y, camera_z], axis=1)
    pose[:3, 3] = camera_position
    return pose


def _load_canonical_mesh(path: Path, device: torch.device) -> Mesh:
    loaded = trimesh.load(path, process=False)
    tm = loaded.to_mesh() if isinstance(loaded, trimesh.Scene) else loaded
    if tm is None or len(tm.vertices) == 0 or len(tm.faces) == 0:
        raise ValueError(f"Empty mesh: {path}")

    mesh = Mesh(
        v=torch.as_tensor(np.asarray(tm.vertices), dtype=torch.float32, device=device),
        f=torch.as_tensor(np.asarray(tm.faces), dtype=torch.int32, device=device),
        device=device,
    )

    # Canonical PLY files normally contain baked vertex colors.  UV textures
    # are also retained when trimesh exposes both UV coordinates and an image.
    visual = getattr(tm, "visual", None)
    uv = getattr(visual, "uv", None)
    material = getattr(visual, "material", None)
    texture = getattr(material, "image", None)
    if uv is not None and texture is not None and len(uv) == len(tm.vertices):
        texture = np.asarray(texture.convert("RGB") if hasattr(texture, "convert") else texture)
        mesh.vt = torch.as_tensor(np.asarray(uv), dtype=torch.float32, device=device)
        mesh.ft = mesh.f.clone()
        mesh.albedo = torch.as_tensor(texture[..., :3], dtype=torch.float32, device=device) / 255.0
    else:
        try:
            colors = np.asarray(visual.to_color().vertex_colors)[..., :3]
            if len(colors) == len(tm.vertices):
                mesh.vc = torch.as_tensor(colors, dtype=torch.float32, device=device) / 255.0
        except Exception:
            pass
    return mesh


def _edge_intersection(a, b, distance_a, distance_b):
    denominator = distance_a - distance_b
    safe_denominator = torch.where(
        denominator.abs() > 1e-12,
        denominator,
        torch.ones_like(denominator),
    )
    t = (distance_a / safe_denominator).unsqueeze(-1)
    return a + t * (b - a)


def _candidate_values(values: torch.Tensor, signed_distance: torch.Tensor) -> torch.Tensor:
    """Original triangle corners plus intersections for edges 01, 12, 20."""
    v0, v1, v2 = values[:, 0], values[:, 1], values[:, 2]
    d0, d1, d2 = signed_distance[:, 0], signed_distance[:, 1], signed_distance[:, 2]
    return torch.stack(
        [
            v0,
            v1,
            v2,
            _edge_intersection(v0, v1, d0, d1),
            _edge_intersection(v1, v2, d1, d2),
            _edge_intersection(v2, v0, d2, d0),
        ],
        dim=1,
    )


def clip_mesh(mesh: Mesh, plane_normal: torch.Tensor, plane_offset: float) -> Mesh:
    """Exactly clip triangles to ``dot(normal, position) <= offset``.

    Crossing edges get new interpolated vertices.  The open cut is deliberately
    not capped, so the original internal surfaces remain visible to the camera.
    """
    faces = mesh.f.long()
    positions = mesh.v[faces]
    signed_distance = positions @ plane_normal - plane_offset
    inside = signed_distance <= 1e-7
    case = (
        inside[:, 0].to(torch.int64)
        + 2 * inside[:, 1].to(torch.int64)
        + 4 * inside[:, 2].to(torch.int64)
    )

    # Candidate order: vertex 0, 1, 2, edge 01, edge 12, edge 20.
    # The templates preserve the winding of the original triangle.
    templates = {
        1: ((5, 0, 3),),
        2: ((3, 1, 4),),
        3: ((5, 0, 1), (5, 1, 4)),
        4: ((5, 4, 2),),
        5: ((0, 3, 4), (0, 4, 2)),
        6: ((5, 3, 1), (5, 1, 2)),
        7: ((0, 1, 2),),
    }

    position_candidates = _candidate_values(positions, signed_distance)
    color_candidates = None
    if mesh.vc is not None and mesh.vc.shape[0] == mesh.v.shape[0]:
        color_candidates = _candidate_values(mesh.vc[faces], signed_distance)

    uv_candidates = None
    if (
        mesh.vt is not None
        and mesh.ft is not None
        and mesh.ft.shape[0] == mesh.f.shape[0]
    ):
        uv_candidates = _candidate_values(mesh.vt[mesh.ft.long()], signed_distance)

    output_positions = []
    output_colors = []
    output_uvs = []
    for case_id, triangles in templates.items():
        selected = case == case_id
        if not torch.any(selected):
            continue
        triangle_indices = torch.as_tensor(triangles, dtype=torch.long, device=mesh.v.device)
        output_positions.append(position_candidates[selected][:, triangle_indices].reshape(-1, 3))
        if color_candidates is not None:
            output_colors.append(color_candidates[selected][:, triangle_indices].reshape(-1, 3))
        if uv_candidates is not None:
            output_uvs.append(uv_candidates[selected][:, triangle_indices].reshape(-1, 2))

    if not output_positions:
        raise ValueError("Cutting plane removed the entire mesh")

    vertices = torch.cat(output_positions, dim=0)
    output_faces = torch.arange(vertices.shape[0], device=vertices.device, dtype=torch.int32).reshape(-1, 3)
    clipped = Mesh(v=vertices, f=output_faces, device=vertices.device)

    if output_colors:
        clipped.vc = torch.cat(output_colors, dim=0)
    if output_uvs:
        clipped.vt = torch.cat(output_uvs, dim=0)
        clipped.ft = output_faces.clone()
        clipped.albedo = mesh.albedo
    return clipped


def _render_one(
    mesh: Mesh,
    pose_np: np.ndarray,
    fov_degrees: float,
    resolution: int,
    context,
):
    device = mesh.v.device
    faces = mesh.f.int()
    pose = torch.as_tensor(pose_np, dtype=torch.float32, device=device)
    projection = torch.as_tensor(
        get_perspective(fov_degrees, aspect=1.0), dtype=torch.float32, device=device
    )

    homogeneous = F.pad(mesh.v.float(), (0, 1), value=1.0).unsqueeze(0)
    camera_vertices = homogeneous @ torch.inverse(pose).T
    clip_vertices = camera_vertices @ projection.T
    rast, rast_db = dr.rasterize(
        context, clip_vertices, faces, (resolution, resolution)
    )
    foreground = rast[..., 3:] > 0

    if mesh.vc is not None:
        color, _ = dr.interpolate(mesh.vc.float().unsqueeze(0), rast, faces)
    elif mesh.albedo is not None and mesh.vt is not None and mesh.ft is not None:
        texcoord, texcoord_db = dr.interpolate(
            mesh.vt.float().unsqueeze(0),
            rast,
            mesh.ft.int(),
            rast_db=rast_db,
            diff_attrs="all",
        )
        texcoord = torch.remainder(texcoord, 1.0)
        color = dr.texture(
            mesh.albedo.float().unsqueeze(0),
            texcoord,
            uv_da=texcoord_db,
            filter_mode="linear",
        )
    else:
        color = torch.full(
            (1, resolution, resolution, 3), 0.7, dtype=torch.float32, device=device
        )

    color = torch.where(foreground, color, torch.zeros_like(color))
    alpha = foreground.float()
    # Clipped triangles are intentionally de-indexed.  Keeping alpha binary
    # avoids treating their shared edges as silhouettes during antialiasing and
    # keeps the RGB mask exactly aligned with the depth map.
    rgba = torch.cat([color, alpha], dim=-1).clamp(0.0, 1.0)

    positive_vertex_depth = -camera_vertices[..., 2:3]
    depth, _ = dr.interpolate(positive_vertex_depth, rast, faces)
    depth = torch.where(foreground, depth, torch.zeros_like(depth))

    rgba_np = (rgba[0].detach().cpu().numpy() * 255.0).round().astype(np.uint8)
    depth_np = depth[0, ..., 0].detach().cpu().numpy().astype(np.float16)
    return rgba_np, depth_np


def _is_complete(folder: Path, resolution: int, radius: float, fov_degrees: float) -> bool:
    transforms_path = folder / "transforms.json"
    if not transforms_path.is_file():
        return False
    try:
        manifest = json.loads(transforms_path.read_text(encoding="utf-8"))
        if manifest.get("recipe_version") != RECIPE_VERSION:
            return False
        if manifest.get("resolution") != resolution:
            return False
        if abs(float(manifest.get("camera_radius")) - radius) > 1e-8:
            return False
        if abs(float(manifest.get("fov_degrees")) - fov_degrees) > 1e-8:
            return False
        frames = manifest.get("frames", [])
        if len(frames) != NUM_VIEWS:
            return False
        return all(
            (folder / frame["file_path"]).is_file()
            and (folder / frame["depth_path"]).is_file()
            for frame in frames
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _atomic_replace(source: Path, destination: Path) -> None:
    if not destination.exists():
        os.replace(source, destination)
        return

    backup = destination.with_name(destination.name + f".old-{os.getpid()}")
    if backup.exists():
        shutil.rmtree(backup)
    os.replace(destination, backup)
    try:
        os.replace(source, destination)
    except Exception:
        os.replace(backup, destination)
        raise
    shutil.rmtree(backup)


def _render_object(
    sample_id: str,
    data_dir: Path,
    resolution: int,
    radius: float,
    fov_degrees: float,
    context,
    device: torch.device,
) -> None:
    mesh_path = data_dir / "renders" / sample_id / "mesh.ply"
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Canonical mesh not found: {mesh_path}")

    output_root = data_dir / "renders_internal_v1"
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{sample_id}.tmp-", dir=output_root))
    final = output_root / sample_id

    try:
        mesh = _load_canonical_mesh(mesh_path, device)
        bbox_min = mesh.v.min(dim=0).values.detach().cpu().tolist()
        bbox_max = mesh.v.max(dim=0).values.detach().cpu().tolist()
        frames = []
        frame_index = 0

        for axis_name, normal_values in AXES:
            normal = torch.tensor(normal_values, dtype=torch.float32, device=device)
            projected = mesh.v @ normal
            near = float(projected.max().item())
            far = float(projected.min().item())
            camera_position = np.asarray(normal_values, dtype=np.float32) * radius
            pose = _camera_pose(camera_position)

            for cut_fraction in CUT_FRACTIONS:
                plane_offset = near - cut_fraction * (near - far)
                clipped = clip_mesh(mesh, normal, plane_offset)
                rgba, depth = _render_one(
                    clipped, pose, fov_degrees, resolution, context
                )

                image_name = f"{frame_index:03d}.png"
                depth_name = f"{frame_index:03d}_depth.npy"
                Image.fromarray(rgba, mode="RGBA").save(temporary / image_name)
                np.save(temporary / depth_name, depth, allow_pickle=False)
                frames.append(
                    {
                        "file_path": image_name,
                        "depth_path": depth_name,
                        "camera_angle_x": float(np.deg2rad(fov_degrees)),
                        "transform_matrix": pose.tolist(),
                        "axis": axis_name,
                        "cut_fraction": cut_fraction,
                        "plane_normal": list(normal_values),
                        "plane_offset": plane_offset,
                        "keep": "le",
                    }
                )
                frame_index += 1

        manifest = {
            "recipe_version": RECIPE_VERSION,
            "resolution": resolution,
            "camera_radius": radius,
            "fov_degrees": fov_degrees,
            "camera_convention": "OpenGL c2w; -Z forward; +Y up",
            "depth_convention": "positive camera depth (-camera-space Z); background=0; float16",
            "plane_equation": "dot(plane_normal, world_point) <= plane_offset",
            "cut_fraction_convention": "fraction of bbox depth removed from the camera-near side",
            "cut_fractions": list(CUT_FRACTIONS),
            "aabb": [bbox_min, bbox_max],
            "frames": frames,
        }
        (temporary / "transforms.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        if not _is_complete(temporary, resolution, radius, fov_degrees):
            raise RuntimeError("Internal render output validation failed")
        _atomic_replace(temporary, final)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _read_instance_ids(data_dir: Path, instances: str | None) -> list[str]:
    if instances:
        instances_path = Path(instances)
        text = (
            instances_path.read_text(encoding="utf-8")
            if instances_path.is_file()
            else instances
        )
        values = [value.strip() for line in text.splitlines() for value in line.split(",")]
        ids = [value for value in values if value]
    else:
        metadata_path = data_dir / "metadata.csv"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"metadata.csv not found: {metadata_path}; provide --instances instead"
            )
        with metadata_path.open(newline="", encoding="utf-8") as handle:
            rows = csv.DictReader(handle)
            if rows.fieldnames is None or "sha256" not in rows.fieldnames:
                raise ValueError(f"metadata.csv has no sha256 column: {metadata_path}")
            ids = [str(row["sha256"]).strip() for row in rows if row.get("sha256")]

    # Preserve deterministic input order while dropping accidental duplicates.
    return list(dict.fromkeys(ids))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument(
        "--instances",
        type=str,
        default=None,
        help="Comma-separated IDs or a text file containing one ID per line",
    )
    parser.add_argument("--resolution", type=int, default=518)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--fov_degrees", type=float, default=40.0)
    parser.add_argument("--override", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("CUDA is required by nvdiffrast")
    if args.resolution <= 0 or args.radius <= 0:
        parser.error("--resolution and --radius must be positive")
    if not 0 < args.fov_degrees < 180:
        parser.error("--fov_degrees must be between 0 and 180")

    sample_ids = _read_instance_ids(args.data_dir, args.instances)
    if not sample_ids:
        parser.error("No instances selected")

    output_root = args.data_dir / "renders_internal_v1"
    output_root.mkdir(parents=True, exist_ok=True)
    error_log = output_root / "errors.log"
    device = torch.device("cuda")
    context = dr.RasterizeCudaContext(device=device)
    failures = []

    for index, sample_id in enumerate(sample_ids, start=1):
        final = output_root / sample_id
        if not args.override and _is_complete(
            final, args.resolution, args.radius, args.fov_degrees
        ):
            print(f"[{index}/{len(sample_ids)}] {sample_id}: already complete")
            continue
        try:
            _render_object(
                sample_id,
                args.data_dir,
                args.resolution,
                args.radius,
                args.fov_degrees,
                context,
                device,
            )
            print(f"[{index}/{len(sample_ids)}] {sample_id}: rendered {NUM_VIEWS} views")
        except Exception:
            failures.append(sample_id)
            details = f"[{sample_id}]\n{traceback.format_exc()}\n"
            with error_log.open("a", encoding="utf-8") as handle:
                handle.write(details)
            print(f"[{index}/{len(sample_ids)}] {sample_id}: FAILED (see {error_log})")

    if failures:
        raise SystemExit(f"Failed to render {len(failures)} object(s): {', '.join(failures)}")
    print(f"Internal renders ready in {output_root}")


if __name__ == "__main__":
    main()
