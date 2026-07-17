import os
import json
import copy
import sys
import importlib
import argparse
import shutil
import traceback
import threading
from functools import partial
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import trimesh
from easydict import EasyDict as edict
from PIL import Image

from utils import sphere_hammersley_sequence

import nvdiffrast.torch as dr
from kiui.mesh import Mesh
from kiui.cam import get_perspective


_LOG_LOCK = threading.Lock()


def _log(log_file: str, level: str, sha256: str, message: str):
    ts = datetime.now().isoformat(timespec='seconds')
    line = f"[{ts}] [{level}] [{sha256}] {message}\n"
    with _LOG_LOCK:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(line)


def _is_complete_output(output_folder: str, expected_num_views: int) -> bool:
    if not os.path.isdir(output_folder):
        return False
    if not os.path.exists(os.path.join(output_folder, 'mesh.ply')):
        return False
    transforms_path = os.path.join(output_folder, 'transforms.json')
    if not os.path.exists(transforms_path):
        return False
    png_count = len([x for x in os.listdir(output_folder) if x.lower().endswith('.png') and x[:3].isdigit()])
    if png_count != expected_num_views:
        return False
    try:
        with open(transforms_path, 'r') as f:
            transforms = json.load(f)
        return len(transforms.get('frames', [])) == expected_num_views
    except Exception:
        return False


def _count_kd_maps(file_path: str) -> int:
    if not file_path.lower().endswith('.obj'):
        return 0
    if not os.path.exists(file_path):
        return 0

    obj_dir = os.path.dirname(file_path)
    mtl_files = []
    with open(file_path, 'r', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if s.lower().startswith('mtllib '):
                parts = s.split(maxsplit=1)
                if len(parts) == 2:
                    mtl_files.append(os.path.normpath(os.path.join(obj_dir, parts[1])))

    count = 0
    for mtl in mtl_files:
        if not os.path.exists(mtl):
            continue
        with open(mtl, 'r', errors='ignore') as mf:
            for line in mf:
                if line.strip().lower().startswith('map_kd '):
                    count += 1
    return count


def _load_mesh_compat(file_path: str, log_file: str, sha256: str) -> Mesh:
    #kd_maps = _count_kd_maps(file_path)
    #if kd_maps <= 1:
    #    return Mesh.load(file_path, resize=False, renormal=True)

    #_log(log_file, 'WARN', sha256, f"multi-material OBJ detected ({kd_maps} map_Kd), baking to vertex colors: {file_path}")
    data = trimesh.load(file_path, process=False)
    tm = data.to_mesh() if isinstance(data, trimesh.Scene) else data

    if tm is None or len(tm.vertices) == 0 or len(tm.faces) == 0:
        _log(log_file, 'WARN', sha256, 'trimesh bake failed, falling back to Mesh.load()')
        return Mesh.load(file_path, resize=False, renormal=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mesh = Mesh(
        v=torch.tensor(np.asarray(tm.vertices), dtype=torch.float32, device=device),
        f=torch.tensor(np.asarray(tm.faces), dtype=torch.int32, device=device),
        device=device,
    )

    try:
        color_vis = tm.visual.to_color()
        vc = getattr(color_vis, 'vertex_colors', None)
        if vc is not None and len(vc) == len(tm.vertices):
            vc = np.asarray(vc)[..., :3].astype(np.float32) / 255.0
            mesh.vc = torch.tensor(vc, dtype=torch.float32, device=device)
    except Exception as e:
        _log(log_file, 'WARN', sha256, f'vertex color bake failed: {e}')

    mesh.auto_normal()
    return mesh


def _normalize_mesh_blender_style(mesh: Mesh):
    v = mesh.v
    bbox_min = v.min(dim=0).values
    bbox_max = v.max(dim=0).values
    scale = 1.0 / torch.max(bbox_max - bbox_min).item()
    mesh.v = mesh.v * scale

    bbox_min = mesh.v.min(dim=0).values
    bbox_max = mesh.v.max(dim=0).values
    offset = -(bbox_min + bbox_max) / 2.0
    mesh.v = mesh.v + offset
    return float(scale), offset.detach().cpu().numpy().tolist()


def _rotate_mesh_world_pos90_x_inplace(mesh: Mesh):
    """Convert mesh from Y-up to Blender-style Z-up world."""
    x = mesh.v[:, 0]
    y = mesh.v[:, 1]
    z = mesh.v[:, 2]
    # +90 deg around X (right-handed): x'=x, y'=-z, z'=y
    mesh.v = torch.stack([x, -z, y], dim=-1)

    if mesh.vn is not None:
        nx = mesh.vn[:, 0]
        ny = mesh.vn[:, 1]
        nz = mesh.vn[:, 2]
        mesh.vn = torch.stack([nx, -nz, ny], dim=-1)


def _blender_track_to_cam2world_rotation(campos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Build cam2world rotation equivalent to Blender TRACK_TO (-Z, UP_Y)."""
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    fallback_y = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    forward = target - campos
    n = np.linalg.norm(forward)
    if n < 1e-12:
        raise ValueError('camera position equals target; cannot build view basis')
    forward = forward / n

    z_cam_world = -forward
    y_cam_world = world_up - np.dot(world_up, z_cam_world) * z_cam_world
    y_norm = np.linalg.norm(y_cam_world)
    if y_norm < 1e-6:
        y_cam_world = fallback_y - np.dot(fallback_y, z_cam_world) * z_cam_world
        y_norm = np.linalg.norm(y_cam_world)
        if y_norm < 1e-6:
            x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            y_cam_world = x_axis - np.dot(x_axis, z_cam_world) * z_cam_world
            y_norm = np.linalg.norm(y_cam_world)
    y_cam_world = y_cam_world / y_norm

    x_cam_world = np.cross(y_cam_world, z_cam_world)
    x_cam_world = x_cam_world / np.linalg.norm(x_cam_world)
    return np.stack([x_cam_world, y_cam_world, z_cam_world], axis=1).astype(np.float32)


def _build_views(num_views):
    yaws = []
    pitchs = []
    offset = (np.random.rand(), np.random.rand())
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views, offset)
        yaws.append(float(y))
        pitchs.append(float(p))
    radius = [2.0] * num_views
    fov = [40 / 180 * np.pi] * num_views
    return [
        {"yaw": y, "pitch": p, "radius": r, "fov": f}
        for y, p, r, f in zip(yaws, pitchs, radius, fov)
    ]


def _build_axis_cutout_views(
    num_views: int,
    radius: float = 2.0,
    fov_deg: float = 40.0,
    jitter_deg: float = 8.0,
):
    """Build cutout views on principal axes with slight deterministic jitter.

    The first 6 views are exact ±X/±Y/±Z. Additional views repeat the same
    axis order with small yaw/pitch offsets so they are nearby-but-distinct.
    """
    fov = fov_deg / 180 * np.pi
    jitter = np.deg2rad(jitter_deg)
    base_views = [
        {"yaw": 0.0, "pitch": 0.0, "radius": radius, "fov": fov, "axis": "+x"},
        {"yaw": np.pi, "pitch": 0.0, "radius": radius, "fov": fov, "axis": "-x"},
        {"yaw": np.pi / 2, "pitch": 0.0, "radius": radius, "fov": fov, "axis": "+y"},
        {"yaw": -np.pi / 2, "pitch": 0.0, "radius": radius, "fov": fov, "axis": "-y"},
        {"yaw": 0.0, "pitch": np.pi / 2, "radius": radius, "fov": fov, "axis": "+z"},
        {"yaw": 0.0, "pitch": -np.pi / 2, "radius": radius, "fov": fov, "axis": "-z"},
    ]

    views = []
    for i in range(max(0, num_views)):
        base = copy.deepcopy(base_views[i % len(base_views)])
        cycle_idx = i // len(base_views)
        if cycle_idx > 0:
            # Deterministic small offsets that alternate sign and grow mildly.
            mag = min(1.0, 0.35 + 0.15 * cycle_idx) * jitter
            sign_yaw = -1.0 if (cycle_idx + (i % 2)) % 2 == 0 else 1.0
            sign_pitch = -1.0 if (cycle_idx + ((i + 1) % 2)) % 2 == 0 else 1.0
            yaw_jit = sign_yaw * mag
            pitch_jit = sign_pitch * (0.65 * mag)

            # Keep near pole views stable by perturbing yaw slightly and clamping pitch.
            base["yaw"] = float(base["yaw"] + yaw_jit)
            base["pitch"] = float(np.clip(base["pitch"] + pitch_jit, -np.pi / 2 + 0.08, np.pi / 2 - 0.08))
        views.append(base)
    return views


def _render_views(
    mesh: Mesh,
    views,
    resolution: int,
    ssaa: float = 1.0,
):
    device = mesh.v.device
    f = mesh.f.int()
    render_resolution = max(int(round(resolution * max(ssaa, 1.0))), resolution)

    poses = []
    proj = []
    for v in views:
        yaw, pitch, radius, fov = v["yaw"], v["pitch"], v["radius"], v["fov"]
        campos = np.array([
            radius * np.cos(yaw) * np.cos(pitch),
            radius * np.sin(yaw) * np.cos(pitch),
            radius * np.sin(pitch),
        ], dtype=np.float32)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = _blender_track_to_cam2world_rotation(campos, np.zeros(3, dtype=np.float32))
        pose[:3, 3] = campos
        poses.append(pose)
        proj.append(get_perspective(np.rad2deg(fov), aspect=1.0))

    poses = torch.from_numpy(np.stack(poses, axis=0)).to(device=device, dtype=torch.float32)
    proj = torch.from_numpy(np.stack(proj, axis=0)).to(device=device, dtype=torch.float32)

    v_h = F.pad(mesh.v, pad=(0, 1), mode="constant", value=1.0).float().unsqueeze(0).expand(len(views), -1, -1)
    v_cam = torch.bmm(v_h, torch.inverse(poses).transpose(1, 2))
    v_clip = torch.bmm(v_cam, proj.transpose(1, 2))

    glctx = dr.RasterizeCudaContext(device=device)
    rast, rast_db = dr.rasterize(glctx, v_clip, f, (render_resolution, render_resolution))

    alpha = (rast[..., 3:] > 0).float()
    alpha = dr.antialias(alpha, rast, v_clip, f).clamp(0, 1)

    if mesh.vc is not None:
        vc = mesh.vc.unsqueeze(0).expand(len(views), -1, -1).contiguous()
        albedo, _ = dr.interpolate(vc, rast, f)
        albedo = torch.where(rast[..., 3:] > 0, albedo, torch.zeros_like(albedo))
    elif mesh.albedo is not None and mesh.vt is not None and mesh.ft is not None:
        vt = mesh.vt.unsqueeze(0).expand(len(views), -1, -1).contiguous()
        texc, texc_db = dr.interpolate(vt, rast, mesh.ft.int(), rast_db=rast_db, diff_attrs='all')
        # Some ShapeNet assets use tiled UVs outside [0, 1]. Wrap to avoid
        # out-of-domain sampling that can look flat/incorrect.
        texc = torch.remainder(texc, 1.0)
        # texc[..., 1] = 1.0 - texc[..., 1]
        albedo = mesh.albedo.unsqueeze(0).expand(len(views), -1, -1, -1).contiguous()
        albedo = dr.texture(albedo, texc, filter_mode="linear")
        albedo = torch.where(rast[..., 3:] > 0, albedo, torch.zeros_like(albedo))
    else:
        albedo = None

    if albedo is None:
        # fixed fast fallback when texture/vertex color is unavailable
        albedo = torch.ones((len(views), render_resolution, render_resolution, 3), device=device, dtype=torch.float32) * 0.7
        albedo = torch.where(rast[..., 3:] > 0, albedo, torch.zeros_like(albedo))

    rgb = albedo

    rgb = torch.where(rast[..., 3:] > 0, rgb, torch.zeros_like(rgb))

    rgba = torch.cat([rgb, alpha], dim=-1).clamp(0, 1)
    rgba = dr.antialias(rgba, rast, v_clip, f).clamp(0, 1)

    # super-sampling downscale for better edge quality
    if render_resolution != resolution:
        rgba = rgba.permute(0, 3, 1, 2)
        rgba = F.interpolate(rgba, size=(resolution, resolution), mode='area')
        rgba = rgba.permute(0, 2, 3, 1).contiguous()

    return rgba, poses


def _slice_mesh_far_half(mesh: Mesh, campos: np.ndarray) -> Mesh:
    """Slice mesh by a plane through origin and keep only the far half.

    The plane normal is the camera principal axis (toward origin), and we drop
    geometry closer to the camera.
    """
    device = mesh.v.device
    vertices = mesh.v
    faces = mesh.f.int()
    if faces.numel() == 0:
        return Mesh(v=vertices.new_zeros((0, 3)), f=faces.new_zeros((0, 3), dtype=torch.int32), device=device)

    normal = torch.tensor(-campos, dtype=vertices.dtype, device=device)
    normal = normal / normal.norm().clamp_min(1e-8)

    face_centers = vertices[faces.long()].mean(dim=1)
    # Keep far half: signed distance >= 0 along camera forward axis.
    keep_mask = (face_centers @ normal) >= 0

    kept_faces = faces[keep_mask]
    if kept_faces.shape[0] == 0:
        return Mesh(v=vertices.new_zeros((0, 3)), f=faces.new_zeros((0, 3), dtype=torch.int32), device=device)

    used_vertices = torch.unique(kept_faces.reshape(-1).long())
    remap = torch.full((vertices.shape[0],), -1, dtype=torch.int64, device=device)
    remap[used_vertices] = torch.arange(used_vertices.shape[0], dtype=torch.int64, device=device)

    sliced = Mesh(
        v=vertices[used_vertices],
        f=remap[kept_faces.long()].int(),
        device=device,
    )

    if mesh.vn is not None and mesh.vn.shape[0] == vertices.shape[0]:
        sliced.vn = mesh.vn[used_vertices]
    if mesh.vc is not None and mesh.vc.shape[0] == vertices.shape[0]:
        sliced.vc = mesh.vc[used_vertices]

    # Keep UV/albedo texturing if present.
    if mesh.vt is not None and mesh.ft is not None and mesh.ft.shape[0] == faces.shape[0]:
        kept_ft = mesh.ft.int()[keep_mask]
        if kept_ft.shape[0] > 0:
            used_vt = torch.unique(kept_ft.reshape(-1).long())
            remap_vt = torch.full((mesh.vt.shape[0],), -1, dtype=torch.int64, device=device)
            remap_vt[used_vt] = torch.arange(used_vt.shape[0], dtype=torch.int64, device=device)
            sliced.vt = mesh.vt[used_vt]
            sliced.ft = remap_vt[kept_ft.long()].int()
            sliced.albedo = mesh.albedo

    sliced.auto_normal()
    return sliced


def _render_cutout_views(
    mesh: Mesh,
    views,
    resolution: int,
    ssaa: float = 1.0,
):
    """Render the same views as normal rendering, but with per-view cutout meshes."""
    cutout_rgba = []
    cutout_poses = []
    for v in views:
        yaw, pitch, radius = v["yaw"], v["pitch"], v["radius"]
        campos = np.array([
            radius * np.cos(yaw) * np.cos(pitch),
            radius * np.sin(yaw) * np.cos(pitch),
            radius * np.sin(pitch),
        ], dtype=np.float32)

        sliced_mesh = _slice_mesh_far_half(mesh, campos)
        rgba_i, poses_i = _render_views(sliced_mesh, [v], resolution, ssaa=ssaa)
        cutout_rgba.append(rgba_i[0])
        cutout_poses.append(poses_i[0])

    cutout_rgba = torch.stack(cutout_rgba, dim=0)
    cutout_poses = torch.stack(cutout_poses, dim=0)
    return cutout_rgba, cutout_poses

def _render(
    file_path,
    sha256,
    output_dir,
    num_views,
    cutout_num_views,
    resolution,
    denoise,
    ssaa,
    log_file,
    override=False,
):
    final_folder = os.path.join(output_dir, 'renders', sha256)
    tmp_folder = final_folder + '.tmp'
    expected_num_views = num_views + cutout_num_views

    try:
        if not override and _is_complete_output(final_folder, expected_num_views):
            return {'sha256': sha256, 'rendered': True}

        if override and os.path.exists(final_folder):
            _log(log_file, 'INFO', sha256, 'override enabled, deleting existing output and re-rendering')
            shutil.rmtree(final_folder, ignore_errors=True)

        if os.path.exists(final_folder):
            _log(log_file, 'WARN', sha256, 'incomplete existing output found, deleting and rebuilding')
            shutil.rmtree(final_folder, ignore_errors=True)
        if os.path.exists(tmp_folder):
            shutil.rmtree(tmp_folder, ignore_errors=True)
        os.makedirs(tmp_folder, exist_ok=True)

        mesh = _load_mesh_compat(file_path, log_file=log_file, sha256=sha256)
        _rotate_mesh_world_pos90_x_inplace(mesh)
        scale, offset = _normalize_mesh_blender_style(mesh)

        views = _build_views(num_views)
        rgba, poses = _render_views(mesh, views, resolution, ssaa=ssaa)

        imgs = (rgba.detach().cpu().numpy() * 255).astype(np.uint8)
        for i in range(num_views):
            Image.fromarray(imgs[i], mode='RGBA').save(os.path.join(tmp_folder, f'{i:03d}.png'))

        cutout_views = []
        cutout_poses_np = None
        if cutout_num_views > 0:
            cutout_views = _build_axis_cutout_views(cutout_num_views)
            cutout_rgba, cutout_poses = _render_cutout_views(mesh, cutout_views, resolution, ssaa=ssaa)
            cutout_imgs = (cutout_rgba.detach().cpu().numpy() * 255).astype(np.uint8)
            for i in range(cutout_num_views):
                file_idx = num_views + i
                Image.fromarray(cutout_imgs[i], mode='RGBA').save(os.path.join(tmp_folder, f'{file_idx:03d}.png'))
            cutout_poses_np = cutout_poses.detach().cpu().numpy()

        # Export exactly the same rotated mesh used for rendering.
        mesh.write(os.path.join(tmp_folder, 'mesh.ply'))

        to_export = {
            'aabb': [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            'scale': scale,
            'offset': offset,
            'frames': [],
        }
        poses_np = poses.detach().cpu().numpy()
        for i, v in enumerate(views):
            to_export['frames'].append(
                {
                    'file_path': f'{i:03d}.png',
                    'camera_angle_x': float(v['fov']),
                    'transform_matrix': poses_np[i].tolist(),
                }
            )

        if cutout_poses_np is not None:
            for i, v in enumerate(cutout_views):
                file_idx = num_views + i
                to_export['frames'].append(
                    {
                        'file_path': f'{file_idx:03d}.png',
                        'camera_angle_x': float(v['fov']),
                        'transform_matrix': cutout_poses_np[i].tolist(),
                        'cutout': True,
                        'cutout_axis': v.get('axis'),
                    }
                )

        with open(os.path.join(tmp_folder, 'transforms.json'), 'w') as f:
            json.dump(to_export, f, indent=4)

        if not _is_complete_output(tmp_folder, expected_num_views):
            _log(log_file, 'ERROR', sha256, 'output validation failed (missing png/mesh/transforms)')
            shutil.rmtree(tmp_folder, ignore_errors=True)
            return None

        os.replace(tmp_folder, final_folder)
        return {'sha256': sha256, 'rendered': True}
    except Exception as e:
        _log(log_file, 'ERROR', sha256, f'{e} | traceback: {traceback.format_exc().strip()}')
        shutil.rmtree(tmp_folder, ignore_errors=True)
        return None


if __name__ == "__main__":
    dataset_utils = importlib.import_module(f"datasets.{sys.argv[1]}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the metadata")
    parser.add_argument("--filter_low_aesthetic_score", type=float, default=None, help="Filter objects with aesthetic score lower than this value")
    parser.add_argument("--instances", type=str, default=None, help="Instances to process")
    parser.add_argument("--num_views", type=int, default=150, help="Number of views to render")
    parser.add_argument("--resolution", type=int, default=512, help="Render resolution for each image")
    parser.add_argument("--denoise", action="store_true", default=True, help="Kept for CLI compatibility")
    parser.add_argument("--ssaa", type=float, default=2.0, help="Super-sampling anti-aliasing ratio for higher quality")
    parser.add_argument("--cutout_num_views", type=int, default=0, help="Number of principal-axis cutout views to append")
    parser.add_argument("--override", action="store_true", help="Force re-render by deleting existing outputs")
    dataset_utils.add_args(parser)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--max_workers", type=int, default=8)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))
    log_file = os.path.join(opt.output_dir, f'error_render_{opt.rank}.log')

    os.makedirs(os.path.join(opt.output_dir, "renders"), exist_ok=True)

    if not os.path.exists(os.path.join(opt.output_dir, "metadata.csv")):
        raise ValueError("metadata.csv not found")
    metadata = pd.read_csv(os.path.join(opt.output_dir, "metadata.csv"))
    if opt.instances is None:
        metadata = metadata[metadata["local_path"].notna()]
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata["aesthetic_score"] >= opt.filter_low_aesthetic_score]
        if not opt.override and "rendered" in metadata.columns:
            metadata = metadata[metadata["rendered"] == False]
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, "r") as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(",")
        metadata = metadata[metadata["sha256"].isin(instances)]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []

    if not opt.override:
        expected_num_views = opt.num_views + opt.cutout_num_views
        for sha256 in copy.copy(metadata["sha256"].values):
            if _is_complete_output(os.path.join(opt.output_dir, 'renders', sha256), expected_num_views):
                records.append({"sha256": sha256, "rendered": True})
                metadata = metadata[metadata["sha256"] != sha256]

    print(f"Processing {len(metadata)} objects...")

    func = partial(
        _render,
        output_dir=opt.output_dir,
        num_views=opt.num_views,
        cutout_num_views=opt.cutout_num_views,
        resolution=opt.resolution,
        denoise=opt.denoise,
        ssaa=opt.ssaa,
        log_file=log_file,
        override=opt.override,
    )
    rendered = dataset_utils.foreach_instance(metadata, opt.output_dir, func, max_workers=opt.max_workers, desc="Rendering objects")
    rendered = pd.concat([rendered, pd.DataFrame.from_records(records)])
    rendered.to_csv(os.path.join(opt.output_dir, f"rendered_{opt.rank}.csv"), index=False)
