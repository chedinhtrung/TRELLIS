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
from kiui.cam import look_at, get_perspective


_LOG_LOCK = threading.Lock()


def _log(log_file: str, level: str, sha256: str, message: str):
    ts = datetime.now().isoformat(timespec='seconds')
    line = f"[{ts}] [{level}] [{sha256}] {message}\n"
    with _LOG_LOCK:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(line)


def _is_complete_output(output_folder: str, num_views: int) -> bool:
    if not os.path.isdir(output_folder):
        return False
    if not os.path.exists(os.path.join(output_folder, 'mesh.ply')):
        return False
    if not os.path.exists(os.path.join(output_folder, 'transforms.json')):
        return False
    png_count = len([x for x in os.listdir(output_folder) if x.lower().endswith('.png') and x[:3].isdigit()])
    return png_count == num_views


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
    kd_maps = _count_kd_maps(file_path)
    if kd_maps <= 1:
        return Mesh.load(file_path, resize=False, renormal=True)

    _log(log_file, 'WARN', sha256, f"multi-material OBJ detected ({kd_maps} map_Kd), baking to vertex colors: {file_path}")
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
        pose[:3, :3] = look_at(campos, np.zeros(3, dtype=np.float32), opengl=True)
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


def _render(file_path, sha256, output_dir, num_views, resolution, denoise, ssaa, log_file, override=False):
    final_folder = os.path.join(output_dir, 'renders', sha256)
    tmp_folder = final_folder + '.tmp'

    try:
        if not override and _is_complete_output(final_folder, num_views):
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
        scale, offset = _normalize_mesh_blender_style(mesh)

        views = _build_views(num_views)
        rgba, poses = _render_views(mesh, views, resolution, ssaa=ssaa)

        imgs = (rgba.detach().cpu().numpy() * 255).astype(np.uint8)
        for i in range(num_views):
            Image.fromarray(imgs[i], mode='RGBA').save(os.path.join(tmp_folder, f'{i:03d}.png'))

        # keep mesh saving (very important)
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

        with open(os.path.join(tmp_folder, 'transforms.json'), 'w') as f:
            json.dump(to_export, f, indent=4)

        if not _is_complete_output(tmp_folder, num_views):
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
    parser.add_argument("--ssaa", type=float, default=1.5, help="Super-sampling anti-aliasing ratio for higher quality")
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
        for sha256 in copy.copy(metadata["sha256"].values):
            if _is_complete_output(os.path.join(opt.output_dir, 'renders', sha256), opt.num_views):
                records.append({"sha256": sha256, "rendered": True})
                metadata = metadata[metadata["sha256"] != sha256]

    print(f"Processing {len(metadata)} objects...")

    func = partial(
        _render,
        output_dir=opt.output_dir,
        num_views=opt.num_views,
        resolution=opt.resolution,
        denoise=opt.denoise,
        ssaa=opt.ssaa,
        log_file=log_file,
        override=opt.override,
    )
    rendered = dataset_utils.foreach_instance(metadata, opt.output_dir, func, max_workers=opt.max_workers, desc="Rendering objects")
    rendered = pd.concat([rendered, pd.DataFrame.from_records(records)])
    rendered.to_csv(os.path.join(opt.output_dir, f"rendered_{opt.rank}.csv"), index=False)
