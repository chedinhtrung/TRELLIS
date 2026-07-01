import argparse
import copy
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from easydict import EasyDict as edict

import utils3d

from datasets.ShapeNet import get_metadata, foreach_instance


def _read_binvox(path: Path):
    with path.open("rb") as fp:
        header = fp.readline().decode("ascii", errors="replace").strip()
        if not header.startswith("#binvox"):
            raise ValueError(f"Not a binvox file: {path}")
        dim_line = fp.readline().decode("ascii", errors="replace").strip().split()
        translate_line = fp.readline().decode("ascii", errors="replace").strip().split()
        scale_line = fp.readline().decode("ascii", errors="replace").strip().split()
        data_line = fp.readline().decode("ascii", errors="replace").strip()
        if dim_line[0] != "dim" or translate_line[0] != "translate" or scale_line[0] != "scale" or data_line != "data":
            raise ValueError(f"Unexpected binvox header in {path}")
        dims = tuple(int(v) for v in dim_line[1:4])
        raw = np.frombuffer(fp.read(), dtype=np.uint8)
        translate = np.asarray([float(v) for v in translate_line[1:4]], dtype=np.float32)
        scale = float(scale_line[1])

    if raw.size % 2 != 0:
        raise ValueError(f"Corrupt binvox RLE payload in {path}")

    values = raw[0::2].astype(np.bool_)
    counts = raw[1::2].astype(np.int64)
    dense = np.repeat(values, counts)
    expected = int(np.prod(dims))
    if dense.size != expected:
        raise ValueError(f"Binvox payload size mismatch in {path}: got {dense.size}, expected {expected}")
    return dense.reshape(dims), translate, scale


def _or_downsample(grid: np.ndarray, resolution: int = 64) -> np.ndarray:
    if grid.ndim != 3 or len(set(grid.shape)) != 1:
        raise ValueError(f"Expected cubic 3D grid, got {grid.shape}")
    source_resolution = grid.shape[0]
    if source_resolution == resolution:
        return grid.astype(bool, copy=False)
    if source_resolution % resolution != 0:
        raise ValueError(f"Cannot integer downsample {source_resolution} to {resolution}")
    factor = source_resolution // resolution
    reshaped = grid.reshape(
        resolution, factor,
        resolution, factor,
        resolution, factor,
    )
    return reshaped.any(axis=(1, 3, 5))


def _grid_to_positions(grid: np.ndarray, translate: np.ndarray, scale: float) -> np.ndarray:
    coords = np.argwhere(grid)
    if coords.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    dims = np.asarray(grid.shape, dtype=np.float32)
    unit = (coords.astype(np.float32) + 0.5) / dims
    return unit * scale + translate


def _voxelize(file, sha256, output_dir, out_folder):
    mesh_path = Path(file)
    source = mesh_path.with_name("model_normalized.surface.binvox")
    if not source.exists():
        raise FileNotFoundError(f"Missing surface binvox: {source}")

    grid, translate, scale = _read_binvox(source)
    grid = _or_downsample(grid, 64)
    vertices = _grid_to_positions(grid, translate, scale)
    target = Path(output_dir) / out_folder / f"{sha256}.ply"
    utils3d.io.write_ply(str(target), vertices)
    return {"sha256": sha256, "voxelized": True, "num_voxels": len(vertices)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory that contains metadata.csv')
    parser.add_argument('--out_folder', type=str, default='voxels',
                        help='Folder name to write voxel PLY files into')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--max_workers', type=int, default=1)
    
    opt = parser.parse_args()
    opt = edict(vars(opt))

    dataset_dir = Path(opt.output_dir)
    out_dir = dataset_dir / opt.out_folder
    os.makedirs(out_dir, exist_ok=True)

    metadata = get_metadata(dataset_dir)
    if opt.instances is None:
        if 'rendered' not in metadata.columns:
            raise ValueError('metadata.csv does not have "rendered" column, please run "build_metadata.py" first')
        metadata = metadata[metadata['rendered'] == True]
        if 'voxelized' in metadata.columns and opt.out_folder == 'voxels':
            metadata = metadata[metadata['voxelized'] == False]
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    records = []

    for sha256 in copy.copy(metadata['sha256'].values):
        if os.path.exists(out_dir / f'{sha256}.ply'):
            pts = utils3d.io.read_ply(str(out_dir / f'{sha256}.ply'))[0]
            records.append({'sha256': sha256, 'voxelized': True, 'num_voxels': len(pts)})
            metadata = metadata[metadata['sha256'] != sha256]

    print(f'Processing {len(metadata)} objects...')

    func = partial(_voxelize, output_dir=opt.output_dir, out_folder=opt.out_folder)
    voxelized = foreach_instance(metadata, opt.output_dir, func, max_workers=opt.max_workers, desc='Voxelizing')
    voxelized = pd.concat([voxelized, pd.DataFrame.from_records(records)])
    voxelized.to_csv(dataset_dir / f'binvox_surface_to_{opt.out_folder}.csv', index=False)

    if opt.out_folder == 'voxels':
        ok_ids = set(voxelized.loc[voxelized['voxelized'] == True, 'sha256'].values)
        counts = {
            row['sha256']: int(row['num_voxels'])
            for _, row in voxelized.iterrows()
            if row.get('voxelized', False) == True and pd.notna(row.get('num_voxels'))
        }
        metadata = get_metadata(dataset_dir)
        metadata['voxelized'] = metadata['sha256'].isin(ok_ids)
        metadata['num_voxels'] = metadata.apply(
            lambda row: counts.get(row['sha256'], row.get('num_voxels', 0)),
            axis=1,
        )
        metadata.to_csv(dataset_dir / 'metadata.csv', index=False)

