from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from plyfile import PlyData, PlyElement


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHAPENET_ROOT = REPO_ROOT / "ShapeNet"
DEFAULT_DATASET_DIR = REPO_ROOT / "datasets" / "ShapeNetInternals_small"  # Where the processed ShapeNet dataset is stored
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "shapenet_internals_stage1_reconstruction"  
FEATURE_MODEL = "dinov2_vitl14_reg"
SS_LATENT_MODEL = "ss_enc_conv3d_16l8_fp16"
SLAT_LATENT_MODEL = "dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16"
CATEGORIES = ("bus", "cabinet", "cars", "file_cabinet")


def ensure_dir(path: Path) -> None:
    """
    Create a directory if it is missing.
    """
    path.mkdir(parents=True, exist_ok=True)


def read_metadata(dataset_dir: Path) -> pd.DataFrame:
    """
    Load the TRELLIS-style metadata.csv for a converted dataset.

    dataset_dir is expected to be a preprocessed dataset folder such as
    datasets/ShapeNetInternals_small. The function fails early with a
    clear error if the metadata file has not been created yet, because almost
    every later preprocessing/evaluation step depends on this manifest.
    """
    path = dataset_dir / "metadata.csv"
    if not path.exists():
        raise FileNotFoundError(f"metadata.csv not found: {path}")
    return pd.read_csv(path)


def grid_to_positions(grid: np.ndarray) -> np.ndarray:
    """
    Convert occupied voxel indices to a sparse 3D point cloud. TRELLIS stores sparse voxels as 
    points (.ply), not as a dense 64 x 64 x 64 boolean grid. 
    """
    coords = np.argwhere(grid)
    if coords.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    resolution = grid.shape[0]
    return ((coords.astype(np.float32) + 0.5) / resolution - 0.5).astype(np.float32)


def positions_to_grid(positions: np.ndarray, resolution: int = 64) -> np.ndarray:
    """
    Convert normalized voxel-center positions back to a boolean grid.
    """
    grid = np.zeros((resolution, resolution, resolution), dtype=bool)
    if positions.size == 0:
        return grid
    coords = np.floor((positions + 0.5) * resolution).astype(np.int64)
    coords = np.clip(coords, 0, resolution - 1)
    grid[coords[:, 0], coords[:, 1], coords[:, 2]] = True
    return grid


def write_ply_points(path: Path, points: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)

    vertices = np.empty(
        len(points),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
        ],
    )

    vertices["x"] = points[:, 0]
    vertices["y"] = points[:, 1]
    vertices["z"] = points[:, 2]

    ply = PlyData(
        [PlyElement.describe(vertices, "vertex")],
        text=False,  # False = binary, True = ASCII
    )

    ply.write(str(path))



def read_ply_points(path: Path) -> np.ndarray:
    ply = PlyData.read(path)
    v = ply["vertex"]
    points = np.stack([v["x"], v["y"], v["z"]], axis=1)
    return points.astype(np.float32)


def artifact_exists(path: Path) -> bool:
    """
    Return whether an expected artifact file exists and is non-empty.
    """
    return path.exists() and path.stat().st_size > 0


def run_command(
    cmd: list[str],
    *,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
    log_path: Path | None = None,
    keep_going: bool = False,
) -> int:
    """
    Run another Python/script/terminal command, show its output live, optionally 
    save that output to a log file, and stop if the command fails.

    Example usage: 
    run_command([
        "python",
        "dataset_toolkits/render.py",
        "--dataset_dir",
        "datasets/ShapeNetInternals_small"
    ])
    """
    display = " ".join(cmd)
    print(f"\n[stage_1] {display}", flush=True)
    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)
    with subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=proc_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        log_fp = None
        try:
            if log_path is not None:
                ensure_dir(log_path.parent)
                log_fp = log_path.open("a", encoding="utf-8")
                log_fp.write(f"\n$ {display}\n")
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                if log_fp is not None:
                    log_fp.write(line)
            proc.wait()
        finally:
            if log_fp is not None:
                log_fp.close()
    if proc.returncode != 0 and not keep_going:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return int(proc.returncode)

