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


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHAPENET_ROOT = REPO_ROOT / "ShapeNet"
DEFAULT_DATASET_DIR = REPO_ROOT / "datasets" / "ShapeNetInternals_small"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "shapenet_internals_stage1_reconstruction"
FEATURE_MODEL = "dinov2_vitl14_reg"
SS_LATENT_MODEL = "ss_enc_conv3d_16l8_fp16"
SLAT_LATENT_MODEL = "dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16"
CATEGORIES = ("bus", "cabinet", "cars", "file_cabinet")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_metadata(dataset_dir: Path) -> pd.DataFrame:
    path = dataset_dir / "metadata.csv"
    if not path.exists():
        raise FileNotFoundError(f"metadata.csv not found: {path}")
    return pd.read_csv(path)


def write_metadata(dataset_dir: Path, metadata: pd.DataFrame) -> None:
    ensure_dir(dataset_dir)
    metadata.to_csv(dataset_dir / "metadata.csv", index=False)


def stable_id(category: str, object_id: str) -> str:
    return f"{category}__{object_id}"


def parse_stable_id(sample_id: str) -> tuple[str, str]:
    if "__" not in sample_id:
        raise ValueError(f"Expected '<category>__<object_id>', got {sample_id}")
    return sample_id.split("__", 1)


def read_score_rows(shapenet_root: Path, category: str) -> dict[str, dict[str, str]]:
    path = shapenet_root / f"{category}_center_box_scores.csv"
    if not path.exists():
        return {}
    with path.open(newline="") as fp:
        return {row["model_id"]: row for row in csv.DictReader(fp)}


def split_ids(ids: list[str]) -> dict[str, list[str]]:
    ids = list(ids)
    n = len(ids)
    if n == 0:
        return {"train": [], "val": [], "test": []}
    n_val = max(1, round(n * 0.15)) if n >= 3 else 0
    n_test = max(1, round(n * 0.15)) if n >= 3 else 0
    n_train = max(1, n - n_val - n_test)
    return {
        "train": ids[:n_train],
        "val": ids[n_train:n_train + n_val],
        "test": ids[n_train + n_val:],
    }


def write_splits(dataset_dir: Path, ids: list[str]) -> None:
    split_dir = dataset_dir / "splits"
    ensure_dir(split_dir)
    splits = split_ids(ids)
    for split, split_ids_ in splits.items():
        (split_dir / f"{split}.txt").write_text(
            "".join(f"{sample_id}\n" for sample_id in split_ids_),
            encoding="utf-8",
        )


def assign_split(sample_id: str, dataset_dir: Path) -> str:
    for split in ("train", "val", "test"):
        path = dataset_dir / "splits" / f"{split}.txt"
        if path.exists() and sample_id in set(path.read_text(encoding="utf-8").splitlines()):
            return split
    return "train"


def read_binvox(path: Path) -> tuple[np.ndarray, dict[str, object]]:
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
        translate = tuple(float(v) for v in translate_line[1:4])
        scale = float(scale_line[1])
        raw = np.frombuffer(fp.read(), dtype=np.uint8)
    if raw.size % 2 != 0:
        raise ValueError(f"Corrupt binvox RLE payload in {path}")
    values = raw[0::2].astype(np.bool_)
    counts = raw[1::2].astype(np.int64)
    dense = np.repeat(values, counts)
    expected = int(np.prod(dims))
    if dense.size != expected:
        raise ValueError(f"Binvox payload size mismatch in {path}: got {dense.size}, expected {expected}")
    grid = dense.reshape(dims)
    return grid, {"dims": dims, "translate": translate, "scale": scale}


def or_downsample(grid: np.ndarray, resolution: int) -> np.ndarray:
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


def grid_to_positions(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid)
    if coords.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    resolution = grid.shape[0]
    return ((coords.astype(np.float32) + 0.5) / resolution - 0.5).astype(np.float32)


def positions_to_grid(positions: np.ndarray, resolution: int = 64) -> np.ndarray:
    grid = np.zeros((resolution, resolution, resolution), dtype=bool)
    if positions.size == 0:
        return grid
    coords = np.floor((positions + 0.5) * resolution).astype(np.int64)
    coords = np.clip(coords, 0, resolution - 1)
    grid[coords[:, 0], coords[:, 1], coords[:, 2]] = True
    return grid


def write_ply_points(path: Path, positions: np.ndarray) -> None:
    ensure_dir(path.parent)
    positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    with path.open("w", encoding="ascii") as fp:
        fp.write("ply\n")
        fp.write("format ascii 1.0\n")
        fp.write(f"element vertex {positions.shape[0]}\n")
        fp.write("property float x\n")
        fp.write("property float y\n")
        fp.write("property float z\n")
        fp.write("end_header\n")
        for x, y, z in positions:
            fp.write(f"{x:.8f} {y:.8f} {z:.8f}\n")


def read_ply_points(path: Path) -> np.ndarray:
    with path.open("r", encoding="ascii", errors="replace") as fp:
        vertex_count = None
        for line in fp:
            line = line.strip()
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            if line == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"PLY vertex count not found: {path}")
        rows = []
        for _ in range(vertex_count):
            parts = fp.readline().split()
            if len(parts) < 3:
                raise ValueError(f"Unexpected PLY vertex row in {path}")
            rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.asarray(rows, dtype=np.float32)


def artifact_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def run_command(
    cmd: list[str],
    *,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
    log_path: Path | None = None,
    keep_going: bool = False,
) -> int:
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


def python_cmd(script: Path, *args: str) -> list[str]:
    return [sys.executable, str(script), *args]


def write_json(path: Path, data: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def list_existing(ids: Iterable[str], root: Path, suffix: str) -> list[str]:
    return [sample_id for sample_id in ids if artifact_exists(root / f"{sample_id}{suffix}")]
