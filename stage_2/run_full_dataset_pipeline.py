from __future__ import annotations

import argparse
import csv
import hashlib
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE2_ROOT = REPO_ROOT / "stage_2"
DEFAULT_SHAPENET_ROOT = REPO_ROOT / "ShapeNet"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "datasets" / "ShapeNetInternals"
DEFAULT_REPORT_PATH = STAGE2_ROOT / "dataset_report.md"
FEATURE_MODEL = "dinov2_vitl14_reg"
SS_LATENT_MODEL = "ss_enc_conv3d_16l8_fp16"
SLAT_LATENT_MODEL = "dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16"
DEFAULT_SLAT_ENCODER = "microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16"
DEFAULT_SS_ENCODER = "microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16"
CATEGORIES = ("bus", "cabinet", "cars", "file_cabinet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the full ShapeNetInternals TRELLIS dataset.")
    parser.add_argument("--shapenet-root", type=Path, default=DEFAULT_SHAPENET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--split-seed", type=int, default=20260627)
    parser.add_argument("--voxel-resolution", type=int, default=64)
    parser.add_argument("--voxel-workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument("--render-views", type=int, default=150)
    parser.add_argument("--cond-views", type=int, default=24)
    parser.add_argument("--render-workers", type=int, default=4)
    parser.add_argument("--cond-render-workers", type=int, default=4)
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--ss-encoder", default=DEFAULT_SS_ENCODER)
    parser.add_argument("--slat-encoder", default=DEFAULT_SLAT_ENCODER)
    parser.add_argument("--overwrite-metadata", action="store_true")
    parser.add_argument("--overwrite-voxels", action="store_true")
    parser.add_argument("--rerun-completed-stages", action="store_true")
    parser.add_argument("--skip-render-cond", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--skip-ss-latents", action="store_true")
    parser.add_argument("--skip-slat-latents", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


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


def read_score_rows(shapenet_root: Path, category: str) -> dict[str, dict[str, str]]:
    path = shapenet_root / f"{category}_center_box_scores.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fp:
        return {row["model_id"]: row for row in csv.DictReader(fp)}


def deterministic_split(ids: list[str], val_ratio: float, test_ratio: float, seed: int) -> dict[str, list[str]]:
    ids = list(ids)
    keyed = []
    for sample_id in ids:
        digest = hashlib.sha1(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()
        keyed.append((digest, sample_id))
    keyed.sort()
    shuffled = [sample_id for _, sample_id in keyed]
    n = len(shuffled)
    n_val = int(round(n * val_ratio))
    n_test = int(round(n * test_ratio))
    if n >= 3:
        n_val = max(1, n_val)
        n_test = max(1, n_test)
    n_train = max(0, n - n_val - n_test)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def write_splits(dataset_dir: Path, ids_by_category: dict[str, list[str]], val_ratio: float, test_ratio: float, seed: int) -> dict[str, str]:
    split_lookup: dict[str, str] = {}
    splits = {"train": [], "val": [], "test": []}
    for category, ids in ids_by_category.items():
        category_splits = deterministic_split(ids, val_ratio, test_ratio, seed)
        for split, split_ids in category_splits.items():
            splits[split].extend(split_ids)
            for sample_id in split_ids:
                split_lookup[sample_id] = split
    split_dir = dataset_dir / "splits"
    ensure_dir(split_dir)
    for split, split_ids in splits.items():
        split_ids = sorted(split_ids)
        (split_dir / f"{split}.txt").write_text("".join(f"{sample_id}\n" for sample_id in split_ids), encoding="utf-8")
    return split_lookup


def prepare_metadata(args: argparse.Namespace) -> pd.DataFrame:
    metadata_path = args.output_dir / "metadata.csv"
    if metadata_path.exists() and not args.overwrite_metadata:
        print(f"metadata.csv already exists, reusing: {metadata_path}")
        return read_metadata(args.output_dir)

    ensure_dir(args.output_dir)
    rows = []
    invalid_rows = []
    ids_by_category: dict[str, list[str]] = {}
    for category in CATEGORIES:
        category_root = args.shapenet_root / category
        score_rows = read_score_rows(args.shapenet_root, category)
        category_ids = []
        object_dirs = sorted(path for path in category_root.iterdir() if path.is_dir()) if category_root.exists() else []
        for object_dir in tqdm(object_dirs, desc=f"Scanning {category}"):
            object_id = object_dir.name
            model_dir = object_dir / "models"
            obj_path = model_dir / "model_normalized.obj"
            surface_path = model_dir / "model_normalized.surface.binvox"
            solid_path = model_dir / "model_normalized.solid.binvox"
            missing = []
            if not obj_path.exists():
                missing.append("models/model_normalized.obj")
            if not surface_path.exists():
                missing.append("models/model_normalized.surface.binvox")
            if missing:
                invalid_rows.append({
                    "category": category,
                    "shapenet_id": object_id,
                    "reason": "missing " + ", ".join(missing),
                })
                continue
            score = score_rows.get(object_id, {})
            row = {
                "sha256": stable_id(category, object_id),
                "file_identifier": object_id,
                "local_path": str(obj_path),
                "category": category,
                "shapenet_id": object_id,
                "source_obj": str(obj_path),
                "surface_binvox": str(surface_path),
                "solid_binvox": str(solid_path) if solid_path.exists() else "",
                "has_solid_binvox": bool(solid_path.exists()),
                "captions": "[]",
                "aesthetic_score": 5.0,
                "inner_face_count": float(score.get("inner_face_count", 0.0) or 0.0),
                "inner_edge_count": float(score.get("inner_edge_count", 0.0) or 0.0),
                "inner_face_ratio": float(score.get("inner_face_ratio", 0.0) or 0.0),
                "inner_edge_ratio": float(score.get("inner_edge_ratio", 0.0) or 0.0),
                "complexity_score": float(score.get("complexity_score", 0.0) or 0.0),
                "rendered": False,
                "voxelized": False,
                "num_voxels": 0,
                "cond_rendered": False,
                f"feature_{FEATURE_MODEL}": False,
                f"ss_latent_{SS_LATENT_MODEL}": False,
                f"latent_{SLAT_LATENT_MODEL}": False,
            }
            rows.append(row)
            category_ids.append(row["sha256"])
        ids_by_category[category] = category_ids

    if not rows:
        raise RuntimeError(f"No valid ShapeNet objects found under {args.shapenet_root}")

    split_lookup = write_splits(args.output_dir, ids_by_category, args.val_ratio, args.test_ratio, args.split_seed)
    for row in rows:
        row["split"] = split_lookup.get(row["sha256"], "train")

    metadata = pd.DataFrame(rows).sort_values(["category", "sha256"]).reset_index(drop=True)
    write_metadata(args.output_dir, metadata)
    (args.output_dir / "all_ids.txt").write_text("".join(f"{sample_id}\n" for sample_id in metadata["sha256"]), encoding="utf-8")
    pd.DataFrame(invalid_rows).to_csv(args.output_dir / "invalid_objects.csv", index=False)
    print(f"Wrote {len(metadata)} valid samples to {metadata_path}")
    if invalid_rows:
        print(f"Excluded {len(invalid_rows)} invalid objects; see {args.output_dir / 'invalid_objects.csv'}")
    return metadata


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
    return dense.reshape(dims), {"dims": dims, "translate": translate, "scale": scale}


def or_downsample(grid: np.ndarray, resolution: int) -> np.ndarray:
    if grid.ndim != 3 or len(set(grid.shape)) != 1:
        raise ValueError(f"Expected cubic 3D grid, got {grid.shape}")
    source_resolution = grid.shape[0]
    if source_resolution == resolution:
        return grid.astype(bool, copy=False)
    if source_resolution % resolution != 0:
        raise ValueError(f"Cannot integer downsample {source_resolution} to {resolution}")
    factor = source_resolution // resolution
    reshaped = grid.reshape(resolution, factor, resolution, factor, resolution, factor)
    return reshaped.any(axis=(1, 3, 5))


def grid_to_positions(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid)
    if coords.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    resolution = grid.shape[0]
    return ((coords.astype(np.float32) + 0.5) / resolution - 0.5).astype(np.float32)


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


def read_ply_vertex_count(path: Path) -> int:
    with path.open("r", encoding="ascii", errors="replace") as fp:
        for line in fp:
            line = line.strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header":
                break
    return 0


def convert_one_voxel(row: dict[str, object], output_dir: Path, resolution: int, overwrite: bool) -> dict[str, object]:
    sample_id = str(row["sha256"])
    source = Path(str(row["surface_binvox"]))
    target = output_dir / "voxels" / f"{sample_id}.ply"
    if target.exists() and not overwrite:
        return {"sha256": sample_id, "ok": True, "status": "exists", "num_voxels": read_ply_vertex_count(target)}
    if not source.exists():
        return {"sha256": sample_id, "ok": False, "status": f"missing {source}"}
    try:
        grid, _ = read_binvox(source)
        downsampled = or_downsample(grid, resolution)
        positions = grid_to_positions(downsampled)
        write_ply_points(target, positions)
        return {"sha256": sample_id, "ok": True, "status": "converted", "num_voxels": int(positions.shape[0])}
    except Exception as exc:
        return {"sha256": sample_id, "ok": False, "status": str(exc)}


def convert_voxels(args: argparse.Namespace) -> pd.DataFrame:
    metadata = read_metadata(args.output_dir)
    ensure_dir(args.output_dir / "voxels")
    records = []
    rows = metadata.to_dict("records")
    with ThreadPoolExecutor(max_workers=max(1, args.voxel_workers)) as executor:
        futures = [
            executor.submit(convert_one_voxel, row, args.output_dir, args.voxel_resolution, args.overwrite_voxels)
            for row in rows
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Converting surface binvox"):
            records.append(future.result())

    records_df = pd.DataFrame.from_records(records)
    records_df.to_csv(args.output_dir / "binvox_surface_to_voxels.csv", index=False)
    ok_lookup = dict(zip(records_df["sha256"], records_df["ok"]))
    count_lookup = dict(zip(records_df["sha256"], records_df.get("num_voxels", pd.Series(dtype=int)).fillna(0)))
    metadata["voxelized"] = metadata["sha256"].map(ok_lookup).fillna(False).astype(bool)
    metadata["num_voxels"] = metadata["sha256"].map(count_lookup).fillna(0).astype(int)
    write_metadata(args.output_dir, metadata)
    print(f"Voxel conversion OK: {int(records_df['ok'].sum())} / {len(records_df)}")
    return records_df


def artifact_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def refresh_metadata(output_dir: Path) -> pd.DataFrame:
    metadata = read_metadata(output_dir)
    for col, default in [
        ("rendered", False),
        ("voxelized", False),
        ("cond_rendered", False),
        (f"feature_{FEATURE_MODEL}", False),
        (f"ss_latent_{SS_LATENT_MODEL}", False),
        (f"latent_{SLAT_LATENT_MODEL}", False),
        ("num_voxels", 0),
    ]:
        if col not in metadata.columns:
            metadata[col] = default

    rendered = []
    cond_rendered = []
    voxelized = []
    num_voxels = []
    features = []
    ss_latents = []
    latents = []
    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Refreshing metadata"):
        sample_id = row["sha256"]
        rendered.append(artifact_exists(output_dir / "renders" / sample_id / "transforms.json"))
        cond_rendered.append(artifact_exists(output_dir / "renders_cond" / sample_id / "transforms.json"))
        voxel_path = output_dir / "voxels" / f"{sample_id}.ply"
        has_voxels = artifact_exists(voxel_path)
        voxelized.append(has_voxels)
        num_voxels.append(read_ply_vertex_count(voxel_path) if has_voxels else 0)
        features.append(artifact_exists(output_dir / "features" / FEATURE_MODEL / f"{sample_id}.npz"))
        ss_latents.append(artifact_exists(output_dir / "ss_latents" / SS_LATENT_MODEL / f"{sample_id}.npz"))
        latents.append(artifact_exists(output_dir / "latents" / SLAT_LATENT_MODEL / f"{sample_id}.npz"))

    metadata["rendered"] = rendered
    metadata["cond_rendered"] = cond_rendered
    metadata["voxelized"] = voxelized
    metadata["num_voxels"] = np.asarray(num_voxels, dtype=np.int64)
    metadata[f"feature_{FEATURE_MODEL}"] = features
    metadata[f"ss_latent_{SS_LATENT_MODEL}"] = ss_latents
    metadata[f"latent_{SLAT_LATENT_MODEL}"] = latents
    write_metadata(output_dir, metadata)
    return metadata


def python_cmd(script: Path, *args: str) -> list[str]:
    return [sys.executable, str(script), *args]


def run_command(cmd: list[str], *, cwd: Path, env: dict[str, str], log_path: Path, keep_going: bool, dry_run: bool) -> int:
    display = " ".join(cmd)
    print(f"\n[stage_2] {display}", flush=True)
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as log_fp:
        log_fp.write(f"\n$ {display}\n")
        if dry_run:
            log_fp.write("[dry-run] command not executed\n")
            return 0
        with subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                log_fp.write(line)
            code = proc.wait()
            log_fp.write(f"[exit {code}]\n")
    if code != 0 and not keep_going:
        raise RuntimeError(f"Command failed with exit code {code}: {display}")
    return code


def stage_complete(metadata: pd.DataFrame, column: str) -> bool:
    return column in metadata.columns and len(metadata) > 0 and bool(metadata[column].fillna(False).all())


def run_existing_pipeline_stage(args: argparse.Namespace, metadata: pd.DataFrame, stage: str, cmd: list[str], column: str) -> pd.DataFrame:
    if not args.rerun_completed_stages and stage_complete(metadata, column):
        print(f"{stage} already complete for all {len(metadata)} objects; skipping.")
        return metadata
    env = os.environ.copy()
    env["PYTHONPATH"] = str(STAGE2_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["SPCONV_ALGO"] = env.get("SPCONV_ALGO", "native")
    run_command(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        log_path=args.output_dir / "logs" / "stage2_pipeline.log",
        keep_going=args.keep_going,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        metadata = refresh_metadata(args.output_dir)
    return metadata


def dir_size(path: Path) -> tuple[int, str]:
    if not path.exists():
        return 0, "0B"
    try:
        text = subprocess.check_output(["du", "-sh", str(path)], text=True).split()[0]
    except Exception:
        total = 0
        for root, _, files in os.walk(path):
            for name in files:
                try:
                    total += (Path(root) / name).stat().st_size
                except OSError:
                    pass
        return total, human_size(total)
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total, text


def human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}TB"


def first_missing_reason(row: pd.Series) -> str:
    checks = [
        ("voxelized", "surface binvox to voxel PLY missing/failed"),
        ("cond_rendered", "conditioning render missing/failed"),
        ("rendered", "multiview render missing/failed"),
        (f"feature_{FEATURE_MODEL}", "DINO feature extraction missing/failed"),
        (f"ss_latent_{SS_LATENT_MODEL}", "sparse-structure latent encoding missing/failed"),
        (f"latent_{SLAT_LATENT_MODEL}", "SLAT latent encoding missing/failed"),
    ]
    for col, reason in checks:
        if col not in row or not bool(row[col]):
            return reason
    return ""


def collect_failures(output_dir: Path, metadata: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in metadata.iterrows():
        reason = first_missing_reason(row)
        if reason:
            records.append({
                "sha256": row.get("sha256", ""),
                "category": row.get("category", ""),
                "shapenet_id": row.get("shapenet_id", ""),
                "split": row.get("split", ""),
                "reason": reason,
            })
    failures = pd.DataFrame.from_records(records)
    failures.to_csv(output_dir / "failures_stage2.csv", index=False)
    return failures


def write_report(args: argparse.Namespace, metadata: pd.DataFrame | None = None) -> None:
    ensure_dir(args.report_path.parent)
    if metadata is None:
        try:
            metadata = refresh_metadata(args.output_dir)
        except FileNotFoundError:
            metadata = pd.DataFrame()
    failures = collect_failures(args.output_dir, metadata) if not metadata.empty else pd.DataFrame()
    successful = len(metadata) - len(failures)
    artifact_cols = [
        "voxelized",
        "cond_rendered",
        "rendered",
        f"feature_{FEATURE_MODEL}",
        f"ss_latent_{SS_LATENT_MODEL}",
        f"latent_{SLAT_LATENT_MODEL}",
    ]
    size_targets = [
        "metadata.csv",
        "splits",
        "voxels",
        "renders_cond",
        "renders",
        f"features/{FEATURE_MODEL}",
        f"ss_latents/{SS_LATENT_MODEL}",
        f"latents/{SLAT_LATENT_MODEL}",
        "logs",
    ]
    lines = [
        "# Stage 2 ShapeNetInternals Dataset Report",
        "",
        f"- ShapeNet root: `{args.shapenet_root}`",
        f"- Output dataset: `{args.output_dir}`",
        f"- Report path: `{args.report_path}`",
        f"- Run mode: `{'dry-run/preflight' if args.dry_run else 'execute'}`",
        "",
        "## Summary",
        "",
        f"- Metadata objects: `{len(metadata)}`",
        f"- Fully successful objects: `{successful}`",
        f"- Failed/incomplete objects: `{len(failures)}`",
        "",
        "## Artifact Counts",
        "",
    ]
    for col in artifact_cols:
        if col in metadata.columns:
            lines.append(f"- `{col}`: `{int(metadata[col].fillna(False).sum())} / {len(metadata)}`")
    lines.extend(["", "## Splits", ""])
    if "split" in metadata.columns:
        for split, count in metadata["split"].value_counts().sort_index().items():
            lines.append(f"- `{split}`: `{count}`")
    lines.extend(["", "## Categories", ""])
    if "category" in metadata.columns:
        for category, count in metadata["category"].value_counts().sort_index().items():
            lines.append(f"- `{category}`: `{count}`")
    lines.extend(["", "## Voxel Statistics", ""])
    if "num_voxels" in metadata.columns and len(metadata) > 0:
        vox = metadata["num_voxels"].fillna(0)
        lines.extend([
            f"- Min voxels: `{int(vox.min())}`",
            f"- Mean voxels: `{float(vox.mean()):.1f}`",
            f"- Max voxels: `{int(vox.max())}`",
        ])
    lines.extend(["", "## Output Folder Sizes", ""])
    for rel in size_targets:
        path = args.output_dir / rel
        _, text = dir_size(path)
        lines.append(f"- `{rel}`: `{text}`")
    lines.extend(["", "## Failed Objects", ""])
    if failures.empty:
        lines.append("No failed or incomplete objects detected.")
    else:
        reason_counts = failures["reason"].value_counts()
        for reason, count in reason_counts.items():
            lines.append(f"- `{reason}`: `{count}`")
        lines.append("")
        lines.append(f"Full failure table: `{args.output_dir / 'failures_stage2.csv'}`")
    invalid_path = args.output_dir / "invalid_objects.csv"
    if invalid_path.exists():
        try:
            invalid = pd.read_csv(invalid_path)
            invalid_count = len(invalid)
        except Exception:
            invalid_count = 0
        lines.extend(["", "## Invalid Source Objects", "", f"- Excluded before preprocessing: `{invalid_count}`", f"- Manifest: `{invalid_path}`"])
    lines.extend([
        "",
        "## Commands",
        "",
        "Full dataset command:",
        "",
        "```bash",
        "/workspace/venv/bin/python stage_2/run_full_dataset_pipeline.py --output-dir /workspace/TRELLIS/datasets/ShapeNetInternals",
        "```",
        "",
        "Useful resume command:",
        "",
        "```bash",
        "/workspace/venv/bin/python stage_2/run_full_dataset_pipeline.py --output-dir /workspace/TRELLIS/datasets/ShapeNetInternals --render-workers 4 --cond-render-workers 4 --voxel-workers 8",
        "```",
    ])
    args.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote report: {args.report_path}")


def run_pipeline(args: argparse.Namespace) -> None:
    ensure_dir(args.output_dir)
    metadata = prepare_metadata(args)
    metadata = refresh_metadata(args.output_dir)

    if args.report_only:
        write_report(args, metadata)
        return

    if not stage_complete(metadata, "voxelized") or args.overwrite_voxels or args.rerun_completed_stages:
        if args.dry_run:
            print("[dry-run] would convert surface binvox files to voxel PLY files")
        else:
            convert_voxels(args)
            metadata = refresh_metadata(args.output_dir)
    else:
        print(f"Voxel conversion already complete for all {len(metadata)} objects; skipping.")

    if not args.skip_render_cond:
        metadata = run_existing_pipeline_stage(
            args,
            metadata,
            "conditioning renders",
            [
                sys.executable, str(REPO_ROOT / "dataset_toolkits" / "render_cond.py"),
                "ShapeNetInternals",
                "--output_dir", str(args.output_dir),
                "--num_views", str(args.cond_views),
                "--max_workers", str(args.cond_render_workers),
            ],
            "cond_rendered",
        )

    if not args.skip_render:
        metadata = run_existing_pipeline_stage(
            args,
            metadata,
            "multiview renders",
            [
                sys.executable, str(REPO_ROOT / "dataset_toolkits" / "render.py"),
                "ShapeNetInternals",
                "--output_dir", str(args.output_dir),
                "--num_views", str(args.render_views),
                "--max_workers", str(args.render_workers),
            ],
            "rendered",
        )

    if not args.skip_features:
        metadata = run_existing_pipeline_stage(
            args,
            metadata,
            "DINO features",
            [
                sys.executable, str(REPO_ROOT / "dataset_toolkits" / "extract_feature.py"),
                "--output_dir", str(args.output_dir),
                "--model", FEATURE_MODEL,
                "--batch_size", str(args.feature_batch_size),
            ],
            f"feature_{FEATURE_MODEL}",
        )

    if not args.skip_ss_latents:
        metadata = run_existing_pipeline_stage(
            args,
            metadata,
            "sparse-structure latents",
            [
                sys.executable, str(REPO_ROOT / "dataset_toolkits" / "encode_ss_latent.py"),
                "--output_dir", str(args.output_dir),
                "--enc_pretrained", args.ss_encoder,
            ],
            f"ss_latent_{SS_LATENT_MODEL}",
        )

    if not args.skip_slat_latents:
        metadata = run_existing_pipeline_stage(
            args,
            metadata,
            "SLAT latents",
            [
                sys.executable, str(REPO_ROOT / "dataset_toolkits" / "encode_latent.py"),
                "--output_dir", str(args.output_dir),
                "--feat_model", FEATURE_MODEL,
                "--enc_pretrained", args.slat_encoder,
            ],
            f"latent_{SLAT_LATENT_MODEL}",
        )

    metadata = refresh_metadata(args.output_dir)
    write_report(args, metadata)


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
