#!/usr/bin/env python3
"""Decode old and interior-aware cached SLAT targets with the same mesh decoder."""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import utils3d
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("SPCONV_ALGO", "native")

import trellis.models as models
import trellis.modules.sparse as sp
from stage_2.export_full_pipeline_voxels import load_lora, mesh_to_voxel_points


BASE_LATENT_NAME = "dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16"
NEW_LATENT_NAME = f"{BASE_LATENT_NAME}_internal_v1"
DEFAULT_DECODER = "microsoft/TRELLIS-image-large/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16"


def read_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not ids:
        raise ValueError(f"No sample IDs found in {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate sample IDs found in {path}")
    return ids


def load_latent(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        if "coords" not in data or "feats" not in data:
            raise ValueError(f"{path} must contain coords and feats")
        coords = np.asarray(data["coords"])
        feats = np.asarray(data["feats"], dtype=np.float32)

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Invalid coordinate shape in {path}: {coords.shape}")
    if feats.ndim != 2 or len(feats) != len(coords):
        raise ValueError(f"Invalid feature shape in {path}: {feats.shape}")
    if len(coords) == 0:
        raise ValueError(f"Empty latent in {path}")
    if not np.isfinite(feats).all():
        raise ValueError(f"Non-finite latent features in {path}")
    if len(np.unique(coords, axis=0)) != len(coords):
        raise ValueError(f"Duplicate latent coordinates in {path}")
    return coords, feats


def make_sparse(coords: np.ndarray, feats: np.ndarray, device: torch.device) -> sp.SparseTensor:
    batched_coords = np.concatenate(
        [np.zeros((len(coords), 1), dtype=np.int32), coords.astype(np.int32)], axis=1
    )
    return sp.SparseTensor(
        coords=torch.from_numpy(batched_coords).to(device),
        feats=torch.from_numpy(feats).to(device),
    )


def write_report(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "target",
        "status",
        "slat_voxels",
        "mesh_vertices",
        "mesh_faces",
        "voxelized_points",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--ids_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--decoder_lora_ckpt", type=Path, default=None)
    parser.add_argument("--decoder_pretrained", default=DEFAULT_DECODER)
    parser.add_argument("--base_latent_name", default=BASE_LATENT_NAME)
    parser.add_argument("--new_latent_name", default=NEW_LATENT_NAME)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    if args.resolution <= 0:
        parser.error("--resolution must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("The TRELLIS mesh decoder requires a CUDA GPU")

    ids = read_ids(args.ids_file)
    old_dir = args.data_dir / "latents" / args.base_latent_name
    new_dir = args.data_dir / "latents" / args.new_latent_name
    prediction_root = args.output_dir / "predictions"
    report_path = args.output_dir / "oracle_decode_report.csv"

    device = torch.device("cuda")
    decoder = models.from_pretrained(args.decoder_pretrained).eval().to(device)
    if args.decoder_lora_ckpt is not None:
        load_lora(decoder, args.decoder_lora_ckpt, model_key="decoder")
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)

    rows: list[dict] = []
    failures: list[str] = []
    for sample_id in tqdm(ids, desc="Decoding oracle SLAT targets"):
        try:
            old_coords, old_feats = load_latent(old_dir / f"{sample_id}.npz")
            new_coords, new_feats = load_latent(new_dir / f"{sample_id}.npz")
            if not np.array_equal(old_coords, new_coords):
                raise ValueError("old and new raw SLAT coordinates are not exactly equal")
            if old_feats.shape != new_feats.shape or old_feats.shape[1] != 8:
                raise ValueError(
                    f"expected matching [N, 8] features, got {old_feats.shape} and {new_feats.shape}"
                )
            if old_coords.min() < 0 or old_coords.max() >= args.resolution:
                raise ValueError(
                    f"SLAT coordinates fall outside [0, {args.resolution - 1}]"
                )
        except Exception as error:
            failures.append(f"{sample_id}: {error}")
            rows.append({
                "sample_id": sample_id,
                "target": "pair",
                "status": "failed",
                "slat_voxels": "",
                "mesh_vertices": "",
                "mesh_faces": "",
                "voxelized_points": "",
                "error": str(error),
            })
            continue

        for target, feats in (("old_target", old_feats), ("new_target", new_feats)):
            mesh_path = prediction_root / target / "seed_0" / "mesh" / f"{sample_id}.ply"
            voxel_path = prediction_root / target / "seed_0" / "voxels" / f"{sample_id}.ply"
            row = {
                "sample_id": sample_id,
                "target": target,
                "status": "",
                "slat_voxels": len(old_coords),
                "mesh_vertices": "",
                "mesh_faces": "",
                "voxelized_points": "",
                "error": "",
            }
            try:
                if args.skip_existing and mesh_path.is_file() and voxel_path.is_file():
                    row["status"] = "skipped"
                    rows.append(row)
                    continue

                slat = make_sparse(old_coords, feats, device)
                with torch.inference_mode():
                    decoded = decoder(slat)
                if len(decoded) != 1:
                    raise RuntimeError(f"Decoder returned {len(decoded)} meshes for one latent")
                mesh = decoded[0]
                if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                    raise RuntimeError("Decoder returned an empty mesh")

                mesh_path.parent.mkdir(parents=True, exist_ok=True)
                voxel_path.parent.mkdir(parents=True, exist_ok=True)
                utils3d.io.write_ply(
                    mesh_path,
                    mesh.vertices.detach().cpu().numpy(),
                    mesh.faces.detach().cpu().numpy(),
                )
                points = mesh_to_voxel_points(mesh, args.resolution)
                utils3d.io.write_ply(voxel_path, points)

                row.update(
                    status="decoded",
                    mesh_vertices=len(mesh.vertices),
                    mesh_faces=len(mesh.faces),
                    voxelized_points=len(points),
                )
                del slat, decoded, mesh
                torch.cuda.empty_cache()
            except Exception as error:
                row["status"] = "failed"
                row["error"] = str(error)
                failures.append(f"{sample_id}/{target}: {error}")
            rows.append(row)

    write_report(report_path, rows)
    print(f"Wrote predictions to {prediction_root}")
    print(f"Wrote report to {report_path}")
    if failures:
        raise RuntimeError(f"Oracle decoding failed for {len(failures)} item(s); see {report_path}")


if __name__ == "__main__":
    main()
