#!/usr/bin/env python3
"""Build interior-aware SLAT targets from visibility-filtered cutaway features."""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import trellis.models as models
import trellis.modules.sparse as sp


BASE_FEATURE_NAME = "dinov2_vitl14_reg"
INTERNAL_FEATURE_NAME = "dinov2_vitl14_reg_internal_v1"
BASE_LATENT_NAME = "dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16"
OUTPUT_LATENT_NAME = f"{BASE_LATENT_NAME}_internal_v1"
ENCODER = "microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16"


def read_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not ids:
        raise ValueError(f"No instance IDs found in {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate instance IDs found in {path}")
    return ids


def check_coords(coords: np.ndarray, label: str) -> list[tuple[int, int, int]]:
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{label} coordinates must have shape [N, 3], got {coords.shape}")
    keys = [tuple(map(int, coord)) for coord in coords]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{label} contains duplicate coordinates")
    return keys


def interior_mask(coords: np.ndarray, margin: int) -> np.ndarray:
    """Match compare_internals.py: behind both extrema along every axis line."""
    if margin < 1:
        raise ValueError("route margin must be at least 1")

    coords = np.asarray(coords, dtype=np.int32)
    mask = np.ones(len(coords), dtype=bool)
    for axis in range(3):
        other = [index for index in range(3) if index != axis]
        groups: dict[tuple[int, int], list[int]] = {}
        for index, coord in enumerate(coords):
            key = (int(coord[other[0]]), int(coord[other[1]]))
            groups.setdefault(key, []).append(index)
        for indices in groups.values():
            values = coords[indices, axis]
            keep = (values - values.min() >= margin) & (values.max() - values >= margin)
            mask[np.asarray(indices)] &= keep
    return mask


def reorder(values: np.ndarray, source_keys: list[tuple[int, int, int]], target_keys: list[tuple[int, int, int]], label: str) -> np.ndarray:
    if set(source_keys) != set(target_keys):
        missing = len(set(target_keys) - set(source_keys))
        extra = len(set(source_keys) - set(target_keys))
        raise ValueError(f"{label} coordinate support differs (missing={missing}, extra={extra})")
    source_index = {key: index for index, key in enumerate(source_keys)}
    return np.asarray(values)[[source_index[key] for key in target_keys]]


def load_npz(path: Path, required: tuple[str, ...]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"{path} is missing arrays: {missing}")
        return {name: np.asarray(data[name]) for name in required}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--route_margin", type=int, default=2)
    parser.add_argument("--override", action="store_true")
    parser.add_argument("--base_feature_name", default=BASE_FEATURE_NAME)
    parser.add_argument("--internal_feature_name", default=INTERNAL_FEATURE_NAME)
    parser.add_argument("--base_latent_name", default=BASE_LATENT_NAME)
    parser.add_argument("--output_latent_name", default=OUTPUT_LATENT_NAME)
    parser.add_argument("--enc_pretrained", default=ENCODER)
    args = parser.parse_args()

    if args.route_margin < 1:
        parser.error("--route_margin must be at least 1")
    if not torch.cuda.is_available():
        raise RuntimeError("The TRELLIS SLAT encoder requires a CUDA GPU")

    ids = read_ids(args.instances)
    base_feature_dir = args.data_dir / "features" / args.base_feature_name
    internal_feature_dir = args.data_dir / "features" / args.internal_feature_name
    base_latent_dir = args.data_dir / "latents" / args.base_latent_name
    output_dir = args.data_dir / "latents" / args.output_latent_name
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_source = base_latent_dir / "stats.json"
    if stats_source.is_file():
        shutil.copyfile(stats_source, output_dir / "stats.json")

    pending = [sample_id for sample_id in ids if args.override or not (output_dir / f"{sample_id}.npz").is_file()]
    encoder = None
    if pending:
        encoder = models.from_pretrained(args.enc_pretrained).eval().cuda()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)

    processed = 0
    skipped = len(ids) - len(pending)
    for sample_id in tqdm(pending, desc="Building internal SLAT targets"):
        base_feature = load_npz(
            base_feature_dir / f"{sample_id}.npz", ("indices", "patchtokens")
        )
        internal_feature = load_npz(
            internal_feature_dir / f"{sample_id}.npz", ("indices", "patchtokens", "view_count")
        )
        base_latent = load_npz(
            base_latent_dir / f"{sample_id}.npz", ("coords", "feats")
        )

        base_coords = np.asarray(base_feature["indices"])
        base_tokens = np.asarray(base_feature["patchtokens"], dtype=np.float32)
        cut_coords = np.asarray(internal_feature["indices"])
        cut_tokens = np.asarray(internal_feature["patchtokens"], dtype=np.float32)
        cut_counts = np.asarray(internal_feature["view_count"]).reshape(-1)
        old_coords = np.asarray(base_latent["coords"])
        old_feats = np.asarray(base_latent["feats"], dtype=np.float32)

        base_keys = check_coords(base_coords, "base features")
        cut_keys = check_coords(cut_coords, "internal features")
        old_keys = check_coords(old_coords, "base latent")
        if base_tokens.ndim != 2 or len(base_tokens) != len(base_coords):
            raise ValueError(f"{sample_id}: invalid base feature shape {base_tokens.shape}")
        if cut_tokens.ndim != 2 or len(cut_tokens) != len(cut_coords):
            raise ValueError(f"{sample_id}: invalid internal feature shape {cut_tokens.shape}")
        if cut_tokens.shape[1] != base_tokens.shape[1]:
            raise ValueError(
                f"{sample_id}: feature widths differ: base={base_tokens.shape}, internal={cut_tokens.shape}"
            )
        if len(cut_counts) != len(cut_coords):
            raise ValueError(f"{sample_id}: view_count length does not match internal coordinates")
        if old_feats.ndim != 2 or len(old_feats) != len(old_coords):
            raise ValueError(f"{sample_id}: invalid base latent shape {old_feats.shape}")
        if not np.isfinite(base_tokens).all() or not np.isfinite(cut_tokens).all() or not np.isfinite(old_feats).all():
            raise ValueError(f"{sample_id}: non-finite input features or latent")
        if not np.isfinite(cut_counts).all() or (cut_counts < 0).any():
            raise ValueError(f"{sample_id}: invalid internal view counts")

        cut_tokens = reorder(cut_tokens, cut_keys, base_keys, "internal features")
        cut_counts = reorder(cut_counts, cut_keys, base_keys, "internal view counts")
        if set(base_keys) != set(old_keys):
            missing = len(set(old_keys) - set(base_keys))
            extra = len(set(base_keys) - set(old_keys))
            raise ValueError(
                f"{sample_id}: feature/latent coordinate support differs "
                f"(missing={missing}, extra={extra})"
            )

        feature_route = interior_mask(base_coords, args.route_margin)
        replace = feature_route & (cut_counts > 0)
        hybrid_tokens = base_tokens.copy()
        hybrid_tokens[replace] = cut_tokens[replace]

        sparse_input = sp.SparseTensor(
            feats=torch.from_numpy(hybrid_tokens).float(),
            coords=torch.cat(
                [
                    torch.zeros(len(base_coords), 1, dtype=torch.int32),
                    torch.from_numpy(base_coords.astype(np.int32)),
                ],
                dim=1,
            ),
        ).cuda()
        with torch.inference_mode():
            candidate = encoder(sparse_input, sample_posterior=False)
        candidate_feats = candidate.feats.float().cpu().numpy()
        candidate_coords = candidate.coords[:, 1:].cpu().numpy()
        if not np.isfinite(candidate_feats).all():
            raise ValueError(f"{sample_id}: encoder produced non-finite latents")

        candidate_keys = check_coords(candidate_coords, "candidate latent")
        candidate_feats = reorder(candidate_feats, candidate_keys, old_keys, "candidate/base latent")
        if candidate_feats.shape != old_feats.shape:
            raise ValueError(
                f"{sample_id}: candidate/base latent shapes differ: "
                f"candidate={candidate_feats.shape}, base={old_feats.shape}"
            )
        cut_counts_latent = reorder(cut_counts, base_keys, old_keys, "view counts/base latent")
        route = interior_mask(old_coords, args.route_margin)

        new_feats = old_feats.copy()
        new_feats[route] = candidate_feats[route]
        np.savez_compressed(
            output_dir / f"{sample_id}.npz",
            coords=old_coords.astype(np.uint8),
            feats=new_feats.astype(np.float32),
            route_mask=route,
            cut_view_count=cut_counts_latent.astype(np.uint16),
        )
        processed += 1

    manifest = {
        "version": "internal_v1",
        "base_feature_name": args.base_feature_name,
        "internal_feature_name": args.internal_feature_name,
        "base_latent_name": args.base_latent_name,
        "output_latent_name": args.output_latent_name,
        "encoder": args.enc_pretrained,
        "route_margin": args.route_margin,
        "stats_source": str(stats_source),
        "stats_copied": stats_source.is_file(),
        "instances": str(args.instances),
        "num_instances": len(ids),
    }
    with (output_dir / "target_info.json").open("w") as file:
        json.dump(manifest, file, indent=2)

    print(f"Built {processed} targets; skipped {skipped} existing targets")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
