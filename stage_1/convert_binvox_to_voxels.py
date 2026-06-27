from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import (
    DEFAULT_DATASET_DIR,
    grid_to_positions,
    or_downsample,
    read_binvox,
    read_metadata,
    write_metadata,
    write_ply_points,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ShapeNet binvox grids into TRELLIS 64^3 sparse voxel PLY files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--kind", choices=("surface", "solid"), default="surface")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--out-folder", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = read_metadata(args.output_dir)
    out_folder = args.out_folder
    if out_folder is None:
        out_folder = "voxels" if args.kind == "surface" else "voxels_solid"
    out_dir = args.output_dir / out_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for _, row in metadata.iterrows():
        sample_id = row["sha256"]
        source_col = f"{args.kind}_binvox"
        source = Path(str(row.get(source_col, "")))
        target = out_dir / f"{sample_id}.ply"
        if target.exists() and not args.overwrite:
            records.append((sample_id, True, "exists", None))
            continue
        if not source.exists():
            records.append((sample_id, False, f"missing {source_col}", None))
            continue
        try:
            grid, _ = read_binvox(source)
            downsampled = or_downsample(grid, args.resolution)
            positions = grid_to_positions(downsampled)
            write_ply_points(target, positions)
            records.append((sample_id, True, "converted", int(positions.shape[0])))
        except Exception as exc:
            records.append((sample_id, False, str(exc), None))

    ok_ids = {sample_id for sample_id, ok, _, _ in records if ok}
    if args.kind == "surface" and out_folder == "voxels":
        metadata["voxelized"] = metadata["sha256"].isin(ok_ids)
        counts = {}
        for sample_id, ok, _, num_voxels in records:
            if ok and num_voxels is not None:
                counts[sample_id] = num_voxels
        metadata["num_voxels"] = metadata.apply(
            lambda row: counts.get(row["sha256"], row.get("num_voxels", 0)),
            axis=1,
        )
        write_metadata(args.output_dir, metadata)

    records_df = pd.DataFrame(records, columns=["sha256", "ok", "status", "num_voxels"])
    records_df.to_csv(args.output_dir / f"binvox_{args.kind}_to_{out_folder}.csv", index=False)

    failures = records_df[records_df["ok"] == False]
    print(f"Converted {records_df['ok'].sum()} / {len(records_df)} {args.kind} binvox grids to {out_folder}/")
    if not failures.empty:
        print("Failures:")
        print(failures.to_string(index=False))


if __name__ == "__main__":
    main()
