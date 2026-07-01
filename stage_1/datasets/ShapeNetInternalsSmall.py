from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--stage1_dataset_adapter_note",
        default="ShapeNetInternalsSmall",
        help=argparse.SUPPRESS,
    )


def get_metadata(output_dir: str, **kwargs) -> pd.DataFrame:
    path = Path(output_dir) / "metadata.csv"
    if not path.exists():
        raise FileNotFoundError(f"metadata.csv not found: {path}")
    return pd.read_csv(path)


def foreach_instance(metadata, output_dir, func, max_workers=None, desc="Processing objects") -> pd.DataFrame:
    records = []
    items = metadata.to_dict("records")
    max_workers = max_workers or max(1, os.cpu_count() or 1)

    with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(total=len(items), desc=desc) as pbar:
        def worker(row):
            sample_id = row["sha256"]
            try:
                local_path = row["local_path"]
                path = Path(local_path)
                if not path.is_absolute():
                    path = Path(output_dir) / path
                record = func(str(path), sample_id)
                if record is not None:
                    records.append(record)
            except Exception as exc:
                print(f"Error processing object {sample_id}: {exc}", flush=True)
            finally:
                pbar.update()

        executor.map(worker, items)
        executor.shutdown(wait=True)

    return pd.DataFrame.from_records(records)
