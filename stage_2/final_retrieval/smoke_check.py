#!/usr/bin/env python3
"""Compare a tiny frozen-policy rerun against the recorded final outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FINAL_METHOD = "retrieval_unified"
MARGIN = 2
FLOAT_FIELDS = (
    "internal_precision",
    "internal_recall",
    "internal_f1",
    "internal_f05",
)
EXACT_FIELDS = (
    "category",
    "selected_id",
    "selected_rank",
    "policy_mode",
    "used_objective1_fallback",
    "gate_reason",
    "pred_internal_voxels",
    "fused_internal_voxels",
    "transferred_voxels",
)


def read_reference(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing smoke reference: {path}")
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Empty smoke reference: {path}")
    return {row["sample_id"]: row for row in rows}


def read_actual(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing smoke output: {path}")
    with path.open(newline="") as file:
        rows = [
            row for row in csv.DictReader(file)
            if row["method"] == FINAL_METHOD and int(row["margin"]) == MARGIN
        ]
    return {row["sample_id"]: row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify deterministic frozen-policy smoke-test metrics."
    )
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-10)
    args = parser.parse_args()
    if args.atol < 0:
        parser.error("--atol cannot be negative")

    reference = read_reference(args.reference)
    actual = read_actual(args.actual)
    missing = sorted(set(reference) - set(actual))
    if missing:
        raise RuntimeError(f"Smoke output is missing reference IDs: {missing}")
    actual = {sample_id: actual[sample_id] for sample_id in reference}

    failures = []
    for sample_id in reference:
        expected = reference[sample_id]
        observed = actual[sample_id]
        for field in EXACT_FIELDS:
            if observed[field] != expected[field]:
                failures.append(
                    f"{sample_id} {field}: {observed[field]!r} != {expected[field]!r}"
                )
        for field in FLOAT_FIELDS:
            delta = abs(float(observed[field]) - float(expected[field]))
            if delta > args.atol:
                failures.append(
                    f"{sample_id} {field}: abs delta {delta:.3e} > {args.atol:.3e}"
                )
    if failures:
        preview = "\n  ".join(failures[:20])
        raise RuntimeError(f"Frozen-policy smoke test changed:\n  {preview}")

    accepted = sum(
        int(row["used_objective1_fallback"]) == 0 for row in actual.values()
    )
    fallback = len(actual) - accepted
    print(
        f"PASS: {len(actual)} frozen-policy samples match the recorded final run "
        f"(accepted={accepted}, fallback={fallback}, atol={args.atol:g})"
    )


if __name__ == "__main__":
    main()
