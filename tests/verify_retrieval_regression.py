#!/usr/bin/env python3
"""Compare a fixed retrieval rerun against the recorded final outputs."""

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


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty regression CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify deterministic frozen-policy retrieval outputs."
    )
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-10)
    args = parser.parse_args()
    if args.atol < 0:
        parser.error("--atol cannot be negative")

    expected_rows = read_rows(args.expected)
    actual_rows = [
        row
        for row in read_rows(args.actual)
        if row.get("method") == FINAL_METHOD
        and int(row.get("margin", -1)) == MARGIN
    ]
    expected = {row["sample_id"]: row for row in expected_rows}
    actual = {row["sample_id"]: row for row in actual_rows}
    if len(expected) != len(expected_rows):
        raise ValueError("Expected regression CSV contains duplicate sample IDs")
    if len(actual) != len(actual_rows):
        raise ValueError("Actual regression CSV contains duplicate sample IDs")
    missing = set(expected) - set(actual)
    if missing:
        raise RuntimeError(
            f"Regression output is missing expected sample IDs: {sorted(missing)}"
        )
    actual = {sample_id: actual[sample_id] for sample_id in expected}

    failures: list[str] = []
    for sample_id, expected_row in expected.items():
        observed = actual[sample_id]
        for field in EXACT_FIELDS:
            if observed.get(field) != expected_row.get(field):
                failures.append(
                    f"{sample_id} {field}: "
                    f"{observed.get(field)!r} != {expected_row.get(field)!r}"
                )
        for field in FLOAT_FIELDS:
            delta = abs(float(observed[field]) - float(expected_row[field]))
            if delta > args.atol:
                failures.append(
                    f"{sample_id} {field}: abs delta {delta:.3e} > {args.atol:.3e}"
                )
    if failures:
        raise RuntimeError("Retrieval regression changed:\n  " + "\n  ".join(failures[:20]))

    accepted = sum(
        int(row["used_objective1_fallback"]) == 0 for row in actual.values()
    )
    print(
        f"PASS: {len(actual)} frozen-policy cases match the recorded final run "
        f"(accepted={accepted}, fallback={len(actual) - accepted}, "
        f"atol={args.atol:g})"
    )


if __name__ == "__main__":
    main()
