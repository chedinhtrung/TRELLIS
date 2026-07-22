#!/usr/bin/env python3
"""Small deterministic tests for interior metrics and ID selection."""

from __future__ import annotations

import csv
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from interior_reconstruction.trellis_finetuning_interior import (
    evaluate_internals,
    select_ids,
)


class InteriorFineTuningTests(unittest.TestCase):
    def test_interior_extracts_center_of_solid_cube(self) -> None:
        cube = {
            (x, y, z)
            for x in range(3)
            for y in range(3)
            for z in range(3)
        }
        self.assertEqual(evaluate_internals.interior(cube, 1), {(1, 1, 1)})
        self.assertEqual(evaluate_internals.interior(cube, 2), set())

    def test_identical_voxels_score_perfectly(self) -> None:
        cube = {
            (x, y, z)
            for x in range(3)
            for y in range(3)
            for z in range(3)
        }
        metrics = evaluate_internals.score_pair(cube, cube, 1)
        for name in (
            "voxel_iou",
            "exterior_iou",
            "internal_precision",
            "internal_recall",
            "internal_f1",
        ):
            self.assertEqual(metrics[name], 1.0)

    def test_empty_voxels_have_well_defined_metrics(self) -> None:
        metrics = evaluate_internals.score_pair(set(), set(), 1)
        self.assertEqual(metrics["voxel_iou"], 1.0)
        self.assertEqual(metrics["internal_precision"], 1.0)
        self.assertEqual(metrics["internal_recall"], 1.0)
        self.assertEqual(metrics["internal_f1"], 1.0)

    def test_zero_per_category_selects_every_unique_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "metadata.csv"
            output = root / "ids.txt"
            with metadata.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=("sha256", "category"))
                writer.writeheader()
                writer.writerows([
                    {"sha256": "car_a", "category": "car"},
                    {"sha256": "bus_a", "category": "bus"},
                ])
            argv = [
                "select_ids.py",
                "--metadata",
                str(metadata),
                "--output",
                str(output),
                "--per-category",
                "0",
            ]
            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                select_ids.main()
            self.assertEqual(output.read_text().splitlines(), ["car_a", "bus_a"])


if __name__ == "__main__":
    unittest.main()
