#!/usr/bin/env python3
"""Small deterministic preflight checks for retrieval-v2 geometry."""

from __future__ import annotations

import unittest
from collections import Counter

from retrieval_v2 import (
    FEATURE_NAMES,
    Alignment,
    ComponentPreset,
    align_voxels,
    choose_conservative_policy,
    find_alignment,
    fit_ridge,
    hybrid_fusion,
    predict_quality,
    tolerant_f1,
    transfer_supported_components,
    unmatched_voxels,
)


class RetrievalV2Tests(unittest.TestCase):
    def test_tolerance_includes_exact_one_voxel_boundary(self) -> None:
        left = {(10, 10, 10)}
        right = {(11, 10, 10)}
        self.assertEqual(tolerant_f1(left, right, 1.0), 1.0)
        self.assertEqual(unmatched_voxels(left, right, 1.0), set())
        self.assertEqual(unmatched_voxels({(9, 10, 10)}, right, 1.0), {(9, 10, 10)})

    def test_alignment_is_small_and_improves_shifted_surface(self) -> None:
        source = {(x, 10, 10) for x in range(10, 30)}
        target = {(x + 2, 10, 10) for x in range(10, 30)}
        identity_score = tolerant_f1(target, source, 1.0)
        alignment = find_alignment(source, target, 64)
        aligned = align_voxels(source, alignment, 64)
        self.assertGreaterEqual(tolerant_f1(target, aligned, 1.0), identity_score)
        self.assertTrue(all(0.88 <= value <= 1.12 for value in alignment.scale))

    def test_transfer_never_invents_voxels(self) -> None:
        donor = {(x, 20, 20) for x in range(10, 30)}
        identity = Alignment((1.0, 1.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0, 1.0)
        support = Counter(donor)
        support.update(donor)
        transferred, source, _ = transfer_supported_components(
            donor,
            identity,
            set(donor),
            support,
            ComponentPreset("test", 4, 0.9, 1, 0.9),
            100,
            64,
        )
        self.assertEqual(transferred, donor)
        self.assertEqual(source, donor)

    def test_hybrid_replaces_locally_without_fragment_loss(self) -> None:
        base = {(x, 20, 20) for x in range(5, 36)}
        donor = {(x, 20, 20) for x in range(15, 26)}
        safe = set(base)
        fused, preserved, _ = hybrid_fusion(base, donor, safe, 24, 0.2, 100)
        self.assertTrue(donor <= fused)
        self.assertIn((5, 20, 20), preserved)
        self.assertIn((35, 20, 20), preserved)
        self.assertNotIn((14, 20, 20), preserved)

    def test_safety_mask_does_not_delete_existing_base(self) -> None:
        base = {(x, 20, 20) for x in range(5, 36)}
        donor = {(x, 20, 20) for x in range(15, 26)}
        safe = base - {(5, 20, 20)}
        fused, preserved, _ = hybrid_fusion(base, donor, safe, 24, 0.2, 100)
        self.assertIn((5, 20, 20), fused)
        self.assertIn((5, 20, 20), preserved)

    def test_reranker_serialization_math(self) -> None:
        rows = [
            {name: float(index + feature) for feature, name in enumerate(FEATURE_NAMES)}
            for index in range(10)
        ]
        model = fit_ridge(rows, [index / 10 for index in range(10)])
        self.assertGreater(predict_quality(rows[-1], model), predict_quality(rows[0], model))

    def test_policy_fallback_survives_overfilled_baseline(self) -> None:
        baseline = [{
            "internal_precision": 0.40,
            "core_fraction": 0.20,
            "pred_internal_voxels": 150,
            "gt_internal_voxels": 100,
        }]
        fallback = {
            "name": "fallback",
            "micro_internal_ratio": 1.50,
            "internal_precision": 0.40,
            "mean_core_fraction": 0.20,
            "internal_f1": 0.45,
            "internal_f05": 0.41,
            "fallback_fraction": 1.0,
        }
        rejected = {
            "name": "unsafe",
            "micro_internal_ratio": 1.60,
            "internal_precision": 0.39,
            "mean_core_fraction": 0.22,
            "internal_f1": 0.50,
            "internal_f05": 0.45,
            "fallback_fraction": 0.0,
        }
        selected = choose_conservative_policy(
            [fallback, rejected], baseline, max_internal_ratio=1.15
        )
        self.assertEqual(selected["name"], "fallback")


if __name__ == "__main__":
    unittest.main()
