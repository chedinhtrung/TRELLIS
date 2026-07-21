#!/usr/bin/env python3
"""Deterministic preflight checks for the category-blind unified selector."""

from __future__ import annotations

import json
import sys
import types
import unittest
from unittest import mock

from retrieval_unified import (
    COMPONENT_ACTION,
    STRUCTURAL_ACTION,
    UNIFIED_FEATURE_NAMES,
    extract_champion_operators,
    extract_donor_experts,
    fit_selector_knn,
    fit_selector_ridge,
    generate_unified_candidates,
    predict_selector,
    predict_selector_ensemble,
    predict_selector_distribution,
    select_unified_candidate,
    shortlist_records,
)
from retrieval_v2 import Alignment, FEATURE_NAMES, connected_components


def features(value: float, structural: bool = False) -> dict[str, float]:
    row = {name: float(value) for name in UNIFIED_FEATURE_NAMES}
    row["is_structural"] = float(structural)
    row["image_similarity"] = 0.5 + value / 10.0
    return row


class RetrievalUnifiedTests(unittest.TestCase):
    def test_feature_schema_is_category_blind(self) -> None:
        self.assertNotIn("category", UNIFIED_FEATURE_NAMES)
        self.assertNotIn("category_id", UNIFIED_FEATURE_NAMES)

    def test_champion_recipes_drop_old_rerankers_and_gates(self) -> None:
        v2 = {
            "method": "retrieval_v2_confidence_gated_hybrid",
            "categories": {
                "car": {
                    "reranker": {"unused": True},
                    "component_preset": {"name": "balanced"},
                    "max_internal_to_exterior_ratio": 0.75,
                    "max_expansion": 1.25,
                    "base_min_component_voxels": 64,
                    "base_max_core_fraction": 0.15,
                    "minimum_transfer_coverage": 0.75,
                }
            },
        }
        v21 = {
            "method": "retrieval_v21_category_structural",
            "categories": {
                "bus": {
                    "mode": "structural_fragments",
                    "reranker": {"unused": True},
                    "structural_preset": {"name": "structural"},
                    "transfer_mode": "fragments",
                    "fusion_mode": "replace",
                    "max_internal_to_exterior_ratio": 0.52,
                    "max_expansion": 1.25,
                    "minimum_score_margin": 0.01,
                }
            },
        }
        operators = extract_champion_operators(v2, v21)
        self.assertEqual(set(operators), {COMPONENT_ACTION, STRUCTURAL_ACTION})
        for recipe in operators.values():
            self.assertNotIn("reranker", recipe)
            self.assertFalse(any(key.startswith("minimum_") for key in recipe))

    def test_all_donor_experts_are_applied_as_one_universal_bank(self) -> None:
        payload = {
            "method": "retrieval_v21_category_structural",
            "categories": {
                category: {"reranker": {"name": category}}
                for category in ("bus", "cabinet", "car", "file_cabinet")
            },
        }
        experts = extract_donor_experts(payload)
        self.assertEqual(len(experts), 4)
        self.assertEqual(
            [expert["source"] for expert in experts],
            ["bus", "cabinet", "car", "file_cabinet"],
        )

    def test_shortlist_retains_every_expert_winner(self) -> None:
        records = []
        for index in range(5):
            donor_features = {name: 0.0 for name in FEATURE_NAMES}
            if index < 4:
                donor_features[FEATURE_NAMES[index]] = 1.0
            records.append({
                "retrieved_id": f"donor_{index}",
                "rank": index + 1,
                "image_similarity": 0.5,
                "features": donor_features,
            })
        base_model = {
            "feature_names": list(FEATURE_NAMES),
            "mean": [0.0] * len(FEATURE_NAMES),
            "scale": [1.0] * len(FEATURE_NAMES),
            "coefficient": [0.0] * len(FEATURE_NAMES),
            "intercept": 0.0,
        }
        experts = []
        for index in range(4):
            model = dict(base_model)
            model["coefficient"] = [
                float(feature_index == index)
                for feature_index in range(len(FEATURE_NAMES))
            ]
            experts.append({"source": f"expert_{index}", "model": model})
        selected = shortlist_records(records, base_model, 4, experts)
        self.assertEqual(
            {row["retrieved_id"] for row in selected},
            {"donor_0", "donor_1", "donor_2", "donor_3"},
        )

    def test_shared_selector_learns_improvement_direction(self) -> None:
        rows = []
        labels = []
        for index in range(30):
            value = index / 30.0
            rows.append(features(value, structural=index % 2 == 0))
            labels.append(value - 0.4)
        model = fit_selector_ridge(rows, labels, ridge=1.0)
        self.assertGreater(
            predict_selector(features(0.9), model),
            predict_selector(features(0.1), model),
        )
        mean, std = predict_selector_ensemble(features(0.8), [model, model])
        self.assertAlmostEqual(mean, predict_selector(features(0.8), model))
        self.assertEqual(std, 0.0)

    def test_global_gate_accepts_only_confident_valid_candidate(self) -> None:
        rows = [features(value) for value in (0.0, 0.25, 0.5, 0.75, 1.0)]
        model = fit_selector_ridge(rows, [-0.2, -0.1, 0.0, 0.1, 0.2], ridge=1.0)
        candidates = [
            {
                "valid": True,
                "features": features(0.9),
                "image_similarity": 0.9,
                "rank": 1,
            },
            {
                "valid": True,
                "features": features(0.1, structural=True),
                "image_similarity": 0.8,
                "rank": 2,
            },
        ]
        selected, fallback, reason, margin = select_unified_candidate(
            candidates, [model], 0.02, 0.0, 0.5
        )
        self.assertIs(selected, candidates[0])
        self.assertFalse(fallback)
        self.assertEqual(reason, "accepted")
        self.assertGreater(margin, 0.0)
        _selected, fallback, reason, _margin = select_unified_candidate(
            candidates, [model], 1.0, 0.0, 0.5
        )
        self.assertTrue(fallback)
        self.assertIn("low_predicted_improvement", reason)

    def test_knn_selector_returns_local_mean_and_uncertainty(self) -> None:
        rows = [features(value) for value in (0.0, 0.1, 0.9, 1.0)]
        model = fit_selector_knn(rows, [-0.2, -0.1, 0.1, 0.2], neighbors=2)
        low, low_std = predict_selector_distribution(features(0.05), model)
        high, high_std = predict_selector_distribution(features(0.95), model)
        self.assertLess(low, 0.0)
        self.assertGreater(high, 0.0)
        self.assertGreaterEqual(low_std, 0.0)
        self.assertGreaterEqual(high_std, 0.0)
        restored = json.loads(json.dumps(model))
        restored_high, _ = predict_selector_distribution(features(0.95), restored)
        self.assertAlmostEqual(high, restored_high)

    def test_no_valid_candidate_fails_closed(self) -> None:
        rows = [features(value) for value in (0.0, 0.5, 1.0)]
        model = fit_selector_ridge(rows, [-0.1, 0.0, 0.1], ridge=1.0)
        candidate = {
            "valid": False,
            "features": features(1.0),
            "image_similarity": 0.9,
            "rank": 1,
        }
        selected, fallback, reason, _margin = select_unified_candidate(
            [candidate], [model], 0.0, 0.0, 0.5
        )
        self.assertIs(selected, candidate)
        self.assertTrue(fallback)
        self.assertEqual(reason, "no_valid_candidate")

    def test_both_operators_generate_safe_coherent_hypotheses(self) -> None:
        donor = {
            (x, y, 30)
            for x in range(20, 30)
            for y in range(20, 30)
        }
        base = {
            (x, y, 28)
            for x in range(20, 30)
            for y in range(20, 30)
        }
        exterior = {(x, 10, 10) for x in range(64)}
        safe = donor | base
        alignment = Alignment(
            scale=(1.0, 1.0, 1.0),
            source_center=(0.0, 0.0, 0.0),
            mapped_center=(0.0, 0.0, 0.0),
            strength=1.0,
            exterior_tolerant_f1=1.0,
        )
        records = []
        candidate_internals = {}
        donor_components = {}
        for rank, donor_id in enumerate(("donor_a", "donor_b"), start=1):
            candidate_internals[donor_id] = donor
            donor_components[donor_id] = connected_components(donor)
            records.append({
                "retrieved_id": donor_id,
                "rank": rank,
                "image_similarity": 1.0 - rank / 100.0,
                "alignment": alignment,
                "aligned_internal": donor,
                "features": {
                    "image_similarity": 1.0 - rank / 100.0,
                    "exterior_tolerant_f1": 1.0,
                    "objective1_internal_tolerant_f1": 0.5,
                    "candidate_medoid_f1": 1.0,
                    "safe_retained_fraction": 1.0,
                    "log_internal_size_ratio": 0.0,
                },
                "shared_donor_score": 0.5,
                "shared_donor_rank": rank,
                "expert_scores": {index: 0.5 for index in range(4)},
                "expert_ranks": {index: rank for index in range(4)},
            })
        operators = {
            COMPONENT_ACTION: {
                "action": COMPONENT_ACTION,
                "component_preset": {"name": "balanced"},
                "max_internal_to_exterior_ratio": 2.0,
                "max_expansion": 1.25,
                "base_min_component_voxels": 64,
                "base_max_core_fraction": 0.15,
            },
            STRUCTURAL_ACTION: {
                "action": STRUCTURAL_ACTION,
                "structural_preset": {"name": "structural"},
                "transfer_mode": "fragments",
                "fusion_mode": "replace",
                "max_internal_to_exterior_ratio": 2.0,
                "max_expansion": 1.25,
                "base_min_component_voxels": 64,
                "base_max_core_fraction": 0.25,
            },
        }
        fake_compare = types.SimpleNamespace(
            interior=lambda voxels, margin: set(voxels) - exterior
        )
        with mock.patch.dict(sys.modules, {"compare_internals": fake_compare}):
            candidates = generate_unified_candidates(
                records,
                candidate_internals,
                base,
                exterior,
                safe,
                operators,
                64,
                2,
                donor_components,
                records,
            )
        self.assertEqual(len(candidates), 4)
        self.assertEqual(
            {candidate["action"] for candidate in candidates},
            {COMPONENT_ACTION, STRUCTURAL_ACTION},
        )
        for candidate in candidates:
            self.assertTrue(candidate["transferred"])
            self.assertLessEqual(candidate["transferred"], safe)
            self.assertLessEqual(exterior, candidate["prediction"])
            self.assertTrue(candidate["valid"])


if __name__ == "__main__":
    unittest.main()
