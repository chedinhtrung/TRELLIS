"""Fixed method choices established before the final unified calibration.

These are ordinary final-method hyperparameters, not runtime dependencies on
the exploratory v2/v2.1 policy files. The unified selector is still fitted
from train-only hypotheses by
:mod:`interior_reconstruction.retrieval_refinement.calibrate`.
"""

from __future__ import annotations

from copy import deepcopy


VIEW_INDEX = 18
MODEL_NAME = "dinov2_vitl14_reg"
RESOLUTION = 64
MARGIN = 2
TOP_K = 20
DONOR_FIT_QUERIES_PER_CATEGORY = 48


_OPERATORS = {
    "component_hybrid": {
        "action": "component_hybrid",
        "component_preset": {
            "name": "balanced",
            "min_component_voxels": 24,
            "min_retained_fraction": 0.7,
            "min_other_support": 1,
            "min_supported_fraction": 0.2,
        },
        "max_internal_to_exterior_ratio": 0.749229293550758,
        "max_expansion": 1.25,
        "base_min_component_voxels": 64,
        "base_max_core_fraction": 0.15,
    },
    "structural_replace": {
        "action": "structural_replace",
        "structural_preset": {
            "name": "structural",
            "min_fragment_voxels": 24,
            "min_other_support": 1,
            "min_supported_fraction": 0.1,
            "max_core_fraction": 0.5,
        },
        "transfer_mode": "fragments",
        "fusion_mode": "replace",
        "max_internal_to_exterior_ratio": 0.5178417638291312,
        "max_expansion": 1.25,
        "base_min_component_voxels": 64,
        "base_max_core_fraction": 0.25,
    },
}


def operator_config() -> dict:
    """Return an isolated copy so query processing cannot mutate defaults."""
    return deepcopy(_OPERATORS)
