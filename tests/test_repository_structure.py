#!/usr/bin/env python3
"""Dependency-free checks for the final repository structure and entrypoints."""

from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT / "interior_reconstruction"


class RepositoryStructureTests(unittest.TestCase):
    def test_only_final_project_directories_remain(self) -> None:
        for legacy in ("stage_1", "stage_2", "stage_3"):
            self.assertFalse((REPO_ROOT / legacy).exists(), legacy)
        self.assertEqual(
            {path.name for path in PROJECT_ROOT.iterdir() if path.is_dir()},
            {
                "data_preparation",
                "retrieval_refinement",
                "trellis_finetuning_interior",
            },
        )

    def test_production_packages_contain_only_final_files(self) -> None:
        expected = {
            "data_preparation": {
                "README.md",
                "__init__.py",
                "run_pipeline.sh",
            },
            "trellis_finetuning_interior": {
                "__init__.py",
                "config.sh",
                "evaluate_internals.py",
                "generate_predictions.py",
                "run_decoder_finetuning.sh",
                "run_flow_finetuning.sh",
                "run_view18_full.sh",
                "run_view18_train.sh",
                "select_ids.py",
                "validate_dataset.py",
            },
            "retrieval_refinement": {
                "README.md",
                "__init__.py",
                "calibrate.py",
                "config.py",
                "dino_index.py",
                "donor_models.py",
                "evaluate.py",
                "geometry.py",
                "pipeline.py",
                "run_view18.sh",
                "support.py",
            },
        }
        for package, expected_files in expected.items():
            folder = PROJECT_ROOT / package
            actual = {
                path.name
                for path in folder.iterdir()
                if path.is_file() and path.name != ".DS_Store"
            }
            self.assertEqual(actual, expected_files, package)

    def test_final_training_configs_are_valid_and_exclusive(self) -> None:
        config_dir = REPO_ROOT / "configs" / "finetune"
        expected = {
            "ss_flow_img_shapenet_internals_lora.json",
            "slat_flow_img_shapenet_internals_lora.json",
            "slat_vae_enc_dec_mesh_shapenet_internals_lora.json",
        }
        self.assertEqual(
            {path.name for path in config_dir.glob("*.json")}, expected
        )
        for path in config_dir.glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("models", payload)
            self.assertIn("dataset", payload)
            self.assertIn("trainer", payload)
            self.assertGreater(int(payload["trainer"]["args"]["max_steps"]), 0)

    def test_python_and_notebook_sources_parse(self) -> None:
        roots = (PROJECT_ROOT, REPO_ROOT / "tests")
        for root in roots:
            for path in root.rglob("*.py"):
                with self.subTest(path=path.relative_to(REPO_ROOT)):
                    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        notebook = REPO_ROOT / "notebooks" / "visualize_results.ipynb"
        payload = json.loads(notebook.read_text(encoding="utf-8"))
        self.assertEqual(payload.get("nbformat"), 4)
        self.assertTrue(payload.get("cells"))

    def test_final_runners_are_portable_and_use_view18(self) -> None:
        scripts = (
            PROJECT_ROOT / "data_preparation" / "run_pipeline.sh",
            PROJECT_ROOT
            / "trellis_finetuning_interior"
            / "run_flow_finetuning.sh",
            PROJECT_ROOT
            / "trellis_finetuning_interior"
            / "run_decoder_finetuning.sh",
            PROJECT_ROOT
            / "trellis_finetuning_interior"
            / "run_view18_full.sh",
            PROJECT_ROOT
            / "trellis_finetuning_interior"
            / "run_view18_train.sh",
            PROJECT_ROOT / "retrieval_refinement" / "run_view18.sh",
        )
        for path in scripts:
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(REPO_ROOT)):
                self.assertTrue(source.startswith("#!/usr/bin/env bash"))
                self.assertIn("set -euo pipefail", source)
                self.assertNotIn("/workspace/TRELLIS", source)
                self.assertNotIn("stage_", source)
        self.assertIn(
            "VIEW_INDEX=18",
            (PROJECT_ROOT / "trellis_finetuning_interior" / "run_view18_full.sh")
            .read_text(encoding="utf-8"),
        )
        self.assertIn(
            "--view-index 18",
            (PROJECT_ROOT / "retrieval_refinement" / "run_view18.sh")
            .read_text(encoding="utf-8"),
        )

    def test_documentation_references_only_current_layout(self) -> None:
        paths = [
            REPO_ROOT / "README.md",
            PROJECT_ROOT / "data_preparation" / "README.md",
            PROJECT_ROOT / "retrieval_refinement" / "README.md",
        ]
        forbidden = ("stage_1", "stage_2", "stage_3", "final_retrieval")
        for path in paths:
            source = path.read_text(encoding="utf-8")
            for term in forbidden:
                self.assertNotIn(term, source, f"{term} remains in {path}")

    def test_validation_code_is_outside_production_packages(self) -> None:
        production_names = [
            path.name.lower()
            for path in PROJECT_ROOT.rglob("*")
            if path.is_file()
        ]
        self.assertFalse(any("smoke" in name for name in production_names))
        self.assertFalse(any(name.startswith("test_") for name in production_names))
        self.assertTrue((REPO_ROOT / "tests" / "test_retrieval_refinement.py").is_file())


if __name__ == "__main__":
    unittest.main()
