#!/usr/bin/env python3
"""Validate every external artifact needed by the two final pipelines."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_CATEGORIES = {"bus", "cabinet", "car", "file_cabinet"}
EXPECTED_POLICY_METHOD = "retrieval_unified_adaptive"
VIEW_INDEX = 18
RESOLUTION = 64
MARGIN = 2
TOP_K = 20


def nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def require_file(path: Path, label: str, errors: list[str]) -> None:
    if not nonempty_file(path):
        errors.append(f"missing or empty {label}: {path}")


def require_directory(path: Path, label: str, errors: list[str]) -> None:
    if not path.is_dir():
        errors.append(f"missing {label}: {path}")


def read_metadata(path: Path, errors: list[str]) -> dict[str, str]:
    require_file(path, "metadata", errors)
    if not nonempty_file(path):
        return {}
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except Exception as exc:
        errors.append(f"cannot read {path}: {exc}")
        return {}
    if not rows:
        errors.append(f"metadata contains no rows: {path}")
        return {}
    required = {"sha256", "category"}
    missing = required - set(rows[0])
    if missing:
        errors.append(f"{path} is missing columns: {sorted(missing)}")
        return {}
    output: dict[str, str] = {}
    for row in rows:
        sample_id = row["sha256"].strip()
        category = row["category"].strip()
        if not sample_id or category not in EXPECTED_CATEGORIES:
            errors.append(f"invalid metadata row in {path}: {row}")
            continue
        if sample_id in output:
            errors.append(f"duplicate metadata ID in {path}: {sample_id}")
        output[sample_id] = category
    return output


def read_ids(path: Path, label: str, errors: list[str]) -> list[str]:
    require_file(path, label, errors)
    if not nonempty_file(path):
        return []
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not ids:
        errors.append(f"empty ID list: {path}")
    if len(ids) != len(set(ids)):
        errors.append(f"duplicate IDs in {path}")
    return ids


def require_predictions(
    directory: Path,
    ids: list[str],
    label: str,
    errors: list[str],
) -> None:
    require_directory(directory, label, errors)
    if not directory.is_dir():
        return
    missing = [sample_id for sample_id in ids if not nonempty_file(directory / f"{sample_id}.ply")]
    if missing:
        errors.append(
            f"{label} is missing {len(missing)}/{len(ids)} files; "
            f"first IDs: {missing[:5]}"
        )


def validate_policy(path: Path, errors: list[str]) -> int:
    require_file(path, "frozen policy", errors)
    if not nonempty_file(path):
        return TOP_K
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"cannot read frozen policy {path}: {exc}")
        return TOP_K
    expected_scalars = {
        "method": EXPECTED_POLICY_METHOD,
        "view_index": VIEW_INDEX,
        "resolution": RESOLUTION,
        "margin": MARGIN,
        "category_feature_used": False,
        "same_category_gallery": True,
    }
    for key, expected in expected_scalars.items():
        if payload.get(key) != expected:
            errors.append(
                f"policy {key}={payload.get(key)!r}, expected {expected!r}"
            )
    top_k = int(payload.get("top_k", -1))
    if top_k != TOP_K:
        errors.append(f"policy top_k={top_k}, expected {TOP_K}")
    if set(payload.get("operators", {})) != {
        "component_hybrid",
        "structural_replace",
    }:
        errors.append("policy does not contain exactly the two final operators")
    selector = payload.get("selector", {})
    if len(selector.get("models", [])) != 1:
        errors.append("policy must contain one final selector fitted on all valid rows")
    if int(selector.get("neighbors", -1)) != 8:
        errors.append("policy selector must use K=8")
    if int(payload.get("donor_shortlist", -1)) != 8:
        errors.append("policy donor shortlist must contain eight candidates")
    if int(payload.get("cross_validation", {}).get("folds", -1)) != 5:
        errors.append("policy must record five-fold query-level cross-validation")
    expert_sources = {
        expert.get("source") for expert in payload.get("donor_experts", [])
    }
    if expert_sources != EXPECTED_CATEGORIES:
        errors.append(
            f"policy donor experts are {sorted(expert_sources)}, "
            f"expected {sorted(EXPECTED_CATEGORIES)}"
        )
    return top_k if top_k > 0 else TOP_K


def validate_rankings(
    path: Path,
    query_ids: list[str],
    test_categories: dict[str, str],
    train_categories: dict[str, str],
    top_k: int,
    errors: list[str],
) -> None:
    require_file(path, "DINO rankings", errors)
    if not nonempty_file(path):
        return
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {
                "query_mode",
                "sample_id",
                "category",
                "rank",
                "retrieved_id",
                "image_cosine_similarity",
            }
            missing = required - set(reader.fieldnames or [])
            if missing:
                errors.append(f"ranking CSV is missing columns: {sorted(missing)}")
                return
            grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in reader:
                grouped[row["sample_id"]].append(row)
    except Exception as exc:
        errors.append(f"cannot read rankings {path}: {exc}")
        return

    query_set = set(query_ids)
    if set(grouped) != query_set:
        errors.append(
            "ranking query IDs do not match the held-out ID list: "
            f"missing={len(query_set - set(grouped))}, "
            f"extra={len(set(grouped) - query_set)}"
        )
    for sample_id in query_ids:
        rows = grouped.get(sample_id, [])
        if len(rows) != top_k:
            errors.append(
                f"{sample_id}: expected {top_k} donors, found {len(rows)}"
            )
            continue
        try:
            ranks = sorted(int(row["rank"]) for row in rows)
        except ValueError:
            errors.append(f"{sample_id}: non-integer donor rank")
            continue
        if ranks != list(range(1, top_k + 1)):
            errors.append(f"{sample_id}: donor ranks are not 1..{top_k}")
        category = test_categories.get(sample_id)
        for row in rows:
            donor_id = row["retrieved_id"]
            if row["query_mode"] != "dino_view018":
                errors.append(f"{sample_id}: ranking was not built from view 18")
                break
            if row["category"] != category:
                errors.append(f"{sample_id}: ranking query category mismatch")
                break
            if train_categories.get(donor_id) != category:
                errors.append(
                    f"{sample_id}: donor {donor_id} violates same-category retrieval"
                )
                break


def validate_fixture(
    ids_path: Path,
    expected_path: Path,
    test_ids: list[str],
    test_categories: dict[str, str],
    errors: list[str],
) -> None:
    fixture_ids = read_ids(ids_path, "regression ID fixture", errors)
    if len(fixture_ids) != 8:
        errors.append(f"regression fixture must contain 8 IDs, found {len(fixture_ids)}")
    counts = Counter(test_categories.get(sample_id) for sample_id in fixture_ids)
    if counts != Counter({category: 2 for category in EXPECTED_CATEGORIES}):
        errors.append(f"regression fixture is not two samples per category: {counts}")
    unknown = set(fixture_ids) - set(test_ids)
    if unknown:
        errors.append(f"regression IDs are not in the held-out split: {sorted(unknown)}")
    require_file(expected_path, "regression reference", errors)
    if nonempty_file(expected_path):
        with expected_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        expected_ids = [row.get("sample_id", "") for row in rows]
        if set(expected_ids) != set(fixture_ids) or len(rows) != 8:
            errors.append("regression reference IDs do not match the fixture ID list")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Validate all artifacts needed to reproduce the final pipelines."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repo_root / "datasets" / "ShapeNetTRELLIS_full",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=repo_root / "results" / "objective1_full",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=repo_root / "results" / "objective1_view18_train",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=repo_root / "results" / "objective1_view18_full",
    )
    parser.add_argument(
        "--dino-dir",
        type=Path,
        default=repo_root / "results" / "dino_retrieval_view18",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=repo_root / "results" / "retrieval_unified_view18" / "policy.json",
    )
    parser.add_argument(
        "--fixture-ids",
        type=Path,
        default=repo_root / "tests" / "fixtures" / "retrieval_refinement" / "ids.txt",
    )
    parser.add_argument(
        "--fixture-expected",
        type=Path,
        default=repo_root
        / "tests"
        / "fixtures"
        / "retrieval_refinement"
        / "expected.csv",
    )
    args = parser.parse_args()

    errors: list[str] = []
    train_dir = args.dataset_root / "train"
    test_dir = args.dataset_root / "test"
    train_categories = read_metadata(train_dir / "metadata.csv", errors)
    test_categories = read_metadata(test_dir / "metadata.csv", errors)
    if train_categories and set(train_categories.values()) != EXPECTED_CATEGORIES:
        errors.append("training metadata does not contain all four categories")
    if test_categories and set(test_categories.values()) != EXPECTED_CATEGORIES:
        errors.append("held-out metadata does not contain all four categories")

    train_ids = read_ids(
        args.train_output / "selected_ids.txt", "train prediction ID list", errors
    )
    test_ids = read_ids(
        args.test_output / "selected_ids.txt", "held-out prediction ID list", errors
    )
    if train_categories and set(train_ids) != set(train_categories):
        errors.append(
            "train prediction IDs do not exactly match training metadata "
            f"({len(train_ids)} vs {len(train_categories)})"
        )
    if test_categories and set(test_ids) != set(test_categories):
        errors.append(
            "held-out prediction IDs do not exactly match test metadata "
            f"({len(test_ids)} vs {len(test_categories)})"
        )

    require_predictions(
        args.train_output / "predictions" / "objective1" / "seed_42" / "voxels",
        train_ids,
        "fine-tuned train voxels",
        errors,
    )
    test_prediction_root = (
        args.test_output / "predictions" / "objective1" / "seed_42"
    )
    require_predictions(
        test_prediction_root / "voxels",
        test_ids,
        "fine-tuned held-out voxels",
        errors,
    )
    require_predictions(
        test_prediction_root / "mesh",
        test_ids,
        "fine-tuned held-out meshes",
        errors,
    )

    checkpoints = (
        args.checkpoint_root / "ss_flow" / "ckpts" / "denoiser_lora_final.pt",
        args.checkpoint_root / "slat_flow" / "ckpts" / "denoiser_lora_final.pt",
        args.checkpoint_root / "decoder" / "ckpts" / "decoder_lora_final.pt",
    )
    for checkpoint in checkpoints:
        require_file(checkpoint, "fine-tuning checkpoint", errors)

    embeddings = (
        args.dino_dir
        / "embeddings"
        / "train_view018_dinov2_vitl14_reg.npz"
    )
    require_file(embeddings, "train DINO embedding cache", errors)
    top_k = validate_policy(args.policy, errors)
    if train_categories and test_categories:
        validate_rankings(
            args.dino_dir / "rankings.csv",
            test_ids,
            test_categories,
            train_categories,
            top_k,
            errors,
        )
        validate_fixture(
            args.fixture_ids,
            args.fixture_expected,
            test_ids,
            test_categories,
            errors,
        )
    else:
        require_file(args.dino_dir / "rankings.csv", "DINO rankings", errors)
        require_file(args.fixture_ids, "regression ID fixture", errors)
        require_file(args.fixture_expected, "regression reference", errors)

    if errors:
        preview = "\n  - ".join(errors[:50])
        suffix = f"\n  ... and {len(errors) - 50} more" if len(errors) > 50 else ""
        raise SystemExit(
            f"Reproduction input validation failed ({len(errors)} errors):\n"
            f"  - {preview}{suffix}"
        )

    print("PASS: reproduction inputs are complete and mutually consistent")
    print(f"  train/test shapes: {len(train_ids)}/{len(test_ids)}")
    print(f"  categories: {', '.join(sorted(EXPECTED_CATEGORIES))}")
    print(f"  DINO view/top-k: {VIEW_INDEX}/{top_k}")
    print(f"  policy: {args.policy}")


if __name__ == "__main__":
    main()
