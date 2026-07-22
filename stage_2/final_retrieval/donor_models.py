"""Fit the final donor-ranking model bank from raw train-only feature rows."""

from __future__ import annotations

from collections import defaultdict

from .geometry import FEATURE_NAMES, fit_query_centered_ridge, fit_ridge


EXPECTED_CATEGORIES = ("bus", "cabinet", "car", "file_cabinet")


def fit_donor_models(rows: list[dict], ridge: float) -> tuple[dict, list[dict]]:
    """Fit the shared ranker and four universal experts used by the selector.

    The car expert keeps the absolute-quality objective used by the successful
    component pipeline.  The three structural experts and shared ranker use a
    query-centered ranking objective.  All four experts are evaluated for
    every query; category is not an inference feature or branch.
    """
    if not rows:
        raise ValueError("donor model fitting requires non-empty feature rows")
    required = {
        "sample_id", "category", "target_internal_f1", *FEATURE_NAMES,
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"donor feature rows are missing columns: {sorted(missing)}")

    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_category[str(row["category"])].append(row)
    if tuple(sorted(by_category)) != EXPECTED_CATEGORIES:
        raise ValueError(
            f"expected donor rows for {EXPECTED_CATEGORIES}, got {tuple(sorted(by_category))}"
        )

    labels = [float(row["target_internal_f1"]) for row in rows]
    query_ids = [str(row["sample_id"]) for row in rows]
    shared = fit_query_centered_ridge(rows, labels, query_ids, ridge)
    training_ids = sorted(set(query_ids))
    selected_by_category = {
        category: sorted({str(row["sample_id"]) for row in category_rows})
        for category, category_rows in sorted(by_category.items())
    }
    counts = {category: len(ids) for category, ids in selected_by_category.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"donor fitting queries are not category-balanced: {counts}")
    shared["balanced_queries_per_category"] = next(iter(counts.values()))
    shared["training_query_ids"] = training_ids
    shared["training_query_counts"] = counts

    experts = []
    for category in EXPECTED_CATEGORIES:
        category_rows = by_category[category]
        category_labels = [float(row["target_internal_f1"]) for row in category_rows]
        if category == "car":
            model = fit_ridge(category_rows, category_labels, ridge)
        else:
            model = fit_query_centered_ridge(
                category_rows,
                category_labels,
                [str(row["sample_id"]) for row in category_rows],
                ridge,
            )
        experts.append({"source": category, "model": model})
    return shared, experts

