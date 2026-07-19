#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np

from compare_internals import interior, read_voxels


SPLITS = ("train", "val", "test")
VOXEL_RESOLUTION = 64
FEATURE_RESOLUTION = 16
REPRESENTATIVES_PER_CLUSTER = 5


def load_sklearn():
    try:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.metrics import silhouette_score
    except ImportError as exc:
        raise SystemExit("scikit-learn is required: pip install scikit-learn") from exc
    return PCA, KMeans, silhouette_score


def make_feature(voxels, margin):
    """Max-pool the 64^3 internal occupancy mask to a flat 16^3 vector."""
    internal_voxels = interior(voxels, margin)
    grid = np.zeros(
        (FEATURE_RESOLUTION, FEATURE_RESOLUTION, FEATURE_RESOLUTION),
        dtype=np.float32,
    )
    if internal_voxels:
        coordinates = np.asarray(list(internal_voxels), dtype=np.int32)
        coordinates //= VOXEL_RESOLUTION // FEATURE_RESOLUTION
        grid[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]] = 1.0
    return grid.reshape(-1), len(internal_voxels)


def load_split(dataset_root, split, margin):
    split_dir = dataset_root / split
    metadata_path = split_dir / "metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")

    with metadata_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No rows found in {metadata_path}")
    if "sha256" not in rows[0] or "category" not in rows[0]:
        raise ValueError(f"{metadata_path} must contain sha256 and category columns")

    samples = []
    seen_ids = set()
    print(f"Reading {split}: {len(rows)} objects")
    for row in rows:
        sample_id = row["sha256"].strip()
        category = row["category"].strip()
        if not sample_id or not category:
            raise ValueError(f"Empty sha256 or category in {metadata_path}")
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate sha256 in {metadata_path}: {sample_id}")
        seen_ids.add(sample_id)

        voxel_path = split_dir / "voxels" / f"{sample_id}.ply"
        if not voxel_path.is_file():
            raise FileNotFoundError(f"Missing voxel file: {voxel_path}")
        voxels = read_voxels(voxel_path, VOXEL_RESOLUTION)
        feature, internal_count = make_feature(voxels, margin)
        samples.append({
            "sha256": sample_id,
            "category": category,
            "feature": feature,
            "internal_voxels": internal_count,
        })
    return samples


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Cluster GT internal occupancy separately inside each object category."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--margin", type=int, default=4)
    parser.add_argument("--clusters", type=int, default=2)
    parser.add_argument("--pca-components", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.margin < 1:
        parser.error("--margin must be at least 1")
    if args.clusters < 2:
        parser.error("--clusters must be at least 2")
    if args.pca_components < 1:
        parser.error("--pca-components must be at least 1")

    PCA, KMeans, silhouette_score = load_sklearn()
    samples_by_split = {
        split: load_split(args.dataset_root, split, args.margin)
        for split in SPLITS
    }
    train_categories = sorted({sample["category"] for sample in samples_by_split["train"]})
    if not train_categories:
        raise ValueError("The training split contains no categories")

    assignment_rows = {split: [] for split in SPLITS}
    summary_rows = []
    representative_rows = []

    for category in train_categories:
        category_samples = {
            split: [
                sample for sample in samples_by_split[split]
                if sample["category"] == category
            ]
            for split in SPLITS
        }
        train_samples = category_samples["train"]
        if len(train_samples) <= args.clusters:
            raise ValueError(
                f"Category {category} needs more training objects than "
                f"--clusters={args.clusters}; found {len(train_samples)}"
            )

        train_features = np.stack([sample["feature"] for sample in train_samples])
        distinct_features = len({feature.tobytes() for feature in train_features})
        if distinct_features < args.clusters:
            raise ValueError(
                f"Category {category} has only {distinct_features} distinct internal features"
            )

        num_components = min(
            args.pca_components,
            len(train_samples) - 1,
            train_features.shape[1],
        )
        pca = PCA(n_components=num_components, random_state=args.seed)
        train_projected = pca.fit_transform(train_features)
        kmeans = KMeans(
            n_clusters=args.clusters,
            n_init=10,
            random_state=args.seed,
        )
        train_labels = kmeans.fit_predict(train_projected)
        unique_labels = np.unique(train_labels)
        if len(unique_labels) != args.clusters:
            raise ValueError(f"K-means produced fewer than {args.clusters} clusters for {category}")

        silhouette = float(silhouette_score(train_projected, train_labels))
        explained_variance = float(np.sum(pca.explained_variance_ratio_))
        labels_by_split = {}
        distances_by_split = {}

        for split in SPLITS:
            split_samples = category_samples[split]
            if not split_samples:
                labels_by_split[split] = np.empty(0, dtype=np.int32)
                distances_by_split[split] = np.empty(0, dtype=np.float32)
                continue

            features = np.stack([sample["feature"] for sample in split_samples])
            projected = train_projected if split == "train" else pca.transform(features)
            labels = train_labels if split == "train" else kmeans.predict(projected)
            distances = np.linalg.norm(projected - kmeans.cluster_centers_[labels], axis=1)
            labels_by_split[split] = labels
            distances_by_split[split] = distances

            for sample, label in zip(split_samples, labels):
                assignment_rows[split].append({
                    "sha256": sample["sha256"],
                    "category": category,
                    "cluster": int(label),
                    "pseudo_category": f"{category}_{int(label)}",
                })

        for cluster in range(args.clusters):
            train_indices = np.flatnonzero(train_labels == cluster)
            ordered_indices = train_indices[
                np.argsort(distances_by_split["train"][train_indices])
            ]
            representative = train_samples[int(ordered_indices[0])]
            summary_rows.append({
                "category": category,
                "cluster": cluster,
                "pseudo_category": f"{category}_{cluster}",
                "margin": args.margin,
                "clusters": args.clusters,
                "pca_components": num_components,
                "seed": args.seed,
                "train_samples": int(np.sum(labels_by_split["train"] == cluster)),
                "val_samples": int(np.sum(labels_by_split["val"] == cluster)),
                "test_samples": int(np.sum(labels_by_split["test"] == cluster)),
                "mean_train_internal_voxels": float(np.mean([
                    train_samples[int(index)]["internal_voxels"]
                    for index in train_indices
                ])),
                "silhouette_score": silhouette,
                "pca_explained_variance": explained_variance,
                "representative_sha256": representative["sha256"],
            })

            for rank, index in enumerate(
                ordered_indices[:REPRESENTATIVES_PER_CLUSTER],
                start=1,
            ):
                representative_rows.append({
                    "category": category,
                    "cluster": cluster,
                    "rank": rank,
                    "sha256": train_samples[int(index)]["sha256"],
                    "distance_to_centroid": float(distances_by_split["train"][index]),
                })

        print(
            f"{category}: PCA {num_components} components, "
            f"explained variance={explained_variance:.3f}, silhouette={silhouette:.3f}"
        )

    label_fields = ["sha256", "category", "cluster", "pseudo_category"]
    for split in SPLITS:
        rows = sorted(assignment_rows[split], key=lambda row: (row["category"], row["sha256"]))
        write_csv(args.output_dir / f"{split}_labels.csv", rows, label_fields)

    write_csv(
        args.output_dir / "cluster_summary.csv",
        summary_rows,
        [
            "category",
            "cluster",
            "pseudo_category",
            "margin",
            "clusters",
            "pca_components",
            "seed",
            "train_samples",
            "val_samples",
            "test_samples",
            "mean_train_internal_voxels",
            "silhouette_score",
            "pca_explained_variance",
            "representative_sha256",
        ],
    )
    write_csv(
        args.output_dir / "cluster_representatives.csv",
        representative_rows,
        ["category", "cluster", "rank", "sha256", "distance_to_centroid"],
    )
    print(f"Wrote clustering results to {args.output_dir}")


if __name__ == "__main__":
    main()
