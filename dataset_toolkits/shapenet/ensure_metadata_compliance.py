from pathlib import Path
import argparse
import pandas as pd


def category_to_caption(category):
    category = str(category).replace("_", " ").replace("-", " ").strip()

    if not category or category.lower() == "nan":
        category = "object"

    return f"a 3D model of a {category}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--overwrite-captions", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.metadata)

    if "rendered" not in df.columns:
        df["rendered"] = False
    if "voxelized" not in df.columns:
        df["voxelized"] = False
    if "num_voxels" not in df.columns:
        df["num_voxels"] = 0
    if "cond_rendered" not in df.columns:
        df["cond_rendered"] = False
    if "aesthetic_score" not in df.columns:
        df["aesthetic_score"] = 5.0

    if "captions" not in df.columns or args.overwrite_captions:
        df["captions"] = df["category"].apply(category_to_caption)

    df.to_csv(args.metadata, index=False)
    print(f"Updated {args.metadata}")
    print(df.columns.tolist())


if __name__ == "__main__":
    main()