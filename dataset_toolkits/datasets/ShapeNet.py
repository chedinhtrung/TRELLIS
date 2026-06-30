from pathlib import Path
import pandas as pd

def category_to_caption(category):
    category = str(category).replace("_", " ").replace("-", " ").strip()
    if not category or category.lower() == "nan":
        category = "object"

    return f"a 3D model of a {category}"

def ensure_column(df, name, default):
    """
        fill in the column required with the default value if it does not exist
        this is due to sometimes TRELLIS expect columns in the metadata just for compliance
        even though we might not need those columns, e.g aesthetic_score
    """
    if name not in df.columns:
        df[name] = default
    return df

def add_args(parser):
    pass

def foreach_instance(metadata, output_dir, func, max_workers=1, desc=None):
    records = []
    root = Path(output_dir).resolve()

    for _, row in metadata.iterrows():
        local_path = Path(row["local_path"])

        if not local_path.is_absolute():
            local_path = root / local_path

        record = func(str(local_path), row["sha256"])

        if record is not None:
            records.append(record)

    return pd.DataFrame.from_records(records)

def get_metadata(output_dir, **kwargs):
    meta_path = Path(output_dir) / "metadata.csv"

    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.csv not found: {meta_path}")

    metadata = pd.read_csv(meta_path)

    for stage_csv in Path(output_dir).glob("*_*.csv"):
        if stage_csv.name == "metadata.csv":
            continue

        stage = pd.read_csv(stage_csv)
        if "sha256" not in stage.columns:
            continue

        metadata = metadata.set_index("sha256")
        stage = stage.set_index("sha256")

        metadata.update(stage)

        # Add new columns that metadata did not have yet
        for col in stage.columns:
            if col not in metadata.columns:
                metadata[col] = stage[col]

        metadata = metadata.reset_index()
    

    if "captions" not in metadata.columns:
        metadata["captions"] = metadata["category"].apply(category_to_caption)

    if "aesthetic_score" not in metadata.columns:
        metadata["aesthetic_score"] = 5.0

    return metadata