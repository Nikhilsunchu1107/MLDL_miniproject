from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def validate_manifest(df: pd.DataFrame) -> None:
    required = {"image_id", "file_path", "gender"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df["image_id"].duplicated().any():
        raise ValueError("Duplicate image_id values found in included manifest")

    if df["gender"].isna().any() or df["gender"].astype(str).str.strip().eq("").any():
        raise ValueError("Missing gender labels found in included manifest")

    labels = set(df["gender"].astype(str).str.strip().str.lower().unique().tolist())
    if labels != {"male", "female"}:
        raise ValueError(f"Unexpected gender labels: {sorted(labels)}")


def split_dataset(df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=random_state,
        stratify=df["gender"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_df["gender"],
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    out = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
    out = out.sort_values(["split", "image_id"]).reset_index(drop=True)
    return out


def build_summary(split_df: pd.DataFrame, random_state: int) -> dict[str, object]:
    total = int(len(split_df))
    summary: dict[str, object] = {
        "total_samples": total,
        "random_state": random_state,
        "split_counts": split_df["split"].value_counts().sort_index().to_dict(),
        "split_gender_counts": {},
    }

    for split_name in ["train", "val", "test"]:
        subset = split_df[split_df["split"] == split_name]
        counts = subset["gender"].value_counts().sort_index().to_dict()
        summary["split_gender_counts"][split_name] = counts

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create stratified 70/15/15 train/val/test split assignments"
    )
    parser.add_argument(
        "--included-manifest",
        type=Path,
        default=Path("outputs/data_pipeline/dataset_manifest_included.csv"),
        help="Path to quality-passed dataset manifest",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/data_pipeline"),
        help="Output directory for split artifacts",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for deterministic split creation",
    )
    args = parser.parse_args()

    included = pd.read_csv(args.included_manifest)
    validate_manifest(included)

    split_df = split_dataset(included, random_state=args.random_state)
    summary = build_summary(split_df, random_state=args.random_state)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    split_path = args.out_dir / "train_val_test_split.csv"
    summary_path = args.out_dir / "split_summary.json"

    split_df.to_csv(split_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Stratified split creation passed integrity checks.")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Split counts: {summary['split_counts']}")
    print(f"Split gender counts: {summary['split_gender_counts']}")
    print(f"Saved: {split_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
