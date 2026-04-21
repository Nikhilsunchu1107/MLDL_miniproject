from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_demographics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"image_id", "gender"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required demographics columns: {sorted(missing)}")

    out = df[["image_id", "gender"]].copy()
    out["image_id"] = out["image_id"].astype(str).str.strip()
    out["gender"] = out["gender"].astype(str).str.strip().str.lower()

    if out["image_id"].duplicated().any():
        raise ValueError("Duplicate image_id values found in demographics mapping")
    if out["gender"].isna().any() or out["gender"].eq("").any():
        raise ValueError("Missing gender labels in demographics mapping")

    return out


def load_quality(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"image_id", "include_in_training", "exclusion_reason"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required quality columns: {sorted(missing)}")

    out = df[["image_id", "include_in_training", "exclusion_reason"]].copy()
    out["image_id"] = out["image_id"].astype(str).str.strip()
    out["quality_pass"] = pd.to_numeric(
        out["include_in_training"], errors="raise"
    ).astype(int)
    out["exclusion_reason"] = out["exclusion_reason"].fillna("").astype(str)

    if out["image_id"].duplicated().any():
        raise ValueError("Duplicate image_id values found in quality manifest")
    values = set(out["quality_pass"].unique().tolist())
    if not values.issubset({0, 1}):
        raise ValueError(f"quality_pass values must be 0/1, found: {sorted(values)}")

    return out[["image_id", "quality_pass", "exclusion_reason"]]


def build_file_index(images_root: Path) -> dict[str, str]:
    files = [p for p in images_root.rglob("*") if p.is_file()]
    index: dict[str, str] = {}
    duplicates: set[str] = set()

    for path in files:
        name = path.name
        rel = str(path.relative_to(images_root))
        if name in index and index[name] != rel:
            duplicates.add(name)
        else:
            index[name] = rel

    if duplicates:
        sample = ", ".join(sorted(duplicates)[:10])
        raise ValueError(
            f"Duplicate filenames found under images root. Sample: {sample}"
        )

    return index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build dataset manifest from demographics and quality artifacts"
    )
    parser.add_argument(
        "--demographics-csv",
        type=Path,
        default=Path("outputs/data_pipeline/demographics_mapping.csv"),
        help="Path to demographics mapping CSV from task 1.1",
    )
    parser.add_argument(
        "--quality-csv",
        type=Path,
        default=Path("outputs/data_pipeline/quality_filter_manifest.csv"),
        help="Path to quality manifest CSV from task 1.2",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=Path("datasets/Dataset/Original UWF Image"),
        help="Root folder containing source fundus images",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/data_pipeline"),
        help="Output directory for manifest artifacts",
    )
    args = parser.parse_args()

    demographics = load_demographics(args.demographics_csv)
    quality = load_quality(args.quality_csv)
    file_index = build_file_index(args.images_root)

    merged = demographics.merge(quality, on="image_id", how="outer", indicator=True)
    only_demo = merged[merged["_merge"] == "left_only"]["image_id"].tolist()
    only_quality = merged[merged["_merge"] == "right_only"]["image_id"].tolist()
    if only_demo or only_quality:
        raise ValueError(
            "Demographics and quality mappings do not align. "
            f"left_only={len(only_demo)} right_only={len(only_quality)}"
        )

    manifest = merged.drop(columns=["_merge"]).copy()
    manifest["relative_file_path"] = manifest["image_id"].map(file_index)
    if manifest["relative_file_path"].isna().any():
        missing_paths = (
            manifest.loc[manifest["relative_file_path"].isna(), "image_id"]
            .head(10)
            .tolist()
        )
        raise ValueError(
            f"Could not resolve file paths for image IDs. Sample: {missing_paths}"
        )

    manifest["file_path"] = manifest["relative_file_path"].apply(
        lambda rel: str((args.images_root / rel).resolve())
    )
    manifest["label_missing"] = manifest["gender"].isna() | manifest["gender"].eq("")

    included = manifest[manifest["quality_pass"] == 1].copy()
    if included["label_missing"].any():
        bad = included.loc[included["label_missing"], "image_id"].head(10).tolist()
        raise ValueError(f"Included samples with missing labels found. Sample: {bad}")

    manifest = (
        manifest[
            [
                "image_id",
                "file_path",
                "relative_file_path",
                "gender",
                "quality_pass",
                "exclusion_reason",
            ]
        ]
        .sort_values("image_id")
        .reset_index(drop=True)
    )
    included = manifest[manifest["quality_pass"] == 1].copy().reset_index(drop=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "dataset_manifest.csv"
    included_path = args.out_dir / "dataset_manifest_included.csv"
    summary_path = args.out_dir / "dataset_manifest_summary.json"

    manifest.to_csv(manifest_path, index=False)
    included.to_csv(included_path, index=False)

    summary = {
        "total_rows": int(len(manifest)),
        "included_rows": int(len(included)),
        "excluded_rows": int(len(manifest) - len(included)),
        "included_gender_counts": included["gender"].value_counts().to_dict(),
        "full_gender_counts": manifest["gender"].value_counts().to_dict(),
        "missing_labels_in_included": int(included["gender"].isna().sum()),
        "manifest_path": str(manifest_path),
        "included_manifest_path": str(included_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Dataset manifest build passed integrity checks.")
    print(f"Total rows: {len(manifest)}")
    print(f"Included rows: {len(included)}")
    print(f"Included gender counts: {included['gender'].value_counts().to_dict()}")
    print(f"Saved: {manifest_path}")
    print(f"Saved: {included_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
