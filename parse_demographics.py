from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

EXPECTED_COUNTS = {"male": 364, "female": 336}


def normalize_gender(value: object) -> str:
    raw = str(value).strip().lower()
    mapping = {
        "male": "male",
        "m": "male",
        "female": "female",
        "f": "female",
    }
    if raw in mapping:
        return mapping[raw]
    raise ValueError(f"Unsupported gender label: {value!r}")


def find_column(columns: list[str], target: str) -> str:
    normalized = {column.strip().lower(): column for column in columns}
    if target not in normalized:
        raise KeyError(f"Missing required column: {target!r}. Found: {columns}")
    return normalized[target]


def build_mapping(excel_path: Path) -> pd.DataFrame:
    df = pd.read_excel(excel_path)
    image_col = find_column(list(df.columns), "image id")
    sex_col = find_column(list(df.columns), "sex")

    mapping = df[[image_col, sex_col]].copy()
    mapping.columns = ["image_id", "gender_raw"]
    mapping["image_id"] = mapping["image_id"].astype(str).str.strip()

    if mapping["image_id"].eq("").any() or mapping["image_id"].isna().any():
        raise ValueError("Found missing image IDs in demographics file")
    if mapping["gender_raw"].isna().any():
        raise ValueError("Found missing gender labels in demographics file")

    mapping["gender"] = mapping["gender_raw"].map(normalize_gender)

    duplicate_rows = mapping[mapping.duplicated("image_id", keep=False)].sort_values(
        "image_id"
    )
    if not duplicate_rows.empty:
        conflicts = duplicate_rows.groupby("image_id")["gender"].nunique().gt(1).any()
        if conflicts:
            raise ValueError("Found duplicate image IDs with conflicting gender labels")
        raise ValueError("Found duplicate image IDs in demographics file")

    return (
        mapping[["image_id", "gender"]].sort_values("image_id").reset_index(drop=True)
    )


def verify_image_ids(mapping: pd.DataFrame, images_root: Path) -> None:
    image_files = {path.name for path in images_root.rglob("*.jpg") if path.is_file()}
    missing = sorted(set(mapping["image_id"]) - image_files)
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(
            f"{len(missing)} mapped image IDs not found in image directory. Sample: {preview}"
        )


def verify_class_counts(mapping: pd.DataFrame) -> dict[str, int]:
    observed = {
        "male": int((mapping["gender"] == "male").sum()),
        "female": int((mapping["gender"] == "female").sum()),
    }
    if observed != EXPECTED_COUNTS:
        raise ValueError(
            f"Class count mismatch. Expected {EXPECTED_COUNTS}, got {observed}"
        )
    return observed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse demographics Excel into image_id -> gender mapping"
    )
    parser.add_argument(
        "--excel-path",
        type=Path,
        default=Path("datasets/Dataset/Demographics of the participants.xlsx"),
        help="Path to demographics Excel file",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=Path("datasets/Dataset/Original UWF Image"),
        help="Root folder containing labeled fundus images",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/data_pipeline"),
        help="Output directory for parsed mappings",
    )
    args = parser.parse_args()

    mapping = build_mapping(args.excel_path)
    verify_image_ids(mapping, args.images_root)
    counts = verify_class_counts(mapping)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "demographics_mapping.csv"
    json_path = args.out_dir / "demographics_mapping.json"
    summary_path = args.out_dir / "demographics_mapping_summary.json"

    mapping.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(mapping.to_dict(orient="records"), indent=2), encoding="utf-8"
    )
    summary_path.write_text(
        json.dumps(
            {
                "total": int(len(mapping)),
                "male": counts["male"],
                "female": counts["female"],
                "excel_path": str(args.excel_path),
                "images_root": str(args.images_root),
                "csv_path": str(csv_path),
                "json_path": str(json_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Demographics parsing passed integrity checks.")
    print(f"Total samples: {len(mapping)}")
    print(f"Male: {counts['male']} | Female: {counts['female']}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
