from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

QUALITY_COLUMNS = [
    "field of view",
    "contrast",
    "illumination",
    "artifacts",
    "overall quality",
]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(column).strip().lower() for column in out.columns]
    return out


def require_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def normalize_image_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def ensure_binary(df: pd.DataFrame, columns: list[str], context: str) -> None:
    for column in columns:
        values = set(
            pd.to_numeric(df[column], errors="coerce").dropna().astype(int).unique()
        )
        if not values.issubset({0, 1}):
            raise ValueError(
                f"Column {column!r} in {context} is not binary 0/1. Found: {sorted(values)}"
            )


def load_ground_truth(path: Path) -> pd.DataFrame:
    df = normalize_columns(pd.read_excel(path))
    require_columns(df, ["image id", *QUALITY_COLUMNS])
    df = df[["image id", *QUALITY_COLUMNS]].copy()
    df["image id"] = normalize_image_id(df["image id"])
    ensure_binary(df, QUALITY_COLUMNS, "Ground Truth")
    for column in QUALITY_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="raise").astype(int)
    if df["image id"].duplicated().any():
        raise ValueError("Duplicate image IDs found in Ground Truth.xlsx")
    return df


def load_iqa(path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    frames: list[pd.DataFrame] = []
    for sheet in xls.sheet_names:
        raw = normalize_columns(pd.read_excel(path, sheet_name=sheet))
        require_columns(raw, ["image id", *QUALITY_COLUMNS])
        raw = raw[["image id", *QUALITY_COLUMNS]].copy()
        raw["image id"] = normalize_image_id(raw["image id"])
        ensure_binary(raw, QUALITY_COLUMNS, f"IQA sheet {sheet}")
        for column in QUALITY_COLUMNS:
            raw[column] = pd.to_numeric(raw[column], errors="raise").astype(int)
        if raw["image id"].duplicated().any():
            raise ValueError(f"Duplicate image IDs found in IQA sheet {sheet!r}")
        renamed = raw.rename(
            columns={
                "overall quality": f"{sheet}_overall_quality",
            }
        )[["image id", f"{sheet}_overall_quality"]]
        frames.append(renamed)

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="image id", how="inner", validate="one_to_one")

    merged = merged.rename(columns={"image id": "image_id"})
    annotator_columns = [c for c in merged.columns if c.endswith("_overall_quality")]
    merged["iqa_pass_votes"] = merged[annotator_columns].sum(axis=1)
    merged["iqa_majority_pass"] = (merged["iqa_pass_votes"] >= 2).astype(int)
    return merged


def load_demographics_ids(path: Path) -> set[str]:
    df = pd.read_csv(path)
    if "image_id" not in df.columns:
        raise ValueError(f"Expected image_id column in demographics mapping: {path}")
    return set(normalize_image_id(df["image_id"]))


def build_quality_manifest(
    ground_truth: pd.DataFrame,
    iqa: pd.DataFrame,
    demographics_ids: set[str],
) -> pd.DataFrame:
    gt = ground_truth.rename(columns={"image id": "image_id"}).copy()
    gt = gt.rename(
        columns={column: f"gt_{column.replace(' ', '_')}" for column in QUALITY_COLUMNS}
    )

    manifest = gt.merge(iqa, on="image_id", how="inner", validate="one_to_one")
    manifest["in_demographics_mapping"] = manifest["image_id"].isin(demographics_ids)

    if not manifest["in_demographics_mapping"].all():
        missing = manifest.loc[
            ~manifest["in_demographics_mapping"], "image_id"
        ].tolist()
        preview = ", ".join(missing[:10])
        raise ValueError(
            "Quality metadata contains image IDs missing from demographics mapping. "
            f"Count={len(missing)} Sample={preview}"
        )

    # Deterministic filter rule for task 1.2:
    # include image when Ground Truth overall quality == 1.
    manifest["include_in_training"] = (manifest["gt_overall_quality"] == 1).astype(int)
    manifest["exclusion_reason"] = manifest["include_in_training"].map(
        {1: "", 0: "ground_truth_overall_quality_0"}
    )

    annotator_columns = [c for c in manifest.columns if c.endswith("_overall_quality")]
    manifest["iqa_all_pass"] = (
        manifest[annotator_columns].sum(axis=1) == len(annotator_columns)
    ).astype(int)
    manifest["iqa_all_fail"] = (manifest[annotator_columns].sum(axis=1) == 0).astype(
        int
    )
    manifest["gt_vs_iqa_majority_agree"] = (
        manifest["gt_overall_quality"] == manifest["iqa_majority_pass"]
    ).astype(int)

    return manifest.sort_values("image_id").reset_index(drop=True)


def write_outputs(manifest: pd.DataFrame, out_dir: Path) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "quality_filter_manifest.csv"
    summary_path = out_dir / "quality_filter_summary.json"
    rule_path = out_dir / "quality_filter_rule.md"

    manifest.to_csv(manifest_path, index=False)

    total = int(len(manifest))
    included = int((manifest["include_in_training"] == 1).sum())
    excluded = total - included
    summary = {
        "total_images": total,
        "included_images": included,
        "excluded_images": excluded,
        "inclusion_rate": included / total if total else 0.0,
        "rule": "include image iff Ground Truth overall quality == 1",
        "exclusion_reason_counts": manifest["exclusion_reason"]
        .value_counts(dropna=False)
        .to_dict(),
        "ground_truth_overall_quality_counts": manifest["gt_overall_quality"]
        .value_counts()
        .to_dict(),
        "iqa_majority_pass_counts": manifest["iqa_majority_pass"]
        .value_counts()
        .to_dict(),
        "gt_iqa_majority_agreement_rate": float(
            manifest["gt_vs_iqa_majority_agree"].mean()
        ),
        "manifest_path": str(manifest_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rule_md = "\n".join(
        [
            "# Quality Filter Rule (Task 1.2)",
            "",
            "Deterministic training inclusion rule:",
            "",
            "- Include image if and only if `Ground Truth.xlsx` `Overall quality == 1`.",
            "- Exclude image when `Overall quality == 0`.",
            "",
            "Why this rule:",
            "",
            "- `Ground Truth.xlsx` is the primary quality metadata source in the project plan.",
            "- Rule is simple, deterministic, and reproducible.",
            "- `Individual Quality Assessment.xlsx` is parsed for agreement diagnostics only and does not override inclusion.",
            "",
            f"Output manifest: `{manifest_path}`",
            f"Output summary: `{summary_path}`",
        ]
    )
    rule_path.write_text(rule_md + "\n", encoding="utf-8")

    return manifest_path, summary_path, rule_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse quality metadata and produce deterministic filtering outputs"
    )
    parser.add_argument(
        "--ground-truth-path",
        type=Path,
        default=Path("datasets/Dataset/Ground Truth.xlsx"),
        help="Path to Ground Truth.xlsx",
    )
    parser.add_argument(
        "--iqa-path",
        type=Path,
        default=Path("datasets/Dataset/Individual Quality Assessment.xlsx"),
        help="Path to Individual Quality Assessment.xlsx",
    )
    parser.add_argument(
        "--demographics-mapping-csv",
        type=Path,
        default=Path("outputs/data_pipeline/demographics_mapping.csv"),
        help="Path to demographics mapping CSV from task 1.1",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/data_pipeline"),
        help="Output directory",
    )
    args = parser.parse_args()

    gt = load_ground_truth(args.ground_truth_path)
    iqa = load_iqa(args.iqa_path)
    demographics_ids = load_demographics_ids(args.demographics_mapping_csv)

    manifest = build_quality_manifest(gt, iqa, demographics_ids)
    manifest_path, summary_path, rule_path = write_outputs(manifest, args.out_dir)

    included = int((manifest["include_in_training"] == 1).sum())
    excluded = int((manifest["include_in_training"] == 0).sum())
    print("Quality metadata parsing passed integrity checks.")
    print(f"Total images: {len(manifest)}")
    print(f"Included: {included} | Excluded: {excluded}")
    print("Filter rule: include iff Ground Truth overall quality == 1")
    print(f"Saved: {manifest_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {rule_path}")


if __name__ == "__main__":
    main()
