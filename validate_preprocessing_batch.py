from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from image_decode import robust_decode_image
from preprocessing import (
    apply_circular_mask,
    apply_clahe_on_green_channel,
    normalize_imagenet,
    resize_to_224,
)

DATASET_IMAGES = Path(
    "datasets/retinal-disease-detection-002/Diabetic Retinopathy/train/images"
)
OUT_DIR = Path("outputs/preprocessing_validation")
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def list_images(images_dir: Path) -> list[Path]:
    paths = [
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS
    ]
    return sorted(paths, key=lambda p: p.name)


def evenly_spaced_indices(total: int, count: int) -> list[int]:
    if total <= 0 or count <= 0:
        return []
    if count >= total:
        return list(range(total))
    return sorted({int(round(i * (total - 1) / (count - 1))) for i in range(count)})


def process_one(path: Path) -> dict[str, object]:
    decoded = robust_decode_image(path, mode="RGB")
    original = np.array(decoded.image, dtype=np.uint8)
    masked = apply_circular_mask(original)
    clahe = apply_clahe_on_green_channel(masked)
    resized = resize_to_224(clahe)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    normalized = normalize_imagenet(tensor)

    return {
        "path": path,
        "encoding_hint": decoded.encoding_hint,
        "decoder": decoded.decoder,
        "original": original,
        "masked": masked,
        "clahe": clahe,
        "resized": resized,
        "normalized": normalized,
    }


def save_panel(path: Path, out_path: Path, processed: dict[str, object]) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(processed["original"])
    axes[0].set_title(f"Original: {path.name}")
    axes[1].imshow(processed["masked"])
    axes[1].set_title("Circular Mask")
    axes[2].imshow(processed["clahe"])
    axes[2].set_title("CLAHE Green")
    axes[3].imshow(processed["resized"])
    axes[3].set_title("Resized 224x224")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_report(
    out_dir: Path,
    sample_size: int,
    total_available: int,
    successes: list[dict[str, object]],
    failures: list[dict[str, str]],
    panel_paths: list[Path],
) -> None:
    widths = [int(item["original"].shape[1]) for item in successes]
    heights = [int(item["original"].shape[0]) for item in successes]
    decoder_counts: dict[str, int] = {}
    encoding_counts: dict[str, int] = {}

    for item in successes:
        decoder = str(item["decoder"])
        decoder_counts[decoder] = decoder_counts.get(decoder, 0) + 1
        encoding = str(item["encoding_hint"])
        encoding_counts[encoding] = encoding_counts.get(encoding, 0) + 1

    summary = {
        "total_available_images": total_available,
        "sample_size_requested": sample_size,
        "processed": len(successes) + len(failures),
        "success_count": len(successes),
        "failure_count": len(failures),
        "resolution_stats": {
            "width_min": min(widths) if widths else None,
            "width_max": max(widths) if widths else None,
            "height_min": min(heights) if heights else None,
            "height_max": max(heights) if heights else None,
        },
        "decoders": decoder_counts,
        "encodings": encoding_counts,
        "failures": failures,
        "panel_files": [str(path) for path in panel_paths],
    }

    (out_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    lines: list[str] = []
    lines.append("# Preprocessing Validation Report (Progress 0.3)")
    lines.append("")
    lines.append("## Run Configuration")
    lines.append(f"- Dataset path: `{DATASET_IMAGES}`")
    lines.append(f"- Total available images: **{total_available}**")
    lines.append(f"- Requested sample size: **{sample_size}**")
    lines.append("")
    lines.append("## Results")
    lines.append(f"- Processed images: **{summary['processed']}**")
    lines.append(f"- Successes: **{summary['success_count']}**")
    lines.append(f"- Failures: **{summary['failure_count']}**")
    lines.append(
        "- Resolution range observed (W x H): "
        f"**{summary['resolution_stats']['width_min']}..{summary['resolution_stats']['width_max']}** x "
        f"**{summary['resolution_stats']['height_min']}..{summary['resolution_stats']['height_max']}**"
    )
    lines.append(f"- Decoder usage: `{decoder_counts}`")
    lines.append(f"- Encoding hints: `{encoding_counts}`")
    lines.append("")
    lines.append("## Visual Spot Checks")
    lines.append(
        f"- Generated panel images: **{len(panel_paths)}** in `{out_dir}` "
        "(original, mask, CLAHE, resize)."
    )
    for panel_path in panel_paths:
        lines.append(f"- `{panel_path}`")

    lines.append("")
    lines.append("## Failure Details")
    if failures:
        for failure in failures:
            lines.append(
                f"- `{failure['file']}` -> `{failure['error_type']}`: {failure['message']}"
            )
    else:
        lines.append("- None")

    (out_dir / "validation_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate preprocessing pipeline on retinal-disease-detection-002"
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DATASET_IMAGES,
        help="Directory containing validation images",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=75,
        help="Number of images to process (50-100 recommended)",
    )
    parser.add_argument(
        "--panel-count",
        type=int,
        default=10,
        help="How many representative visual panels to save",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Output directory for panels and reports",
    )
    args = parser.parse_args()

    images = list_images(args.images_dir)
    if not images:
        raise SystemExit(f"No supported images found in: {args.images_dir}")

    sample_size = min(args.sample_size, len(images))
    selected = images[:sample_size]
    panel_indices = set(evenly_spaced_indices(sample_size, args.panel_count))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    successes: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    panel_paths: list[Path] = []

    print(
        f"Validating preprocessing on {sample_size} / {len(images)} images from {args.images_dir}"
    )

    for index, path in enumerate(selected):
        try:
            processed = process_one(path)
            shape_ok = tuple(processed["normalized"].shape) == (3, 224, 224)
            if not shape_ok:
                raise RuntimeError(
                    f"Unexpected tensor shape: {tuple(processed['normalized'].shape)}"
                )

            successes.append(processed)
            print(
                f"OK [{index + 1:03d}/{sample_size:03d}] {path.name} "
                f"decoder={processed['decoder']} "
                f"hint={processed['encoding_hint']} "
                f"size={processed['original'].shape[1]}x{processed['original'].shape[0]}"
            )

            if index in panel_indices:
                panel_name = f"panel_{index + 1:03d}_{path.stem}.png"
                panel_path = args.out_dir / panel_name
                save_panel(path, panel_path, processed)
                panel_paths.append(panel_path)
        except Exception as error:
            failures.append(
                {
                    "file": path.name,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
            print(
                f"FAIL [{index + 1:03d}/{sample_size:03d}] {path.name} "
                f"{type(error).__name__}: {error}"
            )

    write_report(
        out_dir=args.out_dir,
        sample_size=sample_size,
        total_available=len(images),
        successes=successes,
        failures=failures,
        panel_paths=panel_paths,
    )

    print(
        f"Validation complete. Success={len(successes)} Failures={len(failures)} "
        f"Report={args.out_dir / 'validation_report.md'}"
    )

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
