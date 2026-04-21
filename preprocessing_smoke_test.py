from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from preprocessing import preprocess_image, save_rgb

DATASET_ROOT = Path("datasets/Dataset/Original UWF Image")
OUT_DIR = Path("outputs/preprocessing_sanity")


def samples() -> list[Path]:
    return [
        DATASET_ROOT / "AMD" / "AMD-001.jpg",
        DATASET_ROOT / "DR" / "DR-001.jpg",
        DATASET_ROOT / "Healthy" / "Healthy-001.jpg",
    ]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0

    for sample in samples():
        if not sample.exists():
            print(f"MISSING: {sample}")
            failures += 1
            continue

        result = preprocess_image(sample)
        shape_ok = tuple(result.tensor.shape) == (3, 224, 224)
        rng = (float(result.tensor.min()), float(result.tensor.max()))

        print(
            f"OK: {sample.name:<14} shape={tuple(result.tensor.shape)} "
            f"shape_ok={shape_ok} range=({rng[0]:.3f}, {rng[1]:.3f})"
        )

        stem = sample.stem
        save_rgb(OUT_DIR / f"{stem}_original.png", result.original)
        save_rgb(OUT_DIR / f"{stem}_masked.png", result.masked)
        save_rgb(OUT_DIR / f"{stem}_clahe.png", result.clahe)
        save_rgb(OUT_DIR / f"{stem}_resized.png", result.resized)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(result.original)
        axes[0].set_title("Original")
        axes[1].imshow(result.masked)
        axes[1].set_title("Masked")
        axes[2].imshow(result.clahe)
        axes[2].set_title("CLAHE Green")
        axes[3].imshow(result.resized)
        axes[3].set_title("Resized 224x224")

        for ax in axes:
            ax.axis("off")

        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{stem}_panel.png", dpi=120)
        plt.close(fig)

        if not shape_ok:
            failures += 1

    if failures > 0:
        raise SystemExit(1)

    print(f"Smoke test passed. Outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
