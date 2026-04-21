from __future__ import annotations

import random
from pathlib import Path

import torch

from gender_dataset import EvalTransform, GenderDataset, TrainTransform

MANIFEST_PATH = Path("outputs/data_pipeline/dataset_manifest.csv")
OUT_DIR = Path("outputs/dataset_sanity")


def validate_samples(dataset: GenderDataset, indices: list[int], name: str) -> int:
    failures = 0
    for idx in indices:
        try:
            tensor, label = dataset[idx]
            shape_ok = tuple(tensor.shape) == (3, 224, 224)
            dtype_ok = tensor.dtype == torch.float32
            label_ok = label in {0, 1}
            finite_ok = bool(torch.isfinite(tensor).all())

            print(
                f"{name} idx={idx:03d} shape={tuple(tensor.shape)} "
                f"dtype={tensor.dtype} label={label} "
                f"shape_ok={shape_ok} dtype_ok={dtype_ok} "
                f"label_ok={label_ok} finite_ok={finite_ok}"
            )

            if not (shape_ok and dtype_ok and label_ok and finite_ok):
                failures += 1
        except Exception as error:
            failures += 1
            print(f"{name} idx={idx:03d} FAILED: {type(error).__name__}: {error}")

    return failures


def main() -> None:
    random.seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    eval_dataset = GenderDataset(manifest_path=MANIFEST_PATH, transform=EvalTransform())
    train_dataset = GenderDataset(
        manifest_path=MANIFEST_PATH, transform=TrainTransform()
    )

    if len(eval_dataset) < 20:
        raise SystemExit("Need at least 20 samples for smoke test")

    indices = random.sample(range(len(eval_dataset)), 20)
    print(f"Dataset size: {len(eval_dataset)}")
    print(f"Sample indices: {indices}")

    failures = 0
    failures += validate_samples(eval_dataset, indices, name="eval")
    failures += validate_samples(train_dataset, indices, name="train")

    for i, idx in enumerate(indices[:10], start=1):
        eval_dataset.visualize(idx, OUT_DIR / f"eval_panel_{i:02d}_idx_{idx:03d}.png")
        train_dataset.visualize(idx, OUT_DIR / f"train_panel_{i:02d}_idx_{idx:03d}.png")

    print(f"Saved visual spot-check panels to: {OUT_DIR}")

    if failures > 0:
        raise SystemExit(1)

    print("GenderDataset smoke test passed.")


if __name__ == "__main__":
    main()
