from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from image_decode import robust_decode_image
from preprocessing import (
    apply_circular_mask,
    apply_clahe_on_green_channel,
    normalize_imagenet,
    resize_to_224,
)


@dataclass
class EvalTransform:
    def __call__(self, image_rgb: np.ndarray) -> np.ndarray:
        return image_rgb


@dataclass
class TrainTransform:
    horizontal_flip_p: float = 0.5
    rotation_degrees: float = 15.0
    brightness_jitter: float = 0.2
    contrast_jitter: float = 0.2
    crop_min_scale: float = 0.9

    def __call__(self, image_rgb: np.ndarray) -> np.ndarray:
        out = image_rgb.copy()
        rng = np.random.default_rng()

        if rng.random() < self.horizontal_flip_p:
            out = np.ascontiguousarray(np.fliplr(out))

        angle = float(rng.uniform(-self.rotation_degrees, self.rotation_degrees))
        h, w = out.shape[:2]
        center = (w / 2.0, h / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        out = cv2.warpAffine(
            out,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        brightness = float(rng.uniform(-self.brightness_jitter, self.brightness_jitter))
        contrast = float(rng.uniform(-self.contrast_jitter, self.contrast_jitter))
        out_f = out.astype(np.float32)
        out_f = (out_f - 127.5) * (1.0 + contrast) + 127.5 + brightness * 255.0
        out = np.clip(out_f, 0, 255).astype(np.uint8)

        scale = float(rng.uniform(self.crop_min_scale, 1.0))
        crop_h = max(1, int(h * scale))
        crop_w = max(1, int(w * scale))
        top = int(rng.integers(0, h - crop_h + 1))
        left = int(rng.integers(0, w - crop_w + 1))
        cropped = out[top : top + crop_h, left : left + crop_w]
        out = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        return out


class GenderDataset(Dataset):
    def __init__(
        self,
        manifest_path: str
        | Path = "outputs/data_pipeline/dataset_manifest_included.csv",
        dataset_root: str | Path | None = None,
        transform: TrainTransform | EvalTransform | None = None,
        include_only_quality_pass: bool = True,
        verify_paths: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.dataset_root = Path(dataset_root) if dataset_root else None
        self.transform = transform if transform is not None else EvalTransform()

        df = pd.read_csv(self.manifest_path)
        required = {"image_id", "gender"}
        if not required.issubset(df.columns):
            raise ValueError(f"Manifest missing required columns: {sorted(required)}")

        if include_only_quality_pass and "quality_pass" in df.columns:
            df = df[df["quality_pass"] == 1].copy()

        if "file_path" in df.columns:
            df["resolved_file_path"] = df["file_path"].astype(str)
        elif "relative_file_path" in df.columns:
            if self.dataset_root is None:
                raise ValueError(
                    "dataset_root is required when manifest has only relative paths"
                )
            df["resolved_file_path"] = df["relative_file_path"].apply(
                lambda rel: str((self.dataset_root / str(rel)).resolve())
            )
        else:
            raise ValueError("Manifest must contain file_path or relative_file_path")

        label_map = {"female": 0, "male": 1}
        df["gender"] = df["gender"].astype(str).str.strip().str.lower()
        invalid = sorted(
            set(df[~df["gender"].isin(label_map.keys())]["gender"].tolist())
        )
        if invalid:
            raise ValueError(f"Unsupported gender labels in manifest: {invalid}")

        df["label"] = df["gender"].map(label_map).astype(int)

        if verify_paths:
            missing = [
                p for p in df["resolved_file_path"].tolist() if not Path(p).exists()
            ]
            if missing:
                preview = ", ".join(missing[:5])
                raise FileNotFoundError(
                    f"Missing image files in manifest: {len(missing)}. Sample: {preview}"
                )

        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def _process(self, file_path: Path) -> dict[str, np.ndarray | torch.Tensor]:
        decoded = robust_decode_image(file_path, mode="RGB")
        original = np.array(decoded.image, dtype=np.uint8)
        augmented = self.transform(original)
        masked = apply_circular_mask(augmented)
        clahe = apply_clahe_on_green_channel(masked)
        resized = resize_to_224(clahe)

        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensor = normalize_imagenet(tensor)
        return {
            "original": original,
            "augmented": augmented,
            "masked": masked,
            "clahe": clahe,
            "resized": resized,
            "tensor": tensor,
        }

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[index]
        file_path = Path(str(row["resolved_file_path"]))
        processed = self._process(file_path)
        return processed["tensor"], int(row["label"])

    def visualize(self, index: int, out_path: str | Path) -> None:
        row = self.df.iloc[index]
        file_path = Path(str(row["resolved_file_path"]))
        processed = self._process(file_path)

        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        axes[0].imshow(processed["original"])
        axes[0].set_title(f"Original\n{file_path.name}")
        axes[1].imshow(processed["augmented"])
        axes[1].set_title("Augmented")
        axes[2].imshow(processed["masked"])
        axes[2].set_title("Mask")
        axes[3].imshow(processed["clahe"])
        axes[3].set_title("CLAHE")
        axes[4].imshow(processed["resized"])
        axes[4].set_title(f"Resized 224\nlabel={row['gender']}")

        for ax in axes:
            ax.axis("off")

        fig.tight_layout()
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
