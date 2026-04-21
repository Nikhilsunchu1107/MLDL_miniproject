from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from image_decode import robust_decode_image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PreprocessResult:
    original: np.ndarray
    masked: np.ndarray
    clahe: np.ndarray
    resized: np.ndarray
    tensor: torch.Tensor


def apply_circular_mask(
    image_rgb: np.ndarray,
    center: tuple[int, int] | None = None,
    radius: int | None = None,
) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    if radius is None:
        radius = int(min(h, w) * 0.48)

    yy, xx = np.ogrid[:h, :w]
    mask = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius**2

    out = image_rgb.copy()
    out[~mask] = 0
    return out


def apply_clahe_on_green_channel(
    image_rgb: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    out = image_rgb.copy()
    green = out[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    out[:, :, 1] = clahe.apply(green)
    return out


def resize_to_224(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.resize(image_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)


def normalize_imagenet(tensor_chw: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor_chw.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor_chw.dtype).view(3, 1, 1)
    return (tensor_chw - mean) / std


def preprocess_image(path: str | Path) -> PreprocessResult:
    decoded = robust_decode_image(path, mode="RGB")
    original = np.array(decoded.image, dtype=np.uint8)
    masked = apply_circular_mask(original)
    clahe = apply_clahe_on_green_channel(masked)
    resized = resize_to_224(clahe)

    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    tensor = normalize_imagenet(tensor)

    return PreprocessResult(
        original=original,
        masked=masked,
        clahe=clahe,
        resized=resized,
        tensor=tensor,
    )


def save_rgb(path: str | Path, rgb: np.ndarray) -> None:
    Image.fromarray(rgb).save(path)
