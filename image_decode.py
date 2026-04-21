from __future__ import annotations

import argparse
import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


class ImageDecodeError(RuntimeError):
    pass


def sniff_image_encoding(data: bytes) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if data.startswith((b"II*\x00", b"MM\x00*")):
        return "TIFF"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    return "UNKNOWN"


@dataclass(frozen=True)
class DecodedImage:
    image: Image.Image
    encoding_hint: str
    decoder: str


def robust_decode_image(path: str | Path, mode: str = "RGB") -> DecodedImage:
    path = Path(path)
    data = path.read_bytes()
    encoding_hint = sniff_image_encoding(data)

    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            pil_format = img.format or encoding_hint
            output = img.convert(mode) if mode else img.copy()
            return DecodedImage(image=output, encoding_hint=pil_format, decoder="PIL")
    except (UnidentifiedImageError, OSError, ValueError) as pil_error:
        arr = np.frombuffer(data, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise ImageDecodeError(
                f"Failed to decode image: {path} (PIL error: {pil_error})"
            ) from pil_error

        if decoded.ndim == 2:
            output = Image.fromarray(decoded, mode="L")
        else:
            rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            output = Image.fromarray(rgb, mode="RGB")

        if mode and output.mode != mode:
            output = output.convert(mode)

        return DecodedImage(image=output, encoding_hint=encoding_hint, decoder="OpenCV")


def _default_samples(dataset_root: Path) -> list[Path]:
    return [
        dataset_root / "AMD" / "AMD-001.jpg",
        dataset_root / "DR" / "DR-001.jpg",
        dataset_root / "Healthy" / "Healthy-001.jpg",
    ]


def run_smoke_test(dataset_root: Path) -> int:
    print(f"Dataset root: {dataset_root}")
    failures = 0

    for sample in _default_samples(dataset_root):
        if not sample.exists():
            print(f"MISSING  | {sample}")
            failures += 1
            continue

        try:
            decoded = robust_decode_image(sample, mode="RGB")
            print(
                "OK      | "
                f"{sample.name:<14} "
                f"hint={decoded.encoding_hint:<6} "
                f"decoder={decoded.decoder:<6} "
                f"size={decoded.image.size}"
            )
        except Exception as error:
            failures += 1
            print(f"FAILED  | {sample.name} -> {error}")

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Robust TIFF-in-JPG image decoder smoke test"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/Dataset/Original UWF Image"),
        help="Path to category folders (AMD, DR, Healthy, ...)",
    )
    args = parser.parse_args()

    failures = run_smoke_test(args.dataset_root)
    if failures > 0:
        raise SystemExit(1)
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
