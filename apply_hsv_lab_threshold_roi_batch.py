from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

try:
    from skimage import color as skcolor
    from skimage.morphology import (
        binary_closing,
        binary_opening,
        disk,
        remove_small_holes,
        remove_small_objects,
    )

    SKIMAGE_AVAILABLE = True
except Exception:
    skcolor = None
    binary_closing = None
    binary_opening = None
    disk = None
    remove_small_holes = None
    remove_small_objects = None
    SKIMAGE_AVAILABLE = False

try:
    import cv2

    CV2_AVAILABLE = True
except Exception:
    cv2 = None
    CV2_AVAILABLE = False


TARGET_SIZE = (224, 224)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def preprocess_center_square_resize_224(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    width, height = img.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    square = img.crop((left, top, left + side, top + side))
    return square.resize(TARGET_SIZE, Image.BILINEAR)


def get_hsv_channels(image_uint8: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hsv = np.array(Image.fromarray(image_uint8, mode="RGB").convert("HSV"), dtype=np.float32)
    return hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]


def get_lab_channels(image_uint8: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if SKIMAGE_AVAILABLE:
        lab = skcolor.rgb2lab(image_uint8 / 255.0).astype(np.float32)
        return lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    if CV2_AVAILABLE:
        lab = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
        l = lab[:, :, 0] * (100.0 / 255.0)
        a = lab[:, :, 1] - 128.0
        b = lab[:, :, 2] - 128.0
        return l, a, b
    return None, None, None


def method_hsv_lab_threshold(image_uint8: np.ndarray) -> np.ndarray:
    _, s, v = get_hsv_channels(image_uint8)
    _, a, _ = get_lab_channels(image_uint8)
    if a is None:
        raise RuntimeError("LAB conversion unavailable.")
    mask = (
        (s >= 20)
        & (v >= 35)
        & ~((v >= 235) & (s <= 30))
        & (a >= 6.0)
    )
    return mask


def component_stats(mask_bool: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]], np.ndarray]:
    labeled, num = ndi.label(mask_bool.astype(np.uint8))
    height, width = mask_bool.shape
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0
    yy, xx = np.indices(mask_bool.shape)

    central_region = np.zeros_like(mask_bool, dtype=bool)
    cy0, cy1 = int(height * 0.25), int(height * 0.75)
    cx0, cx1 = int(width * 0.25), int(width * 0.75)
    central_region[cy0:cy1, cx0:cx1] = True

    stats: list[dict[str, object]] = []
    for comp_id in range(1, num + 1):
        comp_mask = labeled == comp_id
        area = int(comp_mask.sum())
        cy = float(yy[comp_mask].mean())
        cx = float(xx[comp_mask].mean())
        dist = math.sqrt((cy - center_y) ** 2 + (cx - center_x) ** 2)
        dist_norm = dist / max(math.sqrt(center_y**2 + center_x**2), 1e-6)
        center_overlap = float((comp_mask & central_region).sum()) / max(float(central_region.sum()), 1.0)
        score = (2.0 * (area / float(mask_bool.size))) + (2.5 * center_overlap) - (1.25 * dist_norm)
        stats.append(
            {
                "component_id": comp_id,
                "mask": comp_mask,
                "area": area,
                "center_overlap": center_overlap,
                "distance_norm": dist_norm,
                "score": score,
            }
        )
    return labeled, stats, central_region


def clean_mask(mask_input: np.ndarray) -> tuple[np.ndarray, dict[str, float | bool | int]]:
    mask = np.asarray(mask_input).astype(bool)

    if SKIMAGE_AVAILABLE:
        mask = binary_opening(mask, disk(3))
        mask = binary_closing(mask, disk(5))
        mask = remove_small_holes(mask, area_threshold=256)
        mask = remove_small_objects(mask, min_size=250)
    elif CV2_AVAILABLE:
        kernel_open = np.ones((3, 3), np.uint8)
        kernel_close = np.ones((5, 5), np.uint8)
        mask_uint8 = (mask.astype(np.uint8) * 255)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel_open)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_close)
        mask = mask_uint8 > 0

    mask = ndi.binary_fill_holes(mask)
    _, stats, _ = component_stats(mask)

    filtered_stats = [item for item in stats if int(item["area"]) >= 200]
    if len(filtered_stats) == 0:
        return np.zeros_like(mask, dtype=bool), {
            "touches_border": False,
            "number_of_components": 0,
            "center_overlap_ratio": 0.0,
            "mask_area_ratio": 0.0,
        }

    best = sorted(filtered_stats, key=lambda item: float(item["score"]), reverse=True)[0]
    final_mask = np.asarray(best["mask"]).astype(bool)
    touches_border = bool(
        final_mask[0, :].any()
        or final_mask[-1, :].any()
        or final_mask[:, 0].any()
        or final_mask[:, -1].any()
    )
    quality = {
        "touches_border": touches_border,
        "number_of_components": len(filtered_stats),
        "center_overlap_ratio": float(best["center_overlap"]),
        "mask_area_ratio": float(final_mask.mean()),
    }
    return final_mask, quality


def failure_flag(mask_bool: np.ndarray, quality: dict[str, float | bool | int]) -> bool:
    if mask_bool.sum() == 0:
        return True
    if float(quality["mask_area_ratio"]) < 0.20:
        return True
    if float(quality["mask_area_ratio"]) > 0.95:
        return True
    if float(quality["center_overlap_ratio"]) < 0.08:
        return True
    return False


def apply_background_fill(image_uint8: np.ndarray, mask_bool: np.ndarray, mode: str = "gray") -> np.ndarray:
    image_float = image_uint8.astype(np.float32)
    if mode == "black":
        bg = np.zeros_like(image_float)
    elif mode == "mean":
        mean_color = image_float.mean(axis=(0, 1), keepdims=True)
        bg = np.ones_like(image_float) * mean_color
    else:
        bg = np.ones_like(image_float) * 127.0
    out = bg.copy()
    out[mask_bool] = image_float[mask_bool]
    return np.clip(out, 0, 255).astype(np.uint8)


def process_image(path: Path, background_mode: str) -> tuple[np.ndarray, dict[str, object]]:
    img = Image.open(path).convert("RGB")
    processed = preprocess_center_square_resize_224(img)
    image_uint8 = np.array(processed)

    raw_mask = method_hsv_lab_threshold(image_uint8)
    final_mask, quality = clean_mask(raw_mask)
    failed = failure_flag(final_mask, quality)

    if failed or not final_mask.any():
        output_uint8 = image_uint8.copy()
    else:
        output_uint8 = apply_background_fill(image_uint8, final_mask, mode=background_mode)

    metadata: dict[str, object] = {
        "segmentation_failed": failed,
        "mask_area_ratio": float(quality["mask_area_ratio"]),
        "center_overlap_ratio": float(quality["center_overlap_ratio"]),
        "number_of_components": int(quality["number_of_components"]),
        "touches_border": bool(quality["touches_border"]),
    }
    return output_uint8, metadata


def process_folder(source_dir: Path, output_dir: Path, background_mode: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "processing_summary.csv"

    image_paths = sorted(
        path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "filename",
                "output_file",
                "segmentation_failed",
                "mask_area_ratio",
                "center_overlap_ratio",
                "number_of_components",
                "touches_border",
            ],
        )
        writer.writeheader()

        for image_path in image_paths:
            output_uint8, metadata = process_image(image_path, background_mode=background_mode)
            output_path = output_dir / image_path.name
            Image.fromarray(output_uint8).save(output_path, quality=95)
            writer.writerow(
                {
                    "filename": image_path.name,
                    "output_file": output_path.name,
                    **metadata,
                }
            )

    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply center-square 224x224 ROI preprocessing and hsv_lab_threshold background removal."
    )
    parser.add_argument("source_dir", type=Path, help="Folder containing source images.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Folder to write processed images into. Defaults to '<source> - hsv_lab_threshold_roi_224'.",
    )
    parser.add_argument(
        "--background-mode",
        choices=["gray", "black", "mean"],
        default="gray",
        help="Background fill mode used after segmentation.",
    )
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    if not source_dir.exists() or not source_dir.is_dir():
        raise SystemExit(f"Source folder not found: {source_dir}")

    output_dir = args.output_dir.resolve() if args.output_dir else source_dir.parent / f"{source_dir.name} - hsv_lab_threshold_roi_224"
    summary_path = process_folder(source_dir, output_dir, background_mode=args.background_mode)

    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
