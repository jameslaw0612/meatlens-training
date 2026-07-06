#!/usr/bin/env python3
"""
Generate dataset CSV files for pork image folders and assign per-image time frames.

Columns:
- image_file_name
- sample_number
- meat_part
- label
- time_frame
- file_destination
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Optional

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
LABEL_ORDER = {"fresh": 0, "not fresh": 1, "spoiled": 2}
OUTPUT_COLUMNS = [
    "image_file_name",
    "sample_number",
    "meat_part",
    "label",
    "time_frame",
    "file_destination",
]


@dataclass(frozen=True)
class DatasetConfig:
    folder_name: str
    meat_part: Optional[str]
    sample_number: Optional[str]
    has_time_frame: bool


DATASETS = [
    DatasetConfig("Pork Belly - Sample 3", "Pork Belly", "3", True),
    DatasetConfig("Pork Belly - Sample 4", "Pork Belly", "4", True),
    DatasetConfig("Pork Shoulder - sample 1", "Pork Shoulder", "1", True),
    DatasetConfig("Pork Shoulder - sample 2", "Pork Shoulder", "2", True),
    DatasetConfig("Public Dataset", None, None, False),
]


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def normalize_label(name: str) -> str:
    lower = name.strip().lower()
    if "not fresh" in lower or "half-fresh" in lower or "half fresh" in lower:
        return "not fresh"
    if "fresh" in lower:
        return "fresh"
    if "spoiled" in lower:
        return "spoiled"
    return lower


def parse_hour_range(folder_name: str) -> tuple[float, float]:
    nums = [int(n) for n in re.findall(r"\d+", folder_name)]
    if len(nums) >= 2:
        start = float(nums[-2])
        end = float(nums[-1])
        if end < start:
            start, end = end, start
        return start, end
    raise ValueError(f"Could not find hour range in folder: {folder_name}")


def format_hhmmss(hours: float) -> str:
    total_seconds = int(round(hours * 3600))
    td = timedelta(seconds=total_seconds)
    total_hours = (td.days * 24) + (td.seconds // 3600)
    minutes = (td.seconds % 3600) // 60
    seconds = td.seconds % 60
    return f"{total_hours:02d}:{minutes:02d}:{seconds:02d}"


def assign_time_frames(files: list[Path], start_hour: float, end_hour: float) -> list[str]:
    count = len(files)
    if count == 0:
        return []
    if count == 1:
        return [format_hhmmss(end_hour)]

    step = (end_hour - start_hour) / (count - 1)
    frames = []
    for idx in range(count):
        hour_value = start_hour + (idx * step)
        if idx == count - 1:
            hour_value = end_hour
        frames.append(format_hhmmss(hour_value))
    return frames


def infer_public_meat_part(file_name: str) -> str:
    lower = file_name.lower()
    keyword_map = {
        "belly": "Pork Belly",
        "shoulder": "Pork Shoulder",
        "loin": "Pork Loin",
        "ham": "Pork Ham",
        "ribs": "Pork Ribs",
        "rib": "Pork Ribs",
    }
    for key, part in keyword_map.items():
        if key in lower:
            return part
    return "Unknown"


def sorted_images(folder: Path) -> list[Path]:
    files = [p for p in folder.iterdir() if is_image(p)]
    return sorted(files, key=lambda p: (p.stat().st_mtime, p.name.lower()))


def build_rows_for_dataset(root: Path, cfg: DatasetConfig) -> list[dict[str, str]]:
    dataset_path = root / cfg.folder_name
    if not dataset_path.exists():
        print(f"Warning: dataset folder not found, skipping: {dataset_path}")
        return []

    label_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    label_dirs = sorted(label_dirs, key=lambda d: LABEL_ORDER.get(normalize_label(d.name), 999))

    rows: list[dict[str, str]] = []
    for label_dir in label_dirs:
        label = normalize_label(label_dir.name)
        images = sorted_images(label_dir)
        if not images:
            continue

        if cfg.has_time_frame:
            start_hr, end_hr = parse_hour_range(label_dir.name)
            time_frames = assign_time_frames(images, start_hr, end_hr)
        else:
            time_frames = ["" for _ in images]

        for img, time_frame in zip(images, time_frames):
            rows.append(
                {
                    "image_file_name": img.name,
                    "sample_number": cfg.sample_number or "",
                    "meat_part": cfg.meat_part or infer_public_meat_part(img.name),
                    "label": label,
                    "time_frame": time_frame,
                    "file_destination": str(img.resolve()),
                }
            )

    rows.sort(
        key=lambda r: (
            LABEL_ORDER.get(r["label"], 999),
            r["time_frame"] or "99:99:99",
            r["image_file_name"].lower(),
        )
    )
    return rows


def safe_csv_name(folder_name: str) -> str:
    name = re.sub(r"[^a-z0-9]+", "_", folder_name.lower()).strip("_")
    return f"{name}.csv"


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def generate_all(root: Path, output_dir: Path, write_combined: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, str]] = []

    for cfg in DATASETS:
        rows = build_rows_for_dataset(root, cfg)
        if not rows:
            continue
        csv_path = output_dir / safe_csv_name(cfg.folder_name)
        write_csv(csv_path, rows)
        print(f"Wrote {len(rows)} rows to {csv_path}")
        all_rows.extend(rows)

    if write_combined and all_rows:
        combined_path = output_dir / "all_datasets_combined.csv"
        write_csv(combined_path, all_rows)
        print(f"Wrote {len(all_rows)} rows to {combined_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CSVs from pork image datasets.")
    parser.add_argument("--root", default=".", help="Root directory containing the dataset folders.")
    parser.add_argument("--output-dir", default="csv_outputs", help="Directory to place generated CSV files.")
    parser.add_argument(
        "--no-combined",
        action="store_true",
        help="Do not generate a combined CSV file.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir).resolve()
    generate_all(root, output_dir, write_combined=not args.no_combined)


if __name__ == "__main__":
    main()
