#!/usr/bin/env python3
"""Pair original images with their ROI versions.

Scans a dataset folder (e.g. "Pork Shoulder - sample 1"), finds original
images and their ROI counterparts (filename suffix, default `_roi`), and
writes a CSV with one row per paired image.

Output columns: file_name, roi_file, meat_part, sample_number, time, label, folder
"""
from __future__ import annotations
import argparse
import csv
import os
import re
from datetime import timedelta
from typing import List

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp', '.gif'}
DEFAULT_HOURS_PER_FOLDER = 8


def is_image(fname: str) -> bool:
    return os.path.splitext(fname)[1].lower() in IMAGE_EXTS


def parse_start_hour(folder_name: str) -> int:
    m = re.search(r"(\d+)\s*-\s*(\d+)", folder_name)
    if m:
        return int(m.group(1))
    lower = folder_name.lower()
    if 'not fresh' in lower:
        return 8
    if 'fresh' in lower and 'not fresh' not in lower:
        return 0
    m2 = re.search(r'level\s*(\d+)', lower)
    if m2:
        level = int(m2.group(1))
        return 16 + (level - 1) * 8
    return 9999


def label_from_folder(folder_name: str) -> str:
    lower = folder_name.lower()
    if 'not fresh' in lower:
        return 'not fresh'
    if 'fresh' in lower and 'not fresh' not in lower:
        return 'fresh'
    if 'spoiled' in lower:
        return 'spoiled'
    return 'unknown'


def format_seconds_as_hhmmss(seconds_float: float) -> str:
    sec = int(round(seconds_float))
    td = timedelta(seconds=sec)
    hours = td.seconds // 3600 + td.days * 24
    minutes = (td.seconds % 3600) // 60
    seconds = td.seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def generate_pairs(dataset_dir: str, out_csv: str, meat_part: str = 'Pork Shoulder', sample_number: str = '1', hours_per_folder: float = DEFAULT_HOURS_PER_FOLDER, suffix: str = '_roi') -> int:
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    folders: List[str] = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    folders = sorted(folders, key=parse_start_hour)

    rows = []

    for folder in folders:
        folder_path = os.path.join(dataset_dir, folder)
        files = [f for f in os.listdir(folder_path) if is_image(f) and not os.path.splitext(f)[0].endswith(suffix)]
        if not files:
            continue
        try:
            files = sorted(files, key=lambda f: os.path.getctime(os.path.join(folder_path, f)))
        except Exception:
            files = sorted(files)

        n = len(files)
        seconds_per_folder = hours_per_folder * 3600
        delta = seconds_per_folder / n
        label = label_from_folder(folder)

        for i, fname in enumerate(files):
            seconds_since_start = (i + 1) * delta
            start_hour = parse_start_hour(folder)
            if start_hour >= 1000:
                start_hour = 0
            seconds_total = start_hour * 3600 + seconds_since_start
            time_str = format_seconds_as_hhmmss(seconds_total)

            base, ext = os.path.splitext(fname)
            roi_name = base + suffix + ext
            roi_path = os.path.join(folder_path, roi_name)
            if not os.path.exists(roi_path):
                # skip originals without an ROI counterpart
                continue

            rows.append([fname, roi_name, meat_part, sample_number, time_str, label, folder])

    with open(out_csv, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['file_name', 'roi_file', 'meat_part', 'sample_number', 'time', 'label', 'folder'])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_csv}")
    return len(rows)


def main() -> None:
    p = argparse.ArgumentParser(description='Pair original images with ROI versions')
    p.add_argument('--dataset', default='Pork Shoulder - sample 1', help='Dataset folder path')
    p.add_argument('--output', default='pork_shoulder_sample_pairs.csv', help='Output CSV path')
    p.add_argument('--hours-per-folder', type=float, default=DEFAULT_HOURS_PER_FOLDER, help='Hours per folder')
    p.add_argument('--sample-number', default='1', help='Sample number')
    p.add_argument('--suffix', default='_roi', help='Suffix used for ROI files')
    args = p.parse_args()

    generate_pairs(args.dataset, args.output, sample_number=str(args.sample_number), hours_per_folder=args.hours_per_folder, suffix=args.suffix)


if __name__ == '__main__':
    main()
