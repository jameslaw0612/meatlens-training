#!/usr/bin/env python3
"""Generate CSV listing ROI images (saved next to originals)

Scans a dataset folder's subfolders and includes only files whose base
filename ends with the given suffix (default: _roi). Time and label
calculation mirrors `generate_csv.py`.
"""
from __future__ import annotations
import os
import csv
import re
import argparse
from datetime import timedelta

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


def generate(dataset_dir: str, out_csv: str, meat_part: str = 'Pork Shoulder', sample_number: str = '1', hours_per_folder: float = DEFAULT_HOURS_PER_FOLDER, suffix: str = '_roi'):
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    folders = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    folders = sorted(folders, key=parse_start_hour)

    rows = []

    for folder in folders:
        folder_path = os.path.join(dataset_dir, folder)
        files = [f for f in os.listdir(folder_path) if is_image(f) and os.path.splitext(f)[0].lower().endswith(suffix.lower())]
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
            rows.append([fname, meat_part, sample_number, time_str, label, folder])

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['file_name', 'meat_part', 'sample_number', 'time', 'label', 'folder'])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_csv}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Generate CSV for ROI images')
    p.add_argument('--dataset', default='Pork Shoulder - sample 1', help='Dataset folder path')
    p.add_argument('--output', default='pork_shoulder_sample_1_roi.csv', help='Output CSV path')
    p.add_argument('--hours-per-folder', type=float, default=DEFAULT_HOURS_PER_FOLDER, help='Hours per folder')
    p.add_argument('--sample-number', default='1', help='Sample number')
    p.add_argument('--suffix', default='_roi', help='Suffix used for ROI files')
    args = p.parse_args()

    generate(args.dataset, args.output, sample_number=str(args.sample_number), hours_per_folder=args.hours_per_folder, suffix=args.suffix)
