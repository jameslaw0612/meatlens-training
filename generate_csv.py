#!/usr/bin/env python3
"""
generate_csv.py

Generate a CSV from the "Pork Shoulder - sample 1" folder.

Output columns: file_name, meat_part, sample_number, time, label

Usage:
    python generate_csv.py
    python generate_csv.py --dataset "Pork Shoulder - sample 1" --output pork_shoulder_sample_1.csv
"""

import os
import csv
import re
import argparse
from datetime import timedelta

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp', '.gif'}
# Default hours per folder (can be overridden via CLI)
DEFAULT_HOURS_PER_FOLDER = 8


def is_image(fname):
    return os.path.splitext(fname)[1].lower() in IMAGE_EXTS


def parse_start_hour(folder_name):
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


def label_from_folder(folder_name):
    lower = folder_name.lower()
    if 'not fresh' in lower:
        return 'not fresh'
    if 'fresh' in lower and 'not fresh' not in lower:
        return 'fresh'
    if 'spoiled' in lower:
        return 'spoiled'
    return 'unknown'


def format_seconds_as_hhmmss(seconds_float):
    sec = int(round(seconds_float))
    td = timedelta(seconds=sec)
    hours = td.seconds // 3600 + td.days * 24
    minutes = (td.seconds % 3600) // 60
    seconds = td.seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def generate(dataset_dir, out_csv, meat_part='Pork Shoulder', sample_number='1', hours_per_folder=DEFAULT_HOURS_PER_FOLDER):
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    folders = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    folders = sorted(folders, key=parse_start_hour)

    rows = []

    for folder in folders:
        folder_path = os.path.join(dataset_dir, folder)
        files = [f for f in os.listdir(folder_path) if is_image(f)]
        if not files:
            continue
        # sort chronologically: prefer creation time, fallback to filename
        try:
            files = sorted(files, key=lambda f: os.path.getctime(os.path.join(folder_path, f)))
        except Exception:
            files = sorted(files)

        n = len(files)
        seconds_per_folder = hours_per_folder * 3600
        delta = seconds_per_folder / n
        label = label_from_folder(folder)

        for i, fname in enumerate(files):
            # place first image at delta, last image at exactly hours_per_folder:00:00
            seconds_since_start = (i + 1) * delta
            # compute absolute time by adding folder start hour (e.g., not fresh starts at 8h)
            start_hour = parse_start_hour(folder)
            if start_hour >= 1000:
                start_hour = 0
            seconds_total = start_hour * 3600 + seconds_since_start
            time_str = format_seconds_as_hhmmss(seconds_total)
            # include folder name so downstream tools can group spoiled levels
            rows.append([fname, meat_part, sample_number, time_str, label, folder])

    # write CSV
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['file_name', 'meat_part', 'sample_number', 'time', 'label', 'folder'])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_csv}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Generate CSV for Pork Shoulder samples')
    p.add_argument('--dataset', default='Pork Shoulder - sample 1', help='Dataset folder path (relative or absolute)')
    p.add_argument('--output', default='pork_shoulder_sample_1.csv', help='Output CSV path')
    p.add_argument('--root', default='.', help='Workspace root if dataset path is relative')
    p.add_argument('--hours-per-folder', type=float, default=DEFAULT_HOURS_PER_FOLDER, help='Hours per folder (default: 8)')
    p.add_argument('--sample-number', default='1', help='Sample number (default: 1)')
    args = p.parse_args()

    dataset_dir = args.dataset if os.path.isabs(args.dataset) else os.path.join(args.root, args.dataset)
    hours = args.hours_per_folder
    sample_number = str(args.sample_number)
    generate(dataset_dir, args.output, sample_number=sample_number, hours_per_folder=hours)
