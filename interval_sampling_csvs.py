#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


DATASET_FILE_MAP = {
    "sample1": "pork_shoulder_sample_1.csv",
    "sample2": "pork_shoulder_sample_2.csv",
    "sample3": "pork_belly_sample_3.csv",
    "sample4": "pork_belly_sample_4.csv",
    "public": "public_dataset.csv",
}

# (remove_k, interval_n)
RULES = {
    "sample1": {
        "fresh": (1, 2),
        "not fresh": (2, 3),
        "spoiled": (10, 11),
    },
    "sample2": {
        "fresh": (2, 3),
        "not fresh": (2, 3),
        "spoiled": (10, 11),
    },
    "sample3": {
        "fresh": None,  # no changes
        "not fresh": (2, 3),
        "spoiled": (10, 11),
    },
    "sample4": {
        "fresh": None,  # no changes
        "not fresh": (2, 3),
        "spoiled": (12, 13),
    },
    "public": {
        "fresh": (4, 5),
        "not fresh": (4, 5),
        "spoiled": (3, 4),
    },
}


def should_remove(label_seen_index: int, remove_k: int, interval_n: int) -> bool:
    return (label_seen_index % interval_n) < remove_k


def sample_rows(rows: list[dict[str, str]], dataset_key: str) -> tuple[list[dict[str, str]], dict[str, int], dict[str, int]]:
    rules = RULES[dataset_key]
    label_seen: dict[str, int] = {}
    label_removed: dict[str, int] = {}
    label_kept: dict[str, int] = {}
    kept_rows: list[dict[str, str]] = []

    for row in rows:
        label = (row.get("label") or "").strip().lower()
        rule = rules.get(label)
        idx = label_seen.get(label, 0)
        label_seen[label] = idx + 1

        remove = False
        if rule is not None:
            remove_k, interval_n = rule
            remove = should_remove(idx, remove_k, interval_n)

        if remove:
            label_removed[label] = label_removed.get(label, 0) + 1
            continue

        label_kept[label] = label_kept.get(label, 0) + 1
        kept_rows.append(row)

    return kept_rows, label_kept, label_removed


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply interval sampling rules to dataset CSVs.")
    parser.add_argument("--input-dir", default="csv_outputs", help="Directory containing original CSVs.")
    parser.add_argument(
        "--output-dir",
        default="csv_outputs_interval_sampled",
        help="Directory to write sampled CSVs.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_rows: list[dict[str, str]] = []
    combined_fields: list[str] | None = None

    for dataset_key, filename in DATASET_FILE_MAP.items():
        in_path = input_dir / filename
        if not in_path.exists():
            print(f"Warning: missing input file, skipping: {in_path}")
            continue

        rows, fieldnames = read_csv(in_path)
        sampled_rows, kept, removed = sample_rows(rows, dataset_key)

        out_path = output_dir / filename
        write_csv(out_path, sampled_rows, fieldnames)
        print(f"{filename}: kept {len(sampled_rows)} / {len(rows)}")
        print(f"  kept by label: {kept}")
        print(f"  removed by label: {removed}")

        combined_rows.extend(sampled_rows)
        if combined_fields is None:
            combined_fields = fieldnames

    if combined_rows and combined_fields:
        combined_path = output_dir / "all_datasets_combined_sampled.csv"
        write_csv(combined_path, combined_rows, combined_fields)
        print(f"Wrote combined sampled CSV: {combined_path} ({len(combined_rows)} rows)")


if __name__ == "__main__":
    main()
