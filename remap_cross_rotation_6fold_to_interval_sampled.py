#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


def extract_sample_number_from_dir(sample_dir_name: str) -> str | None:
    match = re.search(r"(?:Sample|sample)\s*(\d+)", sample_dir_name)
    if match:
        return match.group(1)
    return None


def build_interval_sampled_index(interval_root: Path) -> dict[tuple[str, str], str]:
    matches: defaultdict[tuple[str, str], list[str]] = defaultdict(list)

    for path in interval_root.rglob("*"):
        if not path.is_file():
            continue
        try:
            relative_parts = path.relative_to(interval_root).parts
        except ValueError:
            continue
        if len(relative_parts) < 2:
            continue

        sample_dir_name = relative_parts[0]
        sample_number = extract_sample_number_from_dir(sample_dir_name)
        if sample_number is None:
            continue

        key = (sample_number, path.name)
        matches[key].append(str(path.resolve()))

    ambiguous = {key: values for key, values in matches.items() if len(values) > 1}
    if ambiguous:
        sample_lines = []
        for key, values in list(ambiguous.items())[:10]:
            sample_lines.append(f"{key}: {values}")
        raise SystemExit(
            "Found ambiguous interval-sampled paths for some (sample_number, image_file_name) pairs.\n"
            + "\n".join(sample_lines)
        )

    return {key: values[0] for key, values in matches.items()}


def remap_csv(csv_path: Path, path_index: dict[tuple[str, str], str]) -> int:
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    if "file_destination" not in df.columns:
        return 0

    new_destinations: list[str] = []
    changed_rows = 0

    for _, row in df.iterrows():
        sample_number = str(row.get("sample_number", "")).strip()
        image_file_name = str(row.get("image_file_name", "")).strip()
        key = (sample_number, image_file_name)

        if key not in path_index:
            raise SystemExit(
                f"Missing interval-sampled match for {csv_path.name}: "
                f"sample_number={sample_number!r}, image_file_name={image_file_name!r}"
            )

        new_path = path_index[key]
        old_path = str(row.get("file_destination", "")).strip()
        new_destinations.append(new_path)
        if new_path != old_path:
            changed_rows += 1

    if changed_rows > 0:
        df["file_destination"] = new_destinations
        df.to_csv(csv_path, index=False)

    return changed_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite cross_rotation_6fold CSV file_destination values to the interval sampled image tree."
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("generated_splits/cross_rotation_6fold"),
        help="Directory containing the split CSVs to rewrite in place.",
    )
    parser.add_argument(
        "--interval-root",
        type=Path,
        default=Path("interval sampled"),
        help="Root directory containing the interval-sampled images and subfolders.",
    )
    args = parser.parse_args()

    target_dir = args.target_dir.resolve()
    interval_root = args.interval_root.resolve()

    if not target_dir.exists():
        raise SystemExit(f"Target directory not found: {target_dir}")
    if not interval_root.exists():
        raise SystemExit(f"Interval sampled root not found: {interval_root}")

    path_index = build_interval_sampled_index(interval_root)

    updated_files = 0
    updated_rows = 0

    for csv_path in sorted(target_dir.glob("*.csv")):
        changed_rows = remap_csv(csv_path, path_index)
        if changed_rows == 0:
            continue
        updated_files += 1
        updated_rows += changed_rows
        print(f"Updated {csv_path.name}: {changed_rows} rows")

    print(f"Updated files: {updated_files}")
    print(f"Updated rows: {updated_rows}")


if __name__ == "__main__":
    main()
