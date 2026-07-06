#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite split CSV file_destination values to processed image paths."
    )
    parser.add_argument(
        "--processing-summary",
        type=Path,
        default=Path("processed_hsv_lab_threshold_roi_224/processing_summary.csv"),
        help="CSV that contains original file_destination and processed_output_file columns.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("generated_splits/cross_rotation_interval200_6samples"),
        help="Directory containing split CSVs to rewrite in place.",
    )
    args = parser.parse_args()

    processing_summary_path = args.processing_summary.resolve()
    target_dir = args.target_dir.resolve()

    if not processing_summary_path.exists():
        raise SystemExit(f"Processing summary not found: {processing_summary_path}")
    if not target_dir.exists():
        raise SystemExit(f"Target directory not found: {target_dir}")

    summary_df = pd.read_csv(processing_summary_path, dtype=str).fillna("")
    required_summary_columns = {"file_destination", "processed_output_file"}
    missing_summary_columns = required_summary_columns - set(summary_df.columns)
    if missing_summary_columns:
        raise SystemExit(
            f"Processing summary is missing required columns: {sorted(missing_summary_columns)}"
        )

    path_map = dict(zip(summary_df["file_destination"], summary_df["processed_output_file"]))

    updated_files = 0
    updated_rows = 0

    for csv_path in sorted(target_dir.glob("*.csv")):
        df = pd.read_csv(csv_path, dtype=str).fillna("")
        if "file_destination" not in df.columns:
            continue

        new_destinations = df["file_destination"].map(path_map).fillna(df["file_destination"])
        changed_mask = new_destinations.ne(df["file_destination"])
        changed_rows = int(changed_mask.sum())
        if changed_rows == 0:
            continue

        df["file_destination"] = new_destinations
        df.to_csv(csv_path, index=False)

        updated_files += 1
        updated_rows += changed_rows
        print(f"Updated {csv_path.name}: {changed_rows} rows")

    print(f"Updated files: {updated_files}")
    print(f"Updated rows: {updated_rows}")


if __name__ == "__main__":
    main()
