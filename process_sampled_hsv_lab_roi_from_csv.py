#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd
from PIL import Image

from apply_hsv_lab_threshold_roi_batch import process_image


def normalize_label_folder(label: str) -> str:
    lower = label.strip().lower()
    if lower == "not fresh":
        return "not fresh"
    if lower == "fresh":
        return "fresh"
    if lower == "spoiled":
        return "spoiled"
    return lower


def build_output_path(output_root: Path, sample_number: str, label: str, image_file_name: str) -> Path:
    sample_folder = f"sample {sample_number}"
    label_folder = normalize_label_folder(label)
    return output_root / sample_folder / label_folder / image_file_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process sampled images from CSV using hsv_lab_threshold + 224x224 ROI and organize them by sample/label."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("generated_splits/cross_rotation_interval200_6samples/all_sampled_images.csv"),
        help="CSV listing the sampled input images.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("processed_hsv_lab_threshold_roi_224"),
        help="Root folder where processed images will be written.",
    )
    parser.add_argument(
        "--background-mode",
        choices=["gray", "black", "mean"],
        default="gray",
        help="Background fill mode used after segmentation.",
    )
    args = parser.parse_args()

    input_csv = args.input_csv.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise SystemExit(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv, dtype=str).fillna("")
    required_columns = {
        "image_file_name",
        "sample_number",
        "label",
        "file_destination",
    }
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise SystemExit(f"Input CSV is missing required columns: {missing_columns}")

    summary_path = output_root / "processing_summary.csv"
    processed_count = 0
    failed_count = 0

    with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "image_file_name",
                "sample_number",
                "meat_part",
                "label",
                "time_frame",
                "file_destination",
                "sample_id",
                "pork_cut",
                "split_type",
                "fold",
                "source",
                "processed_output_file",
                "segmentation_failed",
                "mask_area_ratio",
                "center_overlap_ratio",
                "number_of_components",
                "touches_border",
            ],
        )
        writer.writeheader()

        for row in df.to_dict(orient="records"):
            source_path = Path(row["file_destination"])
            if not source_path.exists():
                writer.writerow(
                    {
                        **row,
                        "processed_output_file": "",
                        "segmentation_failed": "missing_source",
                        "mask_area_ratio": "",
                        "center_overlap_ratio": "",
                        "number_of_components": "",
                        "touches_border": "",
                    }
                )
                failed_count += 1
                continue

            output_path = build_output_path(
                output_root=output_root,
                sample_number=row["sample_number"],
                label=row["label"],
                image_file_name=row["image_file_name"],
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)

            output_uint8, metadata = process_image(source_path, background_mode=args.background_mode)
            Image.fromarray(output_uint8).save(output_path, quality=95)

            writer.writerow(
                {
                    **row,
                    "processed_output_file": str(output_path),
                    **metadata,
                }
            )
            processed_count += 1
            if bool(metadata.get("segmentation_failed", False)):
                failed_count += 1

    print(f"Input CSV: {input_csv}")
    print(f"Output root: {output_root}")
    print(f"Summary CSV: {summary_path}")
    print(f"Processed images: {processed_count}")
    print(f"Rows flagged as failed or missing source: {failed_count}")


if __name__ == "__main__":
    main()
