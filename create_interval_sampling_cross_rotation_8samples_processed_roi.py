#!/usr/bin/env python3
import json
import uuid
from pathlib import Path
from textwrap import dedent


def add_markdown(cells: list[dict], text: str) -> None:
    cells.append(
        {
            "cell_type": "markdown",
            "id": uuid.uuid4().hex[:8],
            "metadata": {},
            "source": (dedent(text).strip() + "\n").splitlines(keepends=True),
        }
    )


def add_code(cells: list[dict], text: str) -> None:
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "id": uuid.uuid4().hex[:8],
            "metadata": {},
            "outputs": [],
            "source": (dedent(text).strip() + "\n").splitlines(keepends=True),
        }
    )


def build_notebook() -> dict:
    cells: list[dict] = []

    add_markdown(
        cells,
        """
        # Interval Sampling Cross-Rotation 8 Samples Processed ROI

        This notebook:
        1. scans all 8 sample folders under `interval sampled/`
        2. interval-samples each sample/classification to a maximum of `200`
        3. creates `8` cross-rotation folds
        4. rewrites each split CSV `file_destination` to the matching file under `processed_hsv_lab_threshold_roi_224/`

        Default outputs:
        - sampled CSVs: `csv_outputs_interval_sampled_v5/`
        - split CSVs: `generated_splits/cross_rotation_interval200_8samples_processed_roi/`
        """,
    )

    add_code(
        cells,
        """
        from pathlib import Path

        import pandas as pd
        from IPython.display import Image, display

        from sample_interval_cross_rotation_pipeline import (
            discover_sample_dirs,
            run_pipeline,
            sample_label_counts,
            sample_total_counts,
        )

        PROJECT_ROOT = Path.cwd()
        SAMPLE_ROOT = PROJECT_ROOT / "interval sampled"
        SAMPLED_OUTPUT_DIR = PROJECT_ROOT / "csv_outputs_interval_sampled_v5"
        SPLIT_OUTPUT_DIR = PROJECT_ROOT / "generated_splits" / "cross_rotation_interval200_8samples_processed_roi"
        PROCESSING_SUMMARY_PATH = PROJECT_ROOT / "processed_hsv_lab_threshold_roi_224" / "processing_summary.csv"
        PROCESSED_ROOT = PROJECT_ROOT / "processed_hsv_lab_threshold_roi_224"
        TARGET_COUNT_PER_CLASS = 200
        SEED = 42

        print("Project root:", PROJECT_ROOT)
        print("Sample root:", SAMPLE_ROOT)
        print("Sampled output dir:", SAMPLED_OUTPUT_DIR)
        print("Split output dir:", SPLIT_OUTPUT_DIR)
        print("Processing summary:", PROCESSING_SUMMARY_PATH)
        print("Processed root:", PROCESSED_ROOT)
        print("Target count per class:", TARGET_COUNT_PER_CLASS)
        """,
    )

    add_code(
        cells,
        """
        sample_infos = discover_sample_dirs(SAMPLE_ROOT)
        sample_info_df = pd.DataFrame(
            [
                {
                    "sample_number": info.sample_number,
                    "meat_part": info.meat_part,
                    "pork_cut": info.pork_cut,
                    "sample_id": info.sample_id,
                    "folder_path": str(info.folder_path),
                }
                for info in sample_infos
            ]
        )
        display(sample_info_df)
        print("Total discovered samples:", len(sample_info_df))
        """,
    )

    add_code(
        cells,
        """
        outputs = run_pipeline(
            project_root=PROJECT_ROOT,
            sample_root=SAMPLE_ROOT,
            sampled_output_dir=SAMPLED_OUTPUT_DIR,
            split_output_dir=SPLIT_OUTPUT_DIR,
            target_count_per_class=TARGET_COUNT_PER_CLASS,
            seed=SEED,
        )

        raw_df = outputs["raw_df"]
        sampled_df = outputs["sampled_df"]
        sampling_summary_df = outputs["sampling_summary_df"]
        cross_summary_df = outputs["cross_summary_df"]
        leakage_df = outputs["leakage_df"]

        print("Raw rows:", len(raw_df))
        print("Sampled rows:", len(sampled_df))
        print("Cross-rotation folds:", len(cross_summary_df))
        """,
    )

    add_code(
        cells,
        """
        print("Original counts per sample and classification")
        display(sample_label_counts(raw_df))

        print("Sampled counts per sample and classification")
        display(sample_label_counts(sampled_df))

        print("Original total counts per sample")
        display(sample_total_counts(raw_df))

        print("Sampled total counts per sample")
        display(sample_total_counts(sampled_df))
        """,
    )

    add_code(
        cells,
        """
        def remap_split_destinations_to_processed(split_dir: Path, processing_summary_path: Path) -> tuple[int, int]:
            summary_df = pd.read_csv(processing_summary_path, dtype=str).fillna("")
            required_cols = {"file_destination", "processed_output_file"}
            missing = sorted(required_cols - set(summary_df.columns))
            if missing:
                raise ValueError(f"Processing summary is missing required columns: {missing}")

            path_map = dict(zip(summary_df["file_destination"], summary_df["processed_output_file"]))
            updated_files = 0
            updated_rows = 0

            for csv_path in sorted(split_dir.glob("*.csv")):
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

            return updated_files, updated_rows


        updated_files, updated_rows = remap_split_destinations_to_processed(
            split_dir=SPLIT_OUTPUT_DIR,
            processing_summary_path=PROCESSING_SUMMARY_PATH,
        )
        print("Updated files:", updated_files)
        print("Updated rows:", updated_rows)
        """,
    )

    add_code(
        cells,
        """
        verification_rows = []
        all_ok = True

        for csv_path in sorted(SPLIT_OUTPUT_DIR.glob("fold*.csv")):
            df = pd.read_csv(csv_path, dtype=str).fillna("")
            points_to_processed = df["file_destination"].str.contains(str(PROCESSED_ROOT), regex=False)
            ok = bool(points_to_processed.all())
            all_ok = all_ok and ok
            verification_rows.append(
                {
                    "csv_file": csv_path.name,
                    "rows": len(df),
                    "all_file_destinations_in_processed_root": ok,
                    "non_processed_rows": int((~points_to_processed).sum()),
                }
            )

        verification_df = pd.DataFrame(verification_rows)
        display(verification_df)
        print("All fold split file destinations point to processed root?", all_ok)
        """,
    )

    add_code(
        cells,
        """
        print("Sampling summary")
        display(sampling_summary_df)

        print("Cross-rotation summary")
        display(cross_summary_df)

        print("Leakage check")
        display(leakage_df)
        """,
    )

    add_code(
        cells,
        """
        print("Saved plot files")
        for plot_name, plot_path in outputs["plot_paths"].items():
            print(f"{plot_name}: {plot_path}")
            display(Image(filename=str(plot_path)))
        """,
    )

    add_code(
        cells,
        """
        print("Example processed split rows")
        example_df = pd.read_csv(SPLIT_OUTPUT_DIR / "fold1_train.csv", dtype=str).fillna("")
        display(example_df[["sample_number", "label", "image_file_name", "file_destination"]].head(20))
        """,
    )

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    notebook = build_notebook()
    output_path = Path("interval_sampling_cross_rotation_8samples_processed_roi.ipynb")
    output_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(f"Wrote notebook to {output_path.resolve()}")


if __name__ == "__main__":
    main()
