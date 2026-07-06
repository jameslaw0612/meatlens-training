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
        # Interval Sampling and Cross-Rotation for Samples 1 to 8

        This notebook does four things:
        1. scans the pork sample folders under `interval sampled/`
        2. visualizes the original image counts per sample and classification
        3. interval-samples each sample/classification to the closest possible count up to `200`
        4. creates new cross-rotation CSV splits using the sampled rows

        Default outputs:
        - sampled CSVs: `csv_outputs_interval_sampled_v4/`
        - plots: `csv_outputs_interval_sampled_v4/plots/`
        - cross-rotation CSVs: `generated_splits/cross_rotation_interval200_8samples/`
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
        SAMPLED_OUTPUT_DIR = PROJECT_ROOT / "csv_outputs_interval_sampled_v4"
        SPLIT_OUTPUT_DIR = PROJECT_ROOT / "generated_splits" / "cross_rotation_interval200_8samples"
        TARGET_COUNT_PER_CLASS = 200
        SEED = 42

        print("Project root:", PROJECT_ROOT)
        print("Sample root:", SAMPLE_ROOT)
        print("Sampled output dir:", SAMPLED_OUTPUT_DIR)
        print("Split output dir:", SPLIT_OUTPUT_DIR)
        print("Target count per class:", TARGET_COUNT_PER_CLASS)
        """,
    )

    add_code(
        cells,
        """
        sample_infos = discover_sample_dirs(SAMPLE_ROOT)
        pd.DataFrame(
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
        print("Sampled CSV files")
        for key, path in outputs["sampled_csv_paths"].items():
            print(f"{key}: {path}")
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
    output_path = Path("interval_sampling_cross_rotation_6samples.ipynb")
    output_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(f"Wrote notebook to {output_path.resolve()}")


if __name__ == "__main__":
    main()
