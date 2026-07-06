#!/usr/bin/env python3
import json
import uuid
from pathlib import Path
from textwrap import dedent


def add_md(cells: list[dict], text: str) -> None:
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

    add_md(
        cells,
        """
        # MeatLens 8-Fold Processed ROI Transition-Aware Evaluation

        This notebook performs post-training transition-aware evaluation using the existing
        `processed_roi8_cnn_only_*_predictions.csv` files. It does not rerun training.

        Transition-aware accuracy is a secondary analysis only. Official model evaluation remains
        strict 3-class accuracy, macro F1, and the 3x3 confusion matrix.
        """,
    )

    add_code(
        cells,
        """
        import importlib
        from IPython.display import Image as DisplayImage, display

        import pandas as pd

        import processed_roi8_transition_aware_eval_lib as transition_lib

        transition_lib = importlib.reload(transition_lib)

        print("Library source:", transition_lib.__file__)
        print("Seed metrics path:", transition_lib.SEED_METRICS_PATH)
        print("All sampled images path:", transition_lib.ALL_SAMPLED_IMAGES_PATH)
        print("Output root:", transition_lib.OUTPUT_ROOT)
        """,
    )

    add_code(
        cells,
        """
        seed_metrics_df = transition_lib.load_seed_metrics_df()
        print("Seed metrics rows:", len(seed_metrics_df))
        seed_metrics_df[["fold_name", "seed", "accuracy", "macro_f1", "predictions_path"]].head()
        """,
    )

    add_code(
        cells,
        """
        results = transition_lib.run_transition_aware_evaluation()
        print(results["secondary_note"])
        print("Overall transition-aware accuracy:", results["overall_transition_aware_accuracy"])
        """,
    )

    add_code(
        cells,
        """
        results["summary_df"]
        """,
    )

    add_code(
        cells,
        """
        results["zone_summary_df"]
        """,
    )

    add_code(
        cells,
        """
        results["confusion_bundle"]["cm_5x6"]
        """,
    )

    add_code(
        cells,
        """
        results["confusion_bundle"]["cm_5x5"]
        """,
    )

    add_code(
        cells,
        """
        results["confusion_bundle"]["cm_5x5_norm"]
        """,
    )

    add_code(
        cells,
        """
        results["annotated_df"][
            [
                "fold_name",
                "seed",
                "sample_id",
                "label",
                "image_file_name",
                "position_index",
                "position_ratio",
                "transition_zone",
                "accepted_predictions",
                "predicted_label",
                "is_transition_aware_correct",
            ]
        ].head(20)
        """,
    )

    add_code(
        cells,
        """
        display(DisplayImage(filename=results["graph_paths"]["transition_aware_accuracy_by_fold"]))
        display(DisplayImage(filename=results["graph_paths"]["transition_aware_accuracy_by_transition_zone"]))
        display(DisplayImage(filename=results["graph_paths"]["strict_vs_transition_aware_accuracy_by_fold"]))
        display(DisplayImage(filename=results["graph_paths"]["transition_zone_distribution"]))
        display(DisplayImage(filename=results["graph_paths"]["transition_zone_correctness_bar"]))
        display(DisplayImage(filename=results["confusion_paths"]["transition_zone_confusion_matrix_5x6_png"]))
        display(DisplayImage(filename=results["confusion_paths"]["transition_zone_confusion_matrix_5x5_png"]))
        display(DisplayImage(filename=results["confusion_paths"]["normalized_transition_zone_confusion_matrix_5x5_png"]))
        """,
    )

    add_md(
        cells,
        """
        This notebook keeps the official evaluation unchanged:

        - strict 3-class accuracy
        - macro precision / macro recall / macro F1
        - official 3x3 confusion matrix

        The 5 transition zones are only a secondary analysis layer that recognizes the ordered
        progression of pork freshness within each original class sequence.
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
    output_path = Path("new7_mobilenetv3small_8fold_processed_roi_transition_aware_eval.ipynb")
    notebook = build_notebook()
    output_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(f"Wrote notebook to {output_path.resolve()}")


if __name__ == "__main__":
    main()
