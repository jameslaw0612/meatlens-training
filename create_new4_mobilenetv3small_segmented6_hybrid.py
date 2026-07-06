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
        # MeatLens MobileNetV3Small Segmented 6-Fold Hybrid Notebook

        This notebook prepares the new MeatLens segmented ROI experiment:

        - MobileNetV3Small only
        - 6-fold cross-rotation only
        - already-preprocessed HSV/LAB segmented `224x224` ROI images
        - hybrid CNN + color/texture features
        - new output root only: `training_outputs/mobilenetv3small_segmented6_hybrid/`

        Important preprocessing rule:
        - training does **not** apply `center_square_resize_224`
        - the segmented ROI images are loaded directly
        - resizing happens only as a safety fallback when an image is not already `224x224`
        """,
    )

    add_md(
        cells,
        """
        Feature note:

        The features are extracted from the segmented ROI image, including the neutral/black/gray background introduced by segmentation.
        """,
    )

    add_md(
        cells,
        """
        Transition-aware evaluation note:

        Strict accuracy and macro F1 remain the primary evaluation metrics. Transition-aware metrics are added only as secondary analysis because pork freshness changes gradually and borderline images may visually fall between adjacent classes.
        """,
    )

    add_code(
        cells,
        """
        # ============================================================
        # Load the segmented6 hybrid library from the .py source file
        # This avoids stale embedded notebook code and ensures the
        # latest library version is what the kernel executes.
        # ============================================================
        """,
    )

    add_code(
        cells,
        """
        import importlib
        import inspect

        import pandas as pd
        from IPython.display import Image as DisplayImage, display

        import mobilenetv3small_segmented6_hybrid_lib as segmented6_lib

        segmented6_lib = importlib.reload(segmented6_lib)

        print("Library source:", segmented6_lib.__file__)
        print(
            "Function first lines:",
            {
                "train_segmented6_hybrid_model": segmented6_lib.train_segmented6_hybrid_model.__code__.co_firstlineno,
                "run_single_segmented6_hybrid_experiment": segmented6_lib.run_single_segmented6_hybrid_experiment.__code__.co_firstlineno,
            },
        )
        print(
            "Function source files:",
            {
                "train_segmented6_hybrid_model": inspect.getsourcefile(segmented6_lib.train_segmented6_hybrid_model),
                "run_single_segmented6_hybrid_experiment": inspect.getsourcefile(segmented6_lib.run_single_segmented6_hybrid_experiment),
            },
        )

        sanity_optimizer = segmented6_lib.make_optimizer(1e-3)
        print("Sanity optimizer type:", type(sanity_optimizer))
        print("Sanity optimizer module:", type(sanity_optimizer).__module__)

        if inspect.getsourcefile(segmented6_lib.run_single_segmented6_hybrid_experiment) != segmented6_lib.__file__:
            raise RuntimeError("Notebook is not using the module-backed experiment function.")

        if "legacy" not in type(sanity_optimizer).__module__:
            raise RuntimeError("Notebook did not load the DirectML-safe legacy Adam optimizer.")

        print("Notebook sanity check passed.")

        segmented6_lib.ensure_output_dirs()
        segmented6_lib.print_library_status()
        """,
    )

    add_code(
        cells,
        """
        split_dfs = segmented6_lib.load_all_cross_rotation_splits()
        metadata_validation_df = segmented6_lib.validate_metadata_mapping(split_dfs)
        metadata_validation_df
        """,
    )

    add_code(
        cells,
        """
        quality_bundle = segmented6_lib.build_dataset_quality_summary(split_dfs)
        segmented6_lib.print_quality_tables(quality_bundle)
        quality_bundle["summary_df"].head(20)
        """,
    )

    add_code(
        cells,
        """
        sample_image_figure_path = segmented6_lib.save_sample_visualization(split_dfs)
        print("Saved sample image figure:", sample_image_figure_path)
        display(DisplayImage(filename=str(sample_image_figure_path)))
        """,
    )

    add_code(
        cells,
        """
        smoke_features_df = segmented6_lib.run_feature_extraction_smoke_test(split_dfs)
        smoke_features_df
        """,
    )

    add_code(
        cells,
        """
        print("Dataset quality summary CSV:", segmented6_lib.SEGMENTED6_QUALITY_SUMMARY_PATH)
        print("Feature smoke test CSV:", segmented6_lib.EXTENSION_FEATURES_ROOT / "example_3_segmented_roi_color_texture_features.csv")
        print("Segmented 6-fold split root:", segmented6_lib.SEGMENTED_SPLITS_ROOT)
        print("Output root:", segmented6_lib.EXTENSION_OUTPUT_ROOT)
        """,
    )

    add_code(
        cells,
        """
        if segmented6_lib.RUN_EXTENSION_TRAINING:
            all_results = segmented6_lib.run_segmented6_hybrid_training()
        else:
            print("Segmented 6-fold hybrid training functions are ready. Set RUN_EXTENSION_TRAINING=True to train.")
        """,
    )

    add_code(
        cells,
        """
        regenerate_bundle = segmented6_lib.regenerate_transition_metrics_and_graphs()
        """,
    )

    add_code(
        cells,
        """
        if segmented6_lib.SEGMENTED6_SEED_METRICS_PATH.exists():
            seed_metrics_df = pd.read_csv(segmented6_lib.SEGMENTED6_SEED_METRICS_PATH)
            prediction_df = segmented6_lib.load_prediction_csvs_for_metrics(seed_metrics_df)
            print(segmented6_lib.build_summary_interpretation_text(seed_metrics_df, prediction_df))
        else:
            print("No segmented6 hybrid training results found yet.")
        """,
    )

    add_code(
        cells,
        """
        # debug_result = segmented6_lib.run_single_segmented6_hybrid_experiment(
        #     fold_name="fold1",
        #     seed=42,
        # )
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
    latest_path = Path("new4_mobilenetv3small_segmented6_hybrid_latest.ipynb")
    output_path = Path("new4_mobilenetv3small_segmented6_hybrid.ipynb")

    if latest_path.exists():
        output_path.write_text(latest_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Copied latest notebook to {output_path.resolve()}")
        return

    notebook = build_notebook()
    output_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(f"Wrote notebook to {output_path.resolve()}")


if __name__ == "__main__":
    main()
