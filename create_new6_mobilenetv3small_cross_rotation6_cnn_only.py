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
        # MeatLens MobileNetV3Small Cross-Rotation 6-Fold CNN-Only Notebook

        This notebook replicates the `new5` CNN-only setup, but uses the split CSVs from `generated_splits/cross_rotation_6fold/`.

        - MobileNetV3Small only
        - 6-fold cross-rotation only
        - split source: `generated_splits/cross_rotation_6fold/`
        - current `file_destination` values point into the `interval sampled` image tree
        - non-square ROI images are center-cropped to a square, then resized to `224x224`
        - no handcrafted RGB/HSV/LAB/GLCM feature branch
        - new output root only: `training_outputs/mobilenetv3small_cross_rotation6_cnn_only_centercrop/`
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
        import importlib
        import inspect

        import pandas as pd
        from IPython.display import Image as DisplayImage, display

        import mobilenetv3small_cross_rotation6_cnn_only_lib as segmented6_lib

        segmented6_lib = importlib.reload(segmented6_lib)

        print("Library source:", segmented6_lib.__file__)
        print(
            "Function first lines:",
            {
                "train_segmented6_cnn_only_model": segmented6_lib.train_segmented6_cnn_only_model.__code__.co_firstlineno,
                "run_single_segmented6_cnn_only_experiment": segmented6_lib.run_single_segmented6_cnn_only_experiment.__code__.co_firstlineno,
            },
        )
        print(
            "Function source files:",
            {
                "train_segmented6_cnn_only_model": inspect.getsourcefile(segmented6_lib.train_segmented6_cnn_only_model),
                "run_single_segmented6_cnn_only_experiment": inspect.getsourcefile(segmented6_lib.run_single_segmented6_cnn_only_experiment),
            },
        )

        sanity_optimizer = segmented6_lib.make_optimizer(1e-3)
        print("Sanity optimizer type:", type(sanity_optimizer))
        print("Sanity optimizer module:", type(sanity_optimizer).__module__)

        if inspect.getsourcefile(segmented6_lib.run_single_segmented6_cnn_only_experiment) != segmented6_lib.__file__:
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
        print("Loaded split keys:", sorted(split_dfs.keys()))
        resolved_counts = {
            split_key: int(df["image_path_resolved"].notna().sum())
            for split_key, df in split_dfs.items()
        }
        resolved_counts
        """,
    )

    add_code(
        cells,
        """
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
        print("Dataset quality summary CSV:", segmented6_lib.SEGMENTED6_QUALITY_SUMMARY_PATH)
        print("Cross-rotation 6-fold split root:", segmented6_lib.SEGMENTED_SPLITS_ROOT)
        print("Output root:", segmented6_lib.EXTENSION_OUTPUT_ROOT)
        print("Comparison CSV path:", segmented6_lib.SEGMENTED6_COMPARISON_PATH)
        """,
    )

    add_code(
        cells,
        """
        import inspect

        source = inspect.getsource(segmented6_lib.train_segmented6_cnn_only_model)
        print("Contains ModelCheckpoint?", "ModelCheckpoint" in source)
        print("Contains load_weights?", "load_weights" in source)
        print("Contains .weights.h5?", ".weights.h5" in source)
        print("Contains feature_input?", "feature_input" in source)
        print("Contains Concatenate?", "Concatenate" in source)
        """,
    )

    add_md(
        cells,
        """
        ## Comparison Section

        This section compares the completed hybrid metrics CSV against the CNN-only metrics CSV after CNN-only training has finished.
        """,
    )

    add_code(
        cells,
        """
        comparison_bundle = segmented6_lib.create_hybrid_vs_cnn_only_comparison()
        if comparison_bundle is not None:
            comparison_bundle["comparison_df"]
        """,
    )

    add_md(
        cells,
        """
        ## Manual Full Training Cell
        """,
    )

    add_code(
        cells,
        """
        MANUAL_CONFIRM_RUN_CNN_ONLY_TRAINING = False

        if MANUAL_CONFIRM_RUN_CNN_ONLY_TRAINING:
            RUN_EXTENSION_TRAINING = True
            segmented6_lib.RUN_EXTENSION_TRAINING = RUN_EXTENSION_TRAINING
            all_results = segmented6_lib.run_segmented6_cnn_only_training()
        else:
            print("CNN-only cross_rotation_6fold training is ready but not started.")
            print("Set MANUAL_CONFIRM_RUN_CNN_ONLY_TRAINING = True to train.")
        """,
    )

    add_code(
        cells,
        """
        # debug_result = segmented6_lib.run_single_segmented6_cnn_only_experiment(
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
    output_path = Path("new6_mobilenetv3small_cross_rotation6_cnn_only.ipynb")
    notebook = build_notebook()
    output_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(f"Wrote notebook to {output_path.resolve()}")


if __name__ == "__main__":
    main()
