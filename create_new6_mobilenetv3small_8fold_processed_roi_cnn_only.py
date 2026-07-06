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
        # MeatLens MobileNetV3Small 8-Fold Processed ROI CNN-Only Notebook

        This notebook trains the clean MeatLens comparison model using:

        - MobileNetV3Small only
        - 8-fold cross-rotation
        - interval-sampled 8-sample splits
        - already processed HSV/LAB-threshold segmented ROI images
        - neutralized background
        - `224x224` RGB inputs
        - no handcrafted RGB/HSV/LAB/GLCM feature branch
        - dedicated output root: `training_outputs/mobilenetv3small_8fold_processed_roi_cnn_only/`
        """,
    )

    add_md(
        cells,
        """
        Strict accuracy and macro F1 remain the primary metrics. Top-2 accuracy, adjacent accuracy, severe error rate, and ordinal error are kept as secondary transition-aware metrics because pork freshness changes gradually.
        """,
    )

    add_code(
        cells,
        """
        import importlib
        import inspect

        import pandas as pd
        from IPython.display import Image as DisplayImage, display

        import mobilenetv3small_8fold_processed_roi_cnn_only_lib as segmented6_lib

        segmented6_lib = importlib.reload(segmented6_lib)

        print("Library source:", segmented6_lib.__file__)
        print(
            "Function first lines:",
            {
                "train_mobilenetv3small_8fold_cnn_only_model": segmented6_lib.train_mobilenetv3small_8fold_cnn_only_model.__code__.co_firstlineno,
                "run_single_8fold_cnn_only_experiment": segmented6_lib.run_single_8fold_cnn_only_experiment.__code__.co_firstlineno,
            },
        )
        print(
            "Function source files:",
            {
                "train_mobilenetv3small_8fold_cnn_only_model": inspect.getsourcefile(segmented6_lib.train_mobilenetv3small_8fold_cnn_only_model),
                "run_single_8fold_cnn_only_experiment": inspect.getsourcefile(segmented6_lib.run_single_8fold_cnn_only_experiment),
            },
        )

        sanity_optimizer = segmented6_lib.make_optimizer(1e-3)
        print("Sanity optimizer type:", type(sanity_optimizer))
        print("Sanity optimizer module:", type(sanity_optimizer).__module__)

        if inspect.getsourcefile(segmented6_lib.run_single_8fold_cnn_only_experiment) != segmented6_lib.__file__:
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
        print("USE_TRAINING_AUGMENTATION =", segmented6_lib.USE_TRAINING_AUGMENTATION)
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
        quality_bundle["summary_df"].head(24)
        """,
    )

    add_code(
        cells,
        """
        shape_audit = segmented6_lib.audit_image_shapes(split_dfs)
        shape_audit
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
        print("Seed metrics CSV:", segmented6_lib.PROCESSED_ROI8_SEED_METRICS_PATH)
        print("Split root:", segmented6_lib.SPLIT_ROOT)
        print("Output root:", segmented6_lib.EXTENSION_OUTPUT_ROOT)
        print("Comparison CSV path:", segmented6_lib.PROCESSED_ROI8_COMPARISON_PATH)
        """,
    )

    add_code(
        cells,
        """
        import inspect

        source = inspect.getsource(segmented6_lib.train_mobilenetv3small_8fold_cnn_only_model)

        print("Contains ModelCheckpoint?", "ModelCheckpoint" in source)
        print("Contains load_weights?", "load_weights" in source)
        print("Contains .weights.h5?", ".weights.h5" in source)
        print("Contains feature_input?", "feature_input" in source)
        print("Contains Concatenate?", "Concatenate" in source)
        print("Contains StandardScaler?", "StandardScaler" in source)
        print("Contains graycomatrix?", "graycomatrix" in source)
        print("Contains rgb/hsv/lab/glcm feature extraction?", any(word in source for word in ["glcm", "lab_", "hsv_", "rgb_"]))

        print("EXTENSION_FOLDS length:", len(segmented6_lib.EXTENSION_FOLDS))
        print("SPLIT_ROOT contains expected folder?", "cross_rotation_interval200_8samples_processed_roi" in str(segmented6_lib.SPLIT_ROOT))
        print("IMAGE_CROP_MODE =", segmented6_lib.IMAGE_CROP_MODE)
        print("MODEL_INPUT_MODE =", segmented6_lib.MODEL_INPUT_MODE)
        """,
    )

    add_md(
        cells,
        """
        ## Optional Comparison

        If older result CSVs exist, this section compares:

        - segmented 6-fold hybrid
        - segmented 6-fold CNN-only
        - current 8-fold processed ROI CNN-only
        """,
    )

    add_code(
        cells,
        """
        comparison_bundle = segmented6_lib.create_processed_roi8_vs_previous_models_comparison()
        if comparison_bundle is not None:
            comparison_bundle["comparison_df"]
        """,
    )

    add_code(
        cells,
        """
        # debug_result = segmented6_lib.run_single_8fold_cnn_only_experiment(
        #     fold_name="fold1",
        #     seed=42,
        # )
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
        MANUAL_CONFIRM_RUN_FULL_TRAINING = False

        if MANUAL_CONFIRM_RUN_FULL_TRAINING:
            RUN_EXTENSION_TRAINING = True
            segmented6_lib.RUN_EXTENSION_TRAINING = RUN_EXTENSION_TRAINING
            all_results = segmented6_lib.run_8fold_cnn_only_training()
        else:
            print("8-fold processed ROI CNN-only training is ready but not started.")
            print("Set MANUAL_CONFIRM_RUN_FULL_TRAINING = True to train.")
        """,
    )

    add_md(
        cells,
        """
        ## Manual Summary Regeneration Cell
        """,
    )

    add_code(
        cells,
        """
        MANUAL_CONFIRM_REGENERATE_SUMMARIES = False

        if MANUAL_CONFIRM_REGENERATE_SUMMARIES:
            regenerate_bundle = segmented6_lib.regenerate_metrics_summaries_and_graphs()
        else:
            print("Summary regeneration is ready but not started.")
        """,
    )

    add_md(
        cells,
        """
        ## Completion Check
        """,
    )

    add_code(
        cells,
        """
        expected_runs = [(fold, seed) for fold in segmented6_lib.EXTENSION_FOLDS for seed in segmented6_lib.EXTENSION_RUN_SEEDS]
        print("Expected run count:", len(expected_runs))

        completed_runs = set()
        if segmented6_lib.PROCESSED_ROI8_SEED_METRICS_PATH.exists():
            seed_metrics_df = pd.read_csv(segmented6_lib.PROCESSED_ROI8_SEED_METRICS_PATH)
            completed_runs = set(zip(seed_metrics_df["fold_name"].astype(str), seed_metrics_df["seed"].astype(int)))
            print("Completed run count:", len(completed_runs))
        else:
            print("Completed run count: 0")

        missing_runs = [run for run in expected_runs if run not in completed_runs]
        print("Missing runs:", missing_runs)

        if segmented6_lib.PROCESSED_ROI8_FAILED_RUNS_PATH.exists():
            failed_runs_df = pd.read_csv(segmented6_lib.PROCESSED_ROI8_FAILED_RUNS_PATH)
            print("Failed runs:")
            display(failed_runs_df)
        else:
            print("Failed runs: none logged")
        """,
    )

    add_md(
        cells,
        """
        This notebook trains a CNN-only MobileNetV3Small model using the 8-sample processed segmented ROI dataset. The input images are already segmented and resized to 224×224, so the notebook does not perform segmentation or handcrafted feature extraction during training. Strict accuracy and macro F1 are the primary metrics, while top-2 accuracy, adjacent accuracy, severe error rate, and ordinal error are reported as secondary transition-aware metrics because pork freshness is a gradual process. This model is intended for comparison against earlier hybrid and segmented CNN-only experiments.
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
    output_path = Path("new6_mobilenetv3small_8fold_processed_roi_cnn_only.ipynb")
    notebook = build_notebook()
    output_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(f"Wrote notebook to {output_path.resolve()}")


if __name__ == "__main__":
    main()
