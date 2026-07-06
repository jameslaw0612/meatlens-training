#!/usr/bin/env python3
from __future__ import annotations

import importlib
import inspect

import mobilenetv3small_8fold_processed_roi_cnn_only_lib as segmented6_lib


def main() -> None:
    global segmented6_lib
    segmented6_lib = importlib.reload(segmented6_lib)

    print("Library source:", segmented6_lib.__file__)
    segmented6_lib.ensure_output_dirs()
    segmented6_lib.print_library_status()
    print("USE_TRAINING_AUGMENTATION =", segmented6_lib.USE_TRAINING_AUGMENTATION)

    split_dfs = segmented6_lib.load_all_cross_rotation_splits()
    print("Loaded split keys:", sorted(split_dfs.keys()))

    metadata_validation_df = segmented6_lib.validate_metadata_mapping(split_dfs)
    print(metadata_validation_df.to_string(index=False))

    quality_bundle = segmented6_lib.build_dataset_quality_summary(split_dfs)
    segmented6_lib.print_quality_tables(quality_bundle)

    shape_audit = segmented6_lib.audit_image_shapes(split_dfs)
    print(shape_audit)

    sample_image_figure_path = segmented6_lib.save_sample_visualization(split_dfs)
    print("Saved sample image figure:", sample_image_figure_path)

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


if __name__ == "__main__":
    main()
