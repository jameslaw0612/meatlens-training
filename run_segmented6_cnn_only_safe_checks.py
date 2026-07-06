#!/usr/bin/env python3
import inspect

from mobilenetv3small_segmented6_cnn_only_lib import (
    SEGMENTED6_QUALITY_SUMMARY_PATH,
    SEGMENTED6_COMPARISON_PATH,
    SEGMENTED_SPLITS_ROOT,
    build_dataset_quality_summary,
    ensure_output_dirs,
    load_all_cross_rotation_splits,
    print_library_status,
    print_quality_tables,
    save_sample_visualization,
    train_segmented6_cnn_only_model,
    validate_metadata_mapping,
)


def main() -> None:
    ensure_output_dirs()
    print_library_status()

    print("\n[1] Split loading and path resolution")
    split_dfs = load_all_cross_rotation_splits()
    for split_key, df in sorted(split_dfs.items()):
        resolved_count = int(df["image_path_resolved"].notna().sum())
        print(f"{split_key}: rows={len(df)}, resolved_paths={resolved_count}")

    print("\n[2] Metadata mapping validation")
    metadata_validation_df = validate_metadata_mapping(split_dfs)
    print(metadata_validation_df.to_string(index=False))

    print("\n[3] Dataset quality summary")
    quality_bundle = build_dataset_quality_summary(split_dfs)
    print_quality_tables(quality_bundle)

    print("\n[4] Sample image visualization")
    sample_image_figure_path = save_sample_visualization(split_dfs)
    print("Saved sample image figure:", sample_image_figure_path)

    print("\n[5] Function source verification")
    source = inspect.getsource(train_segmented6_cnn_only_model)
    print("Contains ModelCheckpoint?", "ModelCheckpoint" in source)
    print("Contains load_weights?", "load_weights" in source)
    print("Contains .weights.h5?", ".weights.h5" in source)
    print("Contains feature_input?", "feature_input" in source)
    print("Contains Concatenate?", "Concatenate" in source)

    print("\nPaths")
    print("Segmented split root:", SEGMENTED_SPLITS_ROOT)
    print("Dataset quality summary CSV:", SEGMENTED6_QUALITY_SUMMARY_PATH)
    print("Comparison CSV path:", SEGMENTED6_COMPARISON_PATH)


if __name__ == "__main__":
    main()
