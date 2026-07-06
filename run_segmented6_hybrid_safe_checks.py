#!/usr/bin/env python3
from mobilenetv3small_segmented6_hybrid_lib import (
    EXTENSION_FEATURES_ROOT,
    EXTENSION_OUTPUT_ROOT,
    SEGMENTED6_QUALITY_SUMMARY_PATH,
    SEGMENTED_SPLITS_ROOT,
    build_summary_interpretation_text,
    build_dataset_quality_summary,
    ensure_output_dirs,
    load_prediction_csvs_for_metrics,
    print_library_status,
    print_quality_tables,
    regenerate_transition_metrics_and_graphs,
    run_feature_extraction_smoke_test,
    save_sample_visualization,
    validate_metadata_mapping,
    load_all_cross_rotation_splits,
    load_existing_metrics,
)


def main() -> None:
    ensure_output_dirs()
    print_library_status()

    split_dfs = load_all_cross_rotation_splits()
    metadata_validation_df = validate_metadata_mapping(split_dfs)
    print("\nMetadata mapping validation")
    print(metadata_validation_df.to_string(index=False))

    quality_bundle = build_dataset_quality_summary(split_dfs)
    print()
    print_quality_tables(quality_bundle)

    sample_image_figure_path = save_sample_visualization(split_dfs)
    print()
    print("Saved sample image figure:", sample_image_figure_path)

    smoke_features_df = run_feature_extraction_smoke_test(split_dfs)
    print()
    print("Feature extraction smoke test")
    print(smoke_features_df.to_string(index=False))

    print()
    print("Segmented split root:", SEGMENTED_SPLITS_ROOT)
    print("Output root:", EXTENSION_OUTPUT_ROOT)
    print("Dataset quality summary CSV:", SEGMENTED6_QUALITY_SUMMARY_PATH)
    print("Feature smoke test CSV:", EXTENSION_FEATURES_ROOT / "example_3_segmented_roi_color_texture_features.csv")

    print()
    regenerate_transition_metrics_and_graphs()

    existing_metrics_df = load_existing_metrics()
    if not existing_metrics_df.empty:
        prediction_df = load_prediction_csvs_for_metrics(existing_metrics_df)
        print()
        print(build_summary_interpretation_text(existing_metrics_df, prediction_df))


if __name__ == "__main__":
    main()
