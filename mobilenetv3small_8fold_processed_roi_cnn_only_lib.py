#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
import math
import random
import shutil
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

import mobilenetv3small_segmented6_hybrid_lib as base

tf = base.tf
keras_callbacks = base.keras_callbacks
layers = base.layers
models = base.models
MobileNetV3Small = base.MobileNetV3Small
preprocess_mobilenetv3 = base.preprocess_mobilenetv3

accuracy_score = base.accuracy_score
classification_report = base.classification_report
confusion_matrix = base.confusion_matrix
precision_recall_fscore_support = base.precision_recall_fscore_support
compute_class_weight = base.compute_class_weight

TF_AVAILABLE = base.TF_AVAILABLE
SKLEARN_AVAILABLE = base.SKLEARN_AVAILABLE
SKIMAGE_AVAILABLE = base.SKIMAGE_AVAILABLE
CV2_AVAILABLE = base.CV2_AVAILABLE
JOBLIB_AVAILABLE = base.JOBLIB_AVAILABLE
SEABORN_AVAILABLE = base.SEABORN_AVAILABLE

cv2 = getattr(base, "cv2", None)
plt = base.plt

if TF_AVAILABLE:
    try:
        tf.config.optimizer.set_jit(False)
        print("[INFO] TensorFlow XLA/JIT disabled.")
    except Exception as exc:
        print(f"[WARN] Could not disable TensorFlow XLA/JIT: {exc}")

PROJECT_ROOT = Path.cwd()
TRAINING_OUTPUTS_ROOT = PROJECT_ROOT / "training_outputs"
SPLIT_ROOT = PROJECT_ROOT / "generated_splits" / "cross_rotation_interval200_8samples_processed_roi"
SEGMENTED_SPLITS_ROOT = SPLIT_ROOT
CROSS_ROTATION_ROOT = SPLIT_ROOT
PROCESSED_ROI_ROOT = PROJECT_ROOT / "processed_hsv_lab_threshold_roi_224"

EXTENSION_OUTPUT_ROOT = TRAINING_OUTPUTS_ROOT / "mobilenetv3small_8fold_processed_roi_cnn_only"
EXTENSION_FIGURES_ROOT = EXTENSION_OUTPUT_ROOT / "figures"
EXTENSION_MODELS_ROOT = EXTENSION_OUTPUT_ROOT / "models"
EXTENSION_PREDICTIONS_ROOT = EXTENSION_OUTPUT_ROOT / "predictions"
EXTENSION_GRADCAM_ROOT = EXTENSION_OUTPUT_ROOT / "gradcam"
EXTENSION_LOGS_ROOT = EXTENSION_OUTPUT_ROOT / "logs"

EXTENSION_BACKBONE = "MobileNetV3Small"
EXTENSION_SPLIT_MODE = "cross_rotation_interval200_8samples_processed_roi"
EXTENSION_FOLDS = ["fold1", "fold2", "fold3", "fold4", "fold5", "fold6", "fold7", "fold8"]
EXTENSION_RUN_SEEDS = [42, 123, 2026]

MODEL_INPUT_MODE = "cnn_only"
IMAGE_CROP_MODE = "preprocessed_hsv_lab_threshold_roi_224"
RUN_EXTENSION_TRAINING = False

TARGET_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 3
BATCH_SIZE = 32
EPOCHS_HEAD = 8
EPOCHS_FINE = 20
HEAD_LR = 5e-4
FINE_TUNE_LR = 1e-5
FINE_TUNE_FRACTION = 0.25
TOP_CONFIDENCE_BORDERLINE = 0.90
TOP_CONFIDENCE_LOW = 0.60
USE_TRAINING_AUGMENTATION = True
ALLOW_RESIZE_FALLBACK = False

LABEL_ORDER = ["fresh", "not fresh", "spoiled"]
LABEL_TO_INDEX = {
    "fresh": 0,
    "not fresh": 1,
    "spoiled": 2,
}
INDEX_TO_LABEL = {
    0: "fresh",
    1: "not fresh",
    2: "spoiled",
}

SEGMENTED6_SEED_METRICS_PATH = EXTENSION_OUTPUT_ROOT / "processed_roi8_cnn_only_seed_metrics.csv"
SEGMENTED6_FOLD_SUMMARY_PATH = EXTENSION_OUTPUT_ROOT / "processed_roi8_cnn_only_fold_summary.csv"
SEGMENTED6_SAMPLE_SUMMARY_PATH = EXTENSION_OUTPUT_ROOT / "processed_roi8_cnn_only_sample_summary.csv"
SEGMENTED6_CUT_SUMMARY_PATH = EXTENSION_OUTPUT_ROOT / "processed_roi8_cnn_only_cut_summary.csv"
SEGMENTED6_CAPTURE_SOURCE_SUMMARY_PATH = EXTENSION_OUTPUT_ROOT / "processed_roi8_cnn_only_capture_source_summary.csv"
SEGMENTED6_PHONE_GROUP_SUMMARY_PATH = EXTENSION_OUTPUT_ROOT / "processed_roi8_cnn_only_phone_group_summary.csv"
SEGMENTED6_PER_CLASS_SUMMARY_PATH = EXTENSION_OUTPUT_ROOT / "processed_roi8_cnn_only_per_class_summary.csv"
SEGMENTED6_PRED_DISTRIBUTION_PATH = EXTENSION_OUTPUT_ROOT / "processed_roi8_cnn_only_prediction_distribution.csv"
SEGMENTED6_SIZE_SPEED_PATH = EXTENSION_OUTPUT_ROOT / "processed_roi8_cnn_only_model_size_and_speed.csv"
SEGMENTED6_FAILED_RUNS_PATH = EXTENSION_OUTPUT_ROOT / "failed_processed_roi8_cnn_only_runs.csv"
SEGMENTED6_QUALITY_SUMMARY_PATH = EXTENSION_OUTPUT_ROOT / "processed_roi8_dataset_quality_summary.csv"
SEGMENTED6_COMPARISON_PATH = EXTENSION_OUTPUT_ROOT / "processed_roi8_vs_previous_models_comparison.csv"

HYBRID_SEED_METRICS_PATH = (
    TRAINING_OUTPUTS_ROOT
    / "mobilenetv3small_segmented6_hybrid"
    / "segmented6_hybrid_seed_metrics.csv"
)
SEGMENTED6_CNN_ONLY_SEED_METRICS_PATH = (
    TRAINING_OUTPUTS_ROOT
    / "mobilenetv3small_segmented6_cnn_only"
    / "segmented6_cnn_only_seed_metrics.csv"
)

PRIMARY_RUN_METRICS = list(base.PRIMARY_RUN_METRICS)
SECONDARY_RUN_METRICS = list(base.SECONDARY_RUN_METRICS)

SAMPLE_METADATA = {
    "pork_shoulder_sample_1": {
        "sample_number": 1,
        "pork_cut": "shoulder",
        "capture_source": "researcher_home",
        "phone_group": "old_phone",
    },
    "pork_shoulder_sample_2": {
        "sample_number": 2,
        "pork_cut": "shoulder",
        "capture_source": "partner_home",
        "phone_group": "old_phone",
    },
    "pork_belly_sample_3": {
        "sample_number": 3,
        "pork_cut": "belly",
        "capture_source": "researcher_home",
        "phone_group": "old_phone",
    },
    "pork_belly_sample_4": {
        "sample_number": 4,
        "pork_cut": "belly",
        "capture_source": "partner_home",
        "phone_group": "old_phone",
    },
    "pork_ham_sample_5": {
        "sample_number": 5,
        "pork_cut": "ham",
        "capture_source": "researcher_home",
        "phone_group": "new_phone",
    },
    "pork_ham_sample_6": {
        "sample_number": 6,
        "pork_cut": "ham",
        "capture_source": "partner_home",
        "phone_group": "new_phone",
    },
    "pork_unknown_sample_7": {
        "sample_number": 7,
        "pork_cut": "unknown",
        "capture_source": "unknown",
        "phone_group": "unknown",
    },
    "pork_unknown_sample_8": {
        "sample_number": 8,
        "pork_cut": "unknown",
        "capture_source": "unknown",
        "phone_group": "unknown",
    },
}
SAMPLE_NUMBER_TO_ID = {
    "1": "pork_shoulder_sample_1",
    "2": "pork_shoulder_sample_2",
    "3": "pork_belly_sample_3",
    "4": "pork_belly_sample_4",
    "5": "pork_ham_sample_5",
    "6": "pork_ham_sample_6",
    "7": "pork_unknown_sample_7",
    "8": "pork_unknown_sample_8",
}
IMAGE_PATH_CANDIDATES = [
    "file_destination",
    "image_path",
    "image_path_resolved",
    "path",
    "filename",
    "file_path",
    "image_file_name",
    "roi_file",
]
LOADER_STATS = {
    "already_224": 0,
    "non_224": 0,
    "resized": 0,
    "warning_count": 0,
}


def _base_patch_map() -> dict[str, object]:
    return {
        "PROJECT_ROOT": PROJECT_ROOT,
        "TRAINING_OUTPUTS_ROOT": TRAINING_OUTPUTS_ROOT,
        "SEGMENTED_SPLITS_ROOT": SEGMENTED_SPLITS_ROOT,
        "CROSS_ROTATION_ROOT": CROSS_ROTATION_ROOT,
        "EXTENSION_OUTPUT_ROOT": EXTENSION_OUTPUT_ROOT,
        "EXTENSION_FIGURES_ROOT": EXTENSION_FIGURES_ROOT,
        "EXTENSION_MODELS_ROOT": EXTENSION_MODELS_ROOT,
        "EXTENSION_GRADCAM_ROOT": EXTENSION_GRADCAM_ROOT,
        "EXTENSION_PREDICTIONS_ROOT": EXTENSION_PREDICTIONS_ROOT,
        "EXTENSION_BACKBONE": EXTENSION_BACKBONE,
        "EXTENSION_SPLIT_MODE": EXTENSION_SPLIT_MODE,
        "EXTENSION_FOLDS": EXTENSION_FOLDS,
        "EXTENSION_RUN_SEEDS": EXTENSION_RUN_SEEDS,
        "RUN_EXTENSION_TRAINING": RUN_EXTENSION_TRAINING,
        "TARGET_SIZE": TARGET_SIZE,
        "INPUT_SHAPE": INPUT_SHAPE,
        "NUM_CLASSES": NUM_CLASSES,
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS_HEAD": EPOCHS_HEAD,
        "EPOCHS_FINE": EPOCHS_FINE,
        "HEAD_LR": HEAD_LR,
        "FINE_TUNE_LR": FINE_TUNE_LR,
        "FINE_TUNE_FRACTION": FINE_TUNE_FRACTION,
        "MODEL_INPUT_MODE": MODEL_INPUT_MODE,
        "IMAGE_CROP_MODE": IMAGE_CROP_MODE,
        "LABEL_ORDER": LABEL_ORDER,
        "LABEL_TO_INDEX": LABEL_TO_INDEX,
        "INDEX_TO_LABEL": INDEX_TO_LABEL,
        "SEGMENTED6_SEED_METRICS_PATH": SEGMENTED6_SEED_METRICS_PATH,
        "SEGMENTED6_FOLD_SUMMARY_PATH": SEGMENTED6_FOLD_SUMMARY_PATH,
        "SEGMENTED6_SAMPLE_SUMMARY_PATH": SEGMENTED6_SAMPLE_SUMMARY_PATH,
        "SEGMENTED6_CUT_SUMMARY_PATH": SEGMENTED6_CUT_SUMMARY_PATH,
        "SEGMENTED6_CAPTURE_SOURCE_SUMMARY_PATH": SEGMENTED6_CAPTURE_SOURCE_SUMMARY_PATH,
        "SEGMENTED6_PHONE_GROUP_SUMMARY_PATH": SEGMENTED6_PHONE_GROUP_SUMMARY_PATH,
        "SEGMENTED6_PER_CLASS_SUMMARY_PATH": SEGMENTED6_PER_CLASS_SUMMARY_PATH,
        "SEGMENTED6_PRED_DISTRIBUTION_PATH": SEGMENTED6_PRED_DISTRIBUTION_PATH,
        "SEGMENTED6_SIZE_SPEED_PATH": SEGMENTED6_SIZE_SPEED_PATH,
        "SEGMENTED6_FAILED_RUNS_PATH": SEGMENTED6_FAILED_RUNS_PATH,
        "SEGMENTED6_QUALITY_SUMMARY_PATH": SEGMENTED6_QUALITY_SUMMARY_PATH,
    }


@contextmanager
def patched_base_config():
    patch_map = _base_patch_map()
    previous = {name: getattr(base, name) for name in patch_map}
    for name, value in patch_map.items():
        setattr(base, name, value)
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(base, name, value)


def ensure_output_dirs() -> None:
    for path_obj in [
        EXTENSION_OUTPUT_ROOT,
        EXTENSION_FIGURES_ROOT,
        EXTENSION_MODELS_ROOT,
        EXTENSION_PREDICTIONS_ROOT,
        EXTENSION_GRADCAM_ROOT,
    ]:
        path_obj.mkdir(parents=True, exist_ok=True)


def print_library_status() -> None:
    print("TF_AVAILABLE =", TF_AVAILABLE)
    print("SKLEARN_AVAILABLE =", SKLEARN_AVAILABLE)
    print("SKIMAGE_AVAILABLE =", SKIMAGE_AVAILABLE)
    print("CV2_AVAILABLE =", CV2_AVAILABLE)
    print("JOBLIB_AVAILABLE =", JOBLIB_AVAILABLE)
    print("SEABORN_AVAILABLE =", SEABORN_AVAILABLE)
    print("PROJECT_ROOT =", PROJECT_ROOT)
    print("SEGMENTED_SPLITS_ROOT =", SEGMENTED_SPLITS_ROOT)
    print("EXTENSION_OUTPUT_ROOT =", EXTENSION_OUTPUT_ROOT)
    print("IMAGE_CROP_MODE =", IMAGE_CROP_MODE)
    print("MODEL_INPUT_MODE =", MODEL_INPUT_MODE)
    print("EXTENSION_FOLDS =", EXTENSION_FOLDS)
    print("EXTENSION_RUN_SEEDS =", EXTENSION_RUN_SEEDS)


set_global_seed = base.set_global_seed
detect_image_path_column = base.detect_image_path_column
resolve_image_path = base.resolve_image_path
infer_sample_id = base.infer_sample_id
normalize_sample_id = base.normalize_sample_id
enrich_split_df = base.enrich_split_df
load_split_dataframe = base.load_split_dataframe
combined_splits_dataframe = base.combined_splits_dataframe
validate_metadata_mapping = base.validate_metadata_mapping
reset_loader_stats = base.reset_loader_stats
load_preprocessed_segmented_roi_224 = base.load_preprocessed_segmented_roi_224
count_table_to_rows = base.count_table_to_rows
run_split_integrity_checks = base.run_split_integrity_checks
classification_metrics = base.classification_metrics
build_transition_prediction_dataframe = base.build_transition_prediction_dataframe
compute_transition_metrics_from_prediction_df = base.compute_transition_metrics_from_prediction_df
make_optimizer = base.make_optimizer
freeze_batchnorm_layers = base.freeze_batchnorm_layers
unfreeze_top_backbone_fraction = base.unfreeze_top_backbone_fraction
compute_class_weights_from_labels = base.compute_class_weights_from_labels
save_confusion_matrix_figure = base.save_confusion_matrix_figure
try_convert_to_tflite = base.try_convert_to_tflite
load_prediction_dataframe = base.load_prediction_dataframe
regenerate_prediction_dataframe_if_needed = base.regenerate_prediction_dataframe_if_needed
coerce_prediction_dataframe_types = base.coerce_prediction_dataframe_types
aggregate_summary = base.aggregate_summary
build_prediction_distribution_df = base.build_prediction_distribution_df
save_bar_plot = base.save_bar_plot
save_histogram = base.save_histogram
save_grouped_histogram = base.save_grouped_histogram
save_boxplot = base.save_boxplot
save_scatter_plot = base.save_scatter_plot


def load_all_cross_rotation_splits() -> dict[str, pd.DataFrame]:
    with patched_base_config():
        return base.load_all_cross_rotation_splits()


def build_dataset_quality_summary(split_dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    with patched_base_config():
        return base.build_dataset_quality_summary(split_dfs)


def print_quality_tables(quality_bundle: dict[str, pd.DataFrame]) -> None:
    base.print_quality_tables(quality_bundle)


def save_sample_visualization(split_dfs: dict[str, pd.DataFrame]) -> Path:
    ensure_output_dirs()
    with patched_base_config():
        return base.save_sample_visualization(split_dfs)


def load_prediction_csvs_for_metrics(seed_metrics_df: pd.DataFrame) -> pd.DataFrame:
    with patched_base_config():
        return base.load_prediction_csvs_for_metrics(seed_metrics_df)


def save_segmented6_cnn_only_summary_outputs(
    seed_metrics_df: pd.DataFrame,
    prediction_df: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    if seed_metrics_df.empty:
        empty_df = pd.DataFrame()
        return {
            "fold_summary": empty_df,
            "sample_summary": empty_df,
            "cut_summary": empty_df,
            "capture_source_summary": empty_df,
            "phone_group_summary": empty_df,
            "per_class_summary": empty_df,
            "prediction_distribution_df": empty_df,
            "size_speed_df": empty_df,
        }
    with patched_base_config():
        return base.save_segmented6_hybrid_summary_outputs(seed_metrics_df, prediction_df=prediction_df)


def build_summary_interpretation_text(
    seed_metrics_df: pd.DataFrame,
    prediction_df: pd.DataFrame | None = None,
) -> str:
    if seed_metrics_df.empty:
        return "No segmented6 CNN-only training results found yet."
    return base.build_summary_interpretation_text(seed_metrics_df, prediction_df)


def build_image_array_from_paths(paths: list[str]) -> np.ndarray:
    arrays = []
    for path in paths:
        _, _, image_float = load_preprocessed_segmented_roi_224(path)
        arrays.append(preprocess_mobilenetv3(image_float.copy()))
    if not arrays:
        return np.empty((0, *INPUT_SHAPE), dtype=np.float32)
    return np.stack(arrays).astype(np.float32)


class ImageOnlySequence(tf.keras.utils.Sequence if TF_AVAILABLE else object):
    def __init__(
        self,
        image_paths: list[str],
        labels: np.ndarray | None = None,
        batch_size: int = BATCH_SIZE,
        shuffle: bool = False,
    ):
        self.image_paths = list(image_paths)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int32)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.indices = np.arange(len(self.image_paths))
        self.on_epoch_end()

    def __len__(self) -> int:
        if len(self.indices) == 0:
            return 0
        return int(math.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index: int):
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.indices))
        batch_indices = self.indices[start:end]
        batch_paths = [self.image_paths[idx] for idx in batch_indices]
        batch_images = build_image_array_from_paths(batch_paths)
        if self.labels is None:
            return batch_images
        batch_labels = self.labels[batch_indices]
        return batch_images, batch_labels

    def on_epoch_end(self):
        if self.shuffle and len(self.indices) > 0:
            np.random.shuffle(self.indices)


ValMacroF1Callback = base.ValMacroF1Callback


def compile_model_safely(model, optimizer, loss, metrics, jit_compile: bool = False):
    try:
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            jit_compile=jit_compile,
        )
    except TypeError:
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
        )


def assert_directml_safe_compilation(model) -> None:
    optimizer_type = type(model.optimizer)
    optimizer_module = optimizer_type.__module__
    print(f"[INFO] Compiled optimizer: {optimizer_module}.{optimizer_type.__name__}")
    if "optimizer_experimental" in optimizer_module:
        raise RuntimeError(
            "DirectML-unsafe optimizer detected in the compiled model. "
            "Restart the kernel and rerun the notebook import/reload cell so the latest "
            "mobilenetv3small_segmented6_cnn_only_lib.py code is used."
        )


def build_segmented6_cnn_only_model() -> tf.keras.Model:
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required for model building.")

    image_input = layers.Input(shape=INPUT_SHAPE, name="image_input")

    backbone = MobileNetV3Small(
        include_top=False,
        weights="imagenet",
        input_shape=INPUT_SHAPE,
    )
    backbone._name = "mobilenetv3small_backbone"
    backbone.trainable = False

    x = backbone(image_input, training=False)
    x = layers.GlobalAveragePooling2D(name="image_gap")(x)
    x = layers.Dropout(0.30, name="image_dropout_1")(x)
    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    x = layers.Dropout(0.30, name="image_dropout_2")(x)
    output = layers.Dense(NUM_CLASSES, activation="softmax", name="classification_head")(x)

    return models.Model(
        inputs=image_input,
        outputs=output,
        name="meatlens_segmented6_cnn_only",
    )


def prepare_image_only_bundle_for_fold(
    fold_name: str,
    split_dfs: dict[str, pd.DataFrame],
) -> dict[str, object]:
    train_df = split_dfs[f"{fold_name}_train"].copy()
    val_df = split_dfs[f"{fold_name}_val"].copy()
    test_df = split_dfs[f"{fold_name}_test"].copy()

    bundle = {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_image_paths": train_df["image_path_resolved"].astype(str).tolist(),
        "val_image_paths": val_df["image_path_resolved"].astype(str).tolist(),
        "test_image_paths": test_df["image_path_resolved"].astype(str).tolist(),
        "train_labels": train_df["label"].map(LABEL_TO_INDEX).astype(int).to_numpy(),
        "val_labels": val_df["label"].map(LABEL_TO_INDEX).astype(int).to_numpy(),
        "test_labels": test_df["label"].map(LABEL_TO_INDEX).astype(int).to_numpy(),
    }
    return bundle


def make_run_stem(fold_name: str, seed: int) -> str:
    return f"segmented6_cnn_only_{fold_name}_seed{seed}"


def load_existing_metrics() -> pd.DataFrame:
    if SEGMENTED6_SEED_METRICS_PATH.exists():
        return pd.read_csv(SEGMENTED6_SEED_METRICS_PATH)
    return pd.DataFrame()


def metrics_row_exists(metrics_df: pd.DataFrame, fold_name: str, seed: int) -> bool:
    if metrics_df.empty:
        return False
    mask = (
        metrics_df.get("fold_name", pd.Series(dtype=str)).astype(str).eq(fold_name)
        & metrics_df.get("seed", pd.Series(dtype=str)).astype(str).eq(str(seed))
        & metrics_df.get("model_input_mode", pd.Series(dtype=str)).astype(str).eq(MODEL_INPUT_MODE)
        & metrics_df.get("image_crop_mode", pd.Series(dtype=str)).astype(str).eq(IMAGE_CROP_MODE)
    )
    return bool(mask.any())


def should_skip_run(metrics_df: pd.DataFrame, fold_name: str, seed: int) -> bool:
    return metrics_row_exists(metrics_df, fold_name, seed)


def save_predictions_csv(run_stem: str, prediction_df: pd.DataFrame) -> Path:
    out_path = EXTENSION_PREDICTIONS_ROOT / f"{run_stem}_predictions.csv"
    prediction_df.to_csv(out_path, index=False)
    return out_path


def predict_probabilities_in_batches(
    model: tf.keras.Model,
    image_paths: list[str],
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    predict_sequence = ImageOnlySequence(
        image_paths=image_paths,
        labels=None,
        batch_size=batch_size,
        shuffle=False,
    )
    return model.predict(predict_sequence, verbose=0)


def measure_inference_speed(
    model: tf.keras.Model,
    test_image_paths: list[str],
    warmup: int = 3,
    repeats: int = 10,
) -> tuple[float, float]:
    times = []
    for _ in range(warmup):
        _ = predict_probabilities_in_batches(model, test_image_paths)
    for _ in range(repeats):
        start = time.perf_counter()
        _ = predict_probabilities_in_batches(model, test_image_paths)
        elapsed = time.perf_counter() - start
        times.append((elapsed * 1000.0) / max(len(test_image_paths), 1))
    return float(np.mean(times)), float(np.std(times))


def make_gradcam_heatmap(
    model: tf.keras.Model,
    image_batch: np.ndarray,
    class_index: int | None = None,
) -> np.ndarray:
    backbone = model.get_layer("mobilenetv3small_backbone")
    grad_model = tf.keras.models.Model(
        model.inputs,
        [backbone.output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_batch, training=False)
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.maximum(tf.reduce_max(heatmap), tf.keras.backend.epsilon())
    return heatmap.numpy()


def save_gradcam_preview(model: tf.keras.Model, row: pd.Series, out_path: Path) -> Path:
    original_np, image_uint8, image_float = load_preprocessed_segmented_roi_224(row["image_path_resolved"])
    image_input = preprocess_mobilenetv3(np.expand_dims(image_float, axis=0))
    heatmap = make_gradcam_heatmap(model, image_input)

    fig, axes = base.plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original_np)
    axes[0].set_title("Loaded image")
    axes[1].imshow(image_uint8)
    axes[1].set_title("Segmented ROI 224")
    axes[2].imshow(image_uint8)
    axes[2].imshow(heatmap, cmap="jet", alpha=0.4)
    axes[2].set_title("Grad-CAM")
    for ax in axes:
        ax.axis("off")
    base.plt.tight_layout()
    fig.savefig(out_path, dpi=220)
    base.plt.close(fig)
    return out_path


def train_segmented6_cnn_only_model(
    fold_name: str,
    seed: int,
    bundle: dict[str, object],
) -> tuple[tf.keras.Model, dict[str, object]]:
    _ = fold_name, seed

    class_weights = compute_class_weights_from_labels(bundle["train_labels"])
    model = build_segmented6_cnn_only_model()

    train_sequence = ImageOnlySequence(
        image_paths=bundle["train_image_paths"],
        labels=bundle["train_labels"],
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_sequence = ImageOnlySequence(
        image_paths=bundle["val_image_paths"],
        labels=bundle["val_labels"],
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    callbacks = [
        ValMacroF1Callback(
            val_source=val_sequence,
            val_labels=bundle["val_labels"],
        ),
        keras_callbacks.EarlyStopping(
            monitor="val_f1_macro",
            mode="max",
            patience=4,
            restore_best_weights=True,
        ),
        keras_callbacks.ReduceLROnPlateau(
            monitor="val_f1_macro",
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    compile_model_safely(
        model,
        optimizer=make_optimizer(HEAD_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=False,
    )
    assert_directml_safe_compilation(model)
    history_head = model.fit(
        train_sequence,
        validation_data=val_sequence,
        epochs=EPOCHS_HEAD,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )

    unfreeze_top_backbone_fraction(model, fraction=FINE_TUNE_FRACTION)
    compile_model_safely(
        model,
        optimizer=make_optimizer(FINE_TUNE_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=False,
    )
    assert_directml_safe_compilation(model)
    history_fine = model.fit(
        train_sequence,
        validation_data=val_sequence,
        epochs=EPOCHS_FINE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )

    return model, {
        "history_head": history_head.history,
        "history_fine": history_fine.history,
        "checkpoint_path": "",
        "class_weights_json": json.dumps(class_weights),
    }


def run_single_segmented6_cnn_only_experiment(
    fold_name: str = "fold1",
    seed: int = 42,
    split_dfs: dict[str, pd.DataFrame] | None = None,
) -> dict[str, object]:
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required for segmented CNN-only training.")
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is required for segmented CNN-only training.")

    ensure_output_dirs()
    set_global_seed(seed)
    if split_dfs is None:
        split_dfs = load_all_cross_rotation_splits()

    bundle = prepare_image_only_bundle_for_fold(fold_name, split_dfs)
    train_df = bundle["train_df"]
    val_df = bundle["val_df"]
    test_df = bundle["test_df"]
    run_stem = make_run_stem(fold_name, seed)

    print(f"[DATA] {fold_name} seed={seed}")
    print(f"  train_images count={len(bundle['train_image_paths'])} loaded_per_batch={BATCH_SIZE}")
    print(f"  val_images count={len(bundle['val_image_paths'])} loaded_per_batch={BATCH_SIZE}")
    print(f"  test_images count={len(bundle['test_image_paths'])} loaded_per_batch={BATCH_SIZE}")

    model, train_info = train_segmented6_cnn_only_model(
        fold_name=fold_name,
        seed=seed,
        bundle=bundle,
    )

    y_prob = predict_probabilities_in_batches(
        model,
        bundle["test_image_paths"],
        batch_size=BATCH_SIZE,
    )
    y_pred = y_prob.argmax(axis=1)
    y_true = bundle["test_labels"]
    metric_row = classification_metrics(y_true, y_pred)
    prediction_df = build_transition_prediction_dataframe(test_df, y_true, y_prob)
    transition_metric_row = compute_transition_metrics_from_prediction_df(prediction_df)

    predictions_path = save_predictions_csv(run_stem, prediction_df)

    model_path = EXTENSION_MODELS_ROOT / f"{run_stem}.h5"
    model.save(model_path)
    h5_size_mb = model_path.stat().st_size / (1024 * 1024)

    tflite_path = EXTENSION_MODELS_ROOT / f"{run_stem}.tflite"
    tflite_ok, tflite_size_mb = try_convert_to_tflite(model, tflite_path)
    if not tflite_ok and tflite_path.exists():
        tflite_path.unlink(missing_ok=True)

    inference_mean_ms, inference_std_ms = measure_inference_speed(
        model,
        bundle["test_image_paths"],
    )

    test_sample_ids = sorted(test_df["sample_id"].dropna().astype(str).unique().tolist())
    held_out_sample = test_sample_ids[0] if test_sample_ids else ""
    sample_meta = SAMPLE_METADATA.get(held_out_sample, {})

    result = {
        "timestamp": datetime.now().isoformat(),
        "backbone": EXTENSION_BACKBONE,
        "split_mode": EXTENSION_SPLIT_MODE,
        "fold_name": fold_name,
        "seed": seed,
        "model_input_mode": MODEL_INPUT_MODE,
        "image_crop_mode": IMAGE_CROP_MODE,
        "run_stem": run_stem,
        "model_path": str(model_path),
        "tflite_path": str(tflite_path) if tflite_path.exists() else "",
        "held_out_sample": held_out_sample,
        "held_out_cut": sample_meta.get("pork_cut", ""),
        "capture_source": sample_meta.get("capture_source", ""),
        "phone_group": sample_meta.get("phone_group", ""),
        "test_count": int(len(test_df)),
        "train_count": int(len(train_df)),
        "val_count": int(len(val_df)),
        "predictions_path": str(predictions_path),
        "h5_size_mb": float(h5_size_mb),
        "tflite_size_mb": float(tflite_size_mb) if tflite_size_mb is not None else np.nan,
        "inference_mean_ms_per_image": float(inference_mean_ms),
        "inference_std_ms_per_image": float(inference_std_ms),
        **train_info,
        **metric_row,
        **transition_metric_row,
    }

    cm = np.array(json.loads(result["confusion_matrix_json"]))
    cm_norm = np.array(json.loads(result["normalized_confusion_matrix_json"]))
    save_confusion_matrix_figure(
        cm,
        LABEL_ORDER,
        title=f"{run_stem} Confusion Matrix",
        out_path=EXTENSION_FIGURES_ROOT / f"{run_stem}_confusion_matrix.png",
        normalize=False,
    )
    save_confusion_matrix_figure(
        cm_norm,
        LABEL_ORDER,
        title=f"{run_stem} Normalized Confusion Matrix",
        out_path=EXTENSION_FIGURES_ROOT / f"{run_stem}_normalized_confusion_matrix.png",
        normalize=True,
    )

    return result


def append_failed_run(fold_name: str, seed: int, exc: Exception) -> None:
    row = {
        "timestamp": datetime.now().isoformat(),
        "fold_name": fold_name,
        "seed": seed,
        "model_input_mode": MODEL_INPUT_MODE,
        "image_crop_mode": IMAGE_CROP_MODE,
        "error": repr(exc),
    }
    if SEGMENTED6_FAILED_RUNS_PATH.exists():
        failed_df = pd.read_csv(SEGMENTED6_FAILED_RUNS_PATH)
        failed_df = pd.concat([failed_df, pd.DataFrame([row])], ignore_index=True)
    else:
        failed_df = pd.DataFrame([row])
    failed_df.to_csv(SEGMENTED6_FAILED_RUNS_PATH, index=False)


def copy_best_segmented6_cnn_only_artifacts(seed_metrics_df: pd.DataFrame) -> dict[str, str]:
    if seed_metrics_df.empty:
        return {}

    best_idx = seed_metrics_df["macro_f1"].astype(float).idxmax()
    best_row = seed_metrics_df.loc[best_idx]

    src_model_path = Path(best_row["model_path"])
    src_tflite_path = (
        Path(str(best_row.get("tflite_path", "")))
        if str(best_row.get("tflite_path", "")).strip()
        else None
    )

    best_model_path = EXTENSION_MODELS_ROOT / "meatlens_best_segmented6_cnn_only_mobilenetv3small.h5"
    best_tflite_path = EXTENSION_MODELS_ROOT / "meatlens_best_segmented6_cnn_only_mobilenetv3small.tflite"
    best_metadata_path = EXTENSION_MODELS_ROOT / "meatlens_best_segmented6_cnn_only_mobilenetv3small_metadata.json"

    shutil.copy2(src_model_path, best_model_path)
    if src_tflite_path is not None and src_tflite_path.exists():
        shutil.copy2(src_tflite_path, best_tflite_path)

    metadata = {
        "backbone": EXTENSION_BACKBONE,
        "model_input_mode": MODEL_INPUT_MODE,
        "image_crop_mode": IMAGE_CROP_MODE,
        "input_size": list(INPUT_SHAPE),
        "label_order": LABEL_ORDER,
        "class_index_mapping": {str(idx): label for idx, label in INDEX_TO_LABEL.items()},
        "split_mode": EXTENSION_SPLIT_MODE,
        "fold": best_row["fold_name"],
        "seed": int(best_row["seed"]),
        "held_out_sample": best_row.get("held_out_sample", ""),
        "held_out_cut": best_row.get("held_out_cut", ""),
        "capture_source": best_row.get("capture_source", ""),
        "phone_group": best_row.get("phone_group", ""),
        "accuracy": float(best_row["accuracy"]),
        "macro_f1": float(best_row["macro_f1"]),
        "top_2_accuracy": float(best_row.get("top_2_accuracy", np.nan)),
        "adjacent_accuracy": float(best_row.get("adjacent_accuracy", np.nan)),
        "severe_error_rate": float(best_row.get("severe_error_rate", np.nan)),
        "model_path": str(best_model_path),
        "tflite_path": str(best_tflite_path) if best_tflite_path.exists() else "",
        "timestamp": datetime.now().isoformat(),
        "note": "CNN-only MobileNetV3Small trained on preprocessed HSV/LAB segmented 224x224 ROI images. No handcrafted RGB/HSV/LAB/GLCM feature branch was used.",
    }
    best_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "best_model_path": str(best_model_path),
        "best_tflite_path": str(best_tflite_path) if best_tflite_path.exists() else "",
        "best_metadata_path": str(best_metadata_path),
    }


def _summarize_model_metrics(metrics_df: pd.DataFrame, model_name: str) -> dict[str, object]:
    row: dict[str, object] = {
        "model_name": model_name,
        "runs": int(len(metrics_df)),
    }

    for metric_name in [
        "accuracy",
        "macro_f1",
        "top_2_accuracy",
        "adjacent_accuracy",
        "severe_error_rate",
    ]:
        values = pd.to_numeric(metrics_df.get(metric_name, pd.Series(dtype=float)), errors="coerce")
        row[f"{metric_name}_mean"] = float(values.mean()) if len(values) else np.nan
        row[f"{metric_name}_std"] = float(values.std()) if len(values) else np.nan

    for metric_name in [
        "inference_mean_ms_per_image",
        "h5_size_mb",
        "tflite_size_mb",
    ]:
        values = pd.to_numeric(metrics_df.get(metric_name, pd.Series(dtype=float)), errors="coerce")
        row[f"{metric_name}_mean"] = float(values.mean()) if len(values) else np.nan

    for label_name in LABEL_ORDER:
        safe = label_name.replace(" ", "_")
        recall_values = pd.to_numeric(metrics_df.get(f"{safe}_recall", pd.Series(dtype=float)), errors="coerce")
        f1_values = pd.to_numeric(metrics_df.get(f"{safe}_f1", pd.Series(dtype=float)), errors="coerce")
        row[f"{safe}_recall_mean"] = float(recall_values.mean()) if len(recall_values) else np.nan
        row[f"{safe}_f1_mean"] = float(f1_values.mean()) if len(f1_values) else np.nan

    return row


def _save_model_comparison_bar_plot(
    comparison_df: pd.DataFrame,
    metric_col: str,
    title: str,
    ylabel: str,
    out_path: Path,
    color: str,
) -> None:
    plot_df = comparison_df.copy()
    plot_df[metric_col] = pd.to_numeric(plot_df[metric_col], errors="coerce")
    save_bar_plot(
        plot_df["model_name"],
        plot_df[metric_col],
        title,
        ylabel,
        out_path,
        color,
        xlabel="Model",
    )


def _save_per_class_comparison_plot(
    comparison_df: pd.DataFrame,
    metric_suffix: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    plot_df = comparison_df.copy()
    fig, ax = base.plt.subplots(figsize=(10, 6))
    x = np.arange(len(LABEL_ORDER))
    width = 0.36
    colors = ["#355070", "#2A9D8F"]

    for idx, (_, row) in enumerate(plot_df.iterrows()):
        values = []
        for label_name in LABEL_ORDER:
            safe = label_name.replace(" ", "_")
            values.append(pd.to_numeric(row.get(f"{safe}_{metric_suffix}_mean", np.nan), errors="coerce"))
        ax.bar(
            x + (idx - (len(plot_df) - 1) / 2.0) * width,
            values,
            width=width,
            label=str(row["model_name"]),
            color=colors[idx % len(colors)],
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_ORDER)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel(ylabel)
    ax.legend()
    base.plt.tight_layout()
    fig.savefig(out_path, dpi=220)
    base.plt.close(fig)


def create_hybrid_vs_cnn_only_comparison() -> dict[str, object] | None:
    ensure_output_dirs()
    if not HYBRID_SEED_METRICS_PATH.exists():
        print(f"Hybrid metrics not found: {HYBRID_SEED_METRICS_PATH}")
        return None
    if not SEGMENTED6_SEED_METRICS_PATH.exists():
        print(f"CNN-only metrics not found: {SEGMENTED6_SEED_METRICS_PATH}")
        return None

    hybrid_df = pd.read_csv(HYBRID_SEED_METRICS_PATH)
    cnn_only_df = pd.read_csv(SEGMENTED6_SEED_METRICS_PATH)
    if hybrid_df.empty or cnn_only_df.empty:
        print("Comparison skipped because one metrics CSV is empty.")
        return None

    comparison_df = pd.DataFrame(
        [
            _summarize_model_metrics(hybrid_df, "hybrid"),
            _summarize_model_metrics(cnn_only_df, "cnn_only"),
        ]
    )
    comparison_df.to_csv(SEGMENTED6_COMPARISON_PATH, index=False)

    _save_model_comparison_bar_plot(
        comparison_df,
        "macro_f1_mean",
        "Hybrid vs CNN-only Macro F1",
        "Mean Macro F1",
        EXTENSION_FIGURES_ROOT / "hybrid_vs_cnn_only_macro_f1.png",
        "#355070",
    )
    _save_model_comparison_bar_plot(
        comparison_df,
        "accuracy_mean",
        "Hybrid vs CNN-only Accuracy",
        "Mean Accuracy",
        EXTENSION_FIGURES_ROOT / "hybrid_vs_cnn_only_accuracy.png",
        "#4C9F70",
    )
    _save_model_comparison_bar_plot(
        comparison_df,
        "severe_error_rate_mean",
        "Hybrid vs CNN-only Severe Error Rate",
        "Mean Severe Error Rate",
        EXTENSION_FIGURES_ROOT / "hybrid_vs_cnn_only_severe_error_rate.png",
        "#BC4749",
    )
    _save_model_comparison_bar_plot(
        comparison_df,
        "inference_mean_ms_per_image_mean",
        "Hybrid vs CNN-only Inference Speed",
        "Mean ms/image",
        EXTENSION_FIGURES_ROOT / "hybrid_vs_cnn_only_inference_speed.png",
        "#BC6C25",
    )
    _save_model_comparison_bar_plot(
        comparison_df,
        "h5_size_mb_mean",
        "Hybrid vs CNN-only Model Size",
        "Mean H5 size (MB)",
        EXTENSION_FIGURES_ROOT / "hybrid_vs_cnn_only_model_size.png",
        "#6C9A8B",
    )
    _save_per_class_comparison_plot(
        comparison_df,
        "recall",
        "Hybrid vs CNN-only Per-class Recall",
        "Mean Recall",
        EXTENSION_FIGURES_ROOT / "hybrid_vs_cnn_only_per_class_recall.png",
    )
    _save_per_class_comparison_plot(
        comparison_df,
        "f1",
        "Hybrid vs CNN-only Per-class F1",
        "Mean F1",
        EXTENSION_FIGURES_ROOT / "hybrid_vs_cnn_only_per_class_f1.png",
    )

    return {
        "comparison_df": comparison_df,
        "comparison_csv_path": str(SEGMENTED6_COMPARISON_PATH),
    }


def regenerate_transition_metrics_and_graphs() -> dict[str, object] | None:
    ensure_output_dirs()
    if not SEGMENTED6_SEED_METRICS_PATH.exists():
        print("No segmented6 CNN-only training results found yet. Train first, then rerun this section.")
        return None

    seed_metrics_df = pd.read_csv(SEGMENTED6_SEED_METRICS_PATH).fillna("")
    if seed_metrics_df.empty:
        print("No segmented6 CNN-only training results found yet. Train first, then rerun this section.")
        return None

    updated_rows = []
    prediction_frames = []

    for _, row in seed_metrics_df.iterrows():
        row_dict = row.to_dict()
        predictions_path = str(row_dict.get("predictions_path", "")).strip()
        if predictions_path == "":
            print(f"[WARN] Missing predictions_path for run {row_dict.get('run_stem', '')}")
            updated_rows.append(row_dict)
            continue

        pred_path = Path(predictions_path)
        if not pred_path.exists():
            print(f"[WARN] Prediction CSV not found: {pred_path}")
            updated_rows.append(row_dict)
            continue

        pred_df = load_prediction_dataframe(pred_path)
        pred_df = regenerate_prediction_dataframe_if_needed(pred_df)
        pred_df.to_csv(pred_path, index=False)

        run_metrics = classification_metrics(
            pred_df["true_label"].astype(str).str.strip().str.lower().map(LABEL_TO_INDEX).astype(int).to_numpy(),
            pred_df["predicted_label"].astype(str).str.strip().str.lower().map(LABEL_TO_INDEX).astype(int).to_numpy(),
        )
        run_metrics.update(compute_transition_metrics_from_prediction_df(pred_df))
        for key, value in run_metrics.items():
            row_dict[key] = value
        updated_rows.append(row_dict)

        for col_name in ["fold_name", "seed", "held_out_sample", "held_out_cut", "capture_source", "phone_group", "run_stem"]:
            if col_name not in pred_df.columns:
                pred_df[col_name] = row_dict.get(col_name, "")
        prediction_frames.append(pred_df)

    updated_metrics_df = pd.DataFrame(updated_rows)
    if not updated_metrics_df.empty:
        updated_metrics_df = updated_metrics_df.sort_values(["fold_name", "seed"]).reset_index(drop=True)
    updated_metrics_df.to_csv(SEGMENTED6_SEED_METRICS_PATH, index=False)

    prediction_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    summary_bundle = save_segmented6_cnn_only_summary_outputs(updated_metrics_df, prediction_df=prediction_df)
    interpretation_text = build_summary_interpretation_text(updated_metrics_df, prediction_df)
    print(interpretation_text)
    comparison_bundle = create_hybrid_vs_cnn_only_comparison()

    return {
        "seed_metrics_df": updated_metrics_df,
        "prediction_df": prediction_df,
        "summary_bundle": summary_bundle,
        "comparison_bundle": comparison_bundle,
        "interpretation_text": interpretation_text,
    }


def run_segmented6_cnn_only_training() -> dict[str, object]:
    ensure_output_dirs()
    split_dfs = load_all_cross_rotation_splits()
    existing_metrics_df = load_existing_metrics()
    seed_results = existing_metrics_df.to_dict(orient="records") if not existing_metrics_df.empty else []

    for fold_name in EXTENSION_FOLDS:
        for seed in EXTENSION_RUN_SEEDS:
            if should_skip_run(existing_metrics_df, fold_name, seed):
                print(f"Skipping completed run: {fold_name} seed={seed}")
                continue

            print(f"Running segmented CNN-only experiment: {fold_name} seed={seed}")
            try:
                result = run_single_segmented6_cnn_only_experiment(
                    fold_name=fold_name,
                    seed=seed,
                    split_dfs=split_dfs,
                )
                seed_results.append(result)
                existing_metrics_df = pd.DataFrame(seed_results)
                if not existing_metrics_df.empty:
                    existing_metrics_df = existing_metrics_df.sort_values(["fold_name", "seed"]).reset_index(drop=True)
                existing_metrics_df.to_csv(SEGMENTED6_SEED_METRICS_PATH, index=False)
            except Exception as exc:
                append_failed_run(fold_name, seed, exc)
                print(f"[WARN] Failed run {fold_name} seed={seed}: {exc}")
            finally:
                if TF_AVAILABLE:
                    tf.keras.backend.clear_session()
                gc.collect()

    seed_metrics_df = pd.DataFrame(seed_results)
    if not seed_metrics_df.empty:
        seed_metrics_df = seed_metrics_df.sort_values(["fold_name", "seed"]).reset_index(drop=True)
    seed_metrics_df.to_csv(SEGMENTED6_SEED_METRICS_PATH, index=False)

    summary_bundle = save_segmented6_cnn_only_summary_outputs(seed_metrics_df)
    best_bundle = copy_best_segmented6_cnn_only_artifacts(seed_metrics_df)
    comparison_bundle = create_hybrid_vs_cnn_only_comparison()
    return {
        "seed_metrics_df": seed_metrics_df,
        "summary_bundle": summary_bundle,
        "best_bundle": best_bundle,
        "comparison_bundle": comparison_bundle,
    }


# ---------------------------------------------------------------------------
# 8-fold processed ROI overrides
# ---------------------------------------------------------------------------

PROCESSED_ROI8_SEED_METRICS_PATH = SEGMENTED6_SEED_METRICS_PATH
PROCESSED_ROI8_FOLD_SUMMARY_PATH = SEGMENTED6_FOLD_SUMMARY_PATH
PROCESSED_ROI8_SAMPLE_SUMMARY_PATH = SEGMENTED6_SAMPLE_SUMMARY_PATH
PROCESSED_ROI8_CUT_SUMMARY_PATH = SEGMENTED6_CUT_SUMMARY_PATH
PROCESSED_ROI8_CAPTURE_SOURCE_SUMMARY_PATH = SEGMENTED6_CAPTURE_SOURCE_SUMMARY_PATH
PROCESSED_ROI8_PHONE_GROUP_SUMMARY_PATH = SEGMENTED6_PHONE_GROUP_SUMMARY_PATH
PROCESSED_ROI8_PER_CLASS_SUMMARY_PATH = SEGMENTED6_PER_CLASS_SUMMARY_PATH
PROCESSED_ROI8_PRED_DISTRIBUTION_PATH = SEGMENTED6_PRED_DISTRIBUTION_PATH
PROCESSED_ROI8_SIZE_SPEED_PATH = SEGMENTED6_SIZE_SPEED_PATH
PROCESSED_ROI8_FAILED_RUNS_PATH = SEGMENTED6_FAILED_RUNS_PATH
PROCESSED_ROI8_QUALITY_SUMMARY_PATH = SEGMENTED6_QUALITY_SUMMARY_PATH
PROCESSED_ROI8_COMPARISON_PATH = SEGMENTED6_COMPARISON_PATH


def _base_patch_map() -> dict[str, object]:
    return {
        "PROJECT_ROOT": PROJECT_ROOT,
        "TRAINING_OUTPUTS_ROOT": TRAINING_OUTPUTS_ROOT,
        "SEGMENTED_SPLITS_ROOT": SPLIT_ROOT,
        "CROSS_ROTATION_ROOT": SPLIT_ROOT,
        "EXTENSION_OUTPUT_ROOT": EXTENSION_OUTPUT_ROOT,
        "EXTENSION_FIGURES_ROOT": EXTENSION_FIGURES_ROOT,
        "EXTENSION_MODELS_ROOT": EXTENSION_MODELS_ROOT,
        "EXTENSION_GRADCAM_ROOT": EXTENSION_GRADCAM_ROOT,
        "EXTENSION_PREDICTIONS_ROOT": EXTENSION_PREDICTIONS_ROOT,
        "EXTENSION_BACKBONE": EXTENSION_BACKBONE,
        "EXTENSION_SPLIT_MODE": EXTENSION_SPLIT_MODE,
        "EXTENSION_FOLDS": EXTENSION_FOLDS,
        "EXTENSION_RUN_SEEDS": EXTENSION_RUN_SEEDS,
        "RUN_EXTENSION_TRAINING": RUN_EXTENSION_TRAINING,
        "TARGET_SIZE": TARGET_SIZE,
        "INPUT_SHAPE": INPUT_SHAPE,
        "NUM_CLASSES": NUM_CLASSES,
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS_HEAD": EPOCHS_HEAD,
        "EPOCHS_FINE": EPOCHS_FINE,
        "HEAD_LR": HEAD_LR,
        "FINE_TUNE_LR": FINE_TUNE_LR,
        "FINE_TUNE_FRACTION": FINE_TUNE_FRACTION,
        "MODEL_INPUT_MODE": MODEL_INPUT_MODE,
        "IMAGE_CROP_MODE": IMAGE_CROP_MODE,
        "LABEL_ORDER": LABEL_ORDER,
        "LABEL_TO_INDEX": LABEL_TO_INDEX,
        "INDEX_TO_LABEL": INDEX_TO_LABEL,
        "SEGMENTED6_SEED_METRICS_PATH": SEGMENTED6_SEED_METRICS_PATH,
        "SEGMENTED6_FOLD_SUMMARY_PATH": SEGMENTED6_FOLD_SUMMARY_PATH,
        "SEGMENTED6_SAMPLE_SUMMARY_PATH": SEGMENTED6_SAMPLE_SUMMARY_PATH,
        "SEGMENTED6_CUT_SUMMARY_PATH": SEGMENTED6_CUT_SUMMARY_PATH,
        "SEGMENTED6_CAPTURE_SOURCE_SUMMARY_PATH": SEGMENTED6_CAPTURE_SOURCE_SUMMARY_PATH,
        "SEGMENTED6_PHONE_GROUP_SUMMARY_PATH": SEGMENTED6_PHONE_GROUP_SUMMARY_PATH,
        "SEGMENTED6_PER_CLASS_SUMMARY_PATH": SEGMENTED6_PER_CLASS_SUMMARY_PATH,
        "SEGMENTED6_PRED_DISTRIBUTION_PATH": SEGMENTED6_PRED_DISTRIBUTION_PATH,
        "SEGMENTED6_SIZE_SPEED_PATH": SEGMENTED6_SIZE_SPEED_PATH,
        "SEGMENTED6_FAILED_RUNS_PATH": SEGMENTED6_FAILED_RUNS_PATH,
        "SEGMENTED6_QUALITY_SUMMARY_PATH": SEGMENTED6_QUALITY_SUMMARY_PATH,
        "TOP_CONFIDENCE_BORDERLINE": TOP_CONFIDENCE_BORDERLINE,
        "TOP_CONFIDENCE_LOW": TOP_CONFIDENCE_LOW,
    }


@contextmanager
def patched_base_config():
    patch_map = _base_patch_map()
    previous = {name: getattr(base, name) for name in patch_map}
    for name, value in patch_map.items():
        setattr(base, name, value)
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(base, name, value)


def ensure_output_dirs() -> None:
    for path_obj in [
        EXTENSION_OUTPUT_ROOT,
        EXTENSION_FIGURES_ROOT,
        EXTENSION_MODELS_ROOT,
        EXTENSION_PREDICTIONS_ROOT,
        EXTENSION_GRADCAM_ROOT,
        EXTENSION_LOGS_ROOT,
    ]:
        path_obj.mkdir(parents=True, exist_ok=True)


def print_library_status() -> None:
    print("TF_AVAILABLE =", TF_AVAILABLE)
    print("SKLEARN_AVAILABLE =", SKLEARN_AVAILABLE)
    print("SKIMAGE_AVAILABLE =", SKIMAGE_AVAILABLE)
    print("CV2_AVAILABLE =", CV2_AVAILABLE)
    print("JOBLIB_AVAILABLE =", JOBLIB_AVAILABLE)
    print("SEABORN_AVAILABLE =", SEABORN_AVAILABLE)
    print("PROJECT_ROOT =", PROJECT_ROOT)
    print("SPLIT_ROOT =", SPLIT_ROOT)
    print("EXTENSION_OUTPUT_ROOT =", EXTENSION_OUTPUT_ROOT)
    print("IMAGE_CROP_MODE =", IMAGE_CROP_MODE)
    print("MODEL_INPUT_MODE =", MODEL_INPUT_MODE)
    print("EXTENSION_FOLDS =", EXTENSION_FOLDS)
    print("EXTENSION_RUN_SEEDS =", EXTENSION_RUN_SEEDS)
    print("USE_TRAINING_AUGMENTATION =", USE_TRAINING_AUGMENTATION)
    print("ALLOW_RESIZE_FALLBACK =", ALLOW_RESIZE_FALLBACK)


def reset_loader_stats() -> None:
    for key in LOADER_STATS:
        LOADER_STATS[key] = 0


def detect_image_path_column(df: pd.DataFrame) -> str | None:
    for col_name in IMAGE_PATH_CANDIDATES:
        if col_name in df.columns:
            return col_name
    return None


def resolve_image_path(row: pd.Series, path_col: str, csv_path: Path | None = None) -> str | None:
    raw = row.get(path_col)
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    raw_text = str(raw).strip()
    if raw_text == "":
        return None

    candidates = [Path(raw_text)]
    if csv_path is not None:
        candidates.append(csv_path.parent / raw_text)
    candidates.append(PROJECT_ROOT / raw_text)

    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve(strict=False)
        except Exception:
            resolved = candidate
        if Path(resolved).exists():
            return str(Path(resolved))
    return None


def infer_sample_id(sample_number: object) -> str | None:
    if sample_number is None or (isinstance(sample_number, float) and pd.isna(sample_number)):
        return None
    sample_text = str(sample_number).strip()
    if sample_text.endswith(".0") and sample_text[:-2].isdigit():
        sample_text = sample_text[:-2]
    return SAMPLE_NUMBER_TO_ID.get(sample_text)


def normalize_sample_id(sample_id_value: object, sample_number_value: object) -> str | None:
    if sample_id_value is not None and not (isinstance(sample_id_value, float) and pd.isna(sample_id_value)):
        sample_id_text = str(sample_id_value).strip().lower()
        if sample_id_text not in {"", "nan", "none"}:
            return sample_id_text
    return infer_sample_id(sample_number_value)


def _fallback_metadata_for_sample_number(sample_number: object) -> dict[str, object]:
    sample_text = "" if sample_number is None else str(sample_number).strip().replace(".0", "")
    if sample_text == "7":
        return SAMPLE_METADATA["pork_unknown_sample_7"]
    if sample_text == "8":
        return SAMPLE_METADATA["pork_unknown_sample_8"]
    return {}


def enrich_split_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sample_id"] = out.apply(
        lambda row: normalize_sample_id(row.get("sample_id"), row.get("sample_number")),
        axis=1,
    )
    sample_meta_series = out.apply(
        lambda row: SAMPLE_METADATA.get(str(row.get("sample_id", "")).strip().lower(), _fallback_metadata_for_sample_number(row.get("sample_number"))),
        axis=1,
    )

    def fill_from_meta(column_name: str, default_value: object = "unknown"):
        existing = out[column_name] if column_name in out.columns else pd.Series([np.nan] * len(out), index=out.index)
        return existing.where(existing.notna() & existing.astype(str).str.strip().ne(""), sample_meta_series.map(lambda meta: meta.get(column_name, default_value))).fillna(default_value)

    out["sample_number"] = fill_from_meta("sample_number", np.nan)
    out["sample_number"] = out["sample_number"].astype(str).str.replace(".0", "", regex=False)
    out["pork_cut"] = fill_from_meta("pork_cut", "unknown").astype(str).str.strip()
    out["capture_source"] = fill_from_meta("capture_source", "unknown").astype(str).str.strip()
    out["phone_group"] = fill_from_meta("phone_group", "unknown").astype(str).str.strip()
    if "phone_model" not in out.columns:
        out["phone_model"] = ""
    if "source" not in out.columns:
        out["source"] = ""
    out["label"] = out["label"].astype(str).str.strip().str.lower()
    return out


def load_split_dataframe(csv_path: Path, split_key: str) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing split file: {csv_path}")

    df = pd.read_csv(csv_path)
    df["split_key"] = split_key
    fold_name, partition = split_key.split("_", 1)
    df["split_name"] = fold_name
    df["dataset_partition"] = partition

    path_col = detect_image_path_column(df)
    if path_col is None:
        raise ValueError(
            f"No image path column found in {csv_path}. Tried: {IMAGE_PATH_CANDIDATES}"
        )
    df["image_path_source_column"] = path_col
    df["image_path_resolved"] = df.apply(
        lambda row: resolve_image_path(row, path_col=path_col, csv_path=csv_path),
        axis=1,
    )
    df["missing_image_path"] = df["image_path_resolved"].isna()
    missing_count = int(df["missing_image_path"].sum())
    if missing_count > 0:
        print(f"[WARN] {split_key} has {missing_count} unresolved image paths.")
    df = enrich_split_df(df)
    return df


def load_all_cross_rotation_splits() -> dict[str, pd.DataFrame]:
    split_dfs: dict[str, pd.DataFrame] = {}
    for fold_name in EXTENSION_FOLDS:
        for partition in ["train", "val", "test"]:
            split_key = f"{fold_name}_{partition}"
            csv_path = SPLIT_ROOT / f"{split_key}.csv"
            split_dfs[split_key] = load_split_dataframe(csv_path, split_key)
    return split_dfs


def combined_splits_dataframe(split_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not split_dfs:
        return pd.DataFrame()
    return pd.concat(split_dfs.values(), ignore_index=True)


def validate_metadata_mapping(split_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    combined_df = combined_splits_dataframe(split_dfs)
    rows = []
    for sample_id in sorted(combined_df["sample_id"].dropna().astype(str).unique()):
        sample_df = combined_df.loc[combined_df["sample_id"].astype(str) == sample_id].copy()
        sample_number = sample_df["sample_number"].astype(str).replace("nan", "").iloc[0] if not sample_df.empty else ""
        pork_cut = sample_df["pork_cut"].astype(str).iloc[0] if not sample_df.empty else "unknown"
        capture_source = sample_df["capture_source"].astype(str).iloc[0] if not sample_df.empty else "unknown"
        phone_group = sample_df["phone_group"].astype(str).iloc[0] if not sample_df.empty else "unknown"
        used_placeholder = sample_id.startswith("pork_unknown_sample_") or capture_source == "unknown" or phone_group == "unknown"
        if used_placeholder:
            print(
                f"[WARN] Placeholder metadata used for {sample_id}: "
                f"capture_source={capture_source}, phone_group={phone_group}"
            )
        rows.append(
            {
                "sample_id": sample_id,
                "sample_number": sample_number,
                "pork_cut": pork_cut,
                "capture_source": capture_source,
                "phone_group": phone_group,
                "rows_found": int(len(sample_df)),
                "uses_placeholder_metadata": bool(used_placeholder),
            }
        )
    return pd.DataFrame(rows).sort_values(["sample_number", "sample_id"]).reset_index(drop=True)


def _label_counts(df: pd.DataFrame) -> dict[str, int]:
    counts = df["label"].value_counts().to_dict()
    return {label_name: int(counts.get(label_name, 0)) for label_name in LABEL_ORDER}


def build_dataset_quality_summary(split_dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    summary_rows = []
    for split_key, df in split_dfs.items():
        counts = _label_counts(df)
        summary_rows.append(
            {
                "split_key": split_key,
                "rows": int(len(df)),
                "missing_image_path_rows": int(df["missing_image_path"].sum()) if "missing_image_path" in df.columns else 0,
                "fresh_count": counts["fresh"],
                "not_fresh_count": counts["not fresh"],
                "spoiled_count": counts["spoiled"],
                "unique_samples": int(df["sample_id"].astype(str).nunique()),
            }
        )

    combined_df = combined_splits_dataframe(split_dfs)
    sample_rows = []
    for sample_id, sample_df in combined_df.groupby("sample_id", dropna=False):
        counts = _label_counts(sample_df)
        sample_rows.append(
            {
                "sample_id": str(sample_id),
                "sample_number": str(sample_df["sample_number"].iloc[0]),
                "pork_cut": str(sample_df["pork_cut"].iloc[0]),
                "capture_source": str(sample_df["capture_source"].iloc[0]),
                "phone_group": str(sample_df["phone_group"].iloc[0]),
                "rows": int(len(sample_df)),
                "fresh_count": counts["fresh"],
                "not_fresh_count": counts["not fresh"],
                "spoiled_count": counts["spoiled"],
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("split_key").reset_index(drop=True)
    sample_summary_df = pd.DataFrame(sample_rows).sort_values(["sample_number", "sample_id"]).reset_index(drop=True)

    quality_metric_rows = [
        {"metric_name": "total_rows_all_splits", "metric_value": int(len(combined_df))},
        {"metric_name": "unique_resolved_image_paths", "metric_value": int(combined_df["image_path_resolved"].dropna().astype(str).nunique())},
        {"metric_name": "missing_image_path_rows", "metric_value": int(combined_df["missing_image_path"].sum()) if "missing_image_path" in combined_df.columns else 0},
    ]
    quality_metrics_df = pd.DataFrame(quality_metric_rows)
    quality_metrics_df.to_csv(PROCESSED_ROI8_QUALITY_SUMMARY_PATH, index=False)

    return {
        "summary_df": summary_df,
        "sample_summary_df": sample_summary_df,
        "quality_metrics_df": quality_metrics_df,
    }


def print_quality_tables(quality_bundle: dict[str, pd.DataFrame]) -> None:
    for key in ["summary_df", "sample_summary_df", "quality_metrics_df"]:
        if key in quality_bundle:
            print(f"\n[{key}]")
            print(quality_bundle[key].head(20).to_string(index=False))


def save_sample_visualization(split_dfs: dict[str, pd.DataFrame]) -> Path:
    ensure_output_dirs()
    combined_df = combined_splits_dataframe(split_dfs)
    preview_rows = []
    for label_name in LABEL_ORDER:
        label_df = combined_df.loc[combined_df["label"].astype(str) == label_name].head(3)
        preview_rows.extend(label_df.to_dict(orient="records"))
    preview_df = pd.DataFrame(preview_rows)
    if preview_df.empty:
        raise RuntimeError("No preview rows available for sample visualization.")

    ncols = 3
    nrows = int(math.ceil(len(preview_df) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for ax in axes.ravel():
        ax.axis("off")

    for idx, (_, row) in enumerate(preview_df.iterrows()):
        ax = axes[idx // ncols, idx % ncols]
        image_uint8 = load_preprocessed_hsv_lab_threshold_roi_224(row["image_path_resolved"])
        ax.imshow(image_uint8)
        ax.set_title(f"{row['sample_id']} | {row['label']}")
        ax.axis("off")

    plt.tight_layout()
    out_path = EXTENSION_FIGURES_ROOT / "sample_visualization_grid.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def _resize_image_uint8(image_uint8: np.ndarray) -> np.ndarray:
    pil_img = Image.fromarray(image_uint8)
    return np.array(pil_img.resize(TARGET_SIZE, Image.Resampling.BILINEAR), dtype=np.uint8)


def load_preprocessed_hsv_lab_threshold_roi_224(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    image_uint8 = np.array(img, dtype=np.uint8)
    if image_uint8.shape == (224, 224, 3):
        LOADER_STATS["already_224"] += 1
        return image_uint8

    LOADER_STATS["non_224"] += 1
    if LOADER_STATS["warning_count"] < 20:
        print(f"[WARN] Non-224 processed ROI image: {path} | original shape={image_uint8.shape}")
        LOADER_STATS["warning_count"] += 1
    if not ALLOW_RESIZE_FALLBACK:
        raise ValueError(
            f"Expected 224x224x3 processed ROI image but got {image_uint8.shape} for {path}"
        )

    resized = _resize_image_uint8(image_uint8)
    LOADER_STATS["resized"] += 1
    return resized


def audit_image_shapes(split_dfs: dict[str, pd.DataFrame]) -> dict[str, int]:
    already_224_count = 0
    non_224_count = 0
    unique_paths = sorted(combined_splits_dataframe(split_dfs)["image_path_resolved"].dropna().astype(str).unique().tolist())
    for path in unique_paths:
        img = Image.open(path).convert("RGB")
        shape = np.array(img, dtype=np.uint8).shape
        if shape == (224, 224, 3):
            already_224_count += 1
        else:
            non_224_count += 1
            print(f"[WARN] Shape audit mismatch: {path} -> {shape}")
    print("already_224_count =", already_224_count)
    print("non_224_count =", non_224_count)
    print("unique_images_checked =", len(unique_paths))
    if non_224_count > 0:
        print("[STOP] Some images are not 224x224. The split CSV may be pointing to the wrong folder. Do not train yet.")
    return {
        "already_224_count": already_224_count,
        "non_224_count": non_224_count,
        "unique_images_checked": len(unique_paths),
    }


def _apply_blur_or_sharpen(image_uint8: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() >= 0.20:
        return image_uint8
    if cv2 is not None:
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        blur_kernel = np.ones((3, 3), dtype=np.float32) / 9.0
        kernel = blur_kernel if rng.random() < 0.5 else sharpen_kernel
        filtered = cv2.filter2D(image_uint8, -1, kernel)
    else:
        kernel_values = [1 / 9.0] * 9 if rng.random() < 0.5 else [0, -1, 0, -1, 5, -1, 0, -1, 0]
        pil_img = Image.fromarray(image_uint8)
        filtered = np.array(pil_img.filter(ImageFilter.Kernel((3, 3), kernel_values, scale=None)), dtype=np.uint8)
    blended = (0.90 * image_uint8.astype(np.float32)) + (0.10 * filtered.astype(np.float32))
    return np.clip(blended, 0, 255).astype(np.uint8)


def augment_image_uint8(image_uint8: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    image = image_uint8.copy()
    if rng.random() < 0.30:
        image = image[:, ::-1, :]
    image_f = image.astype(np.float32) / 255.0
    brightness_delta = rng.uniform(-0.04, 0.04)
    contrast_factor = rng.uniform(0.95, 1.05)
    image_f = np.clip((image_f + brightness_delta - 0.5) * contrast_factor + 0.5, 0.0, 1.0)
    if rng.random() < 0.30:
        noise = rng.normal(0.0, 0.01, size=image_f.shape).astype(np.float32)
        image_f = np.clip(image_f + noise, 0.0, 1.0)
    image = np.clip(image_f * 255.0, 0, 255).astype(np.uint8)
    image = _apply_blur_or_sharpen(image, rng)
    return image


def preprocess_image_batch(image_batch_uint8: np.ndarray) -> np.ndarray:
    if len(image_batch_uint8) == 0:
        return np.empty((0, *INPUT_SHAPE), dtype=np.float32)
    image_float = image_batch_uint8.astype(np.float32)
    processed = preprocess_mobilenetv3(image_float.copy())
    return np.asarray(processed, dtype=np.float32)


def load_images_uint8_from_paths(paths: list[str], apply_augmentation: bool = False, seed: int | None = None) -> np.ndarray:
    images = []
    rng = np.random.default_rng(seed)
    for idx, path in enumerate(paths):
        image_uint8 = load_preprocessed_hsv_lab_threshold_roi_224(path)
        if apply_augmentation:
            image_uint8 = augment_image_uint8(image_uint8, np.random.default_rng(int(rng.integers(0, 2**31 - 1)) + idx))
        images.append(image_uint8)
    if not images:
        return np.empty((0, *INPUT_SHAPE), dtype=np.uint8)
    return np.stack(images).astype(np.uint8)


def _approx_mb(arr: np.ndarray) -> float:
    return float(arr.nbytes / (1024 * 1024))


def prepare_bundle_for_fold(
    fold_name: str,
    split_dfs: dict[str, pd.DataFrame],
) -> dict[str, object]:
    train_df = split_dfs[f"{fold_name}_train"].copy()
    val_df = split_dfs[f"{fold_name}_val"].copy()
    test_df = split_dfs[f"{fold_name}_test"].copy()

    missing_rows = int(
        train_df["missing_image_path"].sum()
        + val_df["missing_image_path"].sum()
        + test_df["missing_image_path"].sum()
    )
    if missing_rows > 0:
        raise RuntimeError(
            f"{fold_name} has {missing_rows} rows with unresolved image paths. Do not train until the split CSVs are fixed."
        )

    train_image_paths = train_df["image_path_resolved"].astype(str).tolist()
    val_image_paths = val_df["image_path_resolved"].astype(str).tolist()
    test_image_paths = test_df["image_path_resolved"].astype(str).tolist()

    train_uint8 = load_images_uint8_from_paths(
        train_image_paths,
        apply_augmentation=USE_TRAINING_AUGMENTATION,
        seed=abs(hash((fold_name, "train"))) % (2**31 - 1),
    )
    val_uint8 = load_images_uint8_from_paths(val_image_paths, apply_augmentation=False)
    test_uint8 = load_images_uint8_from_paths(test_image_paths, apply_augmentation=False)

    train_images = preprocess_image_batch(train_uint8)
    val_images = preprocess_image_batch(val_uint8)
    test_images = preprocess_image_batch(test_uint8)

    print(f"train_images shape={train_images.shape}, approx_mb={_approx_mb(train_images):.2f}")
    print(f"val_images shape={val_images.shape}, approx_mb={_approx_mb(val_images):.2f}")
    print(f"test_images shape={test_images.shape}, approx_mb={_approx_mb(test_images):.2f}")

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_images": train_images,
        "val_images": val_images,
        "test_images": test_images,
        "train_labels": train_df["label"].map(LABEL_TO_INDEX).astype(int).to_numpy(),
        "val_labels": val_df["label"].map(LABEL_TO_INDEX).astype(int).to_numpy(),
        "test_labels": test_df["label"].map(LABEL_TO_INDEX).astype(int).to_numpy(),
        "train_image_paths": train_image_paths,
        "val_image_paths": val_image_paths,
        "test_image_paths": test_image_paths,
        "missing_image_path_rows": missing_rows,
    }


def build_mobilenetv3small_cnn_only_model() -> tf.keras.Model:
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required for model building.")

    image_input = layers.Input(shape=INPUT_SHAPE, name="image_input")
    backbone = MobileNetV3Small(
        include_top=False,
        weights="imagenet",
        input_shape=INPUT_SHAPE,
    )
    backbone._name = "mobilenetv3small_backbone"
    backbone.trainable = False

    x = backbone(image_input, training=False)
    x = layers.GlobalAveragePooling2D(name="image_gap")(x)
    x = layers.Dropout(0.30, name="image_dropout")(x)
    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    x = layers.Dropout(0.30, name="dense_dropout")(x)
    output = layers.Dense(NUM_CLASSES, activation="softmax", name="classification_head")(x)

    return models.Model(
        inputs=image_input,
        outputs=output,
        name="meatlens_mobilenetv3small_8fold_processed_roi_cnn_only",
    )


def compile_model_safely(model, optimizer, loss, metrics):
    try:
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            jit_compile=False,
        )
    except TypeError:
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
        )


def make_optimizer(learning_rate):
    legacy_optimizers = getattr(tf.keras.optimizers, "legacy", None)
    if legacy_optimizers is not None and hasattr(legacy_optimizers, "Adam"):
        print("[INFO] Using tf.keras.optimizers.legacy.Adam for DirectML compatibility.")
        return legacy_optimizers.Adam(learning_rate=learning_rate)
    print("[INFO] Using tf.keras.optimizers.Adam fallback.")
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)


def unfreeze_top_backbone_fraction(model, fraction=0.25):
    backbone = model.get_layer("mobilenetv3small_backbone")
    total_layers = len(backbone.layers)
    start_idx = max(0, int(total_layers * (1.0 - fraction)))
    for idx, layer in enumerate(backbone.layers):
        layer.trainable = idx >= start_idx
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


ValMacroF1Callback = base.ValMacroF1Callback


def compute_class_weights_from_labels(labels: np.ndarray) -> dict[int, float]:
    classes = np.array([0, 1, 2], dtype=np.int32)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=np.asarray(labels, dtype=np.int32))
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


def make_run_stem(fold_name: str, seed: int) -> str:
    return f"processed_roi8_cnn_only_{fold_name}_seed{seed}"


def load_existing_metrics() -> pd.DataFrame:
    if SEGMENTED6_SEED_METRICS_PATH.exists():
        return pd.read_csv(SEGMENTED6_SEED_METRICS_PATH)
    return pd.DataFrame()


def metrics_row_exists(metrics_df: pd.DataFrame, fold_name: str, seed: int) -> bool:
    if metrics_df.empty:
        return False
    mask = (
        metrics_df.get("fold_name", pd.Series(dtype=str)).astype(str).eq(str(fold_name))
        & metrics_df.get("seed", pd.Series(dtype=str)).astype(str).eq(str(seed))
        & metrics_df.get("model_input_mode", pd.Series(dtype=str)).astype(str).eq(MODEL_INPUT_MODE)
        & metrics_df.get("image_crop_mode", pd.Series(dtype=str)).astype(str).eq(IMAGE_CROP_MODE)
    )
    return bool(mask.any())


def should_skip_run(metrics_df: pd.DataFrame, fold_name: str, seed: int) -> bool:
    if not metrics_row_exists(metrics_df, fold_name, seed):
        return False
    run_stem = make_run_stem(fold_name, seed)
    model_path = EXTENSION_MODELS_ROOT / f"{run_stem}.h5"
    prediction_path = EXTENSION_PREDICTIONS_ROOT / f"{run_stem}_predictions.csv"
    return model_path.exists() and prediction_path.exists()


def save_predictions_csv(run_stem: str, prediction_df: pd.DataFrame) -> Path:
    out_path = EXTENSION_PREDICTIONS_ROOT / f"{run_stem}_predictions.csv"
    prediction_df.to_csv(out_path, index=False)
    return out_path


def measure_inference_speed(
    model: tf.keras.Model,
    test_images: np.ndarray,
    warmup: int = 3,
    repeats: int = 10,
) -> tuple[float, float]:
    times = []
    for _ in range(warmup):
        _ = model.predict(test_images, verbose=0)
    for _ in range(repeats):
        start = time.perf_counter()
        _ = model.predict(test_images, verbose=0)
        elapsed = time.perf_counter() - start
        times.append((elapsed * 1000.0) / max(len(test_images), 1))
    return float(np.mean(times)), float(np.std(times))


def train_mobilenetv3small_8fold_cnn_only_model(
    fold_name: str,
    seed: int,
    bundle: dict[str, object],
) -> tuple[tf.keras.Model, dict[str, object]]:
    _ = fold_name
    set_global_seed(seed)
    class_weights = compute_class_weights_from_labels(bundle["train_labels"])
    model = build_mobilenetv3small_cnn_only_model()

    callbacks = [
        ValMacroF1Callback(
            val_source=bundle["val_images"],
            val_labels=bundle["val_labels"],
        ),
        keras_callbacks.EarlyStopping(
            monitor="val_f1_macro",
            mode="max",
            patience=4,
            restore_best_weights=True,
        ),
        keras_callbacks.ReduceLROnPlateau(
            monitor="val_f1_macro",
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    compile_model_safely(
        model,
        optimizer=make_optimizer(HEAD_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history_head = model.fit(
        bundle["train_images"],
        bundle["train_labels"],
        validation_data=(bundle["val_images"], bundle["val_labels"]),
        epochs=EPOCHS_HEAD,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )

    unfreeze_top_backbone_fraction(model, fraction=FINE_TUNE_FRACTION)
    compile_model_safely(
        model,
        optimizer=make_optimizer(FINE_TUNE_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history_fine = model.fit(
        bundle["train_images"],
        bundle["train_labels"],
        validation_data=(bundle["val_images"], bundle["val_labels"]),
        epochs=EPOCHS_FINE,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )

    return model, {
        "history_head": history_head.history,
        "history_fine": history_fine.history,
        "checkpoint_path": "",
        "class_weights_json": json.dumps(class_weights),
    }


def save_processed_roi8_summary_outputs(
    seed_metrics_df: pd.DataFrame,
    prediction_df: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    if seed_metrics_df.empty:
        empty_df = pd.DataFrame()
        return {
            "fold_summary": empty_df,
            "sample_summary": empty_df,
            "cut_summary": empty_df,
            "capture_source_summary": empty_df,
            "phone_group_summary": empty_df,
            "per_class_summary": empty_df,
            "prediction_distribution_df": empty_df,
            "size_speed_df": empty_df,
        }
    with patched_base_config():
        return base.save_segmented6_hybrid_summary_outputs(seed_metrics_df, prediction_df=prediction_df)


save_segmented6_cnn_only_summary_outputs = save_processed_roi8_summary_outputs


def run_single_8fold_cnn_only_experiment(
    fold_name: str = "fold1",
    seed: int = 42,
    split_dfs: dict[str, pd.DataFrame] | None = None,
) -> dict[str, object]:
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required for processed ROI CNN-only training.")
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is required for processed ROI CNN-only training.")

    ensure_output_dirs()
    set_global_seed(seed)
    if split_dfs is None:
        split_dfs = load_all_cross_rotation_splits()

    bundle = prepare_bundle_for_fold(fold_name, split_dfs)
    if bundle["missing_image_path_rows"] > 0:
        raise RuntimeError(
            f"{fold_name} still has unresolved image paths. Do not train until the split CSVs are fixed."
        )

    train_df = bundle["train_df"]
    val_df = bundle["val_df"]
    test_df = bundle["test_df"]
    run_stem = make_run_stem(fold_name, seed)

    model, train_info = train_mobilenetv3small_8fold_cnn_only_model(
        fold_name=fold_name,
        seed=seed,
        bundle=bundle,
    )

    y_prob = model.predict(bundle["test_images"], batch_size=BATCH_SIZE, verbose=0)
    y_pred = y_prob.argmax(axis=1)
    y_true = bundle["test_labels"]
    metric_row = classification_metrics(y_true, y_pred)
    prediction_df = build_transition_prediction_dataframe(test_df, y_true, y_prob)
    transition_metric_row = compute_transition_metrics_from_prediction_df(prediction_df)
    predictions_path = save_predictions_csv(run_stem, prediction_df)

    model_path = EXTENSION_MODELS_ROOT / f"{run_stem}.h5"
    model.save(model_path)
    h5_size_mb = model_path.stat().st_size / (1024 * 1024)

    tflite_path = EXTENSION_MODELS_ROOT / f"{run_stem}.tflite"
    tflite_ok, tflite_size_mb = try_convert_to_tflite(model, tflite_path)
    if not tflite_ok and tflite_path.exists():
        tflite_path.unlink(missing_ok=True)

    inference_mean_ms, inference_std_ms = measure_inference_speed(model, bundle["test_images"])

    test_sample_ids = sorted(test_df["sample_id"].dropna().astype(str).unique().tolist())
    held_out_sample = test_sample_ids[0] if test_sample_ids else ""
    sample_meta = {}
    if held_out_sample in SAMPLE_METADATA:
        sample_meta = SAMPLE_METADATA[held_out_sample]
    elif not test_df.empty:
        sample_meta = {
            "pork_cut": str(test_df["pork_cut"].iloc[0]),
            "capture_source": str(test_df["capture_source"].iloc[0]),
            "phone_group": str(test_df["phone_group"].iloc[0]),
        }

    result = {
        "timestamp": datetime.now().isoformat(),
        "model_name": "meatlens_mobilenetv3small_8fold_processed_roi_cnn_only",
        "backbone": EXTENSION_BACKBONE,
        "split_mode": EXTENSION_SPLIT_MODE,
        "fold_name": fold_name,
        "seed": seed,
        "model_input_mode": MODEL_INPUT_MODE,
        "image_crop_mode": IMAGE_CROP_MODE,
        "run_stem": run_stem,
        "model_path": str(model_path),
        "tflite_path": str(tflite_path) if tflite_path.exists() else "",
        "held_out_sample": held_out_sample,
        "held_out_cut": sample_meta.get("pork_cut", ""),
        "capture_source": sample_meta.get("capture_source", ""),
        "phone_group": sample_meta.get("phone_group", ""),
        "test_count": int(len(test_df)),
        "train_count": int(len(train_df)),
        "val_count": int(len(val_df)),
        "predictions_path": str(predictions_path),
        "h5_size_mb": float(h5_size_mb),
        "tflite_size_mb": float(tflite_size_mb) if tflite_size_mb is not None else np.nan,
        "inference_mean_ms_per_image": float(inference_mean_ms),
        "inference_std_ms_per_image": float(inference_std_ms),
        "training_output_root": str(EXTENSION_OUTPUT_ROOT),
        **train_info,
        **metric_row,
        **transition_metric_row,
    }

    cm = np.array(json.loads(result["confusion_matrix_json"]))
    cm_norm = np.array(json.loads(result["normalized_confusion_matrix_json"]))
    save_confusion_matrix_figure(
        cm,
        LABEL_ORDER,
        title=f"{run_stem} Confusion Matrix",
        out_path=EXTENSION_FIGURES_ROOT / f"{run_stem}_confusion_matrix.png",
        normalize=False,
    )
    save_confusion_matrix_figure(
        cm_norm,
        LABEL_ORDER,
        title=f"{run_stem} Normalized Confusion Matrix",
        out_path=EXTENSION_FIGURES_ROOT / f"{run_stem}_normalized_confusion_matrix.png",
        normalize=True,
    )
    return result


def append_failed_run(fold_name: str, seed: int, exc: Exception) -> None:
    row = {
        "timestamp": datetime.now().isoformat(),
        "fold_name": fold_name,
        "seed": seed,
        "model_input_mode": MODEL_INPUT_MODE,
        "image_crop_mode": IMAGE_CROP_MODE,
        "error": repr(exc),
    }
    if SEGMENTED6_FAILED_RUNS_PATH.exists():
        failed_df = pd.read_csv(SEGMENTED6_FAILED_RUNS_PATH)
        failed_df = pd.concat([failed_df, pd.DataFrame([row])], ignore_index=True)
    else:
        failed_df = pd.DataFrame([row])
    failed_df.to_csv(SEGMENTED6_FAILED_RUNS_PATH, index=False)


def copy_best_processed_roi8_artifacts(seed_metrics_df: pd.DataFrame) -> dict[str, str]:
    if seed_metrics_df.empty:
        return {}
    best_idx = seed_metrics_df["macro_f1"].astype(float).idxmax()
    best_row = seed_metrics_df.loc[best_idx]

    src_model_path = Path(best_row["model_path"])
    src_tflite_path = Path(str(best_row.get("tflite_path", ""))) if str(best_row.get("tflite_path", "")).strip() else None

    best_model_path = EXTENSION_MODELS_ROOT / "meatlens_best_processed_roi8_cnn_only_mobilenetv3small.h5"
    best_tflite_path = EXTENSION_MODELS_ROOT / "meatlens_best_processed_roi8_cnn_only_mobilenetv3small.tflite"
    best_metadata_path = EXTENSION_MODELS_ROOT / "meatlens_best_processed_roi8_cnn_only_mobilenetv3small_metadata.json"

    shutil.copy2(src_model_path, best_model_path)
    if src_tflite_path is not None and src_tflite_path.exists():
        shutil.copy2(src_tflite_path, best_tflite_path)

    metadata = {
        "model_name": "meatlens_mobilenetv3small_8fold_processed_roi_cnn_only",
        "backbone": EXTENSION_BACKBONE,
        "model_input_mode": MODEL_INPUT_MODE,
        "image_crop_mode": IMAGE_CROP_MODE,
        "input_shape": list(INPUT_SHAPE),
        "target_size": list(TARGET_SIZE),
        "label_order": LABEL_ORDER,
        "label_to_index": LABEL_TO_INDEX,
        "index_to_label": INDEX_TO_LABEL,
        "split_mode": EXTENSION_SPLIT_MODE,
        "fold_name": best_row["fold_name"],
        "seed": int(best_row["seed"]),
        "held_out_sample": best_row.get("held_out_sample", ""),
        "held_out_cut": best_row.get("held_out_cut", ""),
        "capture_source": best_row.get("capture_source", ""),
        "phone_group": best_row.get("phone_group", ""),
        "accuracy": float(best_row.get("accuracy", np.nan)),
        "macro_precision": float(best_row.get("macro_precision", np.nan)),
        "macro_recall": float(best_row.get("macro_recall", np.nan)),
        "macro_f1": float(best_row.get("macro_f1", np.nan)),
        "top_2_accuracy": float(best_row.get("top_2_accuracy", np.nan)),
        "adjacent_accuracy": float(best_row.get("adjacent_accuracy", np.nan)),
        "severe_error_rate": float(best_row.get("severe_error_rate", np.nan)),
        "mean_absolute_ordinal_error": float(best_row.get("mean_absolute_ordinal_error", np.nan)),
        "model_path": str(best_model_path),
        "tflite_path": str(best_tflite_path) if best_tflite_path.exists() else "",
        "training_output_root": str(EXTENSION_OUTPUT_ROOT),
        "preprocessing_note": "Input images are preprocessed HSV/LAB-threshold segmented ROI images with neutralized background, expected to be 224x224 RGB before MobileNetV3Small preprocessing. No handcrafted RGB/HSV/LAB/GLCM feature branch is used.",
        "timestamp": datetime.now().isoformat(),
    }
    best_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "best_model_path": str(best_model_path),
        "best_tflite_path": str(best_tflite_path) if best_tflite_path.exists() else "",
        "best_metadata_path": str(best_metadata_path),
    }


def _summarize_model_metrics(metrics_df: pd.DataFrame, model_name: str) -> dict[str, object]:
    row = {"model_name": model_name, "runs": int(len(metrics_df))}
    for metric_name in [
        "accuracy",
        "macro_f1",
        "top_2_accuracy",
        "adjacent_accuracy",
        "severe_error_rate",
        "inference_mean_ms_per_image",
        "h5_size_mb",
        "tflite_size_mb",
    ]:
        if metric_name in metrics_df.columns:
            values = pd.to_numeric(metrics_df[metric_name], errors="coerce")
            row[f"{metric_name}_mean"] = float(values.mean())
            row[f"{metric_name}_std"] = float(values.std())
    for label_name in LABEL_ORDER:
        safe = label_name.replace(" ", "_")
        for metric_suffix in ["recall", "f1"]:
            col_name = f"{safe}_{metric_suffix}"
            if col_name in metrics_df.columns:
                values = pd.to_numeric(metrics_df[col_name], errors="coerce")
                row[f"{safe}_{metric_suffix}_mean"] = float(values.mean())
                row[f"{safe}_{metric_suffix}_std"] = float(values.std())
    return row


def create_processed_roi8_vs_previous_models_comparison() -> dict[str, object] | None:
    ensure_output_dirs()
    comparison_rows = []

    if HYBRID_SEED_METRICS_PATH.exists():
        comparison_rows.append(_summarize_model_metrics(pd.read_csv(HYBRID_SEED_METRICS_PATH), "segmented6_hybrid"))
    else:
        print(f"[WARN] Missing old result file: {HYBRID_SEED_METRICS_PATH}")

    if SEGMENTED6_CNN_ONLY_SEED_METRICS_PATH.exists():
        comparison_rows.append(_summarize_model_metrics(pd.read_csv(SEGMENTED6_CNN_ONLY_SEED_METRICS_PATH), "segmented6_cnn_only"))
    else:
        print(f"[WARN] Missing old result file: {SEGMENTED6_CNN_ONLY_SEED_METRICS_PATH}")

    if SEGMENTED6_SEED_METRICS_PATH.exists():
        comparison_rows.append(_summarize_model_metrics(pd.read_csv(SEGMENTED6_SEED_METRICS_PATH), "processed_roi8_cnn_only"))
    else:
        print(f"[WARN] Missing current result file: {SEGMENTED6_SEED_METRICS_PATH}")

    if not comparison_rows:
        return None

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(PROCESSED_ROI8_COMPARISON_PATH, index=False)
    return {
        "comparison_df": comparison_df,
        "comparison_csv_path": str(PROCESSED_ROI8_COMPARISON_PATH),
    }


def regenerate_metrics_summaries_and_graphs() -> dict[str, object] | None:
    ensure_output_dirs()
    if not SEGMENTED6_SEED_METRICS_PATH.exists():
        print("No processed ROI 8-fold training results found yet. Train first, then rerun this section.")
        return None
    seed_metrics_df = pd.read_csv(SEGMENTED6_SEED_METRICS_PATH).fillna("")
    if seed_metrics_df.empty:
        print("No processed ROI 8-fold training results found yet. Train first, then rerun this section.")
        return None

    prediction_df = load_prediction_csvs_for_metrics(seed_metrics_df)
    summary_bundle = save_processed_roi8_summary_outputs(seed_metrics_df, prediction_df=prediction_df)
    comparison_bundle = create_processed_roi8_vs_previous_models_comparison()
    return {
        "seed_metrics_df": seed_metrics_df,
        "prediction_df": prediction_df,
        "summary_bundle": summary_bundle,
        "comparison_bundle": comparison_bundle,
    }


def load_prediction_csvs_for_metrics(seed_metrics_df: pd.DataFrame) -> pd.DataFrame:
    with patched_base_config():
        return base.load_prediction_csvs_for_metrics(seed_metrics_df)


def run_8fold_cnn_only_training() -> dict[str, object]:
    ensure_output_dirs()
    split_dfs = load_all_cross_rotation_splits()
    combined_df = combined_splits_dataframe(split_dfs)
    missing_image_path_rows = int(combined_df["missing_image_path"].sum())
    if missing_image_path_rows > 0:
        raise RuntimeError(
            f"Cannot train because {missing_image_path_rows} rows still have unresolved image paths."
        )

    existing_metrics_df = load_existing_metrics()
    seed_results = existing_metrics_df.to_dict(orient="records") if not existing_metrics_df.empty else []

    for fold_name in EXTENSION_FOLDS:
        for seed in EXTENSION_RUN_SEEDS:
            if should_skip_run(existing_metrics_df, fold_name, seed):
                print(f"Skipping completed run: {fold_name} seed={seed}")
                continue
            print(f"Running processed ROI 8-fold CNN-only experiment: {fold_name} seed={seed}")
            try:
                result = run_single_8fold_cnn_only_experiment(
                    fold_name=fold_name,
                    seed=seed,
                    split_dfs=split_dfs,
                )
                seed_results.append(result)
                existing_metrics_df = pd.DataFrame(seed_results)
                if not existing_metrics_df.empty:
                    existing_metrics_df = existing_metrics_df.sort_values(["fold_name", "seed"]).reset_index(drop=True)
                existing_metrics_df.to_csv(SEGMENTED6_SEED_METRICS_PATH, index=False)
            except Exception as exc:
                append_failed_run(fold_name, seed, exc)
                print(f"[WARN] Failed run {fold_name} seed={seed}: {exc}")
            finally:
                if TF_AVAILABLE:
                    tf.keras.backend.clear_session()
                gc.collect()

    seed_metrics_df = pd.DataFrame(seed_results)
    if not seed_metrics_df.empty:
        seed_metrics_df = seed_metrics_df.sort_values(["fold_name", "seed"]).reset_index(drop=True)
    seed_metrics_df.to_csv(SEGMENTED6_SEED_METRICS_PATH, index=False)

    prediction_df = load_prediction_csvs_for_metrics(seed_metrics_df)
    summary_bundle = save_processed_roi8_summary_outputs(seed_metrics_df, prediction_df=prediction_df)
    best_bundle = copy_best_processed_roi8_artifacts(seed_metrics_df)
    comparison_bundle = create_processed_roi8_vs_previous_models_comparison()
    return {
        "seed_metrics_df": seed_metrics_df,
        "prediction_df": prediction_df,
        "summary_bundle": summary_bundle,
        "best_bundle": best_bundle,
        "comparison_bundle": comparison_bundle,
    }
