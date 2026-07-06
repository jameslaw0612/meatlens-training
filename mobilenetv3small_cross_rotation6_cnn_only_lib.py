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
from PIL import Image

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

PROJECT_ROOT = Path.cwd()
TRAINING_OUTPUTS_ROOT = PROJECT_ROOT / "training_outputs"
SEGMENTED_SPLITS_ROOT = PROJECT_ROOT / "generated_splits" / "cross_rotation_6fold"
CROSS_ROTATION_ROOT = SEGMENTED_SPLITS_ROOT

EXTENSION_OUTPUT_ROOT = TRAINING_OUTPUTS_ROOT / "mobilenetv3small_cross_rotation6_cnn_only_centercrop"
EXTENSION_FIGURES_ROOT = EXTENSION_OUTPUT_ROOT / "figures"
EXTENSION_MODELS_ROOT = EXTENSION_OUTPUT_ROOT / "models"
EXTENSION_PREDICTIONS_ROOT = EXTENSION_OUTPUT_ROOT / "predictions"
EXTENSION_GRADCAM_ROOT = EXTENSION_OUTPUT_ROOT / "gradcam"

EXTENSION_BACKBONE = "MobileNetV3Small"
EXTENSION_SPLIT_MODE = "cross_rotation"
EXTENSION_FOLDS = ["fold1", "fold2", "fold3", "fold4", "fold5", "fold6"]
EXTENSION_RUN_SEEDS = [42, 123, 2026]

MODEL_INPUT_MODE = "cnn_only"
IMAGE_CROP_MODE = "center_square_crop_resize_224"
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

LABEL_ORDER = ["fresh", "not fresh", "spoiled"]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LABEL_ORDER)}
INDEX_TO_LABEL = {idx: label for label, idx in LABEL_TO_INDEX.items()}

SEGMENTED6_SEED_METRICS_PATH = EXTENSION_OUTPUT_ROOT / "cross_rotation6_cnn_only_centercrop_seed_metrics.csv"
SEGMENTED6_FOLD_SUMMARY_PATH = EXTENSION_OUTPUT_ROOT / "cross_rotation6_cnn_only_centercrop_fold_summary.csv"
SEGMENTED6_SAMPLE_SUMMARY_PATH = EXTENSION_OUTPUT_ROOT / "cross_rotation6_cnn_only_centercrop_sample_summary.csv"
SEGMENTED6_CUT_SUMMARY_PATH = EXTENSION_OUTPUT_ROOT / "cross_rotation6_cnn_only_centercrop_cut_summary.csv"
SEGMENTED6_CAPTURE_SOURCE_SUMMARY_PATH = EXTENSION_OUTPUT_ROOT / "cross_rotation6_cnn_only_centercrop_capture_source_summary.csv"
SEGMENTED6_PHONE_GROUP_SUMMARY_PATH = EXTENSION_OUTPUT_ROOT / "cross_rotation6_cnn_only_centercrop_phone_group_summary.csv"
SEGMENTED6_PER_CLASS_SUMMARY_PATH = EXTENSION_OUTPUT_ROOT / "cross_rotation6_cnn_only_centercrop_per_class_summary.csv"
SEGMENTED6_PRED_DISTRIBUTION_PATH = EXTENSION_OUTPUT_ROOT / "cross_rotation6_cnn_only_centercrop_prediction_distribution.csv"
SEGMENTED6_SIZE_SPEED_PATH = EXTENSION_OUTPUT_ROOT / "cross_rotation6_cnn_only_centercrop_model_size_and_speed.csv"
SEGMENTED6_FAILED_RUNS_PATH = EXTENSION_OUTPUT_ROOT / "failed_cross_rotation6_cnn_only_centercrop_runs.csv"
SEGMENTED6_QUALITY_SUMMARY_PATH = EXTENSION_OUTPUT_ROOT / "cross_rotation6_dataset_quality_summary.csv"
SEGMENTED6_COMPARISON_PATH = EXTENSION_OUTPUT_ROOT / "cross_rotation6_hybrid_vs_cnn_only_centercrop_comparison.csv"

HYBRID_SEED_METRICS_PATH = (
    TRAINING_OUTPUTS_ROOT
    / "mobilenetv3small_segmented6_hybrid"
    / "segmented6_hybrid_seed_metrics.csv"
)

PRIMARY_RUN_METRICS = list(base.PRIMARY_RUN_METRICS)
SECONDARY_RUN_METRICS = list(base.SECONDARY_RUN_METRICS)

SAMPLE_METADATA = dict(base.SAMPLE_METADATA)
SAMPLE_NUMBER_TO_ID = dict(base.SAMPLE_NUMBER_TO_ID)
IMAGE_PATH_CANDIDATES = list(base.IMAGE_PATH_CANDIDATES)
TOP_CONFIDENCE_BORDERLINE = base.TOP_CONFIDENCE_BORDERLINE
TOP_CONFIDENCE_LOW = base.TOP_CONFIDENCE_LOW
LOADER_STATS = base.LOADER_STATS


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


def load_preprocessed_segmented_roi_224(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = Image.open(path).convert("RGB")
    original_np = np.array(img, dtype=np.uint8)

    if original_np.shape == (224, 224, 3):
        image_uint8 = original_np
        LOADER_STATS["already_224"] += 1
    else:
        image_uint8 = np.array(base.preprocess_center_square_resize_224(img), dtype=np.uint8)
        LOADER_STATS["resized"] += 1
        if LOADER_STATS["warning_count"] < 10:
            print(f"[WARN] Center-cropped and resized non-224 image: {path} | original shape={original_np.shape}")
            LOADER_STATS["warning_count"] += 1

    image_float = image_uint8.astype(np.float32)
    return original_np, image_uint8, image_float


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
            "mobilenetv3small_cross_rotation6_cnn_only_lib.py code is used."
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
    return f"cross_rotation6_cnn_only_centercrop_{fold_name}_seed{seed}"


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

    best_model_path = EXTENSION_MODELS_ROOT / "meatlens_best_cross_rotation6_cnn_only_centercrop_mobilenetv3small.h5"
    best_tflite_path = EXTENSION_MODELS_ROOT / "meatlens_best_cross_rotation6_cnn_only_centercrop_mobilenetv3small.tflite"
    best_metadata_path = EXTENSION_MODELS_ROOT / "meatlens_best_cross_rotation6_cnn_only_centercrop_mobilenetv3small_metadata.json"

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
