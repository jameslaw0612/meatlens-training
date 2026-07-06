#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
import math
import random
import shutil
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError(f"matplotlib is required: {exc}")

try:
    import seaborn as sns

    SEABORN_AVAILABLE = True
except Exception:
    sns = None
    SEABORN_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras import callbacks as keras_callbacks
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import MobileNetV3Small
    from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as preprocess_mobilenetv3

    TF_AVAILABLE = True
    try:
        tf.config.optimizer.set_jit(False)
        print("[INFO] TensorFlow XLA/JIT disabled.")
    except Exception as exc:
        print(f"[WARN] Could not disable XLA/JIT: {exc}")
except Exception:
    tf = None
    keras_callbacks = None
    layers = None
    models = None
    MobileNetV3Small = None
    preprocess_mobilenetv3 = None
    TF_AVAILABLE = False

try:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.class_weight import compute_class_weight

    SKLEARN_AVAILABLE = True
except Exception:
    accuracy_score = None
    classification_report = None
    confusion_matrix = None
    precision_recall_fscore_support = None
    StandardScaler = None
    compute_class_weight = None
    SKLEARN_AVAILABLE = False

try:
    import joblib

    JOBLIB_AVAILABLE = True
except Exception:
    joblib = None
    JOBLIB_AVAILABLE = False

try:
    import cv2

    CV2_AVAILABLE = True
except Exception:
    cv2 = None
    CV2_AVAILABLE = False

try:
    from skimage import color as skcolor
    from skimage.feature import graycomatrix, graycoprops

    SKIMAGE_AVAILABLE = True
except Exception:
    skcolor = None
    graycomatrix = None
    graycoprops = None
    SKIMAGE_AVAILABLE = False

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path.cwd()
TRAINING_OUTPUTS_ROOT = PROJECT_ROOT / "training_outputs"
SEGMENTED_SPLITS_ROOT = PROJECT_ROOT / "generated_splits" / "cross_rotation_6fold_segmented"
CROSS_ROTATION_ROOT = SEGMENTED_SPLITS_ROOT

EXTENSION_OUTPUT_ROOT = TRAINING_OUTPUTS_ROOT / "mobilenetv3small_segmented6_hybrid"
EXTENSION_FIGURES_ROOT = EXTENSION_OUTPUT_ROOT / "figures"
EXTENSION_MODELS_ROOT = EXTENSION_OUTPUT_ROOT / "models"
EXTENSION_FEATURES_ROOT = EXTENSION_OUTPUT_ROOT / "features"
EXTENSION_GRADCAM_ROOT = EXTENSION_OUTPUT_ROOT / "gradcam"
EXTENSION_PREDICTIONS_ROOT = EXTENSION_OUTPUT_ROOT / "predictions"

EXTENSION_BACKBONE = "MobileNetV3Small"
EXTENSION_SPLIT_MODE = "cross_rotation"
EXTENSION_FOLDS = ["fold1", "fold2", "fold3", "fold4", "fold5", "fold6"]
EXTENSION_RUN_SEEDS = [42, 123, 2026]
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
WEIGHT_DECAY = 1e-4

MODEL_INPUT_MODE = "cnn_plus_color_texture"
IMAGE_CROP_MODE = "preprocessed_segmented_roi_224"

LABEL_ORDER = ["fresh", "not fresh", "spoiled"]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LABEL_ORDER)}
INDEX_TO_LABEL = {idx: label for label, idx in LABEL_TO_INDEX.items()}
GLCM_DISTANCES = [1, 2, 4]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

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
}

SAMPLE_NUMBER_TO_ID = {
    "1": "pork_shoulder_sample_1",
    "2": "pork_shoulder_sample_2",
    "3": "pork_belly_sample_3",
    "4": "pork_belly_sample_4",
    "5": "pork_ham_sample_5",
    "6": "pork_ham_sample_6",
}

IMAGE_PATH_CANDIDATES = [
    "file_destination",
    "image_path",
    "path",
    "filename",
    "file_path",
    "image_file_name",
]

SEGMENTED6_SEED_METRICS_PATH = EXTENSION_OUTPUT_ROOT / "segmented6_hybrid_seed_metrics.csv"
SEGMENTED6_FOLD_SUMMARY_PATH = EXTENSION_OUTPUT_ROOT / "segmented6_hybrid_fold_summary.csv"
SEGMENTED6_SAMPLE_SUMMARY_PATH = EXTENSION_OUTPUT_ROOT / "segmented6_hybrid_sample_summary.csv"
SEGMENTED6_CUT_SUMMARY_PATH = EXTENSION_OUTPUT_ROOT / "segmented6_hybrid_cut_summary.csv"
SEGMENTED6_CAPTURE_SOURCE_SUMMARY_PATH = EXTENSION_OUTPUT_ROOT / "segmented6_hybrid_capture_source_summary.csv"
SEGMENTED6_PHONE_GROUP_SUMMARY_PATH = EXTENSION_OUTPUT_ROOT / "segmented6_hybrid_phone_group_summary.csv"
SEGMENTED6_PER_CLASS_SUMMARY_PATH = EXTENSION_OUTPUT_ROOT / "segmented6_hybrid_per_class_summary.csv"
SEGMENTED6_PRED_DISTRIBUTION_PATH = EXTENSION_OUTPUT_ROOT / "segmented6_hybrid_prediction_distribution.csv"
SEGMENTED6_SIZE_SPEED_PATH = EXTENSION_OUTPUT_ROOT / "segmented6_hybrid_model_size_and_speed.csv"
SEGMENTED6_FAILED_RUNS_PATH = EXTENSION_OUTPUT_ROOT / "failed_segmented6_hybrid_runs.csv"
SEGMENTED6_QUALITY_SUMMARY_PATH = EXTENSION_OUTPUT_ROOT / "segmented6_dataset_quality_summary.csv"

PRIMARY_RUN_METRICS = [
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
]
SECONDARY_RUN_METRICS = [
    "top_2_accuracy",
    "adjacent_accuracy",
    "severe_error_rate",
    "mean_absolute_ordinal_error",
    "borderline_rate",
    "low_confidence_rate",
    "adjacent_pair_combined_confidence_mean",
    "non_adjacent_top2_rate",
]
PER_CLASS_METRICS = ["precision", "recall", "f1", "support"]
TOP_CONFIDENCE_BORDERLINE = 0.90
TOP_CONFIDENCE_LOW = 0.60

LOADER_STATS = {
    "already_224": 0,
    "resized": 0,
    "warning_count": 0,
}


def ensure_output_dirs() -> None:
    for path_obj in [
        EXTENSION_OUTPUT_ROOT,
        EXTENSION_FIGURES_ROOT,
        EXTENSION_MODELS_ROOT,
        EXTENSION_FEATURES_ROOT,
        EXTENSION_GRADCAM_ROOT,
        EXTENSION_PREDICTIONS_ROOT,
    ]:
        path_obj.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if TF_AVAILABLE:
        tf.random.set_seed(seed)


def detect_image_path_column(df: pd.DataFrame) -> str | None:
    for col_name in IMAGE_PATH_CANDIDATES:
        if col_name in df.columns:
            return col_name
    return None


def resolve_image_path(row: pd.Series, path_col: str, csv_path: Path | None = None) -> str | None:
    raw = row.get(path_col)
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    raw_str = str(raw).strip()
    if raw_str == "":
        return None

    candidates = [Path(raw_str)]
    if csv_path is not None:
        candidates.append(csv_path.parent / raw_str)
    candidates.append(PROJECT_ROOT / raw_str)

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


def enrich_split_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sample_id"] = out.apply(
        lambda row: normalize_sample_id(row.get("sample_id"), row.get("sample_number")),
        axis=1,
    )
    out["sample_number"] = out["sample_id"].map(
        lambda sample_id: SAMPLE_METADATA.get(sample_id, {}).get("sample_number", np.nan)
    ).fillna(out.get("sample_number", np.nan))
    out["pork_cut"] = out["sample_id"].map(
        lambda sample_id: SAMPLE_METADATA.get(sample_id, {}).get("pork_cut")
    ).fillna(out.get("pork_cut", "unknown"))
    out["capture_source"] = out["sample_id"].map(
        lambda sample_id: SAMPLE_METADATA.get(sample_id, {}).get("capture_source")
    ).fillna(out.get("capture_source", "unknown"))
    if "phone_group" in out.columns:
        out["phone_group"] = out["phone_group"].fillna(
            out["sample_id"].map(lambda sample_id: SAMPLE_METADATA.get(sample_id, {}).get("phone_group", ""))
        )
    else:
        out["phone_group"] = out["sample_id"].map(
            lambda sample_id: SAMPLE_METADATA.get(sample_id, {}).get("phone_group", "")
        )
    out["label"] = out["label"].astype(str).str.strip().str.lower()
    out["sample_number"] = out["sample_number"].astype(str).str.replace(".0", "", regex=False)
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
        raise ValueError(f"No image path column found in {csv_path}")
    df["image_path_source_column"] = path_col
    df["image_path_resolved"] = df.apply(
        lambda row: resolve_image_path(row, path_col=path_col, csv_path=csv_path),
        axis=1,
    )
    df = enrich_split_df(df)
    return df


def load_all_cross_rotation_splits() -> dict[str, pd.DataFrame]:
    split_dfs: dict[str, pd.DataFrame] = {}
    for fold_name in EXTENSION_FOLDS:
        for part in ["train", "val", "test"]:
            split_key = f"{fold_name}_{part}"
            split_dfs[split_key] = load_split_dataframe(
                CROSS_ROTATION_ROOT / f"{split_key}.csv",
                split_key=split_key,
            )
    return split_dfs


def combined_splits_dataframe(split_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(split_dfs.values(), ignore_index=True)


def validate_metadata_mapping(split_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    combined_df = combined_splits_dataframe(split_dfs)
    rows = []
    for sample_id, meta in SAMPLE_METADATA.items():
        subset = combined_df[combined_df["sample_id"] == sample_id]
        rows.append(
            {
                "sample_id": sample_id,
                "expected_sample_number": str(meta["sample_number"]),
                "observed_rows": int(len(subset)),
                "observed_sample_numbers": ",".join(sorted(subset["sample_number"].astype(str).unique().tolist())),
                "pork_cut": meta["pork_cut"],
                "capture_source": meta["capture_source"],
                "phone_group": meta.get("phone_group", ""),
                "missing_image_path_rows": int(subset["image_path_resolved"].isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def preprocess_center_square_resize_224(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    width, height = img.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    square = img.crop((left, top, left + side, top + side))
    return square.resize(TARGET_SIZE, Image.BILINEAR)


def reset_loader_stats() -> None:
    for key in LOADER_STATS:
        LOADER_STATS[key] = 0


def load_preprocessed_segmented_roi_224(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = Image.open(path).convert("RGB")
    original_np = np.array(img, dtype=np.uint8)

    if original_np.shape == (224, 224, 3):
        image_uint8 = original_np
        LOADER_STATS["already_224"] += 1
    else:
        image_uint8 = np.array(img.resize(TARGET_SIZE, Image.BILINEAR), dtype=np.uint8)
        LOADER_STATS["resized"] += 1
        if LOADER_STATS["warning_count"] < 10:
            print(f"[WARN] Resized non-224 image: {path} | original shape={original_np.shape}")
            LOADER_STATS["warning_count"] += 1
    image_float = image_uint8.astype(np.float32)
    return original_np, image_uint8, image_float


def get_hsv_channels(image_uint8: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hsv = np.array(Image.fromarray(image_uint8, mode="RGB").convert("HSV"), dtype=np.float32)
    return hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]


def get_lab_channels(image_uint8: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if SKIMAGE_AVAILABLE:
        lab = skcolor.rgb2lab(image_uint8 / 255.0).astype(np.float32)
        return lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    if CV2_AVAILABLE:
        lab = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
        l = lab[:, :, 0] * (100.0 / 255.0)
        a = lab[:, :, 1] - 128.0
        b = lab[:, :, 2] - 128.0
        return l, a, b
    return None, None, None


def extract_color_texture_features_from_uint8(image_uint8: np.ndarray) -> dict[str, float]:
    features: dict[str, float] = {}

    rgb_float = image_uint8.astype(np.float32)
    channel_names = ["r", "g", "b"]
    for idx, channel_name in enumerate(channel_names):
        features[f"rgb_{channel_name}_mean"] = float(rgb_float[:, :, idx].mean())
        features[f"rgb_{channel_name}_std"] = float(rgb_float[:, :, idx].std())

    h, s, v = get_hsv_channels(image_uint8)
    hsv_map = {"h": h, "s": s, "v": v}
    for channel_name, channel_values in hsv_map.items():
        features[f"hsv_{channel_name}_mean"] = float(channel_values.mean())
        features[f"hsv_{channel_name}_std"] = float(channel_values.std())

    l, a, b = get_lab_channels(image_uint8)
    if l is not None:
        lab_map = {"l": l, "a": a, "b": b}
        for channel_name, channel_values in lab_map.items():
            features[f"lab_{channel_name}_mean"] = float(channel_values.mean())
            features[f"lab_{channel_name}_std"] = float(channel_values.std())
    else:
        for channel_name in ["l", "a", "b"]:
            features[f"lab_{channel_name}_mean"] = np.nan
            features[f"lab_{channel_name}_std"] = np.nan

    if SKIMAGE_AVAILABLE:
        gray = np.array(Image.fromarray(image_uint8).convert("L"), dtype=np.uint8)
        glcm = graycomatrix(
            gray,
            distances=GLCM_DISTANCES,
            angles=GLCM_ANGLES,
            levels=256,
            symmetric=True,
            normed=True,
        )
        for prop_name in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
            prop = graycoprops(glcm, prop_name)
            features[f"glcm_{prop_name}_mean"] = float(np.mean(prop))
            features[f"glcm_{prop_name}_std"] = float(np.std(prop))
    else:
        for prop_name in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
            features[f"glcm_{prop_name}_mean"] = np.nan
            features[f"glcm_{prop_name}_std"] = np.nan

    return features


def feature_column_names() -> list[str]:
    rgb_cols = [f"rgb_{channel}_{stat}" for channel in ["r", "g", "b"] for stat in ["mean", "std"]]
    hsv_cols = [f"hsv_{channel}_{stat}" for channel in ["h", "s", "v"] for stat in ["mean", "std"]]
    lab_cols = [f"lab_{channel}_{stat}" for channel in ["l", "a", "b"] for stat in ["mean", "std"]]
    glcm_cols = [f"glcm_{prop}_{stat}" for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"] for stat in ["mean", "std"]]
    return rgb_cols + hsv_cols + lab_cols + glcm_cols


def extract_features_for_dataframe(
    df: pd.DataFrame,
    output_path: Path | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    rows = []
    valid_df = df[df["image_path_resolved"].notna()].copy().reset_index(drop=True)
    if limit is not None:
        valid_df = valid_df.head(limit).copy()

    for _, row in valid_df.iterrows():
        _, image_uint8, _ = load_preprocessed_segmented_roi_224(row["image_path_resolved"])
        record = {
            "image_file_name": row.get("image_file_name", ""),
            "sample_number": row.get("sample_number", ""),
            "meat_part": row.get("meat_part", ""),
            "label": row.get("label", ""),
            "time_frame": row.get("time_frame", ""),
            "file_destination": row.get("file_destination", ""),
            "sample_id": row.get("sample_id", ""),
            "pork_cut": row.get("pork_cut", ""),
            "capture_source": row.get("capture_source", ""),
            "phone_group": row.get("phone_group", ""),
            "split_name": row.get("split_name", ""),
            "dataset_partition": row.get("dataset_partition", ""),
            "image_path_resolved": row.get("image_path_resolved", ""),
        }
        record.update(extract_color_texture_features_from_uint8(image_uint8))
        rows.append(record)

    features_df = pd.DataFrame(rows)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_csv(output_path, index=False)
    return features_df


def create_or_load_feature_csv(split_key: str, df: pd.DataFrame, force_recompute: bool = False) -> Path:
    out_path = EXTENSION_FEATURES_ROOT / f"{split_key}_segmented_roi_color_texture_features.csv"
    if out_path.exists() and not force_recompute:
        return out_path
    extract_features_for_dataframe(df, output_path=out_path, limit=None)
    return out_path


def run_feature_extraction_smoke_test(split_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    combined_df = combined_splits_dataframe(split_dfs)
    combined_df = combined_df[combined_df["image_path_resolved"].notna()].drop_duplicates("image_path_resolved")
    sample_df = combined_df.head(3).copy()
    out_path = EXTENSION_FEATURES_ROOT / "example_3_segmented_roi_color_texture_features.csv"
    return extract_features_for_dataframe(sample_df, output_path=out_path, limit=3)


def count_table_to_rows(df: pd.DataFrame, section_name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].astype(str)
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        rows.append({"section": section_name, **row_dict})
    return rows


def run_split_integrity_checks(split_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for fold_name in EXTENSION_FOLDS:
        train_df = split_dfs[f"{fold_name}_train"]
        val_df = split_dfs[f"{fold_name}_val"]
        test_df = split_dfs[f"{fold_name}_test"]

        held_out_samples = sorted(test_df["sample_id"].dropna().astype(str).unique().tolist())
        held_out_sample = held_out_samples[0] if held_out_samples else ""

        train_ids = set(train_df["image_path_resolved"].dropna().astype(str))
        val_ids = set(val_df["image_path_resolved"].dropna().astype(str))
        test_ids = set(test_df["image_path_resolved"].dropna().astype(str))

        rows.append(
            {
                "fold": fold_name,
                "held_out_sample": held_out_sample,
                "test_contains_held_out_sample": bool((test_df["sample_id"] == held_out_sample).any()),
                "held_out_absent_in_train": not bool((train_df["sample_id"] == held_out_sample).any()),
                "held_out_absent_in_val": not bool((val_df["sample_id"] == held_out_sample).any()),
                "train_val_overlap": len(train_ids & val_ids),
                "train_test_overlap": len(train_ids & test_ids),
                "val_test_overlap": len(val_ids & test_ids),
            }
        )
    return pd.DataFrame(rows)


def build_dataset_quality_summary(split_dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    combined_df = combined_splits_dataframe(split_dfs)
    unique_images_df = combined_df.drop_duplicates("image_path_resolved").copy()

    sample_counts = unique_images_df.groupby("sample_id").size().reset_index(name="count")
    label_counts = unique_images_df.groupby("label").size().reset_index(name="count")
    sample_label_counts = unique_images_df.groupby(["sample_id", "label"]).size().reset_index(name="count")
    fold_partition_label_counts = (
        combined_df.groupby(["split_name", "dataset_partition", "label"]).size().reset_index(name="count")
    )
    integrity_df = run_split_integrity_checks(split_dfs)

    reset_loader_stats()
    for image_path in unique_images_df["image_path_resolved"].dropna().tolist():
        load_preprocessed_segmented_roi_224(image_path)
    shape_df = pd.DataFrame(
        [
            {"metric_name": "already_224_count", "metric_value": int(LOADER_STATS["already_224"])},
            {"metric_name": "resized_count", "metric_value": int(LOADER_STATS["resized"])},
            {"metric_name": "unique_images_checked", "metric_value": int(len(unique_images_df))},
        ]
    )

    summary_rows: list[dict[str, object]] = []
    summary_rows.extend(count_table_to_rows(sample_counts, "count_per_sample_id"))
    summary_rows.extend(count_table_to_rows(label_counts, "count_per_label"))
    summary_rows.extend(count_table_to_rows(sample_label_counts, "count_per_sample_id_label"))
    summary_rows.extend(count_table_to_rows(fold_partition_label_counts, "count_per_fold_partition_label"))
    summary_rows.extend(count_table_to_rows(integrity_df, "split_integrity"))
    summary_rows.extend(count_table_to_rows(shape_df, "image_shape_audit"))
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SEGMENTED6_QUALITY_SUMMARY_PATH, index=False)

    resized_count = int(LOADER_STATS["resized"])
    if resized_count > 0:
        print("[WARN] Some segmented ROI images were not 224x224 and were resized. Check preprocessing consistency.")

    return {
        "sample_counts": sample_counts,
        "label_counts": label_counts,
        "sample_label_counts": sample_label_counts,
        "fold_partition_label_counts": fold_partition_label_counts,
        "integrity_df": integrity_df,
        "shape_df": shape_df,
        "summary_df": summary_df,
    }


def select_visual_examples(unique_images_df: pd.DataFrame, max_items: int = 18) -> pd.DataFrame:
    picks = []
    grouped = unique_images_df.groupby(["sample_id", "label"], sort=True)
    for _, group_df in grouped:
        picks.append(group_df.iloc[0])
    if len(picks) == 0:
        return unique_images_df.head(0).copy()
    picks_df = pd.DataFrame(picks).reset_index(drop=True)
    return picks_df.head(max_items)


def save_sample_visualization(split_dfs: dict[str, pd.DataFrame]) -> Path:
    combined_df = combined_splits_dataframe(split_dfs)
    unique_images_df = combined_df[combined_df["image_path_resolved"].notna()].drop_duplicates("image_path_resolved")
    picks_df = select_visual_examples(unique_images_df, max_items=18)
    if picks_df.empty:
        raise ValueError("No valid image paths were found for sample visualization.")

    n = len(picks_df)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, max(5, rows * 4)))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for ax in axes.flatten():
        ax.axis("off")

    for idx, (_, row) in enumerate(picks_df.iterrows()):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        _, image_uint8, _ = load_preprocessed_segmented_roi_224(row["image_path_resolved"])
        ax.imshow(image_uint8)
        ax.set_title(f"{row['sample_id']}\n{row['label']}", fontsize=10)
        ax.axis("off")

    fig.suptitle("Segmented ROI Sample Images", y=0.98)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = EXTENSION_FIGURES_ROOT / "segmented_roi_sample_images.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def print_quality_tables(quality_bundle: dict[str, pd.DataFrame]) -> None:
    print("Count per sample_id")
    print(quality_bundle["sample_counts"].to_string(index=False))
    print()
    print("Count per label")
    print(quality_bundle["label_counts"].to_string(index=False))
    print()
    print("Count per sample_id x label")
    print(quality_bundle["sample_label_counts"].to_string(index=False))
    print()
    print("Count per fold x partition x label")
    print(quality_bundle["fold_partition_label_counts"].to_string(index=False))
    print()
    print("Split integrity")
    print(quality_bundle["integrity_df"].to_string(index=False))
    print()
    print("Image shape audit")
    print(quality_bundle["shape_df"].to_string(index=False))


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


def build_image_array(df: pd.DataFrame) -> np.ndarray:
    arrays = []
    for path in df["image_path_resolved"].tolist():
        _, _, image_float = load_preprocessed_segmented_roi_224(path)
        arrays.append(preprocess_mobilenetv3(image_float.copy()))
    return np.stack(arrays).astype(np.float32)


def build_image_array_from_paths(paths: list[str]) -> np.ndarray:
    arrays = []
    for path in paths:
        _, _, image_float = load_preprocessed_segmented_roi_224(path)
        arrays.append(preprocess_mobilenetv3(image_float.copy()))
    return np.stack(arrays).astype(np.float32)


def estimate_array_memory_mb(array: np.ndarray) -> float:
    return float(array.nbytes / (1024 * 1024))


class HybridImageFeatureSequence(tf.keras.utils.Sequence if TF_AVAILABLE else object):
    def __init__(
        self,
        image_paths: list[str],
        features: np.ndarray,
        labels: np.ndarray | None = None,
        batch_size: int = BATCH_SIZE,
        shuffle: bool = False,
    ):
        self.image_paths = list(image_paths)
        self.features = np.asarray(features, dtype=np.float32)
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
        batch_features = self.features[batch_indices]
        if self.labels is None:
            return {
                "image_input": batch_images,
                "feature_input": batch_features,
            }
        batch_labels = self.labels[batch_indices]
        return (batch_images, batch_features), batch_labels

    def on_epoch_end(self):
        if self.shuffle and len(self.indices) > 0:
            np.random.shuffle(self.indices)


def merge_features_into_split_df(split_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    merge_cols = [
        "image_path_resolved",
        "sample_id",
        "label",
        "dataset_partition",
        "split_name",
    ]
    feature_cols = merge_cols + feature_column_names()
    available_cols = [col for col in feature_cols if col in features_df.columns]
    merged_df = split_df.merge(features_df[available_cols], on=merge_cols, how="left")
    return merged_df


def prepare_feature_bundle_for_fold(
    fold_name: str,
    split_dfs: dict[str, pd.DataFrame],
    force_recompute: bool = False,
) -> dict[str, object]:
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is required for feature scaling and training.")

    bundle: dict[str, object] = {}
    feature_cols = feature_column_names()

    for part in ["train", "val", "test"]:
        split_key = f"{fold_name}_{part}"
        split_df = split_dfs[split_key].copy()
        feature_csv_path = create_or_load_feature_csv(split_key, split_df, force_recompute=force_recompute)
        features_df = pd.read_csv(feature_csv_path)
        merged_df = merge_features_into_split_df(split_df, features_df)
        bundle[f"{part}_df"] = merged_df

    train_df = bundle["train_df"]
    val_df = bundle["val_df"]
    test_df = bundle["test_df"]

    train_feature_df = train_df[feature_cols].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    val_feature_df = val_df[feature_cols].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    test_feature_df = test_df[feature_cols].apply(pd.to_numeric, errors="coerce").astype(np.float32)

    feature_fill_values = train_feature_df.median(axis=0, numeric_only=True)
    feature_fill_values = feature_fill_values.fillna(0.0).astype(np.float32)

    train_feature_df = train_feature_df.fillna(feature_fill_values)
    val_feature_df = val_feature_df.fillna(feature_fill_values)
    test_feature_df = test_feature_df.fillna(feature_fill_values)

    feature_fill_values_path = EXTENSION_FEATURES_ROOT / f"{fold_name}_feature_fill_values.csv"
    feature_fill_values.rename("fill_value").reset_index().rename(columns={"index": "feature_name"}).to_csv(
        feature_fill_values_path,
        index=False,
    )

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_feature_df)
    val_features = scaler.transform(val_feature_df)
    test_features = scaler.transform(test_feature_df)

    bundle.update(
        {
            "feature_columns": feature_cols,
            "scaler": scaler,
            "feature_fill_values": feature_fill_values,
            "feature_fill_values_path": str(feature_fill_values_path),
            "train_features": train_features.astype(np.float32),
            "val_features": val_features.astype(np.float32),
            "test_features": test_features.astype(np.float32),
            "train_image_paths": train_df["image_path_resolved"].astype(str).tolist(),
            "val_image_paths": val_df["image_path_resolved"].astype(str).tolist(),
            "test_image_paths": test_df["image_path_resolved"].astype(str).tolist(),
            "train_labels": train_df["label"].map(LABEL_TO_INDEX).astype(int).to_numpy(),
            "val_labels": val_df["label"].map(LABEL_TO_INDEX).astype(int).to_numpy(),
            "test_labels": test_df["label"].map(LABEL_TO_INDEX).astype(int).to_numpy(),
        }
    )
    return bundle


class ValMacroF1Callback(keras_callbacks.Callback if TF_AVAILABLE else object):
    def __init__(self, val_source, val_labels: np.ndarray):
        super().__init__()
        self.val_source = val_source
        self.val_labels = val_labels
        self.best_f1 = -np.inf

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        preds = self.model.predict(self.val_source, verbose=0)
        pred_labels = preds.argmax(axis=1)
        _, _, f1s, _ = precision_recall_fscore_support(
            self.val_labels,
            pred_labels,
            labels=list(range(NUM_CLASSES)),
            average=None,
            zero_division=0,
        )
        macro_f1 = float(np.mean(f1s))
        logs["val_f1_macro"] = macro_f1
        self.best_f1 = max(self.best_f1, macro_f1)
        print(f" - val_f1_macro: {macro_f1:.4f}")


def make_optimizer(learning_rate: float):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required.")

    legacy_optimizers = getattr(tf.keras.optimizers, "legacy", None)

    if legacy_optimizers is not None and hasattr(legacy_optimizers, "Adam"):
        print("[INFO] Using tf.keras.optimizers.legacy.Adam for DirectML compatibility.")
        return legacy_optimizers.Adam(learning_rate=learning_rate)

    print("[INFO] Using tf.keras.optimizers.Adam fallback.")
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)


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


def assert_directml_safe_compilation(model) -> None:
    optimizer_type = type(model.optimizer)
    optimizer_module = optimizer_type.__module__
    print(f"[INFO] Compiled optimizer: {optimizer_module}.{optimizer_type.__name__}")
    if "optimizer_experimental" in optimizer_module:
        raise RuntimeError(
            "DirectML-unsafe optimizer detected in the compiled model. "
            "Restart the kernel and rerun the notebook import/reload cell so the latest "
            "mobilenetv3small_segmented6_hybrid_lib.py code is used."
        )


def build_segmented6_hybrid_model(num_features: int) -> tf.keras.Model:
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required for model building.")

    image_input = layers.Input(shape=INPUT_SHAPE, name="image_input")
    feature_input = layers.Input(shape=(num_features,), name="feature_input")

    backbone = MobileNetV3Small(
        include_top=False,
        weights="imagenet",
        input_shape=INPUT_SHAPE,
    )
    backbone._name = "mobilenetv3small_backbone"
    backbone.trainable = False

    x_img = backbone(image_input, training=False)
    x_img = layers.GlobalAveragePooling2D(name="image_gap")(x_img)
    x_img = layers.Dropout(0.30, name="image_dropout")(x_img)

    x_feat = layers.Dense(64, activation="relu", name="feature_dense_64")(feature_input)
    x_feat = layers.Dropout(0.20, name="feature_dropout")(x_feat)

    fused = layers.Concatenate(name="fusion_concat")([x_img, x_feat])
    fused = layers.Dense(128, activation="relu", name="fusion_dense_128")(fused)
    fused = layers.Dropout(0.30, name="fusion_dropout")(fused)
    output = layers.Dense(NUM_CLASSES, activation="softmax", name="classification_head")(fused)

    model = models.Model(inputs=[image_input, feature_input], outputs=output, name="meatlens_segmented6_hybrid")
    return model


def safe_load_weights(model: tf.keras.Model, checkpoint_path: Path) -> bool:
    try:
        model.load_weights(str(checkpoint_path))
        print(f"[INFO] Loaded checkpoint weights from: {checkpoint_path}")
        return True
    except Exception as exc:
        print(f"[WARN] Could not load checkpoint weights from {checkpoint_path}: {exc}")
        print("[WARN] Continuing with current in-memory model weights.")
        return False


def freeze_batchnorm_layers(model: tf.keras.Model) -> None:
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        if hasattr(layer, "layers"):
            freeze_batchnorm_layers(layer)


def unfreeze_top_backbone_fraction(model: tf.keras.Model, fraction: float = 0.25) -> None:
    backbone = model.get_layer("mobilenetv3small_backbone")
    total_layers = len(backbone.layers)
    start_idx = max(0, int(total_layers * (1.0 - fraction)))
    for idx, layer in enumerate(backbone.layers):
        layer.trainable = idx >= start_idx
    freeze_batchnorm_layers(backbone)


def compute_class_weights_from_labels(y_train: np.ndarray) -> dict[int, float]:
    if not SKLEARN_AVAILABLE:
        return {idx: 1.0 for idx in range(NUM_CLASSES)}
    unique_classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=unique_classes, y=y_train)
    return {int(class_id): float(weight) for class_id, weight in zip(unique_classes, weights)}


def make_run_stem(fold_name: str, seed: int) -> str:
    return f"segmented6_hybrid_{fold_name}_seed{seed}"


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


def load_existing_metrics() -> pd.DataFrame:
    if SEGMENTED6_SEED_METRICS_PATH.exists():
        return pd.read_csv(SEGMENTED6_SEED_METRICS_PATH)
    return pd.DataFrame()


def should_skip_run(metrics_df: pd.DataFrame, fold_name: str, seed: int) -> bool:
    if not metrics_row_exists(metrics_df, fold_name, seed):
        return False
    row = metrics_df[
        (metrics_df["fold_name"].astype(str) == str(fold_name))
        & (metrics_df["seed"].astype(str) == str(seed))
        & (metrics_df["model_input_mode"].astype(str) == MODEL_INPUT_MODE)
        & (metrics_df["image_crop_mode"].astype(str) == IMAGE_CROP_MODE)
    ].iloc[-1]
    model_path = str(row.get("model_path", "")).strip()
    return model_path != "" and Path(model_path).exists()


def is_adjacent_class_pair(class_a: int, class_b: int) -> bool:
    return abs(int(class_a) - int(class_b)) == 1


def class_pair_is_non_adjacent_extremes(class_a: int, class_b: int) -> bool:
    return {int(class_a), int(class_b)} == {0, 2}


def freshness_score_to_band(score: float) -> str:
    if score >= 70.0:
        return "Good for Consumption"
    if score >= 40.0:
        return "Consume Immediately"
    return "Not Suitable"


def compute_freshness_score(prob_fresh: float, prob_not_fresh: float, prob_spoiled: float) -> float:
    score = (prob_fresh * 100.0) + (prob_not_fresh * 55.0) + (prob_spoiled * 10.0)
    return float(np.clip(score, 0.0, 100.0))


def transition_label_and_recommendation(
    top_class: str,
    top_confidence: float,
    second_class: str,
) -> tuple[str, str]:
    class_pair = {top_class, second_class}
    if top_confidence >= TOP_CONFIDENCE_BORDERLINE:
        if top_class == "fresh":
            return "Likely Fresh", "Good for Consumption"
        if top_class == "not fresh":
            return "Likely Not Fresh", "Consume Immediately / Manual inspection recommended"
        return "Likely Spoiled", "Not Suitable"

    if class_pair == {"fresh", "not fresh"}:
        return "Borderline Fresh / Early Not Fresh", "Consume Immediately / Check carefully"
    if class_pair == {"not fresh", "spoiled"}:
        return "Borderline Spoiled / Advanced Not Fresh", "Not Suitable / Manual inspection recommended"
    if class_pair == {"fresh", "spoiled"}:
        return "Uncertain / Non-adjacent Class Conflict", "No reliable result / Retake image"
    return "Uncertain / Non-adjacent Class Conflict", "No reliable result / Retake image"


def build_transition_prediction_dataframe(
    base_df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> pd.DataFrame:
    out_df = base_df.copy().reset_index(drop=True)
    sorted_indices = np.argsort(-y_prob, axis=1)
    top_indices = sorted_indices[:, 0]
    second_indices = sorted_indices[:, 1]

    prob_fresh = y_prob[:, LABEL_TO_INDEX["fresh"]]
    prob_not_fresh = y_prob[:, LABEL_TO_INDEX["not fresh"]]
    prob_spoiled = y_prob[:, LABEL_TO_INDEX["spoiled"]]

    top_confidence = y_prob[np.arange(len(y_prob)), top_indices]
    second_confidence = y_prob[np.arange(len(y_prob)), second_indices]
    prediction_margin = top_confidence - second_confidence
    top2_combined_confidence = top_confidence + second_confidence
    ordinal_error = np.abs(y_true - top_indices)

    top2_correct = np.array(
        [int(true_idx in row_top2) for true_idx, row_top2 in zip(y_true, sorted_indices[:, :2])],
        dtype=int,
    )
    adjacent_correct = (ordinal_error <= 1).astype(int)
    severe_error = (ordinal_error == 2).astype(int)
    exact_correct = (ordinal_error == 0).astype(int)
    borderline = (top_confidence < TOP_CONFIDENCE_BORDERLINE).astype(int)
    low_confidence = (top_confidence < TOP_CONFIDENCE_LOW).astype(int)
    top2_adjacent_pair = np.array(
        [int(is_adjacent_class_pair(a, b)) for a, b in zip(top_indices, second_indices)],
        dtype=int,
    )
    non_adjacent_top2 = np.array(
        [
            int((top_confidence[idx] < TOP_CONFIDENCE_BORDERLINE) and class_pair_is_non_adjacent_extremes(top_indices[idx], second_indices[idx]))
            for idx in range(len(y_prob))
        ],
        dtype=int,
    )

    transition_labels = []
    recommendations = []
    freshness_scores = []
    freshness_bands = []
    for idx in range(len(out_df)):
        top_class = INDEX_TO_LABEL[int(top_indices[idx])]
        second_class = INDEX_TO_LABEL[int(second_indices[idx])]
        transition_label, recommendation = transition_label_and_recommendation(
            top_class=top_class,
            top_confidence=float(top_confidence[idx]),
            second_class=second_class,
        )
        transition_labels.append(transition_label)
        recommendations.append(recommendation)
        freshness_score = compute_freshness_score(
            float(prob_fresh[idx]),
            float(prob_not_fresh[idx]),
            float(prob_spoiled[idx]),
        )
        freshness_scores.append(freshness_score)
        freshness_bands.append(freshness_score_to_band(freshness_score))

    out_df["true_label"] = [INDEX_TO_LABEL[int(idx)] for idx in y_true]
    out_df["predicted_label"] = [INDEX_TO_LABEL[int(idx)] for idx in top_indices]
    out_df["prob_fresh"] = prob_fresh
    out_df["prob_not_fresh"] = prob_not_fresh
    out_df["prob_spoiled"] = prob_spoiled
    out_df["top_class"] = [INDEX_TO_LABEL[int(idx)] for idx in top_indices]
    out_df["top_confidence"] = top_confidence
    out_df["second_class"] = [INDEX_TO_LABEL[int(idx)] for idx in second_indices]
    out_df["second_confidence"] = second_confidence
    out_df["top2_combined_confidence"] = top2_combined_confidence
    out_df["prediction_margin"] = prediction_margin
    out_df["ordinal_error"] = ordinal_error
    out_df["is_exact_correct"] = exact_correct.astype(bool)
    out_df["is_top2_correct"] = top2_correct.astype(bool)
    out_df["is_adjacent_correct"] = adjacent_correct.astype(bool)
    out_df["is_severe_error"] = severe_error.astype(bool)
    out_df["is_borderline"] = borderline.astype(bool)
    out_df["is_low_confidence"] = low_confidence.astype(bool)
    out_df["transition_label"] = transition_labels
    out_df["recommendation"] = recommendations
    out_df["freshness_score"] = np.clip(np.array(freshness_scores, dtype=np.float32), 0.0, 100.0)
    out_df["freshness_score_band"] = freshness_bands
    out_df["predicted_index"] = top_indices.astype(int)
    out_df["true_index"] = y_true.astype(int)
    out_df["top2_adjacent_pair"] = top2_adjacent_pair.astype(bool)
    out_df["is_non_adjacent_top2"] = non_adjacent_top2.astype(bool)
    return out_df


def compute_transition_metrics_from_prediction_df(pred_df: pd.DataFrame) -> dict[str, float]:
    if pred_df.empty:
        return {metric_name: np.nan for metric_name in SECONDARY_RUN_METRICS}

    borderline_mask = pred_df["top_confidence"].astype(float) < TOP_CONFIDENCE_BORDERLINE
    adjacent_pair_mask = borderline_mask & pred_df["top2_adjacent_pair"].astype(bool)
    valid_adjacent_pair = pred_df.loc[adjacent_pair_mask, "top2_combined_confidence"]

    return {
        "top_2_accuracy": float(pred_df["is_top2_correct"].astype(float).mean()),
        "adjacent_accuracy": float(pred_df["is_adjacent_correct"].astype(float).mean()),
        "severe_error_rate": float(pred_df["is_severe_error"].astype(float).mean()),
        "mean_absolute_ordinal_error": float(pred_df["ordinal_error"].astype(float).mean()),
        "borderline_rate": float(pred_df["is_borderline"].astype(float).mean()),
        "low_confidence_rate": float(pred_df["is_low_confidence"].astype(float).mean()),
        "adjacent_pair_combined_confidence_mean": float(valid_adjacent_pair.astype(float).mean()) if not valid_adjacent_pair.empty else np.nan,
        "non_adjacent_top2_rate": float(pred_df["is_non_adjacent_top2"].astype(float).mean()),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0,
    )
    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        average=None,
        zero_division=0,
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=LABEL_ORDER,
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    cm_norm = cm.astype(np.float32) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    metrics = {
        "accuracy": accuracy,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "classification_report_json": json.dumps(report),
        "confusion_matrix_json": json.dumps(cm.tolist()),
        "normalized_confusion_matrix_json": json.dumps(cm_norm.tolist()),
        "prediction_distribution_json": json.dumps(
            {INDEX_TO_LABEL[idx]: int((y_pred == idx).sum()) for idx in range(NUM_CLASSES)}
        ),
    }
    for idx, label_name in INDEX_TO_LABEL.items():
        safe = label_name.replace(" ", "_")
        metrics[f"{safe}_precision"] = float(per_class_precision[idx])
        metrics[f"{safe}_recall"] = float(per_class_recall[idx])
        metrics[f"{safe}_f1"] = float(per_class_f1[idx])
        metrics[f"{safe}_support"] = int(per_class_support[idx])
    return metrics


def save_confusion_matrix_figure(cm: np.ndarray, labels: list[str], title: str, out_path: Path, normalize: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    if SEABORN_AVAILABLE:
        sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    else:
        ax.imshow(cm, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}", ha="center", va="center")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def predict_probabilities_in_batches(
    model: tf.keras.Model,
    image_paths: list[str],
    features: np.ndarray,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    predict_sequence = HybridImageFeatureSequence(
        image_paths=image_paths,
        features=features,
        labels=None,
        batch_size=batch_size,
        shuffle=False,
    )
    return model.predict(predict_sequence, verbose=0)


def measure_inference_speed(
    model: tf.keras.Model,
    test_image_paths: list[str],
    test_features: np.ndarray,
    warmup: int = 3,
    repeats: int = 10,
) -> tuple[float, float]:
    times = []
    for _ in range(warmup):
        _ = predict_probabilities_in_batches(model, test_image_paths, test_features)
    for _ in range(repeats):
        start = time.perf_counter()
        _ = predict_probabilities_in_batches(model, test_image_paths, test_features)
        elapsed = time.perf_counter() - start
        times.append((elapsed * 1000.0) / max(len(test_image_paths), 1))
    return float(np.mean(times)), float(np.std(times))


def try_convert_to_tflite(model: tf.keras.Model, out_path: Path) -> tuple[bool, float | None]:
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        out_path.write_bytes(tflite_model)
        return True, out_path.stat().st_size / (1024 * 1024)
    except Exception as exc:
        print(f"[WARN] TFLite conversion failed: {exc}")
        return False, None


def save_predictions_csv(
    run_stem: str,
    prediction_df: pd.DataFrame,
) -> Path:
    out_path = EXTENSION_PREDICTIONS_ROOT / f"{run_stem}_predictions.csv"
    prediction_df.to_csv(out_path, index=False)
    return out_path


def make_gradcam_heatmap(model: tf.keras.Model, image_batch: np.ndarray, feature_batch: np.ndarray, class_index: int | None = None) -> np.ndarray:
    backbone = model.get_layer("mobilenetv3small_backbone")
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [backbone.output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([image_batch, feature_batch], training=False)
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


def save_gradcam_preview(model: tf.keras.Model, row: pd.Series, scaler: StandardScaler, feature_columns: list[str], out_path: Path) -> Path:
    feature_df = extract_features_for_dataframe(pd.DataFrame([row]), limit=1)
    feature_values = scaler.transform(feature_df[feature_columns].astype(np.float32))
    original_np, image_uint8, image_float = load_preprocessed_segmented_roi_224(row["image_path_resolved"])
    image_input = preprocess_mobilenetv3(np.expand_dims(image_float, axis=0))
    heatmap = make_gradcam_heatmap(model, image_input, feature_values.astype(np.float32))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original_np)
    axes[0].set_title("Loaded image")
    axes[1].imshow(image_uint8)
    axes[1].set_title("Segmented ROI 224")
    axes[2].imshow(image_uint8)
    axes[2].imshow(heatmap, cmap="jet", alpha=0.4)
    axes[2].set_title("Grad-CAM")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def train_segmented6_hybrid_model(
    fold_name: str,
    seed: int,
    bundle: dict[str, object],
) -> tuple[tf.keras.Model, dict[str, object]]:
    class_weights = compute_class_weights_from_labels(bundle["train_labels"])
    model = build_segmented6_hybrid_model(num_features=len(bundle["feature_columns"]))

    train_sequence = HybridImageFeatureSequence(
        image_paths=bundle["train_image_paths"],
        features=bundle["train_features"],
        labels=bundle["train_labels"],
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_sequence = HybridImageFeatureSequence(
        image_paths=bundle["val_image_paths"],
        features=bundle["val_features"],
        labels=bundle["val_labels"],
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    val_macro_f1_cb = ValMacroF1Callback(
        val_source=val_sequence,
        val_labels=bundle["val_labels"],
    )
    callbacks = [
        val_macro_f1_cb,
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


def run_single_segmented6_hybrid_experiment(
    fold_name: str = "fold1",
    seed: int = 42,
    split_dfs: dict[str, pd.DataFrame] | None = None,
) -> dict[str, object]:
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required for segmented hybrid training.")
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is required for segmented hybrid training.")

    ensure_output_dirs()
    set_global_seed(seed)
    if split_dfs is None:
        split_dfs = load_all_cross_rotation_splits()

    bundle = prepare_feature_bundle_for_fold(fold_name, split_dfs, force_recompute=False)
    train_df = bundle["train_df"]
    val_df = bundle["val_df"]
    test_df = bundle["test_df"]
    run_stem = make_run_stem(fold_name, seed)

    print(f"[MEMORY] {fold_name} seed={seed}")
    print(
        f"  train_images count={len(bundle['train_image_paths'])} loaded_per_batch={BATCH_SIZE}"
    )
    print(
        f"  val_images count={len(bundle['val_image_paths'])} loaded_per_batch={BATCH_SIZE}"
    )
    print(
        f"  test_images count={len(bundle['test_image_paths'])} loaded_per_batch={BATCH_SIZE}"
    )
    print(
        f"  train_features shape={bundle['train_features'].shape} approx_mb={estimate_array_memory_mb(bundle['train_features']):.2f}"
    )
    print(
        f"  val_features shape={bundle['val_features'].shape} approx_mb={estimate_array_memory_mb(bundle['val_features']):.2f}"
    )
    print(
        f"  test_features shape={bundle['test_features'].shape} approx_mb={estimate_array_memory_mb(bundle['test_features']):.2f}"
    )

    model, train_info = train_segmented6_hybrid_model(fold_name=fold_name, seed=seed, bundle=bundle)

    y_prob = predict_probabilities_in_batches(
        model,
        bundle["test_image_paths"],
        bundle["test_features"],
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

    scaler_path = EXTENSION_MODELS_ROOT / f"{run_stem}_scaler.joblib"
    if JOBLIB_AVAILABLE:
        joblib.dump(bundle["scaler"], scaler_path)
    else:
        scaler_path = Path("")

    tflite_path = EXTENSION_MODELS_ROOT / f"{run_stem}.tflite"
    tflite_ok, tflite_size_mb = try_convert_to_tflite(model, tflite_path)
    if not tflite_ok and tflite_path.exists():
        tflite_path.unlink(missing_ok=True)

    inference_mean_ms, inference_std_ms = measure_inference_speed(
        model,
        bundle["test_image_paths"],
        bundle["test_features"],
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
        "scaler_path": str(scaler_path) if str(scaler_path) else "",
        "feature_fill_values_path": str(bundle["feature_fill_values_path"]),
        "feature_columns_json": json.dumps(bundle["feature_columns"]),
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
        df = pd.read_csv(SEGMENTED6_FAILED_RUNS_PATH)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(SEGMENTED6_FAILED_RUNS_PATH, index=False)


def coerce_prediction_dataframe_types(pred_df: pd.DataFrame) -> pd.DataFrame:
    out_df = pred_df.copy()
    numeric_cols = [
        "prob_fresh",
        "prob_not_fresh",
        "prob_spoiled",
        "top_confidence",
        "second_confidence",
        "top2_combined_confidence",
        "prediction_margin",
        "ordinal_error",
        "freshness_score",
    ]
    bool_cols = [
        "is_exact_correct",
        "is_top2_correct",
        "is_adjacent_correct",
        "is_severe_error",
        "is_borderline",
        "is_low_confidence",
        "top2_adjacent_pair",
        "is_non_adjacent_top2",
    ]
    int_cols = ["predicted_index", "true_index"]

    for col in numeric_cols:
        if col in out_df.columns:
            out_df[col] = pd.to_numeric(out_df[col], errors="coerce")
    for col in int_cols:
        if col in out_df.columns:
            out_df[col] = pd.to_numeric(out_df[col], errors="coerce").fillna(0).astype(int)
    for col in bool_cols:
        if col in out_df.columns:
            out_df[col] = out_df[col].astype(str).str.lower().map(
                {"true": True, "false": False, "1": True, "0": False}
            ).fillna(False)
    return out_df


def load_prediction_dataframe(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path).fillna("")
    return coerce_prediction_dataframe_types(df)


def regenerate_prediction_dataframe_if_needed(pred_df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {
        "prob_fresh",
        "prob_not_fresh",
        "prob_spoiled",
        "top_class",
        "top_confidence",
        "second_class",
        "second_confidence",
        "top2_combined_confidence",
        "prediction_margin",
        "ordinal_error",
        "is_top2_correct",
        "is_adjacent_correct",
        "is_severe_error",
        "transition_label",
        "recommendation",
        "freshness_score",
        "freshness_score_band",
    }
    if required_cols.issubset(pred_df.columns):
        return coerce_prediction_dataframe_types(pred_df)

    working_df = pred_df.copy()
    true_label_series = working_df["true_label"] if "true_label" in working_df.columns else working_df["label"]
    y_true = true_label_series.astype(str).str.strip().str.lower().map(LABEL_TO_INDEX).astype(int).to_numpy()
    y_prob = working_df[["prob_fresh", "prob_not_fresh", "prob_spoiled"]].astype(float).to_numpy()
    regenerated_df = build_transition_prediction_dataframe(working_df, y_true, y_prob)
    return coerce_prediction_dataframe_types(regenerated_df)


def save_bar_plot(x, y, title: str, ylabel: str, out_path: Path, color: str, xlabel: str = "", ylim: tuple[float, float] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(x, y, color=color)
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.tick_params(axis="x", rotation=45)
    for bar in bars:
        height = bar.get_height()
        if pd.notna(height):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}" if abs(float(height)) < 10 else f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    plt.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_histogram(values, title: str, xlabel: str, ylabel: str, out_path: Path, bins: int = 20, color: str = "#2E86AB") -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(pd.Series(values).dropna().astype(float), bins=bins, color=color, edgecolor="black", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_grouped_histogram(df: pd.DataFrame, group_col: str, value_col: str, title: str, xlabel: str, out_path: Path, bins: int = 20) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    groups = [label for label in LABEL_ORDER if label in df[group_col].astype(str).unique().tolist()]
    if len(groups) == 0:
        plt.close(fig)
        return
    colors = ["#4C9F70", "#F18F01", "#BC4749"]
    for idx, group_name in enumerate(groups):
        values = pd.to_numeric(df.loc[df[group_col].astype(str) == group_name, value_col], errors="coerce").dropna()
        if values.empty:
            continue
        ax.hist(values, bins=bins, alpha=0.45, label=group_name, color=colors[idx % len(colors)], edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_boxplot(df: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, ylabel: str, out_path: Path, order: list[str] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df = df.copy()
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
    if SEABORN_AVAILABLE:
        sns.boxplot(data=plot_df, x=x_col, y=y_col, order=order, ax=ax)
    else:
        groups = order if order is not None else plot_df[x_col].dropna().astype(str).unique().tolist()
        data = [plot_df.loc[plot_df[x_col].astype(str) == group, y_col].dropna().astype(float).to_numpy() for group in groups]
        ax.boxplot(data, labels=groups)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_scatter_plot(x, y, title: str, xlabel: str, ylabel: str, out_path: Path, color: str = "#2E86AB") -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(pd.to_numeric(pd.Series(x), errors="coerce"), pd.to_numeric(pd.Series(y), errors="coerce"), color=color, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def aggregate_summary(
    df: pd.DataFrame,
    group_cols: list[str],
    extra_first_cols: list[str] | None = None,
) -> pd.DataFrame:
    extra_first_cols = extra_first_cols or []
    agg_spec: dict[str, tuple[str, str]] = {}
    for metric_name in PRIMARY_RUN_METRICS + SECONDARY_RUN_METRICS:
        if metric_name in df.columns:
            agg_spec[f"{metric_name}_mean"] = (metric_name, "mean")
            agg_spec[f"{metric_name}_std"] = (metric_name, "std")
    for col_name in extra_first_cols:
        if col_name in df.columns and col_name not in group_cols:
            agg_spec[col_name] = (col_name, "first")
    summary_df = df.groupby(group_cols, as_index=False).agg(**agg_spec)
    return summary_df


def build_prediction_distribution_df(seed_metrics_df: pd.DataFrame) -> pd.DataFrame:
    prediction_distribution_rows = []
    for _, row in seed_metrics_df.iterrows():
        pred_dist = json.loads(row["prediction_distribution_json"])
        for label_name, count in pred_dist.items():
            prediction_distribution_rows.append(
                {
                    "fold_name": row["fold_name"],
                    "seed": row["seed"],
                    "label": label_name,
                    "predicted_count": count,
                }
            )
    return pd.DataFrame(prediction_distribution_rows)


def load_prediction_csvs_for_metrics(seed_metrics_df: pd.DataFrame) -> pd.DataFrame:
    prediction_frames = []
    for _, row in seed_metrics_df.iterrows():
        predictions_path = str(row.get("predictions_path", "")).strip()
        if predictions_path == "":
            continue
        pred_path = Path(predictions_path)
        if not pred_path.exists():
            continue
        pred_df = load_prediction_dataframe(pred_path)
        for col_name in ["fold_name", "seed", "held_out_sample", "held_out_cut", "capture_source", "phone_group", "run_stem"]:
            if col_name not in pred_df.columns:
                pred_df[col_name] = row.get(col_name, "")
        prediction_frames.append(pred_df)
    if not prediction_frames:
        return pd.DataFrame()
    return pd.concat(prediction_frames, ignore_index=True)


def save_aggregate_confusion_matrices(prediction_df: pd.DataFrame) -> None:
    if prediction_df.empty:
        return
    y_true = prediction_df["true_label"].astype(str).str.strip().str.lower().map(LABEL_TO_INDEX).astype(int).to_numpy()
    y_pred = prediction_df["predicted_label"].astype(str).str.strip().str.lower().map(LABEL_TO_INDEX).astype(int).to_numpy()
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    cm_norm = cm.astype(np.float32) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    save_confusion_matrix_figure(cm, LABEL_ORDER, "Aggregate Confusion Matrix Across All Runs", EXTENSION_FIGURES_ROOT / "aggregate_confusion_matrix_all_runs.png", normalize=False)
    save_confusion_matrix_figure(cm_norm, LABEL_ORDER, "Aggregate Normalized Confusion Matrix Across All Runs", EXTENSION_FIGURES_ROOT / "aggregate_normalized_confusion_matrix_all_runs.png", normalize=True)


def save_transition_metric_graphs(seed_metrics_df: pd.DataFrame) -> None:
    fold_summary = aggregate_summary(seed_metrics_df, ["fold_name"])
    sample_summary = aggregate_summary(seed_metrics_df, ["held_out_sample"], extra_first_cols=["held_out_cut", "capture_source", "phone_group"])
    cut_summary = aggregate_summary(seed_metrics_df, ["held_out_cut"])
    capture_summary = aggregate_summary(seed_metrics_df, ["capture_source"])

    save_bar_plot(fold_summary["fold_name"], fold_summary["top_2_accuracy_mean"], "Top-2 Accuracy by Fold", "Top-2 Accuracy", EXTENSION_FIGURES_ROOT / "top2_accuracy_by_fold.png", "#3D5A80", xlabel="Fold", ylim=(0, 1.05))
    save_bar_plot(fold_summary["fold_name"], fold_summary["adjacent_accuracy_mean"], "Adjacent Accuracy by Fold", "Adjacent Accuracy", EXTENSION_FIGURES_ROOT / "adjacent_accuracy_by_fold.png", "#588157", xlabel="Fold", ylim=(0, 1.05))
    save_bar_plot(fold_summary["fold_name"], fold_summary["severe_error_rate_mean"], "Severe Error Rate by Fold", "Severe Error Rate", EXTENSION_FIGURES_ROOT / "severe_error_rate_by_fold.png", "#BC4749", xlabel="Fold", ylim=(0, 1.05))
    save_bar_plot(fold_summary["fold_name"], fold_summary["mean_absolute_ordinal_error_mean"], "Mean Absolute Ordinal Error by Fold", "Mean Absolute Ordinal Error", EXTENSION_FIGURES_ROOT / "mean_absolute_ordinal_error_by_fold.png", "#6D597A", xlabel="Fold")
    save_bar_plot(fold_summary["fold_name"], fold_summary["borderline_rate_mean"], "Borderline Rate by Fold", "Borderline Rate", EXTENSION_FIGURES_ROOT / "borderline_rate_by_fold.png", "#E09F3E", xlabel="Fold", ylim=(0, 1.05))
    save_bar_plot(fold_summary["fold_name"], fold_summary["low_confidence_rate_mean"], "Low Confidence Rate by Fold", "Low Confidence Rate", EXTENSION_FIGURES_ROOT / "low_confidence_rate_by_fold.png", "#8D99AE", xlabel="Fold", ylim=(0, 1.05))
    save_bar_plot(fold_summary["fold_name"], fold_summary["non_adjacent_top2_rate_mean"], "Non-adjacent Top-2 Rate by Fold", "Non-adjacent Top-2 Rate", EXTENSION_FIGURES_ROOT / "non_adjacent_top2_rate_by_fold.png", "#6A040F", xlabel="Fold", ylim=(0, 1.05))

    save_bar_plot(sample_summary["held_out_sample"], sample_summary["adjacent_accuracy_mean"], "Adjacent Accuracy by Held-out Sample", "Adjacent Accuracy", EXTENSION_FIGURES_ROOT / "adjacent_accuracy_by_held_out_sample.png", "#52796F", xlabel="Held-out sample", ylim=(0, 1.05))
    save_bar_plot(sample_summary["held_out_sample"], sample_summary["severe_error_rate_mean"], "Severe Error Rate by Held-out Sample", "Severe Error Rate", EXTENSION_FIGURES_ROOT / "severe_error_rate_by_held_out_sample.png", "#C1121F", xlabel="Held-out sample", ylim=(0, 1.05))
    save_bar_plot(sample_summary["held_out_sample"], sample_summary["borderline_rate_mean"], "Borderline Rate by Held-out Sample", "Borderline Rate", EXTENSION_FIGURES_ROOT / "borderline_rate_by_held_out_sample.png", "#F4A261", xlabel="Held-out sample", ylim=(0, 1.05))

    save_bar_plot(cut_summary["held_out_cut"], cut_summary["adjacent_accuracy_mean"], "Adjacent Accuracy by Pork Cut", "Adjacent Accuracy", EXTENSION_FIGURES_ROOT / "adjacent_accuracy_by_pork_cut.png", "#386641", xlabel="Pork cut", ylim=(0, 1.05))
    save_bar_plot(cut_summary["held_out_cut"], cut_summary["severe_error_rate_mean"], "Severe Error Rate by Pork Cut", "Severe Error Rate", EXTENSION_FIGURES_ROOT / "severe_error_rate_by_pork_cut.png", "#9D0208", xlabel="Pork cut", ylim=(0, 1.05))
    save_bar_plot(capture_summary["capture_source"], capture_summary["macro_f1_mean"], "Macro F1 by Capture Source", "Macro F1", EXTENSION_FIGURES_ROOT / "macro_f1_by_capture_source.png", "#264653", xlabel="Capture source", ylim=(0, 1.05))


def save_prediction_analysis_graphs(prediction_df: pd.DataFrame, seed_metrics_df: pd.DataFrame) -> None:
    if prediction_df.empty:
        return

    prediction_df = coerce_prediction_dataframe_types(prediction_df)

    per_class_precision = seed_metrics_df.groupby("held_out_sample", as_index=False).first()
    _ = per_class_precision  # keeps lint quiet for notebook-style utility module

    per_class_rows = []
    for label_name in LABEL_ORDER:
        safe = label_name.replace(" ", "_")
        if f"{safe}_precision" in seed_metrics_df.columns:
            per_class_rows.append(
                {
                    "label": label_name,
                    "precision": pd.to_numeric(seed_metrics_df[f"{safe}_precision"], errors="coerce").mean(),
                    "recall": pd.to_numeric(seed_metrics_df[f"{safe}_recall"], errors="coerce").mean(),
                    "f1": pd.to_numeric(seed_metrics_df[f"{safe}_f1"], errors="coerce").mean(),
                    "support": pd.to_numeric(seed_metrics_df[f"{safe}_support"], errors="coerce").mean(),
                }
            )
    per_class_df = pd.DataFrame(per_class_rows)
    if not per_class_df.empty:
        save_bar_plot(per_class_df["label"], per_class_df["precision"], "Per-class Precision", "Precision", EXTENSION_FIGURES_ROOT / "per_class_precision.png", "#355070", xlabel="Class", ylim=(0, 1.05))
        save_bar_plot(per_class_df["label"], per_class_df["recall"], "Per-class Recall", "Recall", EXTENSION_FIGURES_ROOT / "per_class_recall.png", "#6D597A", xlabel="Class", ylim=(0, 1.05))
        save_bar_plot(per_class_df["label"], per_class_df["f1"], "Per-class F1", "F1", EXTENSION_FIGURES_ROOT / "per_class_f1.png", "#B56576", xlabel="Class", ylim=(0, 1.05))
        save_bar_plot(per_class_df["label"], per_class_df["support"], "Per-class Support", "Support", EXTENSION_FIGURES_ROOT / "per_class_support.png", "#4C9F70", xlabel="Class")

    save_histogram(prediction_df["top_confidence"], "Confidence Distribution for All Predictions", "Top Confidence", "Count", EXTENSION_FIGURES_ROOT / "confidence_distribution_all_predictions.png", bins=25, color="#2A9D8F")
    save_grouped_histogram(prediction_df, "true_label", "top_confidence", "Confidence Distribution by True Class", "Top Confidence", EXTENSION_FIGURES_ROOT / "confidence_distribution_by_true_class.png", bins=20)
    save_grouped_histogram(prediction_df, "predicted_label", "top_confidence", "Confidence Distribution by Predicted Class", "Top Confidence", EXTENSION_FIGURES_ROOT / "confidence_distribution_by_predicted_class.png", bins=20)
    save_histogram(prediction_df["prediction_margin"], "Prediction Margin Distribution", "Top-1 minus Top-2 Probability", "Count", EXTENSION_FIGURES_ROOT / "prediction_margin_distribution.png", bins=25, color="#5E548E")

    borderline_by_true = prediction_df.groupby("true_label", as_index=False)["is_borderline"].mean()
    low_conf_by_true = prediction_df.groupby("true_label", as_index=False)["is_low_confidence"].mean()
    save_bar_plot(borderline_by_true["true_label"], borderline_by_true["is_borderline"], "Borderline Rate by True Class", "Borderline Rate", EXTENSION_FIGURES_ROOT / "borderline_rate_by_true_class.png", "#F4A261", xlabel="True class", ylim=(0, 1.05))
    save_bar_plot(low_conf_by_true["true_label"], low_conf_by_true["is_low_confidence"], "Low Confidence Rate by True Class", "Low Confidence Rate", EXTENSION_FIGURES_ROOT / "low_confidence_rate_by_true_class.png", "#8D99AE", xlabel="True class", ylim=(0, 1.05))

    save_histogram(prediction_df["freshness_score"], "Freshness Score Distribution", "Freshness Score", "Count", EXTENSION_FIGURES_ROOT / "freshness_score_distribution.png", bins=25, color="#43AA8B")
    save_grouped_histogram(prediction_df, "true_label", "freshness_score", "Freshness Score Distribution by True Class", "Freshness Score", EXTENSION_FIGURES_ROOT / "freshness_score_distribution_by_true_class.png", bins=20)
    save_boxplot(prediction_df, "true_label", "freshness_score", "Freshness Score by True Class", "True class", "Freshness Score", EXTENSION_FIGURES_ROOT / "freshness_score_boxplot_by_true_class.png", order=LABEL_ORDER)
    save_boxplot(prediction_df, "predicted_label", "freshness_score", "Freshness Score by Predicted Class", "Predicted class", "Freshness Score", EXTENSION_FIGURES_ROOT / "freshness_score_boxplot_by_predicted_class.png", order=LABEL_ORDER)

    save_histogram(prediction_df["ordinal_error"], "Ordinal Error Distribution", "Absolute Ordinal Error", "Count", EXTENSION_FIGURES_ROOT / "ordinal_error_distribution.png", bins=3, color="#9C6644")
    severe_by_fold = prediction_df.groupby("fold_name", as_index=False)["is_severe_error"].sum()
    severe_by_true = prediction_df.groupby("true_label", as_index=False)["is_severe_error"].sum()
    save_bar_plot(severe_by_fold["fold_name"], severe_by_fold["is_severe_error"], "Severe Error Count by Fold", "Severe Error Count", EXTENSION_FIGURES_ROOT / "severe_error_count_by_fold.png", "#AE2012", xlabel="Fold")
    save_bar_plot(severe_by_true["true_label"], severe_by_true["is_severe_error"], "Severe Error Count by True Class", "Severe Error Count", EXTENSION_FIGURES_ROOT / "severe_error_count_by_true_class.png", "#9B2226", xlabel="True class")

    exact_count = int(prediction_df["is_exact_correct"].astype(bool).sum())
    adjacent_only_count = int((prediction_df["is_adjacent_correct"].astype(bool) & ~prediction_df["is_exact_correct"].astype(bool)).sum())
    severe_count = int(prediction_df["is_severe_error"].astype(bool).sum())
    save_bar_plot(
        ["Exact", "Adjacent", "Severe"],
        [exact_count, adjacent_only_count, severe_count],
        "Exact vs Adjacent vs Severe Errors",
        "Count",
        EXTENSION_FIGURES_ROOT / "exact_vs_adjacent_vs_severe_error_bar.png",
        "#577590",
        xlabel="Outcome type",
    )

    transition_label_dist = prediction_df["transition_label"].astype(str).value_counts().reset_index()
    transition_label_dist.columns = ["transition_label", "count"]
    save_bar_plot(transition_label_dist["transition_label"], transition_label_dist["count"], "Transition Label Distribution", "Count", EXTENSION_FIGURES_ROOT / "transition_label_distribution.png", "#6D597A", xlabel="Transition label")

    save_aggregate_confusion_matrices(prediction_df)
    save_scatter_plot(seed_metrics_df["inference_mean_ms_per_image"], seed_metrics_df["macro_f1"], "Macro F1 vs Inference Speed", "Inference mean ms/image", "Macro F1", EXTENSION_FIGURES_ROOT / "macro_f1_vs_inference_speed.png", color="#3A86FF")
    save_scatter_plot(seed_metrics_df["h5_size_mb"], seed_metrics_df["macro_f1"], "Macro F1 vs Model Size", "Model size (MB)", "Macro F1", EXTENSION_FIGURES_ROOT / "macro_f1_vs_model_size.png", color="#8338EC")


def save_segmented6_hybrid_summary_outputs(seed_metrics_df: pd.DataFrame, prediction_df: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    seed_metrics_df = seed_metrics_df.copy()
    numeric_cols = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "top_2_accuracy",
        "adjacent_accuracy",
        "severe_error_rate",
        "mean_absolute_ordinal_error",
        "borderline_rate",
        "low_confidence_rate",
        "adjacent_pair_combined_confidence_mean",
        "non_adjacent_top2_rate",
        "h5_size_mb",
        "tflite_size_mb",
        "inference_mean_ms_per_image",
        "inference_std_ms_per_image",
    ]
    for col in numeric_cols:
        if col in seed_metrics_df.columns:
            seed_metrics_df[col] = pd.to_numeric(seed_metrics_df[col], errors="coerce")

    fold_summary = aggregate_summary(seed_metrics_df, ["fold_name"])
    fold_summary.to_csv(SEGMENTED6_FOLD_SUMMARY_PATH, index=False)

    sample_summary = aggregate_summary(
        seed_metrics_df,
        ["held_out_sample"],
        extra_first_cols=["fold_name", "held_out_cut", "capture_source", "phone_group"],
    )
    sample_summary.to_csv(SEGMENTED6_SAMPLE_SUMMARY_PATH, index=False)

    cut_summary = aggregate_summary(seed_metrics_df, ["held_out_cut"])
    cut_summary.to_csv(SEGMENTED6_CUT_SUMMARY_PATH, index=False)

    capture_source_summary = aggregate_summary(seed_metrics_df, ["capture_source"])
    capture_source_summary.to_csv(SEGMENTED6_CAPTURE_SOURCE_SUMMARY_PATH, index=False)

    if "phone_group" in seed_metrics_df.columns and seed_metrics_df["phone_group"].astype(str).str.strip().ne("").any():
        phone_group_summary = aggregate_summary(seed_metrics_df, ["phone_group"])
    else:
        phone_group_summary = pd.DataFrame(columns=["phone_group"])
    phone_group_summary.to_csv(SEGMENTED6_PHONE_GROUP_SUMMARY_PATH, index=False)

    per_class_rows = []
    for _, row in seed_metrics_df.iterrows():
        for label_name in LABEL_ORDER:
            safe = label_name.replace(" ", "_")
            per_class_rows.append(
                {
                    "fold_name": row["fold_name"],
                    "seed": row["seed"],
                    "label": label_name,
                    "precision": row.get(f"{safe}_precision", np.nan),
                    "recall": row.get(f"{safe}_recall", np.nan),
                    "f1": row.get(f"{safe}_f1", np.nan),
                    "support": row.get(f"{safe}_support", np.nan),
                }
            )
    per_class_summary = pd.DataFrame(per_class_rows)
    per_class_summary.to_csv(SEGMENTED6_PER_CLASS_SUMMARY_PATH, index=False)

    prediction_distribution_df = build_prediction_distribution_df(seed_metrics_df)
    prediction_distribution_df.to_csv(SEGMENTED6_PRED_DISTRIBUTION_PATH, index=False)

    size_speed_df = seed_metrics_df[
        ["fold_name", "seed", "h5_size_mb", "tflite_size_mb", "inference_mean_ms_per_image", "inference_std_ms_per_image"]
    ].copy()
    size_speed_df.to_csv(SEGMENTED6_SIZE_SPEED_PATH, index=False)

    x_labels = seed_metrics_df["fold_name"].astype(str) + "_s" + seed_metrics_df["seed"].astype(str)
    save_bar_plot(x_labels, seed_metrics_df["macro_f1"], "Strict Macro F1 by Fold and Seed", "Macro F1", EXTENSION_FIGURES_ROOT / "macro_f1_by_fold_seed.png", "#2E86AB", xlabel="Fold / seed", ylim=(0, 1.05))
    save_bar_plot(x_labels, seed_metrics_df["accuracy"], "Strict Accuracy by Fold and Seed", "Accuracy", EXTENSION_FIGURES_ROOT / "accuracy_by_fold_seed.png", "#4C9F70", xlabel="Fold / seed", ylim=(0, 1.05))
    save_bar_plot(fold_summary["fold_name"], fold_summary["macro_f1_mean"], "Mean Strict Macro F1 by Fold", "Mean Macro F1", EXTENSION_FIGURES_ROOT / "mean_macro_f1_by_fold.png", "#F18F01", xlabel="Fold", ylim=(0, 1.05))
    save_bar_plot(fold_summary["fold_name"], fold_summary["accuracy_mean"], "Mean Strict Accuracy by Fold", "Mean Accuracy", EXTENSION_FIGURES_ROOT / "mean_accuracy_by_fold.png", "#7F5539", xlabel="Fold", ylim=(0, 1.05))
    save_bar_plot(sample_summary["held_out_sample"], sample_summary["macro_f1_mean"], "Strict Macro F1 by Held-out Sample", "Macro F1", EXTENSION_FIGURES_ROOT / "macro_f1_by_held_out_sample.png", "#A23B72", xlabel="Held-out sample", ylim=(0, 1.05))
    save_bar_plot(cut_summary["held_out_cut"], cut_summary["macro_f1_mean"], "Strict Macro F1 by Pork Cut", "Macro F1", EXTENSION_FIGURES_ROOT / "macro_f1_by_pork_cut.png", "#6A994E", xlabel="Pork cut", ylim=(0, 1.05))
    save_bar_plot(capture_source_summary["capture_source"], capture_source_summary["macro_f1_mean"], "Strict Macro F1 by Capture Source", "Macro F1", EXTENSION_FIGURES_ROOT / "macro_f1_by_capture_source.png", "#3D5A80", xlabel="Capture source", ylim=(0, 1.05))
    if not phone_group_summary.empty:
        save_bar_plot(phone_group_summary["phone_group"], phone_group_summary["macro_f1_mean"], "Strict Macro F1 by Phone Group", "Macro F1", EXTENSION_FIGURES_ROOT / "macro_f1_by_phone_group.png", "#9C6644", xlabel="Phone group", ylim=(0, 1.05))

    prediction_dist_plot = prediction_distribution_df.groupby("label", as_index=False)["predicted_count"].sum()
    save_bar_plot(prediction_dist_plot["label"], prediction_dist_plot["predicted_count"], "Prediction Distribution", "Predicted Count", EXTENSION_FIGURES_ROOT / "prediction_distribution.png", "#E56B6F", xlabel="Predicted label")
    save_bar_plot(x_labels, seed_metrics_df["h5_size_mb"], "Model Size", "H5 size (MB)", EXTENSION_FIGURES_ROOT / "model_size.png", "#6C9A8B", xlabel="Fold / seed")
    save_bar_plot(x_labels, seed_metrics_df["inference_mean_ms_per_image"], "Inference Speed", "Mean ms/image", EXTENSION_FIGURES_ROOT / "inference_speed.png", "#BC6C25", xlabel="Fold / seed")

    best_idx = seed_metrics_df["macro_f1"].idxmax()
    best_row = seed_metrics_df.loc[best_idx]
    best_cm = np.array(json.loads(best_row["confusion_matrix_json"]))
    best_cm_norm = np.array(json.loads(best_row["normalized_confusion_matrix_json"]))
    save_confusion_matrix_figure(best_cm, LABEL_ORDER, "Best Confusion Matrix", EXTENSION_FIGURES_ROOT / "best_confusion_matrix.png", normalize=False)
    save_confusion_matrix_figure(best_cm_norm, LABEL_ORDER, "Best Normalized Confusion Matrix", EXTENSION_FIGURES_ROOT / "best_normalized_confusion_matrix.png", normalize=True)

    save_transition_metric_graphs(seed_metrics_df)
    if prediction_df is not None and not prediction_df.empty:
        save_prediction_analysis_graphs(prediction_df, seed_metrics_df)

    return {
        "fold_summary": fold_summary,
        "sample_summary": sample_summary,
        "cut_summary": cut_summary,
        "capture_source_summary": capture_source_summary,
        "phone_group_summary": phone_group_summary,
        "per_class_summary": per_class_summary,
        "prediction_distribution_df": prediction_distribution_df,
        "size_speed_df": size_speed_df,
    }


def build_run_level_metrics_from_prediction_df(pred_df: pd.DataFrame) -> dict[str, object]:
    pred_df = regenerate_prediction_dataframe_if_needed(pred_df)
    y_true = pred_df["true_label"].astype(str).str.strip().str.lower().map(LABEL_TO_INDEX).astype(int).to_numpy()
    y_pred = pred_df["predicted_label"].astype(str).str.strip().str.lower().map(LABEL_TO_INDEX).astype(int).to_numpy()
    metric_row = classification_metrics(y_true, y_pred)
    metric_row.update(compute_transition_metrics_from_prediction_df(pred_df))
    return metric_row


def regenerate_transition_metrics_and_graphs() -> dict[str, object] | None:
    ensure_output_dirs()
    if not SEGMENTED6_SEED_METRICS_PATH.exists():
        print("No segmented6 hybrid training results found yet. Train first, then run regenerate_transition_metrics_and_graphs().")
        return None

    seed_metrics_df = pd.read_csv(SEGMENTED6_SEED_METRICS_PATH).fillna("")
    if seed_metrics_df.empty:
        print("No segmented6 hybrid training results found yet. Train first, then run regenerate_transition_metrics_and_graphs().")
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

        run_metrics = build_run_level_metrics_from_prediction_df(pred_df)
        for key, value in run_metrics.items():
            row_dict[key] = value
        updated_rows.append(row_dict)

        for col_name in ["fold_name", "seed", "held_out_sample", "held_out_cut", "capture_source", "phone_group", "run_stem"]:
            if col_name not in pred_df.columns:
                pred_df[col_name] = row_dict.get(col_name, "")
        prediction_frames.append(pred_df)

    updated_metrics_df = pd.DataFrame(updated_rows).sort_values(["fold_name", "seed"]).reset_index(drop=True)
    updated_metrics_df.to_csv(SEGMENTED6_SEED_METRICS_PATH, index=False)

    prediction_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    summary_bundle = save_segmented6_hybrid_summary_outputs(updated_metrics_df, prediction_df=prediction_df)
    interpretation_text = build_summary_interpretation_text(updated_metrics_df, prediction_df)
    print(interpretation_text)

    return {
        "seed_metrics_df": updated_metrics_df,
        "prediction_df": prediction_df,
        "summary_bundle": summary_bundle,
        "interpretation_text": interpretation_text,
    }


def build_summary_interpretation_text(seed_metrics_df: pd.DataFrame, prediction_df: pd.DataFrame | None = None) -> str:
    if seed_metrics_df.empty:
        return "No segmented6 hybrid training results found yet."

    seed_metrics_df = seed_metrics_df.copy()
    for metric_name in PRIMARY_RUN_METRICS + SECONDARY_RUN_METRICS:
        if metric_name in seed_metrics_df.columns:
            seed_metrics_df[metric_name] = pd.to_numeric(seed_metrics_df[metric_name], errors="coerce")

    best_idx = seed_metrics_df["macro_f1"].idxmax()
    best_row = seed_metrics_df.loc[best_idx]

    per_class_rows = []
    for label_name in LABEL_ORDER:
        safe = label_name.replace(" ", "_")
        per_class_rows.append(
            {
                "label": label_name,
                "recall_mean": pd.to_numeric(seed_metrics_df.get(f"{safe}_recall", np.nan), errors="coerce").mean(),
                "f1_mean": pd.to_numeric(seed_metrics_df.get(f"{safe}_f1", np.nan), errors="coerce").mean(),
            }
        )
    per_class_df = pd.DataFrame(per_class_rows)
    per_class_df["difficulty_score"] = 1.0 - per_class_df["f1_mean"].fillna(0.0)
    hardest_class = per_class_df.sort_values(["recall_mean", "f1_mean"], ascending=[True, True]).iloc[0]["label"]

    if prediction_df is None or prediction_df.empty:
        adjacent_only_rate = np.nan
        severe_rate = seed_metrics_df["severe_error_rate"].mean() if "severe_error_rate" in seed_metrics_df.columns else np.nan
    else:
        prediction_df = coerce_prediction_dataframe_types(prediction_df)
        adjacent_only_rate = float(
            (prediction_df["is_adjacent_correct"].astype(bool) & ~prediction_df["is_exact_correct"].astype(bool)).astype(float).mean()
        )
        severe_rate = float(prediction_df["is_severe_error"].astype(float).mean())

    error_style = "Most errors are adjacent rather than severe." if pd.notna(adjacent_only_rate) and adjacent_only_rate >= severe_rate else "Severe opposite-end mistakes remain non-negligible."

    lines = [
        f"Best run by strict macro F1: {best_row['run_stem']} | macro F1={float(best_row['macro_f1']):.4f} | accuracy={float(best_row['accuracy']):.4f}",
        f"Mean strict macro F1 across all folds/seeds: {seed_metrics_df['macro_f1'].mean():.4f}",
        f"Mean strict accuracy across all folds/seeds: {seed_metrics_df['accuracy'].mean():.4f}",
        f"Mean top-2 accuracy: {seed_metrics_df['top_2_accuracy'].mean():.4f}" if "top_2_accuracy" in seed_metrics_df.columns else "Mean top-2 accuracy: N/A",
        f"Mean adjacent accuracy: {seed_metrics_df['adjacent_accuracy'].mean():.4f}" if "adjacent_accuracy" in seed_metrics_df.columns else "Mean adjacent accuracy: N/A",
        f"Mean severe error rate: {seed_metrics_df['severe_error_rate'].mean():.4f}" if "severe_error_rate" in seed_metrics_df.columns else "Mean severe error rate: N/A",
        f"Mean ordinal error: {seed_metrics_df['mean_absolute_ordinal_error'].mean():.4f}" if "mean_absolute_ordinal_error" in seed_metrics_df.columns else "Mean ordinal error: N/A",
        f"Borderline rate: {seed_metrics_df['borderline_rate'].mean():.4f}" if "borderline_rate" in seed_metrics_df.columns else "Borderline rate: N/A",
        f"Most difficult true class based on recall/F1: {hardest_class}",
        error_style,
        "Strict accuracy and macro F1 are retained as the primary metrics. Transition-aware metrics are reported only as secondary analysis because pork freshness changes gradually and borderline images may visually lie between adjacent classes.",
    ]
    return "\n".join(lines)


def copy_best_segmented6_hybrid_artifacts(seed_metrics_df: pd.DataFrame) -> dict[str, str]:
    best_idx = seed_metrics_df["macro_f1"].astype(float).idxmax()
    best_row = seed_metrics_df.loc[best_idx]

    src_model_path = Path(best_row["model_path"])
    src_tflite_path = Path(str(best_row.get("tflite_path", ""))) if str(best_row.get("tflite_path", "")).strip() else None

    best_model_path = EXTENSION_MODELS_ROOT / "meatlens_best_segmented6_hybrid_mobilenetv3small.h5"
    best_tflite_path = EXTENSION_MODELS_ROOT / "meatlens_best_segmented6_hybrid_mobilenetv3small.tflite"
    best_metadata_path = EXTENSION_MODELS_ROOT / "meatlens_best_segmented6_hybrid_mobilenetv3small_metadata.json"

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
        "feature_columns": json.loads(best_row["feature_columns_json"]),
        "scaler_path": best_row.get("scaler_path", ""),
        "feature_fill_values_path": best_row.get("feature_fill_values_path", ""),
        "split_mode": EXTENSION_SPLIT_MODE,
        "fold": best_row["fold_name"],
        "seed": int(best_row["seed"]),
        "held_out_sample": best_row.get("held_out_sample", ""),
        "held_out_cut": best_row.get("held_out_cut", ""),
        "capture_source": best_row.get("capture_source", ""),
        "phone_group": best_row.get("phone_group", ""),
        "macro_f1": float(best_row["macro_f1"]),
        "accuracy": float(best_row["accuracy"]),
        "model_path": str(best_model_path),
        "tflite_path": str(best_tflite_path) if best_tflite_path.exists() else "",
        "timestamp": datetime.now().isoformat(),
        "note": "Model trained on preprocessed HSV/LAB segmented 224x224 ROI images.",
    }
    best_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "best_model_path": str(best_model_path),
        "best_tflite_path": str(best_tflite_path) if best_tflite_path.exists() else "",
        "best_metadata_path": str(best_metadata_path),
    }


def run_segmented6_hybrid_training() -> dict[str, object]:
    ensure_output_dirs()
    split_dfs = load_all_cross_rotation_splits()
    existing_metrics_df = load_existing_metrics()
    seed_results = existing_metrics_df.to_dict(orient="records") if not existing_metrics_df.empty else []

    for fold_name in EXTENSION_FOLDS:
        for seed in EXTENSION_RUN_SEEDS:
            if should_skip_run(existing_metrics_df, fold_name, seed):
                print(f"Skipping completed run: {fold_name} seed={seed}")
                continue

            print(f"Running segmented hybrid experiment: {fold_name} seed={seed}")
            try:
                result = run_single_segmented6_hybrid_experiment(
                    fold_name=fold_name,
                    seed=seed,
                    split_dfs=split_dfs,
                )
                seed_results.append(result)
                existing_metrics_df = pd.DataFrame(seed_results).sort_values(["fold_name", "seed"]).reset_index(drop=True)
                existing_metrics_df.to_csv(SEGMENTED6_SEED_METRICS_PATH, index=False)
            except Exception as exc:
                append_failed_run(fold_name, seed, exc)
                print(f"[WARN] Failed run {fold_name} seed={seed}: {exc}")
            finally:
                if TF_AVAILABLE:
                    tf.keras.backend.clear_session()
                gc.collect()

    seed_metrics_df = pd.DataFrame(seed_results).sort_values(["fold_name", "seed"]).reset_index(drop=True)
    seed_metrics_df.to_csv(SEGMENTED6_SEED_METRICS_PATH, index=False)
    summary_bundle = save_segmented6_hybrid_summary_outputs(seed_metrics_df)
    best_bundle = copy_best_segmented6_hybrid_artifacts(seed_metrics_df)
    return {
        "seed_metrics_df": seed_metrics_df,
        "summary_bundle": summary_bundle,
        "best_bundle": best_bundle,
    }
