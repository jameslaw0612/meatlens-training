#!/usr/bin/env python3
from pathlib import Path
import nbformat as nbf


def main() -> None:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            """# MeatLens Thesis Notebook (`new2.ipynb`)

This revision keeps the data-loading/validation/preprocessing workflow and adds **CNN-only training/evaluation scaffolding**.

Important:
- No public dataset is used.
- No full training is run automatically.
- Debug run cell is provided but intentionally not executed.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """# ====================================================
# Libraries / tools
# ====================================================
import os
import gc
import io
import json
import math
import time
import shutil
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, ImageOps

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from sklearn.utils.class_weight import compute_class_weight

# Optional for handcrafted GLCM (not used for training in this revision)
SKIMAGE_AVAILABLE = True
try:
    from skimage.feature import graycomatrix, graycoprops
except Exception as e:
    SKIMAGE_AVAILABLE = False
    graycomatrix = None
    graycoprops = None
    print(f"[WARN] scikit-image unavailable: {e}. GLCM will be skipped.")

# TensorFlow (required for CNN functions)
TF_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

    from tensorflow.keras.applications import MobileNetV3Small, EfficientNetB0, ResNet50, MobileNetV2
    from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as preprocess_mobilenetv3
    from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnetb0
    from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet50
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenetv2
except Exception as e:
    TF_AVAILABLE = False
    tf = None
    print(f"[WARN] TensorFlow unavailable: {e}. Training functions will not run.")

# Optional upload widgets
try:
    import ipywidgets as widgets
    from IPython.display import display
    WIDGETS_AVAILABLE = True
except Exception:
    WIDGETS_AVAILABLE = False
    widgets = None
    def display(x):
        print(x)

print("Libraries loaded.")"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """# ====================================================
# Configuration (debug defaults)
# ====================================================
PROJECT_ROOT = Path.cwd()
SPLIT_ROOT = PROJECT_ROOT / "generated_splits"
CROSS_ROOT = SPLIT_ROOT / "cross_rotation"
RANDOM_ROOT = SPLIT_ROOT / "random_70_15_15"

TRAINING_OUTPUTS = PROJECT_ROOT / "training_outputs"
FEATURE_OUTPUTS = TRAINING_OUTPUTS / "features"
MODEL_OUTPUTS = TRAINING_OUTPUTS / "models"
PLOT_OUTPUTS = TRAINING_OUTPUTS / "plots"

TRAINING_OUTPUTS.mkdir(parents=True, exist_ok=True)
FEATURE_OUTPUTS.mkdir(parents=True, exist_ok=True)
MODEL_OUTPUTS.mkdir(parents=True, exist_ok=True)
PLOT_OUTPUTS.mkdir(parents=True, exist_ok=True)

MODEL_INPUT_MODE = "cnn_only"
SPLIT_MODE = "cross_rotation"
IMAGE_CROP_MODE = "center_crop"
RUN_SEEDS = [42]
BACKBONES = ["MobileNetV2"]
RUN_FULL_TRAINING = False

# For preprocessing comparison later, test:
# IMAGE_CROP_MODE = "center_crop"
# IMAGE_CROP_MODE = "resize_pad"
# IMAGE_CROP_MODE = "resize"

# For final thesis run:
# RUN_SEEDS = [42, 123, 2026]
# BACKBONES = ["MobileNetV3Small", "EfficientNetB0", "ResNet50", "MobileNetV2"]
# SPLIT_MODE = "both"
# RUN_FULL_TRAINING = True

USE_HANDCRAFTED_FEATURES = True
EXTRACT_FEATURES_NOW = False

TARGET_SIZE = (224, 224)
NUM_CLASSES = 3
LABEL_ORDER = ["fresh", "not fresh", "spoiled"]
LABEL_TO_INDEX = {k: i for i, k in enumerate(LABEL_ORDER)}
INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}

SEED_DEFAULT = 42
BATCH_SIZE = 32
EPOCHS_HEAD = 8
EPOCHS_FINE = 20

FINE_TUNE_LR = {
    "MobileNetV3Small": 1e-5,
    "EfficientNetB0": 1e-5,
    "ResNet50": 5e-6,
    "MobileNetV2": 1e-5,
}

SHARPEN_KERNEL = [
    [0.0, -1.0, 0.0],
    [-1.0, 5.0, -1.0],
    [0.0, -1.0, 0.0],
]

print(f"SPLIT_MODE={SPLIT_MODE}")
print(f"IMAGE_CROP_MODE={IMAGE_CROP_MODE}")
print(f"RUN_FULL_TRAINING={RUN_FULL_TRAINING}")
print(f"BACKBONES={BACKBONES}")"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """# ====================================================
# Split files and expected counts
# ====================================================
CROSS_FILES = {
    "fold1_train": CROSS_ROOT / "fold1_train.csv",
    "fold1_val": CROSS_ROOT / "fold1_val.csv",
    "fold1_test": CROSS_ROOT / "fold1_test.csv",
    "fold2_train": CROSS_ROOT / "fold2_train.csv",
    "fold2_val": CROSS_ROOT / "fold2_val.csv",
    "fold2_test": CROSS_ROOT / "fold2_test.csv",
    "fold3_train": CROSS_ROOT / "fold3_train.csv",
    "fold3_val": CROSS_ROOT / "fold3_val.csv",
    "fold3_test": CROSS_ROOT / "fold3_test.csv",
    "fold4_train": CROSS_ROOT / "fold4_train.csv",
    "fold4_val": CROSS_ROOT / "fold4_val.csv",
    "fold4_test": CROSS_ROOT / "fold4_test.csv",
}

RANDOM_FILES = {
    "random_train": RANDOM_ROOT / "random_train.csv",
    "random_val": RANDOM_ROOT / "random_val.csv",
    "random_test": RANDOM_ROOT / "random_test.csv",
}

EXPECTED_CROSS = {
    "fold1": {
        "held_out_sample": "pork_shoulder_sample_1", "held_out_cut": "shoulder",
        "train_count": 1161, "val_count": 205, "test_count": 485,
        "train_labels": {"fresh": 359, "not fresh": 386, "spoiled": 416},
        "val_labels": {"fresh": 63, "not fresh": 68, "spoiled": 74},
        "test_labels": {"fresh": 157, "not fresh": 158, "spoiled": 170},
    },
    "fold2": {
        "held_out_sample": "pork_shoulder_sample_2", "held_out_cut": "shoulder",
        "train_count": 1187, "val_count": 210, "test_count": 454,
        "train_labels": {"fresh": 361, "not fresh": 397, "spoiled": 429},
        "val_labels": {"fresh": 64, "not fresh": 70, "spoiled": 76},
        "test_labels": {"fresh": 154, "not fresh": 145, "spoiled": 155},
    },
    "fold3": {
        "held_out_sample": "pork_belly_sample_3", "held_out_cut": "belly",
        "train_count": 1199, "val_count": 212, "test_count": 440,
        "train_labels": {"fresh": 392, "not fresh": 385, "spoiled": 422},
        "val_labels": {"fresh": 69, "not fresh": 68, "spoiled": 75},
        "test_labels": {"fresh": 118, "not fresh": 159, "spoiled": 163},
    },
    "fold4": {
        "held_out_sample": "pork_belly_sample_4", "held_out_cut": "belly",
        "train_count": 1172, "val_count": 207, "test_count": 472,
        "train_labels": {"fresh": 364, "not fresh": 393, "spoiled": 415},
        "val_labels": {"fresh": 65, "not fresh": 69, "spoiled": 73},
        "test_labels": {"fresh": 150, "not fresh": 150, "spoiled": 172},
    },
}

EXPECTED_RANDOM = {
    "train_count": 1295, "val_count": 278, "test_count": 278,
    "train_labels": {"spoiled": 462, "not fresh": 428, "fresh": 405},
    "val_labels": {"spoiled": 99, "not fresh": 92, "fresh": 87},
    "test_labels": {"spoiled": 99, "not fresh": 92, "fresh": 87},
    "train_cuts": {"belly": 666, "shoulder": 629},
    "val_cuts": {"shoulder": 153, "belly": 125},
    "test_cuts": {"shoulder": 157, "belly": 121},
}

IMAGE_PATH_CANDIDATES = ["image_path", "file_path", "path", "filename", "image_file", "roi_file", "file_destination", "image_file_name"]
ROI_SETS = [
    ("x", "y", "w", "h"),
    ("xmin", "ymin", "xmax", "ymax"),
    ("bbox_x", "bbox_y", "bbox_w", "bbox_h"),
]

print("Split registries prepared.")"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """# ====================================================
# Split loading, path detection, path resolution
# ====================================================

def collect_split_files(split_mode: str):
    files = {}
    if split_mode in ["cross_rotation", "both"]:
        files.update(CROSS_FILES)
    if split_mode in ["random_70_15_15", "both"]:
        files.update(RANDOM_FILES)
    return files


def detect_image_path_column(df: pd.DataFrame):
    for c in IMAGE_PATH_CANDIDATES:
        if c in df.columns:
            return c
    return None


def resolve_image_path(row, path_col, csv_path=None):
    if path_col is None:
        return None
    raw = row.get(path_col, None)
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None

    raw_str = str(raw).strip()
    if raw_str == "":
        return None

    candidates = []
    p = Path(raw_str)
    candidates.append(p)

    if csv_path is not None:
        candidates.append(Path(csv_path).resolve().parent / raw_str)

    candidates.append(PROJECT_ROOT / raw_str)

    for cand in candidates:
        try:
            c = cand.resolve()
        except Exception:
            c = cand
        if c.exists() and c.is_file():
            return str(c)
    return None


def infer_split_type_from_key(key: str):
    return "cross_rotation" if key.startswith("fold") else "random_70_15_15"


def infer_fold_from_key(key: str):
    return key.split("_")[0] if key.startswith("fold") else "random"


def normalize_columns(df: pd.DataFrame, split_key: str):
    out = df.copy()
    if "sample_id" not in out.columns:
        if "sample_number" in out.columns:
            out["sample_id"] = out["sample_number"].astype(str)
        else:
            out["sample_id"] = "unknown"

    if "pork_cut" not in out.columns:
        out["pork_cut"] = out["sample_id"].astype(str).str.contains("shoulder", case=False).map({True: "shoulder", False: "belly"})

    if "split_type" not in out.columns:
        out["split_type"] = infer_split_type_from_key(split_key)

    if "fold" not in out.columns:
        out["fold"] = infer_fold_from_key(split_key)

    if "source" not in out.columns:
        out["source"] = "meatlens"

    return out


ALL_SPLIT_FILES = collect_split_files(SPLIT_MODE)
print("Detected split files:")
for k, p in ALL_SPLIT_FILES.items():
    print(f"- {k}: {p} | exists={p.exists()}")

SPLIT_DATA = {}
path_rows = []

for split_key, csv_path in ALL_SPLIT_FILES.items():
    if not csv_path.exists():
        print(f"[WARN] Missing file: {csv_path}")
        continue

    df = pd.read_csv(csv_path)
    df = normalize_columns(df, split_key)

    path_col = detect_image_path_column(df)
    if path_col is None:
        df["image_path_original"] = None
    else:
        df["image_path_original"] = df[path_col].astype(str)

    df["image_path_resolved"] = df.apply(lambda r: resolve_image_path(r, path_col=path_col, csv_path=csv_path), axis=1)

    missing_mask = df["image_path_resolved"].isna()
    missing_count = int(missing_mask.sum())
    missing_examples = df.loc[missing_mask, "image_path_original"].head(5).tolist() if path_col is not None else []

    SPLIT_DATA[split_key] = {
        "csv_path": csv_path,
        "df": df,
        "path_col": path_col,
        "missing_count": missing_count,
        "missing_examples": missing_examples,
    }

    path_rows.append({
        "split_key": split_key,
        "csv_path": str(csv_path),
        "detected_path_column": path_col,
        "row_count": int(len(df)),
        "missing_image_count": missing_count,
    })

PATH_DETECTION_DF = pd.DataFrame(path_rows)
print("\\nDetected image path columns:")
display(PATH_DETECTION_DF)

print("\\nMissing path examples:")
for k, info in SPLIT_DATA.items():
    print(f"{k}: missing={info['missing_count']}")
    for ex in info["missing_examples"][:3]:
        print(f"  - {ex}")"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """# ====================================================
# Split validation + export CSV summaries
# ====================================================

def validate_splits(split_data: dict):
    summary_rows = []
    missing_rows = []
    integrity_rows = []
    expected_rows = []

    for split_key, info in split_data.items():
        df = info["df"]
        path_col = info["path_col"]

        summary_rows.append({
            "split_key": split_key,
            "split_type": infer_split_type_from_key(split_key),
            "fold": infer_fold_from_key(split_key),
            "row_count": int(len(df)),
            "detected_path_column": path_col,
            "missing_image_count": int(df["image_path_resolved"].isna().sum()),
            "label_distribution": json.dumps(df["label"].value_counts().to_dict()),
            "sample_id_distribution": json.dumps(df["sample_id"].value_counts().to_dict()),
            "sample_number_distribution": json.dumps(df["sample_number"].value_counts().to_dict()) if "sample_number" in df.columns else json.dumps({}),
            "pork_cut_distribution": json.dumps(df["pork_cut"].value_counts().to_dict()),
        })

        missing_rows.append({
            "split_key": split_key,
            "missing_image_count": int(df["image_path_resolved"].isna().sum()),
            "example_missing_paths": json.dumps(info["missing_examples"]),
        })

    # Cross integrity checks
    for fold in ["fold1", "fold2", "fold3", "fold4"]:
        train_key, val_key, test_key = f"{fold}_train", f"{fold}_val", f"{fold}_test"
        if train_key not in split_data or val_key not in split_data or test_key not in split_data:
            continue

        held_out = EXPECTED_CROSS[fold]["held_out_sample"]
        train_df = split_data[train_key]["df"]
        val_df = split_data[val_key]["df"]
        test_df = split_data[test_key]["df"]

        in_train = bool(train_df["sample_id"].eq(held_out).any())
        in_val = bool(val_df["sample_id"].eq(held_out).any())
        in_test = bool(test_df["sample_id"].eq(held_out).any())

        integrity_rows.append({
            "fold": fold,
            "held_out_sample": held_out,
            "held_out_cut": EXPECTED_CROSS[fold]["held_out_cut"],
            "present_in_train": in_train,
            "present_in_val": in_val,
            "present_in_test": in_test,
            "integrity_pass": (in_test and not in_train and not in_val),
        })

    # Expected-vs-observed checks
    for fold in ["fold1", "fold2", "fold3", "fold4"]:
        if fold not in EXPECTED_CROSS:
            continue
        exp = EXPECTED_CROSS[fold]
        for part in ["train", "val", "test"]:
            key = f"{fold}_{part}"
            if key not in split_data:
                continue
            d = split_data[key]["df"]
            obs_count = int(len(d))
            exp_count = int(exp[f"{part}_count"])
            obs_labels = d["label"].value_counts().to_dict()
            exp_labels = exp[f"{part}_labels"]
            expected_rows.append({
                "split_key": key,
                "expected_count": exp_count,
                "observed_count": obs_count,
                "count_match": obs_count == exp_count,
                "expected_labels": json.dumps(exp_labels),
                "observed_labels": json.dumps(obs_labels),
                "label_match": obs_labels == exp_labels,
            })

    for part in ["train", "val", "test"]:
        key = f"random_{part}"
        if key not in split_data:
            continue
        d = split_data[key]["df"]
        obs_count = int(len(d))
        exp_count = int(EXPECTED_RANDOM[f"{part}_count"])
        obs_labels = d["label"].value_counts().to_dict()
        exp_labels = EXPECTED_RANDOM[f"{part}_labels"]
        obs_cuts = d["pork_cut"].value_counts().to_dict()
        exp_cuts = EXPECTED_RANDOM[f"{part}_cuts"]
        expected_rows.append({
            "split_key": key,
            "expected_count": exp_count,
            "observed_count": obs_count,
            "count_match": obs_count == exp_count,
            "expected_labels": json.dumps(exp_labels),
            "observed_labels": json.dumps(obs_labels),
            "label_match": obs_labels == exp_labels,
            "expected_cuts": json.dumps(exp_cuts),
            "observed_cuts": json.dumps(obs_cuts),
            "cut_match": obs_cuts == exp_cuts,
        })

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(missing_rows),
        pd.DataFrame(integrity_rows),
        pd.DataFrame(expected_rows),
    )

SUMMARY_DF, MISSING_DF, INTEGRITY_DF, EXPECTED_CHECK_DF = validate_splits(SPLIT_DATA)

display(SUMMARY_DF)
display(MISSING_DF)
display(INTEGRITY_DF)
display(EXPECTED_CHECK_DF)

SUMMARY_DF.to_csv(TRAINING_OUTPUTS / "split_validation_summary.csv", index=False)
MISSING_DF.to_csv(TRAINING_OUTPUTS / "missing_images_summary.csv", index=False)
INTEGRITY_DF.to_csv(TRAINING_OUTPUTS / "cross_rotation_integrity.csv", index=False)
EXPECTED_CHECK_DF.to_csv(TRAINING_OUTPUTS / "expected_count_checks.csv", index=False)

print(f"Saved: {TRAINING_OUTPUTS / 'split_validation_summary.csv'}")
print(f"Saved: {TRAINING_OUTPUTS / 'missing_images_summary.csv'}")

# Clean dataframes used for modeling (only unresolved paths removed)
CLEAN_SPLIT_DATA = {}
for k, info in SPLIT_DATA.items():
    df = info["df"]
    CLEAN_SPLIT_DATA[k] = df.loc[~df["image_path_resolved"].isna()].copy()

print("\\nClean rows per split:")
for k in sorted(CLEAN_SPLIT_DATA.keys()):
    print(f"{k}: {len(CLEAN_SPLIT_DATA[k])}/{len(SPLIT_DATA[k]['df'])}")"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """# ====================================================
# Validation graphs (requested visual graphs)
# ====================================================
if len(SUMMARY_DF) > 0:
    # 1) rows per split
    plt.figure(figsize=(12, 4))
    tmp = SUMMARY_DF.copy()
    plt.bar(tmp["split_key"], tmp["row_count"])
    plt.title("Rows per split")
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUTS / "rows_per_split.png", dpi=200)
    plt.show()

    # 2) missing image counts per split
    plt.figure(figsize=(12, 4))
    plt.bar(MISSING_DF["split_key"], MISSING_DF["missing_image_count"])
    plt.title("Missing image count per split")
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUTS / "missing_images_per_split.png", dpi=200)
    plt.show()

    # 3) label distribution per split (stacked)
    label_rows = []
    for k, info in CLEAN_SPLIT_DATA.items():
        vc = info["label"].value_counts().to_dict()
        for lbl in LABEL_ORDER:
            label_rows.append({"split_key": k, "label": lbl, "count": int(vc.get(lbl, 0))})
    label_df = pd.DataFrame(label_rows)
    pvt = label_df.pivot(index="split_key", columns="label", values="count").fillna(0)
    pvt.plot(kind="bar", stacked=True, figsize=(12, 5))
    plt.title("Label distribution per generated split")
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUTS / "label_distribution_per_split.png", dpi=200)
    plt.show()

print(f"Validation plots saved to: {PLOT_OUTPUTS}")"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """# ====================================================
# Deterministic preprocessing (non-ROI images)
# ====================================================

def _to_rgb_pil(image_input):
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    return Image.fromarray(np.asarray(image_input)).convert("RGB")


def preprocess_resize(img_pil, target_size=(224, 224)):
    return _to_rgb_pil(img_pil).resize(target_size, Image.BILINEAR)


def preprocess_resize_pad(img_pil, target_size=(224, 224), pad_color=(0, 0, 0)):
    img = _to_rgb_pil(img_pil)
    tw, th = target_size
    w, h = img.size
    scale = min(tw / w, th / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (tw, th), pad_color)
    left = (tw - nw) // 2
    top = (th - nh) // 2
    canvas.paste(resized, (left, top))
    return canvas


def preprocess_center_crop(img_pil, target_size=(224, 224)):
    img = _to_rgb_pil(img_pil)
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    crop = img.crop((left, top, left + side, top + side))
    return crop.resize(target_size, Image.BILINEAR)


def _extract_roi_box(row):
    if row is None:
        return None
    for keys in ROI_SETS:
        if all((k in row and pd.notna(row[k])) for k in keys):
            if keys in [("x", "y", "w", "h"), ("bbox_x", "bbox_y", "bbox_w", "bbox_h")]:
                x, y, w, h = [float(row[k]) for k in keys]
                if w <= 0 or h <= 0:
                    continue
                return (x, y, x + w, y + h)
            if keys == ("xmin", "ymin", "xmax", "ymax"):
                xmin, ymin, xmax, ymax = [float(row[k]) for k in keys]
                if xmax <= xmin or ymax <= ymin:
                    continue
                return (xmin, ymin, xmax, ymax)
    return None


def preprocess_roi_crop(img_pil, row=None, target_size=(224, 224)):
    img = _to_rgb_pil(img_pil)
    box = _extract_roi_box(row)
    if box is None:
        # fallback for now until ROI columns exist
        return preprocess_resize_pad(img, target_size=target_size)

    w, h = img.size
    xmin, ymin, xmax, ymax = box
    xmin = max(0, min(w - 1, int(round(xmin))))
    ymin = max(0, min(h - 1, int(round(ymin))))
    xmax = max(xmin + 1, min(w, int(round(xmax))))
    ymax = max(ymin + 1, min(h, int(round(ymax))))
    roi = img.crop((xmin, ymin, xmax, ymax))
    return roi.resize(target_size, Image.BILINEAR)


def load_image_for_model(path, image_crop_mode="center_crop", target_size=(224, 224), row=None):
    if path is None:
        raise ValueError("Image path is None")
    img = Image.open(path).convert("RGB")
    original_np = np.array(img)

    mode = str(image_crop_mode).lower().strip()
    if mode == "resize":
        proc = preprocess_resize(img, target_size)
    elif mode == "resize_pad":
        proc = preprocess_resize_pad(img, target_size)
    elif mode == "center_crop":
        proc = preprocess_center_crop(img, target_size)
    elif mode == "roi_crop":
        proc = preprocess_roi_crop(img, row=row, target_size=target_size)
    else:
        raise ValueError(f"Unsupported IMAGE_CROP_MODE: {image_crop_mode}")

    proc_np = np.asarray(proc).astype(np.float32) / 255.0
    return proc_np, original_np

print("Preprocessing functions ready. Default mode is center_crop.")"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """# ====================================================
# Example preprocessing visualization (few images)
# ====================================================
examples = []
for split_key in sorted(CLEAN_SPLIT_DATA.keys()):
    d = CLEAN_SPLIT_DATA[split_key]
    if len(d) == 0:
        continue
    for _, r in d.head(2).iterrows():
        examples.append((split_key, r))
        if len(examples) >= 3:
            break
    if len(examples) >= 3:
        break

if len(examples) == 0:
    print("No valid images found.")
else:
    fig, axes = plt.subplots(len(examples), 2, figsize=(10, 4 * len(examples)))
    if len(examples) == 1:
        axes = np.array([axes])

    for i, (split_key, row) in enumerate(examples):
        proc, orig = load_image_for_model(
            path=row["image_path_resolved"],
            image_crop_mode=IMAGE_CROP_MODE,
            target_size=TARGET_SIZE,
            row=row,
        )
        axes[i, 0].imshow(orig)
        axes[i, 0].set_title(f"Original | {split_key}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(proc)
        axes[i, 1].set_title(f"Processed 224x224 | {IMAGE_CROP_MODE}")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig(PLOT_OUTPUTS / "preprocessing_examples.png", dpi=200)
    plt.show()"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """# ====================================================
# Handcrafted features (defined, not used in CNN training yet)
# ====================================================

def _rgb_stats(arr):
    out = {}
    names = ["R", "G", "B"]
    for i, n in enumerate(names):
        ch = arr[:, :, i].astype(np.float32)
        out[f"mean_{n}"] = float(ch.mean())
        out[f"std_{n}"] = float(ch.std())
    return out


def _lab_stats(arr):
    lab = np.array(Image.fromarray(arr, mode="RGB").convert("LAB"), dtype=np.float32)
    out = {}
    names = ["L", "a", "b"]
    for i, n in enumerate(names):
        ch = lab[:, :, i]
        out[f"mean_{n}"] = float(ch.mean())
        out[f"std_{n}"] = float(ch.std())
    return out


def _hsv_stats(arr):
    hsv = np.array(Image.fromarray(arr, mode="RGB").convert("HSV"), dtype=np.float32)
    out = {}
    names = ["H", "S", "V"]
    for i, n in enumerate(names):
        ch = hsv[:, :, i]
        out[f"mean_{n}"] = float(ch.mean())
        out[f"std_{n}"] = float(ch.std())
    return out


def _glcm_stats(gray):
    feats = {}
    if not SKIMAGE_AVAILABLE:
        return feats
    distances = [1, 2, 4]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    g = graycomatrix(gray, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    for p in props:
        vals = graycoprops(g, p).ravel().astype(np.float32)
        feats[f"glcm_{p.lower()}_mean"] = float(vals.mean())
        feats[f"glcm_{p.lower()}_std"] = float(vals.std())
    return feats


def extract_color_texture_features(image_np):
    if image_np.dtype != np.uint8:
        arr = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
    else:
        arr = image_np

    feats = {}
    feats.update(_rgb_stats(arr))
    feats.update(_lab_stats(arr))
    feats.update(_hsv_stats(arr))

    gray = np.array(Image.fromarray(arr, mode="RGB").convert("L"), dtype=np.uint8)
    if SKIMAGE_AVAILABLE:
        feats.update(_glcm_stats(gray))
    else:
        warnings.warn("scikit-image not installed: GLCM skipped.")

    return feats


def build_feature_table(df, image_path_col="image_path_resolved"):
    rows = []
    for idx, row in df.iterrows():
        p = row.get(image_path_col, None)
        if p is None or (isinstance(p, float) and pd.isna(p)):
            continue
        try:
            proc, _ = load_image_for_model(p, image_crop_mode=IMAGE_CROP_MODE, target_size=TARGET_SIZE, row=row)
            feats = extract_color_texture_features(proc)
        except Exception:
            continue

        rec = {
            "row_index": int(idx),
            "image_path_resolved": str(p),
            "label": row.get("label", None),
            "sample_id": row.get("sample_id", None),
            "pork_cut": row.get("pork_cut", None),
            "split_type": row.get("split_type", None),
            "fold": row.get("fold", None),
        }
        rec.update(feats)
        rows.append(rec)
    return pd.DataFrame(rows)

# Example only (3 images), unless EXTRACT_FEATURES_NOW=True
if EXTRACT_FEATURES_NOW:
    for split_key, d in CLEAN_SPLIT_DATA.items():
        ft = build_feature_table(d)
        out = FEATURE_OUTPUTS / f"{split_key}_features.csv"
        ft.to_csv(out, index=False)
        print(f"Saved {out} ({len(ft)} rows)")
else:
    pool = []
    for k, d in CLEAN_SPLIT_DATA.items():
        if len(d) > 0:
            dd = d.copy()
            dd["_split_key"] = k
            pool.append(dd)
    if len(pool) > 0:
        all_pool = pd.concat(pool, ignore_index=True)
        ex = all_pool.sample(n=min(3, len(all_pool)), random_state=SEED_DEFAULT)
        ex_ft = build_feature_table(ex)
        ex_ft.to_csv(FEATURE_OUTPUTS / "example_3_images_features.csv", index=False)
        display(ex_ft)
        print(f"Saved example features: {FEATURE_OUTPUTS / 'example_3_images_features.csv'}")"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """# ====================================================
# CNN-only training/evaluation utilities
# ====================================================

BACKBONE_REGISTRY = {
    "MobileNetV3Small": {
        "constructor": MobileNetV3Small if TF_AVAILABLE else None,
        "preprocess_fn": preprocess_mobilenetv3 if TF_AVAILABLE else None,
        "preprocess_name": "mobilenet_v3.preprocess_input",
    },
    "EfficientNetB0": {
        "constructor": EfficientNetB0 if TF_AVAILABLE else None,
        "preprocess_fn": preprocess_efficientnetb0 if TF_AVAILABLE else None,
        "preprocess_name": "efficientnet.preprocess_input",
    },
    "ResNet50": {
        "constructor": ResNet50 if TF_AVAILABLE else None,
        "preprocess_fn": preprocess_resnet50 if TF_AVAILABLE else None,
        "preprocess_name": "resnet50.preprocess_input",
    },
    "MobileNetV2": {
        "constructor": MobileNetV2 if TF_AVAILABLE else None,
        "preprocess_fn": preprocess_mobilenetv2 if TF_AVAILABLE else None,
        "preprocess_name": "mobilenet_v2.preprocess_input",
    },
}


def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    if TF_AVAILABLE:
        tf.random.set_seed(seed)


def _apply_mild_augmentation_tf(image, seed=42):
    if not TF_AVAILABLE:
        return image

    if tf.random.uniform((), seed=seed) < 0.30:
        image = tf.image.flip_left_right(image)

    image = tf.image.random_brightness(image, max_delta=0.04, seed=seed)
    image = tf.image.random_contrast(image, lower=0.95, upper=1.05, seed=seed)

    if tf.random.uniform((), seed=seed + 1) < 0.30:
        noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.01, seed=seed)
        image = image + noise

    if tf.random.uniform((), seed=seed + 2) < 0.20:
        if tf.random.uniform((), seed=seed + 3) < 0.5:
            k = tf.constant([[1,1,1],[1,1,1],[1,1,1]], dtype=tf.float32) / 9.0
            k = tf.reshape(k, [3,3,1,1])
            k = tf.repeat(k, repeats=3, axis=2)
            image4 = tf.expand_dims(image, axis=0)
            image = tf.nn.depthwise_conv2d(image4, k, strides=[1,1,1,1], padding="SAME")[0]
        else:
            s = tf.constant(SHARPEN_KERNEL, dtype=tf.float32)
            s = tf.reshape(s, [3,3,1,1])
            s = tf.repeat(s, repeats=3, axis=2)
            image4 = tf.expand_dims(image, axis=0)
            image = tf.nn.depthwise_conv2d(image4, s, strides=[1,1,1,1], padding="SAME")[0]

    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def _extract_label_index(df: pd.DataFrame):
    y = df["label"].map(LABEL_TO_INDEX)
    if y.isna().any():
        bad = df.loc[y.isna(), "label"].unique().tolist()
        raise ValueError(f"Unknown labels found: {bad}")
    return y.astype(np.int32).values


def make_cnn_dataset(df, backbone_name="MobileNetV2", training=False, batch_size=32, image_crop_mode="center_crop", target_size=(224,224), seed=42):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not available.")

    if backbone_name not in BACKBONE_REGISTRY:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    preprocess_fn = BACKBONE_REGISTRY[backbone_name]["preprocess_fn"]
    if preprocess_fn is None:
        raise RuntimeError(f"Preprocess function missing for backbone: {backbone_name}")

    work = df.copy()
    work = work.loc[~work["image_path_resolved"].isna()].reset_index(drop=True)

    images = []
    labels = []
    for _, row in work.iterrows():
        path = row["image_path_resolved"]
        try:
            proc, _ = load_image_for_model(path, image_crop_mode=image_crop_mode, target_size=target_size, row=row)
            images.append(proc.astype(np.float32))
            labels.append(int(LABEL_TO_INDEX[row["label"]]))
        except Exception:
            continue

    if len(images) == 0:
        raise ValueError("No valid images after preprocessing.")

    x = np.stack(images).astype(np.float32)
    y = np.asarray(labels, dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((x, y))

    if training:
        ds = ds.shuffle(buffer_size=len(x), seed=seed, reshuffle_each_iteration=True)

    def _map_fn(img, label):
        img = tf.cast(img, tf.float32)
        if training:
            img = _apply_mild_augmentation_tf(img, seed=seed)
        img = preprocess_fn(img * 255.0)
        return img, tf.cast(label, tf.int32)

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, len(x)


def build_cnn_model(backbone_name="MobileNetV2", input_shape=(224,224,3), num_classes=3):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not available.")

    reg = BACKBONE_REGISTRY[backbone_name]
    backbone_ctor = reg["constructor"]
    backbone = backbone_ctor(include_top=False, weights="imagenet", input_shape=input_shape)
    backbone.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.20)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name=f"{backbone_name}_cnn_only")
    return model, backbone


def _make_optimizer(lr, weight_decay=1e-4):
    if hasattr(tf.keras.optimizers, "AdamW"):
        return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    return tf.keras.optimizers.Adam(learning_rate=lr)


class ValF1Callback(tf.keras.callbacks.Callback):
    def __init__(self, val_ds):
        super().__init__()
        self.val_ds = val_ds

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_true = []
        y_pred = []
        for xb, yb in self.val_ds:
            probs = self.model.predict(xb, verbose=0)
            pred = np.argmax(probs, axis=1)
            y_true.extend(yb.numpy().tolist())
            y_pred.extend(pred.tolist())
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        logs["val_f1_macro"] = float(f1)
        print(f" - val_f1_macro: {f1:.4f}")


def train_two_phase_cnn(model, backbone, train_ds, val_ds, class_weight=None, backbone_name="MobileNetV2", run_name="run"):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not available.")

    ckpt_dir = MODEL_OUTPUTS / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{run_name}_best.keras"

    f1_cb = ValF1Callback(val_ds)
    callbacks = [
        f1_cb,
        EarlyStopping(monitor="val_f1_macro", mode="max", patience=8, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=str(ckpt_path), monitor="val_f1_macro", mode="max", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1),
    ]

    model.compile(
        optimizer=_make_optimizer(5e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    hist_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    backbone.trainable = True
    total_layers = len(backbone.layers)
    unfreeze_from = int(total_layers * 0.7)

    for i, layer in enumerate(backbone.layers):
        if i < unfreeze_from:
            layer.trainable = False
        else:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

    fine_lr = FINE_TUNE_LR.get(backbone_name, 1e-5)
    model.compile(
        optimizer=_make_optimizer(fine_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    hist_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINE,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    history = {
        "head": hist_head.history,
        "fine": hist_fine.history,
        "best_checkpoint": str(ckpt_path),
    }
    return model, history


def evaluate_cnn_model(model, test_ds, split_name="unknown", backbone_name="unknown", save_plots=True):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not available.")

    y_true = []
    y_pred = []
    y_prob = []

    for xb, yb in test_ds:
        probs = model.predict(xb, verbose=0)
        pred = np.argmax(probs, axis=1)
        y_true.extend(yb.numpy().tolist())
        y_pred.extend(pred.tolist())
        y_prob.extend(probs.tolist())

    y_true = np.array(y_true, dtype=np.int32)
    y_pred = np.array(y_pred, dtype=np.int32)
    y_prob = np.array(y_prob, dtype=np.float32)

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    cm_norm = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cls_report = classification_report(y_true, y_pred, target_names=LABEL_ORDER, output_dict=True, zero_division=0)
    pred_dist = pd.Series(y_pred).map(INDEX_TO_LABEL).value_counts().to_dict()

    if save_plots:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(cm, cmap="Blues")
        axes[0].set_title(f"Confusion Matrix\\n{split_name} | {backbone_name}")
        axes[0].set_xticks(range(3)); axes[0].set_yticks(range(3))
        axes[0].set_xticklabels(LABEL_ORDER, rotation=30)
        axes[0].set_yticklabels(LABEL_ORDER)

        axes[1].imshow(cm_norm, cmap="Oranges", vmin=0, vmax=1)
        axes[1].set_title("Normalized Confusion Matrix")
        axes[1].set_xticks(range(3)); axes[1].set_yticks(range(3))
        axes[1].set_xticklabels(LABEL_ORDER, rotation=30)
        axes[1].set_yticklabels(LABEL_ORDER)

        plt.tight_layout()
        out_cm = PLOT_OUTPUTS / f"cm_{split_name}_{backbone_name}.png"
        plt.savefig(out_cm, dpi=180)
        plt.close(fig)

    return {
        "accuracy": float(acc),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "normalized_confusion_matrix": cm_norm.tolist(),
        "classification_report": cls_report,
        "prediction_distribution": pred_dist,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def _get_split_triplet(split_mode="cross_rotation", split_name="fold1"):
    if split_mode == "cross_rotation":
        train_key = f"{split_name}_train"
        val_key = f"{split_name}_val"
        test_key = f"{split_name}_test"
    elif split_mode == "random_70_15_15":
        train_key, val_key, test_key = "random_train", "random_val", "random_test"
    else:
        raise ValueError("split_mode must be 'cross_rotation' or 'random_70_15_15'")

    for k in [train_key, val_key, test_key]:
        if k not in CLEAN_SPLIT_DATA:
            raise KeyError(f"Missing split dataframe: {k}")
    return train_key, val_key, test_key


def _compute_class_weights_from_df(df):
    y = _extract_label_index(df)
    classes = np.array([0, 1, 2], dtype=np.int32)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, cw)}


def _measure_model_size_mb(model_path):
    p = Path(model_path)
    if not p.exists():
        return None
    return float(p.stat().st_size / (1024 ** 2))


def _measure_inference_speed(model, sample_batch):
    t0 = time.perf_counter()
    _ = model.predict(sample_batch, verbose=0)
    dt = time.perf_counter() - t0
    bs = sample_batch.shape[0]
    return float((dt / max(bs, 1)) * 1000.0)


def _save_model_metadata(out_json_path, metadata: dict):
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def _convert_to_tflite(h5_path, tflite_path):
    if not TF_AVAILABLE:
        return False
    try:
        model = tf.keras.models.load_model(h5_path, compile=False)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        return True
    except Exception as e:
        print(f"[WARN] TFLite conversion failed: {e}")
        return False


def run_single_cnn_experiment(split_mode="cross_rotation", split_name="fold1", backbone_name="MobileNetV2", seed=42):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required.")

    set_global_seed(seed)
    train_key, val_key, test_key = _get_split_triplet(split_mode=split_mode, split_name=split_name)
    train_df = CLEAN_SPLIT_DATA[train_key].copy()
    val_df = CLEAN_SPLIT_DATA[val_key].copy()
    test_df = CLEAN_SPLIT_DATA[test_key].copy()

    class_weights = _compute_class_weights_from_df(train_df)

    train_ds, n_train = make_cnn_dataset(train_df, backbone_name=backbone_name, training=True, batch_size=BATCH_SIZE, image_crop_mode=IMAGE_CROP_MODE, target_size=TARGET_SIZE, seed=seed)
    val_ds, n_val = make_cnn_dataset(val_df, backbone_name=backbone_name, training=False, batch_size=BATCH_SIZE, image_crop_mode=IMAGE_CROP_MODE, target_size=TARGET_SIZE, seed=seed)
    test_ds, n_test = make_cnn_dataset(test_df, backbone_name=backbone_name, training=False, batch_size=BATCH_SIZE, image_crop_mode=IMAGE_CROP_MODE, target_size=TARGET_SIZE, seed=seed)

    model, backbone = build_cnn_model(backbone_name=backbone_name, input_shape=(224,224,3), num_classes=NUM_CLASSES)

    run_id = f"{split_mode}_{split_name}_{backbone_name}_seed{seed}"
    model, history = train_two_phase_cnn(
        model=model,
        backbone=backbone,
        train_ds=train_ds,
        val_ds=val_ds,
        class_weight=class_weights,
        backbone_name=backbone_name,
        run_name=run_id,
    )

    eval_res = evaluate_cnn_model(model, test_ds, split_name=split_name, backbone_name=backbone_name, save_plots=True)

    run_model_h5 = MODEL_OUTPUTS / f"{run_id}.h5"
    model.save(run_model_h5)

    sample_batch = next(iter(test_ds))[0]
    inference_ms = _measure_inference_speed(model, sample_batch)
    model_size_mb = _measure_model_size_mb(run_model_h5)

    held_out_sample = None
    held_out_cut = None
    if split_mode == "cross_rotation" and split_name in EXPECTED_CROSS:
        held_out_sample = EXPECTED_CROSS[split_name]["held_out_sample"]
        held_out_cut = EXPECTED_CROSS[split_name]["held_out_cut"]

    metadata = {
        "backbone": backbone_name,
        "preprocess_function_name": BACKBONE_REGISTRY[backbone_name]["preprocess_name"],
        "input_size": list(TARGET_SIZE),
        "image_crop_mode": IMAGE_CROP_MODE,
        "model_input_mode": MODEL_INPUT_MODE,
        "split_mode": split_mode,
        "split_name": split_name,
        "label_order": LABEL_ORDER,
        "class_index_mapping": LABEL_TO_INDEX,
        "macro_f1": eval_res["macro_f1"],
        "accuracy": eval_res["accuracy"],
    }
    run_metadata_path = MODEL_OUTPUTS / f"{run_id}_metadata.json"
    _save_model_metadata(run_metadata_path, metadata)

    return {
        "run_id": run_id,
        "split_mode": split_mode,
        "split_name": split_name,
        "backbone": backbone_name,
        "seed": seed,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "held_out_sample": held_out_sample,
        "held_out_cut": held_out_cut,
        "class_weights": class_weights,
        "history": history,
        "metrics": eval_res,
        "model_h5": str(run_model_h5),
        "metadata_json": str(run_metadata_path),
        "model_size_mb": model_size_mb,
        "inference_ms_per_image": inference_ms,
    }


def run_cnn_training(split_mode=None, backbones=None, seeds=None):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required.")

    split_mode = split_mode or SPLIT_MODE
    backbones = backbones or BACKBONES
    seeds = seeds or RUN_SEEDS

    if not RUN_FULL_TRAINING:
        print("RUN_FULL_TRAINING=False -> skipping full training loop.")
        return None

    runs = []
    split_plan = []
    if split_mode in ["cross_rotation", "both"]:
        split_plan.extend([("cross_rotation", f"fold{i}") for i in [1,2,3,4]])
    if split_mode in ["random_70_15_15", "both"]:
        split_plan.append(("random_70_15_15", "random"))

    for seed in seeds:
        for backbone in backbones:
            for smode, sname in split_plan:
                print(f"Running: mode={smode}, split={sname}, backbone={backbone}, seed={seed}")
                res = run_single_cnn_experiment(
                    split_mode=smode,
                    split_name=sname,
                    backbone_name=backbone,
                    seed=seed,
                )
                runs.append(res)

    if len(runs) == 0:
        print("No runs executed.")
        return None

    metric_rows, pred_rows, size_rows, speed_rows = [], [], [], []
    for r in runs:
        m = r["metrics"]
        metric_rows.append({
            "run_id": r["run_id"],
            "seed": r["seed"],
            "backbone": r["backbone"],
            "split_mode": r["split_mode"],
            "split_name": r["split_name"],
            "held_out_sample": r["held_out_sample"],
            "held_out_cut": r["held_out_cut"],
            "accuracy": m["accuracy"],
            "macro_precision": m["macro_precision"],
            "macro_recall": m["macro_recall"],
            "macro_f1": m["macro_f1"],
        })
        pred_rows.append({"run_id": r["run_id"], "prediction_distribution": json.dumps(m["prediction_distribution"])})
        size_rows.append({"run_id": r["run_id"], "model_size_mb": r["model_size_mb"]})
        speed_rows.append({"run_id": r["run_id"], "inference_ms_per_image": r["inference_ms_per_image"]})

    seed_metrics = pd.DataFrame(metric_rows)
    fold_metrics = seed_metrics[seed_metrics["split_mode"] == "cross_rotation"].copy()
    random_metrics = seed_metrics[seed_metrics["split_mode"] == "random_70_15_15"].copy()

    cross_summary = pd.DataFrame()
    if len(fold_metrics) > 0:
        cross_summary = fold_metrics.groupby(["backbone", "seed"]).agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
        ).reset_index()

        aux = fold_metrics.groupby(["backbone", "seed", "held_out_cut"]).agg(
            macro_f1_mean=("macro_f1", "mean"),
            accuracy_mean=("accuracy", "mean"),
        ).reset_index()
        cross_pivot = aux.pivot_table(index=["backbone", "seed"], columns="held_out_cut", values="macro_f1_mean").reset_index()
        cross_summary = cross_summary.merge(cross_pivot, on=["backbone", "seed"], how="left")

    random_summary = pd.DataFrame()
    if len(random_metrics) > 0:
        random_summary = random_metrics.groupby(["backbone", "seed"]).agg(
            accuracy=("accuracy", "mean"),
            macro_precision=("macro_precision", "mean"),
            macro_recall=("macro_recall", "mean"),
            macro_f1=("macro_f1", "mean"),
        ).reset_index()

    backbone_summary = seed_metrics.groupby(["backbone"]).agg(
        accuracy_mean=("accuracy", "mean"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
    ).reset_index()

    pred_df = pd.DataFrame(pred_rows)
    size_df = pd.DataFrame(size_rows)
    speed_df = pd.DataFrame(speed_rows)

    seed_metrics.to_csv(TRAINING_OUTPUTS / "seed_metrics.csv", index=False)
    fold_metrics.to_csv(TRAINING_OUTPUTS / "fold_metrics.csv", index=False)
    cross_summary.to_csv(TRAINING_OUTPUTS / "cross_rotation_summary.csv", index=False)
    random_summary.to_csv(TRAINING_OUTPUTS / "random_baseline_summary.csv", index=False)
    backbone_summary.to_csv(TRAINING_OUTPUTS / "backbone_summary.csv", index=False)
    pred_df.to_csv(TRAINING_OUTPUTS / "prediction_distribution_by_run.csv", index=False)
    size_df.to_csv(TRAINING_OUTPUTS / "model_size_table.csv", index=False)
    speed_df.to_csv(TRAINING_OUTPUTS / "inference_speed_table.csv", index=False)

    best_idx = seed_metrics["macro_f1"].idxmax()
    best_row = seed_metrics.loc[best_idx]
    best_run_id = best_row["run_id"]

    src_h5 = MODEL_OUTPUTS / f"{best_run_id}.h5"
    src_meta = MODEL_OUTPUTS / f"{best_run_id}_metadata.json"
    dst_h5 = MODEL_OUTPUTS / "meatlens_best_model.h5"
    dst_tflite = MODEL_OUTPUTS / "meatlens_best_model.tflite"
    dst_meta = MODEL_OUTPUTS / "meatlens_best_model_metadata.json"

    if src_h5.exists():
        shutil.copy2(src_h5, dst_h5)
    if src_meta.exists():
        shutil.copy2(src_meta, dst_meta)
    _convert_to_tflite(dst_h5, dst_tflite)

    print("Saved training outputs and best model artifacts.")
    return {
        "seed_metrics": seed_metrics,
        "fold_metrics": fold_metrics,
        "cross_summary": cross_summary,
        "random_summary": random_summary,
        "backbone_summary": backbone_summary,
    }

print("CNN training/evaluation functions defined.")"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """# ====================================================
# Final prediction / upload cell (best model inference)
# ====================================================
USER_IMAGE_PATH = "path/to/image.jpg"


def _get_preprocess_fn_by_name(name: str):
    name = str(name)
    if "mobilenet_v3" in name:
        return preprocess_mobilenetv3
    if "efficientnet" in name:
        return preprocess_efficientnetb0
    if "resnet50" in name:
        return preprocess_resnet50
    if "mobilenet_v2" in name:
        return preprocess_mobilenetv2
    return preprocess_mobilenetv2


def _freshness_score_and_recommendation(predicted_class: str, confidence: float):
    if predicted_class == "fresh":
        score = 70 + 30 * confidence
    elif predicted_class == "not fresh":
        score = 40 + 20 * confidence
    else:
        score = 39 - 34 * confidence

    score = float(max(0.0, min(100.0, score)))
    if score >= 70:
        rec = "Good for Consumption"
    elif score >= 40:
        rec = "Consume Immediately"
    else:
        rec = "Not Suitable"
    return score, rec


def predict_with_best_model(user_image_path: str, show_images: bool = True):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow not available.")

    best_model_path = MODEL_OUTPUTS / "meatlens_best_model.h5"
    best_meta_path = MODEL_OUTPUTS / "meatlens_best_model_metadata.json"

    if not best_model_path.exists() or not best_meta_path.exists():
        raise FileNotFoundError("Best model or metadata not found in training_outputs/models/.")

    with open(best_meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    model = tf.keras.models.load_model(best_model_path, compile=False)

    image_crop_mode = meta.get("image_crop_mode", "center_crop")
    input_size = tuple(meta.get("input_size", [224, 224]))
    preprocess_name = meta.get("preprocess_function_name", "mobilenet_v2.preprocess_input")
    preprocess_fn = _get_preprocess_fn_by_name(preprocess_name)

    proc, orig = load_image_for_model(
        user_image_path,
        image_crop_mode=image_crop_mode,
        target_size=input_size,
        row=None,
    )

    x = np.expand_dims(proc.astype(np.float32) * 255.0, axis=0)
    x = preprocess_fn(x)
    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))

    class_map = meta.get("class_index_mapping", LABEL_TO_INDEX)
    idx_to_class = {int(v): k for k, v in class_map.items()}
    pred_class = idx_to_class.get(pred_idx, INDEX_TO_LABEL[pred_idx])
    confidence = float(probs[pred_idx])

    score, recommendation = _freshness_score_and_recommendation(pred_class, confidence)

    if show_images:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].imshow(orig)
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(proc)
        axes[1].set_title(f"Processed ({image_crop_mode})")
        axes[1].axis("off")
        plt.tight_layout()
        plt.show()

    prob_dict = {idx_to_class.get(i, str(i)): float(probs[i]) for i in range(len(probs))}

    print(f"predicted_class: {pred_class}")
    print(f"confidence: {confidence:.4f}")
    print(f"freshness_score: {score:.2f}")
    print(f"recommendation: {recommendation}")
    print(f"class_probabilities: {prob_dict}")
    print("The freshness score is a rule-based decision-support score derived from model confidence, not a direct biochemical measurement.")

    return {
        "predicted_class": pred_class,
        "confidence": confidence,
        "freshness_score": score,
        "recommendation": recommendation,
        "class_probabilities": prob_dict,
    }


if WIDGETS_AVAILABLE:
    print("ipywidgets available: optional upload widget displayed below.")
    uploader = widgets.FileUpload(accept='image/*', multiple=False)
    display(uploader)
else:
    print("ipywidgets not available. Use USER_IMAGE_PATH string path instead.")

# Example manual call (run only after best model exists):
# result = predict_with_best_model(USER_IMAGE_PATH, show_images=True)"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """# ====================================================
# Full training gate (kept OFF by default)
# ====================================================
if RUN_FULL_TRAINING:
    _ = run_cnn_training(split_mode=SPLIT_MODE, backbones=BACKBONES, seeds=RUN_SEEDS)
else:
    print("RUN_FULL_TRAINING=False -> full training is skipped.")"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """# ====================================================
# Debug single experiment (DO NOT EXECUTE until explicitly requested)
# ====================================================
# debug_result = run_single_cnn_experiment(
#     split_mode="cross_rotation",
#     split_name="fold1",
#     backbone_name="MobileNetV2",
#     seed=42,
# )"""
        )
    )

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }

    out_path = Path("new2.ipynb")
    nbf.write(nb, out_path)
    print(f"Revised notebook written: {out_path.resolve()}")


if __name__ == "__main__":
    main()

