#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import mobilenetv3small_8fold_processed_roi_cnn_only_lib as base_lib

plt = base_lib.plt
SEABORN_AVAILABLE = base_lib.SEABORN_AVAILABLE
sns = getattr(base_lib.base, "sns", None)

PROJECT_ROOT = Path.cwd()
OUTPUT_ROOT = PROJECT_ROOT / "training_outputs" / "mobilenetv3small_8fold_processed_roi_cnn_only"
FIGURES_ROOT = OUTPUT_ROOT / "figures"
PREDICTIONS_ROOT = OUTPUT_ROOT / "predictions"
SEED_METRICS_PATH = OUTPUT_ROOT / "processed_roi8_cnn_only_seed_metrics.csv"
ALL_SAMPLED_IMAGES_PATH = (
    PROJECT_ROOT
    / "generated_splits"
    / "cross_rotation_interval200_8samples_processed_roi"
    / "all_sampled_images.csv"
)

TRANSITION_AWARE_PREDICTIONS_PATH = OUTPUT_ROOT / "processed_roi8_cnn_only_transition_aware_predictions.csv"
TRANSITION_AWARE_SUMMARY_PATH = OUTPUT_ROOT / "processed_roi8_cnn_only_transition_aware_summary.csv"
TRANSITION_ZONE_SUMMARY_PATH = OUTPUT_ROOT / "processed_roi8_cnn_only_transition_zone_summary.csv"
TRANSITION_ZONE_CM_5X6_CSV_PATH = OUTPUT_ROOT / "transition_zone_confusion_matrix_5x6.csv"
TRANSITION_ZONE_CM_5X5_CSV_PATH = OUTPUT_ROOT / "transition_zone_confusion_matrix_5x5.csv"
TRANSITION_ZONE_CM_5X5_NORM_CSV_PATH = OUTPUT_ROOT / "normalized_transition_zone_confusion_matrix_5x5.csv"

ZONE_ORDER = [
    "clear_fresh",
    "borderline_fresh_not_fresh",
    "clear_not_fresh",
    "borderline_not_fresh_spoiled",
    "clear_spoiled",
]
PREDICTED_ZONE_ORDER_5X6 = ZONE_ORDER + ["uncertain_non_adjacent_conflict"]
ZONE_ACCEPTED_PREDICTIONS = {
    "clear_fresh": ["fresh"],
    "borderline_fresh_not_fresh": ["fresh", "not fresh"],
    "clear_not_fresh": ["not fresh"],
    "borderline_not_fresh_spoiled": ["not fresh", "spoiled"],
    "clear_spoiled": ["spoiled"],
}
SECONDARY_NOTE = (
    "Transition-aware accuracy is a secondary analysis only. Official model evaluation "
    "remains the 3-class confusion matrix, strict accuracy, macro F1, top-2 accuracy, "
    "adjacent accuracy, and severe error rate."
)


def ensure_output_dirs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)


def _safe_str(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _parse_timeframe(series: pd.Series) -> pd.Series:
    return pd.to_timedelta(_safe_str(series), errors="coerce")


def _build_path_key(series: pd.Series) -> pd.Series:
    return _safe_str(series).str.replace("\\", "/", regex=False).str.lower()


def load_seed_metrics_df() -> pd.DataFrame:
    if not SEED_METRICS_PATH.exists():
        raise FileNotFoundError(f"Missing seed metrics CSV: {SEED_METRICS_PATH}")
    df = pd.read_csv(SEED_METRICS_PATH)
    if df.empty:
        raise RuntimeError("Seed metrics CSV exists but is empty.")
    return df


def load_run_prediction_frames(seed_metrics_df: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, row in seed_metrics_df.iterrows():
        pred_path = Path(str(row.get("predictions_path", "")).strip())
        if not pred_path.exists():
            print(f"[WARN] Missing prediction CSV: {pred_path}")
            continue
        pred_df = pd.read_csv(pred_path)
        pred_df["fold_name"] = row.get("fold_name", "")
        pred_df["seed"] = row.get("seed", "")
        pred_df["held_out_sample"] = row.get("held_out_sample", "")
        pred_df["held_out_cut"] = row.get("held_out_cut", "")
        pred_df["capture_source"] = row.get("capture_source", pred_df.get("capture_source", ""))
        pred_df["phone_group"] = row.get("phone_group", pred_df.get("phone_group", ""))
        pred_df["run_stem"] = row.get("run_stem", "")
        frames.append(pred_df)
    if not frames:
        raise RuntimeError("No prediction CSVs could be loaded from the seed metrics file.")
    out = pd.concat(frames, ignore_index=True)
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce").astype("Int64")
    return out


def load_sequence_reference_df() -> pd.DataFrame:
    if not ALL_SAMPLED_IMAGES_PATH.exists():
        raise FileNotFoundError(f"Missing all_sampled_images CSV: {ALL_SAMPLED_IMAGES_PATH}")
    ref_df = pd.read_csv(ALL_SAMPLED_IMAGES_PATH).copy()
    ref_df["sample_id"] = _safe_str(ref_df["sample_id"]).str.lower()
    ref_df["label"] = _safe_str(ref_df["label"]).str.lower()
    ref_df["time_frame_td"] = _parse_timeframe(ref_df["time_frame"]) if "time_frame" in ref_df.columns else pd.NaT
    if "image_path_resolved" not in ref_df.columns:
        path_col = "file_destination" if "file_destination" in ref_df.columns else "image_path"
        ref_df["image_path_resolved"] = _safe_str(ref_df[path_col])
    ref_df["image_file_name"] = _safe_str(ref_df["image_file_name"])
    ref_df["image_path_key"] = _build_path_key(ref_df["image_path_resolved"])
    ref_df["image_file_name_key"] = _safe_str(ref_df["image_file_name"]).str.lower()
    ref_df["sort_fallback"] = np.where(
        ref_df["image_file_name_key"].ne(""),
        ref_df["image_file_name_key"],
        ref_df["image_path_key"],
    )

    ordered_groups: list[pd.DataFrame] = []
    for (_, _), group_df in ref_df.groupby(["sample_id", "label"], sort=True):
        ordered = group_df.sort_values(
            by=["time_frame_td", "image_file_name_key", "image_path_key"],
            na_position="last",
        ).reset_index(drop=True)
        group_size = len(ordered)
        ordered["position_index"] = np.arange(group_size, dtype=int)
        if group_size <= 1:
            ordered["position_ratio"] = 0.0
        else:
            ordered["position_ratio"] = ordered["position_index"] / float(group_size - 1)
        ordered_groups.append(ordered)

    ordered_df = pd.concat(ordered_groups, ignore_index=True)
    return ordered_df[
        [
            "sample_id",
            "label",
            "image_file_name",
            "image_file_name_key",
            "image_path_resolved",
            "image_path_key",
            "time_frame",
            "position_index",
            "position_ratio",
        ]
    ].copy()


def assign_transition_zone(label: str, position_ratio: float) -> str:
    label = str(label).strip().lower()
    ratio = float(position_ratio)
    if label == "fresh":
        return "clear_fresh" if ratio <= 0.75 else "borderline_fresh_not_fresh"
    if label == "not fresh":
        if ratio <= 0.25:
            return "borderline_fresh_not_fresh"
        if ratio <= 0.75:
            return "clear_not_fresh"
        return "borderline_not_fresh_spoiled"
    if label == "spoiled":
        return "borderline_not_fresh_spoiled" if ratio <= 0.25 else "clear_spoiled"
    raise ValueError(f"Unexpected label for transition zone assignment: {label}")


def assign_predicted_transition_zone(
    top_confidence: float,
    top_class: str,
    second_class: str,
) -> str:
    top_class = str(top_class).strip().lower()
    second_class = str(second_class).strip().lower()
    if float(top_confidence) >= 0.90:
        if top_class == "fresh":
            return "clear_fresh"
        if top_class == "not fresh":
            return "clear_not_fresh"
        if top_class == "spoiled":
            return "clear_spoiled"
    class_pair = {top_class, second_class}
    if class_pair == {"fresh", "not fresh"}:
        return "borderline_fresh_not_fresh"
    if class_pair == {"not fresh", "spoiled"}:
        return "borderline_not_fresh_spoiled"
    if class_pair == {"fresh", "spoiled"}:
        return "uncertain_non_adjacent_conflict"
    return "uncertain_non_adjacent_conflict"


def map_predicted_transition_zone_to_5x5(predicted_transition_zone: str, top_class: str) -> str:
    zone = str(predicted_transition_zone).strip()
    top_class = str(top_class).strip().lower()
    if zone != "uncertain_non_adjacent_conflict":
        return zone
    if top_class == "fresh":
        return "clear_fresh"
    if top_class == "not fresh":
        return "clear_not_fresh"
    if top_class == "spoiled":
        return "clear_spoiled"
    return "clear_not_fresh"


def annotate_transition_aware_predictions(
    prediction_df: pd.DataFrame,
    sequence_ref_df: pd.DataFrame,
) -> pd.DataFrame:
    pred_df = prediction_df.copy()
    pred_df["sample_id"] = _safe_str(pred_df["sample_id"]).str.lower()
    pred_df["label"] = _safe_str(pred_df["label"]).str.lower()
    pred_df["image_file_name"] = _safe_str(pred_df["image_file_name"])
    pred_df["image_path_resolved"] = _safe_str(pred_df["image_path_resolved"])
    pred_df["image_path_key"] = _build_path_key(pred_df["image_path_resolved"])
    pred_df["image_file_name_key"] = _safe_str(pred_df["image_file_name"]).str.lower()

    merge_cols = ["sample_id", "label", "image_path_key"]
    ref_by_path = sequence_ref_df.drop_duplicates(subset=merge_cols)[
        ["sample_id", "label", "image_path_key", "position_index", "position_ratio"]
    ]
    merged = pred_df.merge(ref_by_path, on=merge_cols, how="left")

    missing_mask = merged["position_ratio"].isna()
    if missing_mask.any():
        ref_by_name = sequence_ref_df.drop_duplicates(subset=["sample_id", "label", "image_file_name_key"])[
            ["sample_id", "label", "image_file_name_key", "position_index", "position_ratio"]
        ]
        fallback = merged.loc[missing_mask].drop(columns=["position_index", "position_ratio"]).merge(
            ref_by_name,
            on=["sample_id", "label", "image_file_name_key"],
            how="left",
        )
        merged.loc[missing_mask, "position_index"] = fallback["position_index"].to_numpy()
        merged.loc[missing_mask, "position_ratio"] = fallback["position_ratio"].to_numpy()

    unresolved = int(merged["position_ratio"].isna().sum())
    if unresolved > 0:
        print(f"[WARN] {unresolved} prediction rows could not be matched to a sequence position.")

    merged["position_index"] = pd.to_numeric(merged["position_index"], errors="coerce").astype("Int64")
    merged["position_ratio"] = pd.to_numeric(merged["position_ratio"], errors="coerce")
    merged["transition_zone"] = merged.apply(
        lambda row: assign_transition_zone(row["label"], row["position_ratio"])
        if pd.notna(row["position_ratio"])
        else "",
        axis=1,
    )
    merged["true_transition_zone"] = merged["transition_zone"]
    merged["predicted_transition_zone"] = merged.apply(
        lambda row: assign_predicted_transition_zone(
            row.get("top_confidence", np.nan),
            row.get("top_class", ""),
            row.get("second_class", ""),
        ),
        axis=1,
    )
    merged["predicted_transition_zone_5x5"] = merged.apply(
        lambda row: map_predicted_transition_zone_to_5x5(
            row.get("predicted_transition_zone", ""),
            row.get("top_class", ""),
        ),
        axis=1,
    )
    merged["accepted_predictions"] = merged["transition_zone"].map(
        lambda zone: json.dumps(ZONE_ACCEPTED_PREDICTIONS.get(zone, []))
    )
    merged["is_transition_aware_correct"] = merged.apply(
        lambda row: str(row.get("predicted_label", "")).strip().lower() in ZONE_ACCEPTED_PREDICTIONS.get(row["transition_zone"], []),
        axis=1,
    )
    merged["transition_analysis_note"] = SECONDARY_NOTE
    return merged


def _summarize_accuracy(df: pd.DataFrame, summary_type: str, group_col: str | None = None) -> pd.DataFrame:
    if group_col is None:
        return pd.DataFrame(
            [
                {
                    "summary_type": summary_type,
                    "group_value": "overall",
                    "prediction_rows": int(len(df)),
                    "transition_aware_accuracy": float(df["is_transition_aware_correct"].astype(float).mean()),
                }
            ]
        )

    summary_rows = []
    for group_value, group_df in df.groupby(group_col, dropna=False):
        summary_rows.append(
            {
                "summary_type": summary_type,
                "group_value": str(group_value),
                "prediction_rows": int(len(group_df)),
                "transition_aware_accuracy": float(group_df["is_transition_aware_correct"].astype(float).mean()),
            }
        )
    return pd.DataFrame(summary_rows)


def build_transition_aware_summaries(
    annotated_df: pd.DataFrame,
    seed_metrics_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_level_df = (
        annotated_df.groupby(["fold_name", "seed", "held_out_sample", "held_out_cut", "phone_group"], dropna=False, as_index=False)
        .agg(
            transition_aware_accuracy=("is_transition_aware_correct", lambda s: float(pd.Series(s).astype(float).mean())),
            prediction_rows=("is_transition_aware_correct", "size"),
            pork_cut=("pork_cut", "first"),
        )
    )

    summary_frames = [
        _summarize_accuracy(annotated_df, "overall"),
        run_level_df.groupby("fold_name", as_index=False).agg(
            summary_type=("fold_name", lambda _: "fold"),
            group_value=("fold_name", "first"),
            prediction_rows=("prediction_rows", "sum"),
            transition_aware_accuracy=("transition_aware_accuracy", "mean"),
            transition_aware_accuracy_std=("transition_aware_accuracy", "std"),
        ),
        run_level_df.groupby("seed", as_index=False).agg(
            summary_type=("seed", lambda _: "seed"),
            group_value=("seed", lambda s: str(s.iloc[0])),
            prediction_rows=("prediction_rows", "sum"),
            transition_aware_accuracy=("transition_aware_accuracy", "mean"),
            transition_aware_accuracy_std=("transition_aware_accuracy", "std"),
        ),
        run_level_df.groupby("held_out_sample", as_index=False).agg(
            summary_type=("held_out_sample", lambda _: "held_out_sample"),
            group_value=("held_out_sample", "first"),
            prediction_rows=("prediction_rows", "sum"),
            transition_aware_accuracy=("transition_aware_accuracy", "mean"),
            transition_aware_accuracy_std=("transition_aware_accuracy", "std"),
        ),
        run_level_df.groupby("pork_cut", as_index=False).agg(
            summary_type=("pork_cut", lambda _: "pork_cut"),
            group_value=("pork_cut", "first"),
            prediction_rows=("prediction_rows", "sum"),
            transition_aware_accuracy=("transition_aware_accuracy", "mean"),
            transition_aware_accuracy_std=("transition_aware_accuracy", "std"),
        ),
        run_level_df.groupby("phone_group", as_index=False).agg(
            summary_type=("phone_group", lambda _: "phone_group"),
            group_value=("phone_group", "first"),
            prediction_rows=("prediction_rows", "sum"),
            transition_aware_accuracy=("transition_aware_accuracy", "mean"),
            transition_aware_accuracy_std=("transition_aware_accuracy", "std"),
        ),
    ]

    summary_df = pd.concat(summary_frames, ignore_index=True, sort=False)
    zone_summary_df = annotated_df.groupby("transition_zone", as_index=False).agg(
        prediction_rows=("is_transition_aware_correct", "size"),
        transition_aware_accuracy=("is_transition_aware_correct", lambda s: float(pd.Series(s).astype(float).mean())),
        correct_count=("is_transition_aware_correct", lambda s: int(pd.Series(s).astype(bool).sum())),
        incorrect_count=("is_transition_aware_correct", lambda s: int((~pd.Series(s).astype(bool)).sum())),
    )
    zone_summary_df["accepted_predictions"] = zone_summary_df["transition_zone"].map(
        lambda zone: json.dumps(ZONE_ACCEPTED_PREDICTIONS.get(zone, []))
    )
    zone_summary_df["transition_zone"] = pd.Categorical(zone_summary_df["transition_zone"], categories=ZONE_ORDER, ordered=True)
    zone_summary_df = zone_summary_df.sort_values("transition_zone").reset_index(drop=True)

    fold_transition_df = run_level_df.groupby("fold_name", as_index=False).agg(
        transition_aware_accuracy_mean=("transition_aware_accuracy", "mean"),
        transition_aware_accuracy_std=("transition_aware_accuracy", "std"),
    )
    strict_fold_df = seed_metrics_df.groupby("fold_name", as_index=False).agg(
        strict_accuracy_mean=("accuracy", "mean"),
        strict_accuracy_std=("accuracy", "std"),
    )
    fold_compare_df = fold_transition_df.merge(strict_fold_df, on="fold_name", how="left")
    return summary_df, zone_summary_df, fold_compare_df


def _make_bar_plot(
    x_values,
    y_values,
    title: str,
    ylabel: str,
    out_path: Path,
    color: str = "#355070",
    xlabel: str = "",
    ylim: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([str(x) for x in x_values], pd.to_numeric(pd.Series(y_values), errors="coerce"), color=color, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    if ylim is not None:
        ax.set_ylim(*ylim)
    plt.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _make_grouped_bar_plot(
    fold_compare_df: pd.DataFrame,
    out_path: Path,
) -> None:
    plot_df = fold_compare_df.copy().sort_values("fold_name")
    x = np.arange(len(plot_df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, plot_df["strict_accuracy_mean"], width=width, label="Strict Accuracy", color="#355070")
    ax.bar(x + width / 2, plot_df["transition_aware_accuracy_mean"], width=width, label="Transition-aware Accuracy", color="#2A9D8F")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["fold_name"].astype(str).tolist())
    ax.set_ylim(0, 1.05)
    ax.set_title("Strict vs Transition-aware Accuracy by Fold")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Accuracy")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _make_correctness_plot(zone_summary_df: pd.DataFrame, out_path: Path) -> None:
    plot_df = zone_summary_df.copy()
    x = np.arange(len(plot_df))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width / 2, plot_df["correct_count"], width=width, label="Correct", color="#2A9D8F")
    ax.bar(x + width / 2, plot_df["incorrect_count"], width=width, label="Incorrect", color="#BC4749")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["transition_zone"].astype(str).tolist(), rotation=20)
    ax.set_title("Transition Zone Correctness Counts")
    ax.set_xlabel("Transition Zone")
    ax.set_ylabel("Prediction Rows")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_confusion_matrix_df(
    matrix_df: pd.DataFrame,
    title: str,
    out_path: Path,
    normalize: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    values = matrix_df.to_numpy(dtype=float)
    if SEABORN_AVAILABLE and sns is not None:
        sns.heatmap(
            matrix_df,
            annot=True,
            fmt=".2f" if normalize else "g",
            cmap="Blues",
            cbar=True,
            ax=ax,
        )
    else:
        im = ax.imshow(values, cmap="Blues", aspect="auto")
        fig.colorbar(im, ax=ax)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                text_val = f"{values[i, j]:.2f}" if normalize else f"{int(values[i, j])}"
                ax.text(j, i, text_val, ha="center", va="center", color="black", fontsize=9)
        ax.set_xticks(np.arange(matrix_df.shape[1]))
        ax.set_xticklabels(matrix_df.columns.tolist(), rotation=20)
        ax.set_yticks(np.arange(matrix_df.shape[0]))
        ax.set_yticklabels(matrix_df.index.tolist())
    ax.set_title(title)
    ax.set_xlabel("Predicted Transition Zone")
    ax.set_ylabel("True Transition Zone")
    plt.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def build_transition_zone_confusion_matrices(
    annotated_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    cm_5x6 = pd.crosstab(
        annotated_df["true_transition_zone"],
        annotated_df["predicted_transition_zone"],
        dropna=False,
    ).reindex(index=ZONE_ORDER, columns=PREDICTED_ZONE_ORDER_5X6, fill_value=0)

    cm_5x5 = pd.crosstab(
        annotated_df["true_transition_zone"],
        annotated_df["predicted_transition_zone_5x5"],
        dropna=False,
    ).reindex(index=ZONE_ORDER, columns=ZONE_ORDER, fill_value=0)

    cm_5x5_norm = cm_5x5.div(cm_5x5.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return {
        "cm_5x6": cm_5x6,
        "cm_5x5": cm_5x5,
        "cm_5x5_norm": cm_5x5_norm,
    }


def save_transition_zone_confusion_matrices(
    confusion_bundle: dict[str, pd.DataFrame],
) -> dict[str, str]:
    cm_5x6 = confusion_bundle["cm_5x6"]
    cm_5x5 = confusion_bundle["cm_5x5"]
    cm_5x5_norm = confusion_bundle["cm_5x5_norm"]

    cm_5x6.to_csv(TRANSITION_ZONE_CM_5X6_CSV_PATH)
    cm_5x5.to_csv(TRANSITION_ZONE_CM_5X5_CSV_PATH)
    cm_5x5_norm.to_csv(TRANSITION_ZONE_CM_5X5_NORM_CSV_PATH)

    _plot_confusion_matrix_df(
        cm_5x6,
        "Transition Zone Confusion Matrix 5x6",
        FIGURES_ROOT / "transition_zone_confusion_matrix_5x6.png",
        normalize=False,
    )
    _plot_confusion_matrix_df(
        cm_5x5,
        "Transition Zone Confusion Matrix 5x5",
        FIGURES_ROOT / "transition_zone_confusion_matrix_5x5.png",
        normalize=False,
    )
    _plot_confusion_matrix_df(
        cm_5x5_norm,
        "Normalized Transition Zone Confusion Matrix 5x5",
        FIGURES_ROOT / "normalized_transition_zone_confusion_matrix_5x5.png",
        normalize=True,
    )
    return {
        "transition_zone_confusion_matrix_5x6_csv": str(TRANSITION_ZONE_CM_5X6_CSV_PATH),
        "transition_zone_confusion_matrix_5x5_csv": str(TRANSITION_ZONE_CM_5X5_CSV_PATH),
        "normalized_transition_zone_confusion_matrix_5x5_csv": str(TRANSITION_ZONE_CM_5X5_NORM_CSV_PATH),
        "transition_zone_confusion_matrix_5x6_png": str(FIGURES_ROOT / "transition_zone_confusion_matrix_5x6.png"),
        "transition_zone_confusion_matrix_5x5_png": str(FIGURES_ROOT / "transition_zone_confusion_matrix_5x5.png"),
        "normalized_transition_zone_confusion_matrix_5x5_png": str(FIGURES_ROOT / "normalized_transition_zone_confusion_matrix_5x5.png"),
    }


def generate_transition_aware_graphs(
    summary_df: pd.DataFrame,
    zone_summary_df: pd.DataFrame,
    fold_compare_df: pd.DataFrame,
) -> dict[str, str]:
    ensure_output_dirs()
    fold_summary = summary_df.loc[summary_df["summary_type"] == "fold"].copy().sort_values("group_value")
    _make_bar_plot(
        fold_summary["group_value"],
        fold_summary["transition_aware_accuracy"],
        "Transition-aware Accuracy by Fold",
        "Transition-aware Accuracy",
        FIGURES_ROOT / "transition_aware_accuracy_by_fold.png",
        color="#2A9D8F",
        xlabel="Fold",
        ylim=(0, 1.05),
    )
    _make_bar_plot(
        zone_summary_df["transition_zone"].astype(str),
        zone_summary_df["transition_aware_accuracy"],
        "Transition-aware Accuracy by Transition Zone",
        "Transition-aware Accuracy",
        FIGURES_ROOT / "transition_aware_accuracy_by_transition_zone.png",
        color="#4C956C",
        xlabel="Transition Zone",
        ylim=(0, 1.05),
    )
    _make_grouped_bar_plot(
        fold_compare_df,
        FIGURES_ROOT / "strict_vs_transition_aware_accuracy_by_fold.png",
    )
    _make_bar_plot(
        zone_summary_df["transition_zone"].astype(str),
        zone_summary_df["prediction_rows"],
        "Transition Zone Distribution",
        "Prediction Rows",
        FIGURES_ROOT / "transition_zone_distribution.png",
        color="#6D597A",
        xlabel="Transition Zone",
    )
    _make_correctness_plot(
        zone_summary_df,
        FIGURES_ROOT / "transition_zone_correctness_bar.png",
    )
    return {
        "transition_aware_accuracy_by_fold": str(FIGURES_ROOT / "transition_aware_accuracy_by_fold.png"),
        "transition_aware_accuracy_by_transition_zone": str(FIGURES_ROOT / "transition_aware_accuracy_by_transition_zone.png"),
        "strict_vs_transition_aware_accuracy_by_fold": str(FIGURES_ROOT / "strict_vs_transition_aware_accuracy_by_fold.png"),
        "transition_zone_distribution": str(FIGURES_ROOT / "transition_zone_distribution.png"),
        "transition_zone_correctness_bar": str(FIGURES_ROOT / "transition_zone_correctness_bar.png"),
    }


def run_transition_aware_evaluation() -> dict[str, object]:
    ensure_output_dirs()
    seed_metrics_df = load_seed_metrics_df()
    prediction_df = load_run_prediction_frames(seed_metrics_df)
    sequence_ref_df = load_sequence_reference_df()
    annotated_df = annotate_transition_aware_predictions(prediction_df, sequence_ref_df)

    summary_df, zone_summary_df, fold_compare_df = build_transition_aware_summaries(annotated_df, seed_metrics_df)
    confusion_bundle = build_transition_zone_confusion_matrices(annotated_df)

    annotated_df.to_csv(TRANSITION_AWARE_PREDICTIONS_PATH, index=False)
    summary_df.to_csv(TRANSITION_AWARE_SUMMARY_PATH, index=False)
    zone_summary_df.to_csv(TRANSITION_ZONE_SUMMARY_PATH, index=False)

    graph_paths = generate_transition_aware_graphs(summary_df, zone_summary_df, fold_compare_df)
    confusion_paths = save_transition_zone_confusion_matrices(confusion_bundle)

    overall_accuracy = float(annotated_df["is_transition_aware_correct"].astype(float).mean())
    print(SECONDARY_NOTE)
    print(f"Overall transition-aware accuracy: {overall_accuracy:.4f}")
    print(f"Saved: {TRANSITION_AWARE_PREDICTIONS_PATH}")
    print(f"Saved: {TRANSITION_AWARE_SUMMARY_PATH}")
    print(f"Saved: {TRANSITION_ZONE_SUMMARY_PATH}")

    return {
        "seed_metrics_df": seed_metrics_df,
        "prediction_df": prediction_df,
        "sequence_ref_df": sequence_ref_df,
        "annotated_df": annotated_df,
        "summary_df": summary_df,
        "zone_summary_df": zone_summary_df,
        "fold_compare_df": fold_compare_df,
        "graph_paths": graph_paths,
        "confusion_bundle": confusion_bundle,
        "confusion_paths": confusion_paths,
        "secondary_note": SECONDARY_NOTE,
        "overall_transition_aware_accuracy": overall_accuracy,
    }


if __name__ == "__main__":
    run_transition_aware_evaluation()
