#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
LABEL_ORDER = ["fresh", "not fresh", "spoiled"]
LABEL_SORT = {label: idx for idx, label in enumerate(LABEL_ORDER)}
DEFAULT_LABEL_HOUR_RANGES = {
    "fresh": (0.0, 7.0),
    "not fresh": (7.0, 14.0),
    "spoiled": (14.0, 40.0),
}
BASE_COLUMNS = [
    "image_file_name",
    "sample_number",
    "meat_part",
    "label",
    "time_frame",
    "file_destination",
]
SPLIT_COLUMNS = BASE_COLUMNS + ["sample_id", "pork_cut", "split_type", "fold", "source"]
SOURCE_NAME = "meatlens"


@dataclass(frozen=True)
class SampleInfo:
    sample_number: int
    meat_part: str
    pork_cut: str
    sample_id: str
    folder_path: Path


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def normalize_label(text: str) -> str:
    lower = text.strip().lower()
    if "not fresh" in lower or "half-fresh" in lower or "half fresh" in lower:
        return "not fresh"
    if "fresh" in lower and "not fresh" not in lower:
        return "fresh"
    if "spoiled" in lower:
        return "spoiled"
    return lower


def parse_hour_range(folder_name: str) -> tuple[float, float]:
    numbers = [int(value) for value in re.findall(r"\d+", folder_name)]
    if len(numbers) < 2:
        label = normalize_label(folder_name)
        if label in DEFAULT_LABEL_HOUR_RANGES:
            return DEFAULT_LABEL_HOUR_RANGES[label]
        raise ValueError(f"Could not find hour range in folder name: {folder_name}")
    start_hour = float(numbers[-2])
    end_hour = float(numbers[-1])
    if end_hour < start_hour:
        start_hour, end_hour = end_hour, start_hour
    return start_hour, end_hour


def format_hhmmss(hour_value: float) -> str:
    total_seconds = int(round(hour_value * 3600))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def assign_time_frames(count: int, start_hour: float, end_hour: float) -> list[str]:
    if count <= 0:
        return []
    if count == 1:
        return [format_hhmmss((start_hour + end_hour) / 2.0)]

    step = (end_hour - start_hour) / (count - 1)
    frames = []
    for idx in range(count):
        hour_value = start_hour + (idx * step)
        if idx == count - 1:
            hour_value = end_hour
        frames.append(format_hhmmss(hour_value))
    return frames


def evenly_spaced_indices(total_count: int, target_count: int) -> list[int]:
    if total_count <= 0:
        return []
    if total_count <= target_count:
        return list(range(total_count))
    if target_count <= 1:
        return [total_count // 2]

    indices = []
    for idx in range(target_count):
        position = round(idx * (total_count - 1) / (target_count - 1))
        indices.append(int(position))
    return indices


def sorted_images(folder_path: Path) -> list[Path]:
    images = [path for path in folder_path.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS]
    return sorted(images, key=lambda path: (path.stat().st_mtime, path.name.lower()))


def discover_sample_dirs(project_root: Path) -> list[SampleInfo]:
    sample_infos: list[SampleInfo] = []
    for path in project_root.iterdir():
        if not path.is_dir():
            continue
        match = re.search(r"sample\s*(\d+)", path.name, flags=re.IGNORECASE)
        if match is None:
            continue
        sample_number = int(match.group(1))

        prefix = re.split(r"-\s*sample", path.name, flags=re.IGNORECASE)[0].strip()
        if not prefix.lower().startswith("pork"):
            continue

        meat_part = " ".join(part.capitalize() for part in prefix.split())
        pork_cut = meat_part.lower().replace("pork", "", 1).strip().replace(" ", "_")
        sample_infos.append(
            SampleInfo(
                sample_number=sample_number,
                meat_part=meat_part,
                pork_cut=pork_cut,
                sample_id=f"{slugify(prefix)}_sample_{sample_number}",
                folder_path=path,
            )
        )

    return sorted(sample_infos, key=lambda info: info.sample_number)


def build_dataset_rows(
    sample_infos: Iterable[SampleInfo],
    target_count_per_class: int | None = None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for sample_info in sample_infos:
        label_dirs = [path for path in sample_info.folder_path.iterdir() if path.is_dir()]
        label_dirs = sorted(label_dirs, key=lambda path: LABEL_SORT.get(normalize_label(path.name), 999))

        for label_dir in label_dirs:
            label = normalize_label(label_dir.name)
            if label not in LABEL_SORT:
                continue

            start_hour, end_hour = parse_hour_range(label_dir.name)
            images = sorted_images(label_dir)
            if not images:
                continue

            selected_images = images
            if target_count_per_class is not None:
                selected_indices = evenly_spaced_indices(len(images), target_count_per_class)
                selected_images = [images[idx] for idx in selected_indices]

            time_frames = assign_time_frames(len(selected_images), start_hour, end_hour)
            for image_path, time_frame in zip(selected_images, time_frames):
                rows.append(
                    {
                        "image_file_name": image_path.name,
                        "sample_number": str(sample_info.sample_number),
                        "meat_part": sample_info.meat_part,
                        "label": label,
                        "time_frame": time_frame,
                        "file_destination": str(image_path.resolve()),
                        "sample_id": sample_info.sample_id,
                        "pork_cut": sample_info.pork_cut,
                        "source_folder": label_dir.name,
                    }
                )

    rows.sort(
        key=lambda row: (
            int(row["sample_number"]),
            LABEL_SORT.get(row["label"], 999),
            row["time_frame"],
            row["image_file_name"].lower(),
        )
    )
    return rows


def to_dataframe(rows: list[dict[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def write_per_sample_csvs(sampled_df: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_sample_paths: dict[str, Path] = {}

    for sample_id, group_df in sampled_df.groupby("sample_id", sort=False):
        csv_path = output_dir / f"{sample_id}.csv"
        group_df.loc[:, BASE_COLUMNS].to_csv(csv_path, index=False)
        per_sample_paths[sample_id] = csv_path

    combined_path = output_dir / "all_datasets_combined_sampled.csv"
    sampled_df.loc[:, BASE_COLUMNS].to_csv(combined_path, index=False)
    per_sample_paths["all_datasets_combined_sampled"] = combined_path
    return per_sample_paths


def make_sampling_summary(raw_df: pd.DataFrame, sampled_df: pd.DataFrame) -> pd.DataFrame:
    raw_counts = (
        raw_df.groupby(["sample_number", "meat_part", "label"])
        .size()
        .reset_index(name="raw_count")
    )
    sampled_counts = (
        sampled_df.groupby(["sample_number", "meat_part", "label"])
        .size()
        .reset_index(name="sampled_count")
    )
    summary = raw_counts.merge(sampled_counts, on=["sample_number", "meat_part", "label"], how="left")
    summary["sampled_count"] = summary["sampled_count"].fillna(0).astype(int)
    summary["retention_ratio"] = (summary["sampled_count"] / summary["raw_count"]).round(4)
    summary = summary.sort_values(
        by=["sample_number", "label"],
        key=lambda series: series.map(LABEL_SORT) if series.name == "label" else series,
    )
    return summary


def sample_label_counts(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.groupby(["sample_number", "label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=LABEL_ORDER, fill_value=0)
    )
    counts.index = counts.index.astype(str)
    counts = counts.sort_index(key=lambda values: values.astype(int))
    return counts


def sample_total_counts(df: pd.DataFrame) -> pd.DataFrame:
    totals = (
        df.groupby(["sample_number", "meat_part"])
        .size()
        .reset_index(name="count")
        .sort_values("sample_number", key=lambda values: values.astype(int))
    )
    return totals


def save_plots(raw_df: pd.DataFrame, sampled_df: pd.DataFrame, plots_dir: Path) -> dict[str, Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: dict[str, Path] = {}

    raw_counts = sample_label_counts(raw_df)
    sampled_counts = sample_label_counts(sampled_df)

    ax = raw_counts.plot(
        kind="bar",
        figsize=(12, 6),
        color=["#4caf50", "#ff9800", "#d32f2f"],
    )
    ax.set_title("Original Image Counts per Sample and Classification")
    ax.set_xlabel("Sample Number")
    ax.set_ylabel("Image Count")
    ax.legend(title="Classification")
    plt.xticks(rotation=0)
    plt.tight_layout()
    raw_plot_path = plots_dir / "raw_sample_label_counts.png"
    plt.savefig(raw_plot_path, dpi=220)
    plt.close()
    plot_paths["raw_sample_label_counts"] = raw_plot_path

    ax = sampled_counts.plot(
        kind="bar",
        figsize=(12, 6),
        color=["#4caf50", "#ff9800", "#d32f2f"],
    )
    ax.set_title("Interval-Sampled Counts per Sample and Classification")
    ax.set_xlabel("Sample Number")
    ax.set_ylabel("Image Count")
    ax.legend(title="Classification")
    plt.xticks(rotation=0)
    plt.tight_layout()
    sampled_plot_path = plots_dir / "sampled_sample_label_counts.png"
    plt.savefig(sampled_plot_path, dpi=220)
    plt.close()
    plot_paths["sampled_sample_label_counts"] = sampled_plot_path

    raw_totals = sample_total_counts(raw_df).rename(columns={"count": "raw_count"})
    sampled_totals = sample_total_counts(sampled_df).rename(columns={"count": "sampled_count"})
    totals = raw_totals.merge(sampled_totals, on=["sample_number", "meat_part"], how="left")

    x_labels = totals["sample_number"].astype(str).tolist()
    positions = list(range(len(totals)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([pos - (width / 2) for pos in positions], totals["raw_count"], width=width, label="Original", color="#5b8ff9")
    ax.bar([pos + (width / 2) for pos in positions], totals["sampled_count"], width=width, label="Interval sampled", color="#61dDAa")
    ax.set_title("Original vs Interval-Sampled Total Counts per Sample")
    ax.set_xlabel("Sample Number")
    ax.set_ylabel("Image Count")
    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels)
    ax.legend()
    plt.tight_layout()
    totals_plot_path = plots_dir / "raw_vs_sampled_totals.png"
    plt.savefig(totals_plot_path, dpi=220)
    plt.close()
    plot_paths["raw_vs_sampled_totals"] = totals_plot_path

    comparison_rows: list[dict[str, object]] = []
    for sample_number in raw_counts.index.tolist():
        for label in LABEL_ORDER:
            comparison_rows.append(
                {
                    "sample_number": sample_number,
                    "label": label,
                    "original_count": int(raw_counts.loc[sample_number, label]),
                    "sampled_count": int(sampled_counts.loc[sample_number, label]),
                }
            )
    comparison_df = pd.DataFrame(comparison_rows)

    sample_numbers = sorted(comparison_df["sample_number"].unique(), key=int)
    num_samples = len(sample_numbers)
    num_cols = min(3, max(1, num_samples))
    num_rows = (num_samples + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 4.5 * num_rows), sharey=False)
    if not isinstance(axes, (list, tuple)):
        axes = [axes]
    else:
        axes = list(axes)
    flat_axes = []
    for ax in axes:
        if isinstance(ax, (list, tuple, pd.Series, pd.Index)):
            flat_axes.extend(list(ax))
        elif hasattr(ax, "flatten"):
            flat_axes.extend(list(ax.flatten()))
        else:
            flat_axes.append(ax)
    axes = flat_axes
    colors = {"original_count": "#5b8ff9", "sampled_count": "#61dDAa"}
    for idx, sample_number in enumerate(sample_numbers):
        ax = axes[idx]
        subset = comparison_df[comparison_df["sample_number"] == sample_number]
        x_positions = list(range(len(LABEL_ORDER)))
        ax.bar(
            [pos - (width / 2) for pos in x_positions],
            subset["original_count"],
            width=width,
            label="Original" if idx == 0 else None,
            color=colors["original_count"],
        )
        ax.bar(
            [pos + (width / 2) for pos in x_positions],
            subset["sampled_count"],
            width=width,
            label="Sampled" if idx == 0 else None,
            color=colors["sampled_count"],
        )
        ax.set_title(f"Sample {sample_number}")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(LABEL_ORDER, rotation=0)
        ax.set_ylabel("Image Count")

    if len(axes) > num_samples:
        for ax in axes[num_samples:]:
            ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Original vs Sampled Counts per Classification", y=0.98)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    comparison_plot_path = plots_dir / "raw_vs_sampled_per_classification.png"
    plt.savefig(comparison_plot_path, dpi=220)
    plt.close()
    plot_paths["raw_vs_sampled_per_classification"] = comparison_plot_path

    return plot_paths


def add_split_metadata(df: pd.DataFrame, split_type: str, fold_name: str) -> pd.DataFrame:
    out = df.copy()
    out["split_type"] = split_type
    out["fold"] = fold_name
    out["source"] = SOURCE_NAME
    return out.loc[:, SPLIT_COLUMNS]


def label_count_dict(df: pd.DataFrame) -> dict[str, int]:
    counts = df["label"].value_counts().to_dict()
    return {label: int(counts.get(label, 0)) for label in LABEL_ORDER}


def sort_split_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["label_order"] = out["label"].map(LABEL_SORT).fillna(999)
    out = out.sort_values(
        by=["label_order", "time_frame", "sample_number", "image_file_name"],
        kind="stable",
    ).drop(columns=["label_order"])
    return out.reset_index(drop=True)


def write_combined_sampled_split_csv(sampled_df: pd.DataFrame, output_dir: Path) -> Path:
    combined_df = sampled_df.copy()
    combined_df["split_type"] = "cross_rotation"
    combined_df["fold"] = "all"
    combined_df["source"] = SOURCE_NAME
    combined_df = sort_split_df(combined_df.loc[:, SPLIT_COLUMNS])
    output_path = output_dir / "all_sampled_images.csv"
    combined_df.to_csv(output_path, index=False)
    return output_path


def stratified_train_val_split(
    df: pd.DataFrame,
    val_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []

    for group_index, (_, group_df) in enumerate(df.groupby("label", sort=False)):
        shuffled = group_df.sample(frac=1.0, random_state=seed + group_index).reset_index(drop=True)
        if len(shuffled) <= 1:
            train_parts.append(shuffled)
            continue

        val_count = int(round(len(shuffled) * val_fraction))
        val_count = max(1, val_count)
        val_count = min(val_count, len(shuffled) - 1)

        val_parts.append(shuffled.iloc[:val_count].copy())
        train_parts.append(shuffled.iloc[val_count:].copy())

    train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, val_df


def generate_cross_rotation_splits(
    sampled_df: pd.DataFrame,
    output_dir: Path,
    seed: int = 42,
    val_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    leakage_rows: list[dict[str, object]] = []

    ordered_sample_ids = (
        sampled_df.loc[:, ["sample_id", "sample_number", "pork_cut"]]
        .drop_duplicates()
        .sort_values("sample_number", key=lambda values: values.astype(int))
    )

    for fold_index, sample_row in enumerate(ordered_sample_ids.itertuples(index=False), start=1):
        held_out_sample = sample_row.sample_id
        fold_name = f"fold{fold_index}"

        test_df = sampled_df[sampled_df["sample_id"] == held_out_sample].copy()
        train_pool = sampled_df[sampled_df["sample_id"] != held_out_sample].copy()

        train_df, val_df = stratified_train_val_split(
            train_pool,
            val_fraction=val_fraction,
            seed=seed,
        )

        train_df = sort_split_df(add_split_metadata(train_df, split_type="cross_rotation", fold_name=fold_name))
        val_df = sort_split_df(add_split_metadata(val_df, split_type="cross_rotation", fold_name=fold_name))
        test_df = sort_split_df(add_split_metadata(test_df, split_type="cross_rotation", fold_name=fold_name))

        train_df.to_csv(output_dir / f"{fold_name}_train.csv", index=False)
        val_df.to_csv(output_dir / f"{fold_name}_val.csv", index=False)
        test_df.to_csv(output_dir / f"{fold_name}_test.csv", index=False)

        train_counts = label_count_dict(train_df)
        val_counts = label_count_dict(val_df)
        test_counts = label_count_dict(test_df)

        summary_row: dict[str, object] = {
            "fold": fold_name,
            "held_out_sample": held_out_sample,
            "held_out_cut": sample_row.pork_cut,
            "train_count": int(len(train_df)),
            "val_count": int(len(val_df)),
            "test_count": int(len(test_df)),
            "train_class_counts": json.dumps(train_counts),
            "val_class_counts": json.dumps(val_counts),
            "test_class_counts": json.dumps(test_counts),
        }
        for label in LABEL_ORDER:
            label_key = label.replace(" ", "_")
            summary_row[f"train_{label_key}_count"] = train_counts[label]
            summary_row[f"val_{label_key}_count"] = val_counts[label]
            summary_row[f"test_{label_key}_count"] = test_counts[label]
        summary_rows.append(summary_row)

        leaked_in_train = bool(train_df["sample_id"].eq(held_out_sample).any())
        leaked_in_val = bool(val_df["sample_id"].eq(held_out_sample).any())
        leakage_rows.append(
            {
                "fold": fold_name,
                "held_out_sample": held_out_sample,
                "present_in_train": leaked_in_train,
                "present_in_val": leaked_in_val,
                "status": "PASS" if (not leaked_in_train and not leaked_in_val) else "FAIL",
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    leakage_df = pd.DataFrame(leakage_rows)
    summary_df.to_csv(output_dir / "cross_rotation_summary.csv", index=False)
    leakage_df.to_csv(output_dir / "cross_rotation_leakage_check.csv", index=False)
    return summary_df, leakage_df


def run_pipeline(
    project_root: Path,
    sampled_output_dir: Path,
    split_output_dir: Path,
    target_count_per_class: int = 200,
    seed: int = 42,
    sample_root: Path | None = None,
) -> dict[str, object]:
    scan_root = sample_root if sample_root is not None else project_root
    sample_infos = discover_sample_dirs(scan_root)
    if not sample_infos:
        raise FileNotFoundError(f"No sample folders were found under: {scan_root}")

    raw_df = to_dataframe(build_dataset_rows(sample_infos, target_count_per_class=None))
    sampled_df = to_dataframe(build_dataset_rows(sample_infos, target_count_per_class=target_count_per_class))

    if raw_df.empty or sampled_df.empty:
        raise ValueError("The dataset scan completed, but no image rows were produced.")

    sampled_output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = sampled_output_dir / "plots"
    split_output_dir.mkdir(parents=True, exist_ok=True)

    sampled_csv_paths = write_per_sample_csvs(sampled_df, sampled_output_dir)
    sampling_summary_df = make_sampling_summary(raw_df, sampled_df)
    sampling_summary_path = sampled_output_dir / "sampling_summary.csv"
    sampling_summary_df.to_csv(sampling_summary_path, index=False)
    plot_paths = save_plots(raw_df, sampled_df, plots_dir)
    combined_sampled_split_path = write_combined_sampled_split_csv(sampled_df, split_output_dir)
    cross_summary_df, leakage_df = generate_cross_rotation_splits(sampled_df, split_output_dir, seed=seed)

    return {
        "sample_infos": sample_infos,
        "raw_df": raw_df,
        "sampled_df": sampled_df,
        "sampling_summary_df": sampling_summary_df,
        "cross_summary_df": cross_summary_df,
        "leakage_df": leakage_df,
        "sampled_output_dir": sampled_output_dir,
        "split_output_dir": split_output_dir,
        "sampled_csv_paths": sampled_csv_paths,
        "sampling_summary_path": sampling_summary_path,
        "plot_paths": plot_paths,
        "combined_sampled_split_path": combined_sampled_split_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create interval-sampled CSVs and cross-rotation splits for sample folders."
    )
    parser.add_argument("--project-root", default=".", help="Root directory that contains the sample folders.")
    parser.add_argument(
        "--sample-root",
        default=None,
        help="Optional directory to scan for sample folders. Defaults to --project-root.",
    )
    parser.add_argument(
        "--sampled-output-dir",
        default="csv_outputs_interval_sampled_v3",
        help="Directory for sampled per-sample CSVs and plots.",
    )
    parser.add_argument(
        "--split-output-dir",
        default="generated_splits/cross_rotation_interval200_6samples",
        help="Directory for generated cross-rotation CSVs.",
    )
    parser.add_argument(
        "--target-count-per-class",
        type=int,
        default=200,
        help="Maximum number of interval-sampled images per sample/classification.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/validation split generation.")
    args = parser.parse_args()

    outputs = run_pipeline(
        project_root=Path(args.project_root).resolve(),
        sampled_output_dir=Path(args.sampled_output_dir).resolve(),
        split_output_dir=Path(args.split_output_dir).resolve(),
        target_count_per_class=args.target_count_per_class,
        seed=args.seed,
        sample_root=Path(args.sample_root).resolve() if args.sample_root else None,
    )

    print(f"Discovered samples: {len(outputs['sample_infos'])}")
    print(f"Sampled CSV output: {outputs['sampled_output_dir']}")
    print(f"Cross-rotation output: {outputs['split_output_dir']}")
    print(f"Sampling summary: {outputs['sampling_summary_path']}")
    print(f"Combined sampled split CSV: {outputs['combined_sampled_split_path']}")
    print("Saved plots:")
    for plot_name, plot_path in outputs["plot_paths"].items():
        print(f"  {plot_name}: {plot_path}")


if __name__ == "__main__":
    main()
