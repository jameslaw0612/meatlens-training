#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def normalize_sample_value(value: object) -> str:
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return "Public/NA"
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Export visualization PNGs from dataset CSV.")
    parser.add_argument(
        "--input-csv",
        default="csv_outputs/all_datasets_combined.csv",
        help="Combined CSV to visualize.",
    )
    parser.add_argument(
        "--output-dir",
        default="csv_outputs/plots",
        help="Directory where PNG plots will be saved.",
    )
    args = parser.parse_args()

    csv_path = Path(args.input_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df["sample_number"] = df["sample_number"].apply(normalize_sample_value)
    df["meat_part"] = df["meat_part"].fillna("Unknown").replace("", "Unknown")
    df["time_frame"] = df["time_frame"].fillna("")

    # 1) Counts per pork part and sample
    part_sample_counts = (
        df.groupby(["meat_part", "sample_number"])
        .size()
        .reset_index(name="image_count")
    )
    pivot_counts = part_sample_counts.pivot(
        index="meat_part", columns="sample_number", values="image_count"
    ).fillna(0)

    ax = pivot_counts.plot(kind="bar", figsize=(11, 6))
    ax.set_title("Image Count per Pork Part and Sample")
    ax.set_xlabel("Meat Part")
    ax.set_ylabel("Image Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_dir / "image_count_per_pork_part_and_sample.png", dpi=200)
    plt.close()

    # 2) Overall label distribution
    label_counts = df["label"].value_counts()
    plt.figure(figsize=(7, 5))
    label_counts.plot(kind="bar", color=["#4caf50", "#ff9800", "#e53935"])
    plt.title("Overall Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Image Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_dir / "overall_label_distribution.png", dpi=200)
    plt.close()

    # 3) Per sample (and Public/NA) label counts with exact annotations
    sample_label_counts = (
        df.groupby(["sample_number", "label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["fresh", "not fresh", "spoiled"], fill_value=0)
    )
    sample_order = ["1", "2", "3", "4", "Public/NA"]
    sample_label_counts = sample_label_counts.reindex(
        [s for s in sample_order if s in sample_label_counts.index] +
        [s for s in sample_label_counts.index if s not in sample_order]
    )

    ax = sample_label_counts.plot(
        kind="bar",
        figsize=(11, 6),
        color=["#4caf50", "#ff9800", "#e53935"],
    )
    ax.set_title("Fresh / Not Fresh / Spoiled Counts per Sample (with Public Dataset)")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Image Count")
    ax.legend(title="Label")
    plt.xticks(rotation=0)

    # exact count labels on each bar
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=2, fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / "sample_label_counts_grouped_bar.png", dpi=220)
    plt.close()

    # 4) Time-frame hourly distribution by sample
    timed = df[df["time_frame"] != ""].copy()
    timed["hour_bucket"] = timed["time_frame"].str.slice(0, 2).astype(int)
    hourly = (
        timed.groupby(["sample_number", "label", "hour_bucket"])
        .size()
        .reset_index(name="image_count")
        .sort_values(["sample_number", "label", "hour_bucket"])
    )

    for sample in sorted(hourly["sample_number"].unique(), key=lambda x: str(x)):
        sdf = hourly[hourly["sample_number"] == sample]
        if sdf.empty:
            continue

        plt.figure(figsize=(11, 5))
        for label, color in [("fresh", "#4caf50"), ("not fresh", "#ff9800"), ("spoiled", "#e53935")]:
            ldf = sdf[sdf["label"] == label]
            if not ldf.empty:
                plt.plot(ldf["hour_bucket"], ldf["image_count"], marker="o", label=label, color=color)
        plt.title(f"Hourly Time-Frame Distribution - Sample {sample}")
        plt.xlabel("Hour Bucket")
        plt.ylabel("Image Count")
        plt.legend()
        plt.tight_layout()
        safe_sample = str(sample).replace("/", "_").replace("\\", "_")
        plt.savefig(out_dir / f"timeframe_distribution_sample_{safe_sample}.png", dpi=200)
        plt.close()

    print(f"Saved plots to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
