#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")

SEED = 42
SOURCE = "meatlens"
INPUT_DIR = Path("csv_outputs_interval_sampled_v2")
OUTPUT_DIR = Path("generated_splits")
CROSS_DIR = OUTPUT_DIR / "cross_rotation"
RANDOM_DIR = OUTPUT_DIR / "random_70_15_15"
PLOTS_DIR = OUTPUT_DIR / "plots"

SAMPLE_FILES = {
    "pork_shoulder_sample_1": "pork_shoulder_sample_1.csv",
    "pork_shoulder_sample_2": "pork_shoulder_sample_2.csv",
    "pork_belly_sample_3": "pork_belly_sample_3.csv",
    "pork_belly_sample_4": "pork_belly_sample_4.csv",
}

SAMPLE_TO_CUT = {
    "pork_shoulder_sample_1": "shoulder",
    "pork_shoulder_sample_2": "shoulder",
    "pork_belly_sample_3": "belly",
    "pork_belly_sample_4": "belly",
}

LABEL_ORDER = ["fresh", "not fresh", "spoiled"]
PATH_COLUMN_CANDIDATES = ["file_destination", "image_path", "file_path", "filepath", "path"]


def ensure_dirs() -> None:
    CROSS_DIR.mkdir(parents=True, exist_ok=True)
    RANDOM_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def find_path_column(df: pd.DataFrame) -> str | None:
    for col in PATH_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    return None


def add_metadata_columns(
    df: pd.DataFrame,
    sample_id: str,
    split_type: str,
    fold: str,
) -> pd.DataFrame:
    out = df.copy()
    out["sample_id"] = sample_id
    out["pork_cut"] = SAMPLE_TO_CUT[sample_id]
    out["split_type"] = split_type
    out["fold"] = fold
    out["source"] = SOURCE
    return out


def label_count_dict(df: pd.DataFrame) -> dict[str, int]:
    counts = df["label"].value_counts().to_dict()
    return {label: int(counts.get(label, 0)) for label in LABEL_ORDER}


def format_counts(d: dict[str, int]) -> str:
    return ", ".join([f"{k}={v}" for k, v in d.items()])


def write_df(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def load_input_samples() -> dict[str, pd.DataFrame]:
    samples: dict[str, pd.DataFrame] = {}
    for sample_id, filename in SAMPLE_FILES.items():
        csv_path = INPUT_DIR / filename
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing input CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        samples[sample_id] = add_metadata_columns(
            df=df,
            sample_id=sample_id,
            split_type="input",
            fold="input",
        )
    return samples


def generate_cross_rotation(samples: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[dict[str, object]], list[pd.DataFrame]]:
    summary_rows: list[dict[str, object]] = []
    generated_frames: list[pd.DataFrame] = []
    leakage_checks: list[dict[str, object]] = []

    ordered_samples = list(SAMPLE_FILES.keys())
    for fold_idx, held_out_sample in enumerate(ordered_samples, start=1):
        fold_name = f"fold{fold_idx}"
        held_out_cut = SAMPLE_TO_CUT[held_out_sample]

        test_df = samples[held_out_sample].copy()
        train_pool = pd.concat(
            [samples[sid] for sid in ordered_samples if sid != held_out_sample],
            ignore_index=True,
        )

        train_df, val_df = train_test_split(
            train_pool,
            test_size=0.15,
            random_state=SEED,
            stratify=train_pool["label"],
            shuffle=True,
        )

        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()

        train_df["split_type"] = "cross_rotation"
        val_df["split_type"] = "cross_rotation"
        test_df["split_type"] = "cross_rotation"
        train_df["fold"] = fold_name
        val_df["fold"] = fold_name
        test_df["fold"] = fold_name

        train_path = CROSS_DIR / f"{fold_name}_train.csv"
        val_path = CROSS_DIR / f"{fold_name}_val.csv"
        test_path = CROSS_DIR / f"{fold_name}_test.csv"
        write_df(train_df, train_path)
        write_df(val_df, val_path)
        write_df(test_df, test_path)

        train_counts = label_count_dict(train_df)
        val_counts = label_count_dict(val_df)
        test_counts = label_count_dict(test_df)

        row = {
            "fold": fold_name,
            "held_out_sample": held_out_sample,
            "held_out_cut": held_out_cut,
            "train_count": int(len(train_df)),
            "val_count": int(len(val_df)),
            "test_count": int(len(test_df)),
            "train_class_counts": json.dumps(train_counts),
            "val_class_counts": json.dumps(val_counts),
            "test_class_counts": json.dumps(test_counts),
        }
        for label in LABEL_ORDER:
            safe = label.replace(" ", "_")
            row[f"train_{safe}_count"] = train_counts[label]
            row[f"val_{safe}_count"] = val_counts[label]
            row[f"test_{safe}_count"] = test_counts[label]
        summary_rows.append(row)

        leaked_in_train = train_df["sample_id"].eq(held_out_sample).any()
        leaked_in_val = val_df["sample_id"].eq(held_out_sample).any()
        leakage_checks.append(
            {
                "fold": fold_name,
                "held_out_sample": held_out_sample,
                "present_in_train": bool(leaked_in_train),
                "present_in_val": bool(leaked_in_val),
                "status": "PASS" if (not leaked_in_train and not leaked_in_val) else "FAIL",
            }
        )

        generated_frames.extend([train_df, val_df, test_df])

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(CROSS_DIR / "cross_rotation_summary.csv", index=False)
    leakage_df = pd.DataFrame(leakage_checks)
    leakage_df.to_csv(CROSS_DIR / "cross_rotation_leakage_check.csv", index=False)
    return summary_df, leakage_checks, generated_frames


def generate_random_split(all_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[pd.DataFrame]]:
    train_df, temp_df = train_test_split(
        all_df,
        test_size=0.30,
        random_state=SEED,
        stratify=all_df["label"],
        shuffle=True,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_df["label"],
        shuffle=True,
    )

    for df in (train_df, val_df, test_df):
        df["split_type"] = "random_70_15_15"
        df["fold"] = "random"

    write_df(train_df, RANDOM_DIR / "random_train.csv")
    write_df(val_df, RANDOM_DIR / "random_val.csv")
    write_df(test_df, RANDOM_DIR / "random_test.csv")

    def dist_counts(df: pd.DataFrame, col: str) -> dict[str, int]:
        vals = df[col].value_counts().to_dict()
        return {str(k): int(v) for k, v in vals.items()}

    summary_row = {
        "train_count": int(len(train_df)),
        "val_count": int(len(val_df)),
        "test_count": int(len(test_df)),
        "train_label_counts": json.dumps(dist_counts(train_df, "label")),
        "val_label_counts": json.dumps(dist_counts(val_df, "label")),
        "test_label_counts": json.dumps(dist_counts(test_df, "label")),
        "train_sample_number_counts": json.dumps(dist_counts(train_df, "sample_number")),
        "val_sample_number_counts": json.dumps(dist_counts(val_df, "sample_number")),
        "test_sample_number_counts": json.dumps(dist_counts(test_df, "sample_number")),
        "train_pork_cut_counts": json.dumps(dist_counts(train_df, "pork_cut")),
        "val_pork_cut_counts": json.dumps(dist_counts(val_df, "pork_cut")),
        "test_pork_cut_counts": json.dumps(dist_counts(test_df, "pork_cut")),
    }
    summary_df = pd.DataFrame([summary_row])
    summary_df.to_csv(RANDOM_DIR / "random_split_summary.csv", index=False)
    return train_df, val_df, test_df, summary_df, [train_df, val_df, test_df]


def missing_path_count(df: pd.DataFrame, path_col: str | None) -> tuple[int, int]:
    if path_col is None:
        return 0, 0
    series = df[path_col]
    missing_value_count = int(series.isna().sum() + series.astype(str).str.strip().eq("").sum())
    exists_count = 0
    missing_file_count = 0
    for p in series.dropna().astype(str):
        p = p.strip()
        if not p:
            continue
        if Path(p).exists():
            exists_count += 1
        else:
            missing_file_count += 1
    return missing_value_count, missing_file_count


def make_diagnostics_plot(
    input_samples: dict[str, pd.DataFrame],
    generated_splits_for_plot: list[tuple[str, pd.DataFrame]],
) -> Path:
    input_rows = pd.DataFrame(
        [{"sample_id": k, "count": len(v)} for k, v in input_samples.items()]
    )

    input_label = (
        pd.concat(input_samples.values(), ignore_index=True)
        .groupby(["sample_id", "label"])
        .size()
        .reset_index(name="count")
    )

    split_label_rows = []
    split_sample_rows = []
    split_cut_rows = []
    for split_name, df in generated_splits_for_plot:
        c1 = df["label"].value_counts().to_dict()
        c2 = df["sample_id"].value_counts().to_dict()
        c3 = df["pork_cut"].value_counts().to_dict()
        for k, v in c1.items():
            split_label_rows.append({"split": split_name, "label": k, "count": int(v)})
        for k, v in c2.items():
            split_sample_rows.append({"split": split_name, "sample_id": k, "count": int(v)})
        for k, v in c3.items():
            split_cut_rows.append({"split": split_name, "pork_cut": k, "count": int(v)})

    split_label_df = pd.DataFrame(split_label_rows)
    split_sample_df = pd.DataFrame(split_sample_rows)
    split_cut_df = pd.DataFrame(split_cut_rows)

    fig, axes = plt.subplots(3, 2, figsize=(18, 16))
    axes = axes.flatten()

    # 1) Total rows per input sample
    axes[0].bar(input_rows["sample_id"], input_rows["count"], color="#4c78a8")
    axes[0].set_title("1) Total Rows per Input Sample")
    axes[0].set_ylabel("Rows")
    axes[0].tick_params(axis="x", rotation=25)

    # 2) Label distribution per input sample
    pvt_input_label = input_label.pivot(index="sample_id", columns="label", values="count").fillna(0)
    pvt_input_label.plot(kind="bar", stacked=True, ax=axes[1], colormap="Set2")
    axes[1].set_title("2) Label Distribution per Input Sample")
    axes[1].set_ylabel("Rows")
    axes[1].tick_params(axis="x", rotation=25)

    # 3) Label distribution per generated split
    pvt_split_label = split_label_df.pivot(index="split", columns="label", values="count").fillna(0)
    pvt_split_label.plot(kind="bar", stacked=True, ax=axes[2], colormap="Set3")
    axes[2].set_title("3) Label Distribution per Generated Split")
    axes[2].set_ylabel("Rows")
    axes[2].tick_params(axis="x", rotation=65)

    # 4) Sample distribution per generated split
    pvt_split_sample = split_sample_df.pivot(index="split", columns="sample_id", values="count").fillna(0)
    pvt_split_sample.plot(kind="bar", stacked=True, ax=axes[3], colormap="tab20")
    axes[3].set_title("4) Sample Distribution per Generated Split")
    axes[3].set_ylabel("Rows")
    axes[3].tick_params(axis="x", rotation=65)

    # 5) Cut distribution per generated split
    pvt_split_cut = split_cut_df.pivot(index="split", columns="pork_cut", values="count").fillna(0)
    pvt_split_cut.plot(kind="bar", stacked=True, ax=axes[4], color=["#59a14f", "#e15759"])
    axes[4].set_title("5) Cut Distribution per Generated Split")
    axes[4].set_ylabel("Rows")
    axes[4].tick_params(axis="x", rotation=65)

    # Keep final pane as legend/help text
    axes[5].axis("off")
    axes[5].text(
        0.0,
        1.0,
        "Diagnostics Figure\n- Includes items 1 to 5\n- Items 6 and 7 printed in console and CSV summaries",
        va="top",
        fontsize=12,
    )

    plt.tight_layout()
    out_path = PLOTS_DIR / "meatlens_split_diagnostics.png"
    plt.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def print_section_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_input_sample_stats(samples: dict[str, pd.DataFrame]) -> None:
    print_section_header("1) Total Rows per Input Sample")
    for sample_id, df in samples.items():
        print(f"{sample_id}: {len(df)}")

    print_section_header("2) Label Distribution per Input Sample")
    label_df = (
        pd.concat(samples.values(), ignore_index=True)
        .groupby(["sample_id", "label"])
        .size()
        .reset_index(name="count")
    )
    print(label_df.to_string(index=False))


def print_generated_split_stats(generated_splits_for_plot: list[tuple[str, pd.DataFrame]]) -> None:
    print_section_header("3) Label Distribution per Generated Split")
    for split_name, df in generated_splits_for_plot:
        print(f"{split_name}: {format_counts(label_count_dict(df))}")

    print_section_header("4) Sample Distribution per Generated Split")
    for split_name, df in generated_splits_for_plot:
        counts = df["sample_id"].value_counts().to_dict()
        print(f"{split_name}: {counts}")

    print_section_header("5) Cut Distribution per Generated Split")
    for split_name, df in generated_splits_for_plot:
        counts = df["pork_cut"].value_counts().to_dict()
        print(f"{split_name}: {counts}")


def print_missing_path_stats(generated_splits_for_plot: list[tuple[str, pd.DataFrame]]) -> None:
    print_section_header("6) Missing Image Path Count")
    any_path_col = False
    for split_name, df in generated_splits_for_plot:
        path_col = find_path_column(df)
        if path_col is None:
            continue
        any_path_col = True
        missing_values, missing_files = missing_path_count(df, path_col)
        print(
            f"{split_name} | path_col={path_col} | missing_value_rows={missing_values} | missing_file_rows={missing_files}"
        )
    if not any_path_col:
        print("No path column found in generated splits.")


def print_leakage_confirmation(leakage_checks: Iterable[dict[str, object]]) -> None:
    print_section_header("7) Cross-Rotation Leakage Confirmation")
    for row in leakage_checks:
        print(
            f"{row['fold']} | held_out={row['held_out_sample']} | "
            f"in_train={row['present_in_train']} | in_val={row['present_in_val']} | {row['status']}"
        )


def main() -> None:
    ensure_dirs()
    samples = load_input_samples()

    cross_summary, leakage_checks, cross_frames = generate_cross_rotation(samples)

    all_df = pd.concat(samples.values(), ignore_index=True)
    random_train, random_val, random_test, random_summary, random_frames = generate_random_split(all_df)

    split_frames_for_plot: list[tuple[str, pd.DataFrame]] = []
    for i in range(1, 5):
        fold = f"fold{i}"
        split_frames_for_plot.append((f"{fold}_train", pd.read_csv(CROSS_DIR / f"{fold}_train.csv")))
        split_frames_for_plot.append((f"{fold}_val", pd.read_csv(CROSS_DIR / f"{fold}_val.csv")))
        split_frames_for_plot.append((f"{fold}_test", pd.read_csv(CROSS_DIR / f"{fold}_test.csv")))
    split_frames_for_plot.extend(
        [
            ("random_train", random_train),
            ("random_val", random_val),
            ("random_test", random_test),
        ]
    )

    plot_path = make_diagnostics_plot(samples, split_frames_for_plot)

    print_input_sample_stats(samples)
    print_generated_split_stats(split_frames_for_plot)
    print_missing_path_stats(split_frames_for_plot)
    print_leakage_confirmation(leakage_checks)

    print_section_header("Artifacts")
    print(f"Cross rotation summary: {CROSS_DIR / 'cross_rotation_summary.csv'}")
    print(f"Cross rotation leakage: {CROSS_DIR / 'cross_rotation_leakage_check.csv'}")
    print(f"Random split summary: {RANDOM_DIR / 'random_split_summary.csv'}")
    print(f"Diagnostics plot image: {plot_path}")
    print(f"Seed used: {SEED}")
    print(f"Total combined rows: {len(all_df)}")
    print(f"Cross summary rows: {len(cross_summary)}")
    print(f"Random summary rows: {len(random_summary)}")


if __name__ == "__main__":
    main()
