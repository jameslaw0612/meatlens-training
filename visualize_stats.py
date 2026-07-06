#!/usr/bin/env python3
"""Generate a counts table and visualizations from the combined CSV.

Creates:
- pork_shoulder_label_summary.csv  (label,count,percent)
- pork_shoulder_label_counts.png  (bar chart)
- pork_shoulder_label_counts_by_sample.png (grouped bar chart)

Usage:
    python visualize_stats.py --input pork_shoulder_combined.csv --output-dir .
"""
from __future__ import annotations
import argparse
import csv
import os
from collections import Counter, defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


def read_rows(path: str) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def compute_counts(rows: list[dict]):
    counts = Counter()
    per_sample = defaultdict(Counter)
    total = 0
    for r in rows:
        label = (r.get("label") or "").strip()
        sample = (r.get("sample_number") or "").strip()
        counts[label] += 1
        per_sample[sample][label] += 1
        total += 1
    return counts, per_sample, total


def save_summary_csv(counts: Counter, total: int, outpath: str) -> None:
    with open(outpath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "count", "percent"])
        for label, cnt in counts.items():
            w.writerow([label, cnt, f"{cnt/total*100:.2f}"])


def print_table(counts: Counter, total: int) -> None:
    print("Label counts summary:")
    print(f"Total rows: {total}")
    print(f"{'label':20} {'count':>8} {'percent':>8}")
    for label, cnt in counts.items():
        print(f"{label:20} {cnt:8d} {cnt/total*100:8.2f}%")


def plot_counts(counts: Counter, outpath: str) -> None:
    labels = list(counts.keys())
    values = [counts[l] for l in labels]
    colors = [('#2ca02c' if 'fresh' in l.lower() else '#ff7f0e' if 'not' in l.lower() else '#d62728') for l in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel('Count')
    ax.set_title('Pork Shoulder — label counts')
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, str(v), ha='center', va='bottom')
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_by_sample(per_sample: dict, all_labels: list, outpath: str) -> None:
    samples = sorted(per_sample.keys())
    n = len(all_labels)
    x = list(range(n))
    width = 0.8 / max(1, len(samples))
    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), 4))
    for i, sample in enumerate(samples):
        vals = [per_sample[sample].get(lbl, 0) for lbl in all_labels]
        offsets = [xi + (i - (len(samples) - 1) / 2) * width for xi in x]
        ax.bar(offsets, vals, width=width, label=f'Sample {sample}')
        for ox, v in zip(offsets, vals):
            ax.text(ox, v, str(v), ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels)
    ax.set_ylabel('Count')
    ax.set_title('Label counts by sample')
    ax.legend()
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', default='pork_shoulder_combined.csv', help='Combined CSV path')
    p.add_argument('--output-dir', '-o', default='.', help='Directory to save charts and summary')
    args = p.parse_args()

    inp = args.input
    outdir = args.output_dir
    os.makedirs(outdir, exist_ok=True)

    if not os.path.isfile(inp):
        print(f"Input file not found: {inp}")
        raise SystemExit(1)

    rows = read_rows(inp)
    counts, per_sample, total = compute_counts(rows)

    print_table(counts, total)

    summary_csv = os.path.join(outdir, 'pork_shoulder_label_summary.csv')
    save_summary_csv(counts, total, summary_csv)
    print(f"Wrote summary CSV: {summary_csv}")

    if HAVE_MPL:
        bar_png = os.path.join(outdir, 'pork_shoulder_label_counts.png')
        plot_counts(counts, bar_png)
        print(f"Saved bar chart: {bar_png}")

        all_labels = list(counts.keys())
        by_sample_png = os.path.join(outdir, 'pork_shoulder_label_counts_by_sample.png')
        plot_by_sample(per_sample, all_labels, by_sample_png)
        print(f"Saved per-sample chart: {by_sample_png}")
    else:
        print('matplotlib not available — charts were not created. Install matplotlib to enable charts.')


if __name__ == '__main__':
    main()
