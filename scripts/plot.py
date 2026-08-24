#!/usr/bin/env python3
"""Read a sweep-results CSV and render accuracy/throughput/skew-sensitivity charts."""
import argparse
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt


REQUIRED_COLUMNS = {
    "run_id", "algorithm", "trial", "seed", "cardinality", "skew",
    "stream_length", "top_k_target", "warmup_fraction", "requested_memory_bytes",
    "actual_memory_bytes", "construction_time_ns", "insert_elapsed_ns",
    "items_per_sec", "precision_at_k", "recall_at_k", "f1_at_k",
    "mean_relative_error", "max_relative_error", "underestimate_count",
    "overestimate_count", "workload", "num_heavy", "heavy_jitter",
    "heavy_mass_fraction"
}


def load_results(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        sys.exit(f"error: CSV file not found: {csv_path}")
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        sys.exit(f"error: failed to read CSV file {csv_path}: {e}")


def validate_columns(df: pd.DataFrame, csv_path: str) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        sys.exit(f"error: CSV file {csv_path} is missing columns: {sorted(missing)}")


def filter_workload(df: pd.DataFrame, workload: str, csv_path: str) -> pd.DataFrame:
    present = sorted(df["workload"].unique())
    if workload is not None:
        filtered = df[df["workload"] == workload]
        if filtered.empty:
            sys.exit(f"error: no rows with workload={workload!r} found in {csv_path} (present: {present})")
        return filtered
    if len(present) > 1:
        sys.exit(
            f"error: {csv_path} contains multiple workload types {present} — "
            "pass --workload to select one, plotting a mix would be misleading"
        )
    return df


def plot_accuracy_vs_memory(df: pd.DataFrame, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    grouped = df.groupby(["algorithm", "actual_memory_bytes"])["f1_at_k"].mean().reset_index()
    for algorithm, sub in grouped.groupby("algorithm"):
        sub = sub.sort_values("actual_memory_bytes")
        ax.plot(sub["actual_memory_bytes"], sub["f1_at_k"], marker="o", label=algorithm)
    ax.set_xscale("log")
    ax.set_xlabel("Actual memory (bytes, log scale)")
    ax.set_ylabel("F1 @ k")
    ax.set_title("Accuracy vs. memory budget")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "accuracy_vs_memory.png"), dpi=150)
    plt.close(fig)


def plot_throughput_vs_memory(df: pd.DataFrame, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    grouped = df.groupby(["algorithm", "actual_memory_bytes"])["items_per_sec"].mean().reset_index()
    for algorithm, sub in grouped.groupby("algorithm"):
        sub = sub.sort_values("actual_memory_bytes")
        ax.plot(sub["actual_memory_bytes"], sub["items_per_sec"], marker="o", label=algorithm)
    ax.set_xscale("log")
    ax.set_xlabel("Actual memory (bytes, log scale)")
    ax.set_ylabel("Items / sec")
    ax.set_title("Throughput vs. memory budget")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "throughput_vs_memory.png"), dpi=150)
    plt.close(fig)


def plot_skew_sensitivity(df: pd.DataFrame, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    grouped = df.groupby(["algorithm", "skew"])["f1_at_k"].mean().reset_index()
    for algorithm, sub in grouped.groupby("algorithm"):
        sub = sub.sort_values("skew")
        ax.plot(sub["skew"], sub["f1_at_k"], marker="o", label=algorithm)
    ax.set_xlabel("Zipfian skew")
    ax.set_ylabel("F1 @ k")
    ax.set_title("Skew sensitivity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "skew_sensitivity.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot heavy-hitters sweep results.")
    parser.add_argument("csv_path", help="Path to a sweep-results CSV file")
    parser.add_argument("--out", default="charts", help="Output directory for PNG charts")
    parser.add_argument(
        "--workload",
        choices=["zipfian", "plateau"],
        default=None,
        help="Filter to one workload type. Required if the CSV contains more than one.",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = load_results(args.csv_path)
    validate_columns(df, args.csv_path)
    df = filter_workload(df, args.workload, args.csv_path)

    plot_accuracy_vs_memory(df, args.out)
    plot_throughput_vs_memory(df, args.out)
    plot_skew_sensitivity(df, args.out)

    print(f"wrote 3 charts to {args.out}/")


if __name__ == "__main__":
    main()
