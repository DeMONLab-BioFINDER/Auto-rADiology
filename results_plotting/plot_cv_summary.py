#!/usr/bin/env python3
"""Simple CV summary plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "fold": "#1f77b4",
    "train": "#1f77b4",
    "val": "#d62728",
}
METRICS = ["mae", "rmse", "r2", "eval_metric"]
TITLE_SIZE = 17
LABEL_SIZE = 15
TICK_SIZE = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create simple plots for CV runs.")
    parser.add_argument("--run_dir", required=True, help="Path to CV run directory.")
    return parser.parse_args()


def style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def style_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=TITLE_SIZE, pad=12)
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.grid(alpha=0.28)


def finalize_figure(fig: plt.Figure, rect: tuple[float, float, float, float] = (0.0, 0.0, 0.98, 0.97)) -> None:
    fig.tight_layout(rect=rect)


def save_figure(fig: plt.Figure, out_path: Path, **kwargs) -> bool:
    if out_path.exists():
        print(f"[skip] {out_path} exists")
        return False
    fig.savefig(out_path, **kwargs)
    return True


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed")]
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="last")]
    return df.copy()


def load_csv(path: Path) -> pd.DataFrame:
    return clean_dataframe(pd.read_csv(path))


def plot_fold_metrics(df: pd.DataFrame, out_path: Path) -> None:
    metric_cols = [metric for metric in METRICS if metric in df.columns]
    if not metric_cols:
        return

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.8))
    axes = axes.flatten()
    folds = df["fold"].astype(str).tolist()
    x = np.arange(len(folds))

    for axis, metric in zip(axes, METRICS):
        if metric not in df.columns:
            axis.set_visible(False)
            continue
        values = pd.to_numeric(df[metric], errors="coerce").to_numpy()
        axis.bar(x, values, color=COLORS["fold"])
        style_axes(axis, metric.upper() if metric != "eval_metric" else "Eval Metric", "Fold", "Value")
        axis.set_xticks(x)
        axis.set_xticklabels(folds)
        for x_value, y_value in zip(x, values):
            if np.isfinite(y_value):
                axis.text(x_value, y_value, f"{y_value:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("CV Metrics by Fold", y=0.99, fontsize=TITLE_SIZE)
    finalize_figure(fig, rect=(0.0, 0.0, 0.98, 0.95))
    save_figure(fig, out_path, dpi=300)
    plt.close(fig)


def load_curve_tables(run_dir: Path) -> list[pd.DataFrame]:
    tables = []
    for path in sorted(run_dir.glob("kfold-*/metrics/trainning_metrics_per_epoch.csv")):
        df = load_csv(path)
        if {"epoch", "train_loss"}.issubset(df.columns):
            for column in ["epoch", "train_loss", "val_loss"]:
                if column in df.columns:
                    df[column] = pd.to_numeric(df[column], errors="coerce")
            df = df.dropna(subset=["epoch", "train_loss"])
            if not df.empty:
                tables.append(df[["epoch", "train_loss", "val_loss"]].copy())
    return tables


def plot_single_fold_curve(df: pd.DataFrame, out_path: Path, fold_name: str) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    ax.plot(df["epoch"], df["train_loss"], label="train loss", linewidth=2, color=COLORS["train"])
    if "val_loss" in df.columns and df["val_loss"].notna().any():
        val_df = df.dropna(subset=["val_loss"])
        ax.plot(val_df["epoch"], val_df["val_loss"], label="val loss", linewidth=2, color=COLORS["val"])
    style_axes(ax, f"{fold_name}: Training and Validation Loss", "Epoch", "Loss")
    ax.legend(frameon=False, loc="best")
    finalize_figure(fig)
    save_figure(fig, out_path, dpi=300)
    plt.close(fig)


def plot_all_fold_curves(run_dir: Path, out_dir: Path) -> None:
    for path in sorted(run_dir.glob("kfold-*/metrics/trainning_metrics_per_epoch.csv")):
        df = load_csv(path)
        if not {"epoch", "train_loss"}.issubset(df.columns):
            continue
        for column in ["epoch", "train_loss", "val_loss"]:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")
        df = df.dropna(subset=["epoch", "train_loss"])
        if df.empty:
            continue
        fold_name = path.parts[-3]
        plot_single_fold_curve(df, out_dir / f"{fold_name}_training_validation_loss.png", fold_name)


def plot_mean_cv_curves(curve_tables: list[pd.DataFrame], out_path: Path) -> None:
    if not curve_tables:
        return

    merged_rows = []
    for fold_index, df in enumerate(curve_tables, start=1):
        tmp = df.copy()
        tmp["fold"] = fold_index
        merged_rows.append(tmp)
    long_df = pd.concat(merged_rows, ignore_index=True)

    summary = long_df.groupby("epoch", as_index=False).agg(
        train_loss_mean=("train_loss", "mean"),
        val_loss_mean=("val_loss", "mean"),
    )

    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    x = summary["epoch"].to_numpy()
    train_mean = summary["train_loss_mean"].to_numpy()
    ax.plot(x, train_mean, color=COLORS["train"], linewidth=2, label="Train loss")

    if summary["val_loss_mean"].notna().any():
        val_mean = summary["val_loss_mean"].to_numpy()
        valid = np.isfinite(val_mean)
        ax.plot(x[valid], val_mean[valid], color=COLORS["val"], linewidth=2, label="Validation loss")

    style_axes(ax, "Mean Training and Validation Loss Across Epochs", "Epoch", "Loss")
    ax.legend(frameon=False)
    finalize_figure(fig)
    save_figure(fig, out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    style_matplotlib()
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    out_dir = run_dir / "figures_cv"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics csv: {metrics_path}")

    metrics_df = load_csv(metrics_path)
    metrics_df.to_csv(out_dir / "cv_metrics_table.csv", index=False)
    plot_fold_metrics(metrics_df, out_dir / "cv_metrics_by_fold.png")
    plot_mean_cv_curves(load_curve_tables(run_dir), out_dir / "cv_mean_training_validation_loss.png")
    plot_all_fold_curves(run_dir, out_dir)


if __name__ == "__main__":
    main()
