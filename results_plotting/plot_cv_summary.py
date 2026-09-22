#!/usr/bin/env python3
"""Per-fold summary plot for a k-fold CV run (--run_kfold_cv, e.g. submit_mse-cv5.sh):
reads the fold-level metrics.csv that src/cv.py's kfold_cv() writes and shows the
spread of each performance metric across folds, plus a mean +/- std stats CSV."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from plot_utils import CUSTOM_PALETTE, SEABORN_COLORS, TITLE_SIZE, save_figure, style_axes

REGRESSION_METRICS = [("mae", "MAE"), ("rmse", "RMSE"), ("r2", "R2")]
CLASSIFICATION_METRICS = [("auc", "AUC"), ("acc", "Accuracy")]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a per-fold metric summary plot for a k-fold CV run (metrics.csv)."
    )
    parser.add_argument("run_dir", help="Path to a results run directory trained with --run_kfold_cv.")
    return parser.parse_args()


def load_metrics(run_dir: Path) -> pd.DataFrame:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Could not find {metrics_path} - this run may not have used --run_kfold_cv."
        )
    return pd.read_csv(metrics_path)


def make_fold_metric_panel(df: pd.DataFrame, metric_defs, out_dir: Path, filename: str, title: str) -> bool:
    present = [(k, lbl) for k, lbl in metric_defs if k in df.columns and df[k].notna().any()]
    if not present:
        return False

    fig, axes = plt.subplots(1, len(present), figsize=(4.5 * len(present), 4.5), dpi=300)
    axes = [axes] if len(present) == 1 else list(axes)

    for ax, (key, label), color in zip(axes, present, SEABORN_COLORS):
        values = df[key].dropna()
        sns.stripplot(y=values, ax=ax, color=color, size=9, jitter=0.08)
        ax.axhline(values.mean(), color=color, linestyle="--", linewidth=1.5, alpha=0.7)
        style_axes(ax, f"{label} per fold (n={len(values)})", "", label)

    fig.suptitle(title, fontsize=TITLE_SIZE + 1)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    save_figure(fig, out_dir / filename, dpi=300)
    plt.close(fig)
    return True


def main():
    sns.set_theme(style="whitegrid", palette=CUSTOM_PALETTE)
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    out_dir = run_dir / "figures"
    out_dir.mkdir(exist_ok=True)

    df = load_metrics(run_dir)

    made_any = False
    if make_fold_metric_panel(df, REGRESSION_METRICS, out_dir, "suvr_cv_fold_metrics.png",
                               "Regression metrics across CV folds"):
        made_any = True
    if make_fold_metric_panel(df, CLASSIFICATION_METRICS, out_dir, "visual_read_cv_fold_metrics.png",
                               "Classification metrics across CV folds"):
        made_any = True

    # metrics.csv only has the columns relevant to --targets going forward, but stay
    # defensive (e.g. reading an older metrics.csv) by also dropping any all-NaN column.
    metric_cols = [k for k, _ in REGRESSION_METRICS + CLASSIFICATION_METRICS
                   if k in df.columns and df[k].notna().any()]
    stats = df[metric_cols].agg(["mean", "std"]).T.reset_index().rename(columns={"index": "metric"})
    stats.to_csv(out_dir / "cv_summary_stats.csv", index=False)

    if not made_any:
        print("[WARNING] No non-null metric columns found in metrics.csv - nothing plotted.")
    print(f"[done] figures saved to {out_dir}")


if __name__ == "__main__":
    main()
