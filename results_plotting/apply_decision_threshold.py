#!/usr/bin/env python3
"""Reclassify a visual_read run's predictions at a different decision threshold, without
touching the original results.

The decision threshold is a pure post-hoc choice (see --decision_threshold in run.py /
run_val.py): the continuous 'prob' column is already saved, so trying a different cutoff
never requires retraining or even rerunning inference - just recomputing pred = (prob >=
threshold) and rebuilding the confusion-matrix-derived stats/figures from that.

Usage:
    python results_plotting/apply_decision_threshold.py \\
        /path/to/Eval_Gothenburg_results.csv --threshold 0.4514 --out_dir /path/to/thr_0.4514
"""

import argparse
from pathlib import Path

import pandas as pd

from plot_utils import (
    make_pr_curve_panel,
    make_probability_by_class_panel,
    make_roc_confusion_panel,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_csv", help="Path to an existing Eval_*_results.csv or External_validation_*_results.csv (needs y, prob columns).")
    parser.add_argument("--threshold", type=float, required=True, help="New decision threshold: prob >= threshold -> predicted positive.")
    parser.add_argument("--out_dir", required=True, help="Directory to write the reclassified results/metrics csv and figures into. The original file is never modified.")
    return parser.parse_args()


def reclassify_at_threshold(results_csv: Path, threshold: float, out_dir: Path) -> Path:
    """Recompute 'pred' from the saved 'prob' column at a new threshold, write the
    reclassified results csv + a matching metrics-csv sidecar into out_dir, and rebuild
    the threshold-dependent figures (ROC/confusion panel, PR curve, probability-by-class).
    Returns the path to the new results csv."""
    df = pd.read_csv(results_csv)
    required_cols = {"y", "prob"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{results_csv} is missing columns {sorted(missing)} - not a visual_read results csv.")

    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["pred"] = (df["prob"] >= threshold).astype(int)

    new_results_csv = out_dir / results_csv.name
    df.to_csv(new_results_csv, index=False)

    # Carry over threshold-independent diagnostics (auc, best_thr, best_epoch, ...) from the
    # original metrics csv if present, but refresh 'acc'/'eval_metric'/'decision_threshold' -
    # those are the fields that actually depend on which threshold produced 'pred'
    # (eval_metric = auc + acc, matching compute_metrics()'s definition in src/train.py).
    metrics_row = {"decision_threshold": threshold}
    old_metrics_csv = results_csv.parent / results_csv.name.replace("_results.csv", "_metrics.csv")
    if old_metrics_csv.exists():
        metrics_row = pd.read_csv(old_metrics_csv).iloc[0].to_dict()
        metrics_row["decision_threshold"] = threshold
    metrics_row["acc"] = float((df["pred"] == df["y"]).mean())
    metrics_row["eval_metric"] = float(pd.Series([metrics_row.get("auc"), metrics_row["acc"]]).sum(skipna=True))
    new_metrics_csv = out_dir / results_csv.name.replace("_results.csv", "_metrics.csv")
    pd.DataFrame([metrics_row]).to_csv(new_metrics_csv, index=False)

    y_true = df["y"].to_numpy()
    y_prob = df["prob"].to_numpy()
    y_pred = df["pred"].to_numpy()
    make_roc_confusion_panel(
        y_true, y_prob, y_pred, out_dir,
        stats_csv_path=out_dir / "visual_read_performance_stats.csv",
    )
    make_pr_curve_panel(y_true, y_prob, out_dir)
    make_probability_by_class_panel(y_true, y_prob, threshold, out_dir)

    return new_results_csv


def main():
    args = parse_args()
    results_csv = Path(args.results_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    reclassify_at_threshold(results_csv, args.threshold, out_dir)
    print(f"[done] reclassified at threshold={args.threshold} -> {out_dir}")


if __name__ == "__main__":
    main()
