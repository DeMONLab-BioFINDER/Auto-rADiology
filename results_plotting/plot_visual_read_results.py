#!/usr/bin/env python3
"""Plots for the visual-read (binary classification) model: class balance and
agreement between model predictions and the expert clinical read."""

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns

from plot_utils import (
    CUSTOM_PALETTE,
    find_demo_csv,
    make_class_balance_panel,
    make_roc_confusion_panel,
    resolve_existing_path,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create class-balance and clinical-read-agreement plots for a visual_read run."
    )
    parser.add_argument("run_dir", help="Path to a results run directory trained with --targets visual_read.")
    parser.add_argument("--dataset", default="Gothenburg", help="Dataset name used in the result filenames.")
    parser.add_argument("--group_col", default="site", help="Demographic column to break class balance down by.")
    return parser.parse_args()


def load_predictions(run_dir: Path, dataset: str) -> pd.DataFrame:
    evaluation_dir = run_dir / "evaluation" / dataset
    preds_path = resolve_existing_path(
        [
            run_dir / "validation" / dataset / f"Test_{dataset}_results.csv",
            evaluation_dir / f"Eval_{dataset}_results.csv",
        ],
        "predictions csv",
    )
    df = pd.read_csv(preds_path)
    required_cols = {"y", "prob"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"{preds_path} is missing columns {sorted(missing)} — "
            "this run may not have been trained with a visual_read classification target."
        )
    return df


def main():
    sns.set_theme(style="whitegrid", palette=CUSTOM_PALETTE)

    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    out_dir = run_dir / "figures"
    out_dir.mkdir(exist_ok=True)

    demo_path = find_demo_csv()
    if demo_path is not None:
        df_demo = pd.read_csv(demo_path)
        make_class_balance_panel(df_demo, out_dir, group_col=args.group_col)
    else:
        print("[WARNING] Could not find demographics csv for class balance panel.")

    df_preds = load_predictions(run_dir, args.dataset)
    make_roc_confusion_panel(
        df_preds["y"].to_numpy(),
        df_preds["prob"].to_numpy(),
        out_dir,
        stats_csv_path=out_dir / "visual_read_performance_stats.csv",
    )
    print(f"[done] figures saved to {out_dir}")


if __name__ == "__main__":
    main()
