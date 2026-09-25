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
    make_confusion_category_distribution_panel,
    make_roc_confusion_panel,
    resolve_existing_path,
    resolve_path,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create class-balance and clinical-read-agreement plots for a visual_read run."
    )
    parser.add_argument("run_dir", help="Path to a results run directory trained with --targets visual_read.")
    parser.add_argument("--dataset", default="Gothenburg", help="Dataset name used in the result filenames.")
    parser.add_argument("--group_col", default="site", help="Demographic column to break class balance down by.")
    parser.add_argument(
        "--demo_csv",
        default=None,
        help="Path to the demographics csv (defaults to data/demo.csv if found). Set this "
             "explicitly for a dataset with its own demo file, e.g. demo_test_subset_tau_raw.csv.",
    )
    return parser.parse_args()


def find_zeroshot_results_csv(run_dir: Path, dataset: str) -> Path | None:
    """Results from a run_val.py --few_shot 0 external validation (e.g. reusing
    test_subset as a stand-in unseen dataset), saved under <run_dir>/validation/."""
    matches = sorted((run_dir / "validation").glob(f"External_validation_{dataset}__*_zeroshot_results.csv"))
    return matches[0] if matches else None


def find_dataset_demo_csv(dataset: str) -> Path | None:
    """demo_{dataset}.csv, the same naming convention run_val.py's load_validation_data
    uses (e.g. demo_AVID_unseen.csv) - checked before falling back to the generic
    data/demo.csv, since a run on a non-Gothenburg dataset (AVID, a subset smoke test,
    ...) needs its own demographics for the confusion-category merge to find any
    matching IDs at all."""
    script_path = Path(__file__).resolve()
    demo_candidates = [
        script_path.parents[1] / "data" / f"demo_{dataset}.csv",
        script_path.parents[2] / "data" / f"demo_{dataset}.csv",
    ]
    return next((p for p in demo_candidates if p.exists()), None)


def load_predictions(preds_path: Path) -> pd.DataFrame:
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

    # A zero-shot external validation result (run_val.py) takes priority if present -
    # its figures live under validation/figures/, separate from the run's own held-out
    # test figures in figures/, so the two never overwrite each other.
    zeroshot_path = find_zeroshot_results_csv(run_dir, args.dataset)
    if zeroshot_path is not None:
        preds_path = zeroshot_path
        out_dir = run_dir / "validation" / "figures"
    else:
        evaluation_dir = run_dir / "evaluation" / args.dataset
        preds_path = resolve_existing_path(
            [
                run_dir / "validation" / args.dataset / f"Test_{args.dataset}_results.csv",
                evaluation_dir / f"Eval_{args.dataset}_results.csv",
            ],
            "predictions csv",
        )
        out_dir = run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    demo_path = resolve_path(args.demo_csv, lambda: find_dataset_demo_csv(args.dataset) or find_demo_csv())
    df_demo = None
    if demo_path is not None:
        df_demo = pd.read_csv(demo_path)
        make_class_balance_panel(df_demo, out_dir, group_col=args.group_col)
    else:
        print("[WARNING] Could not find demographics csv for class balance panel.")

    df_preds = load_predictions(preds_path)
    make_roc_confusion_panel(
        df_preds["y"].to_numpy(),
        df_preds["prob"].to_numpy(),
        out_dir,
        stats_csv_path=out_dir / "visual_read_performance_stats.csv",
    )

    if df_demo is not None:
        make_confusion_category_distribution_panel(
            df_preds, df_demo, out_dir,
            value_col="Universal", value_label="Universal tau SUVR",
        )

    print(f"[done] figures saved to {out_dir}")


if __name__ == "__main__":
    main()
