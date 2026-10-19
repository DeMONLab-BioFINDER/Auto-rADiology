#!/usr/bin/env python3
"""Demographic/clinical context for discordant visual_read cases (False Positive /
False Negative), building on list_discordant_cases.py's output.

Run list_discordant_cases.py first for the same run_dir/--dataset - this script reads
its saved visual_read_discordant_cases.csv (for which cases are discordant and their
demographics) rather than re-deriving that itself, and does NOT touch or rely on that
script's three-way quantitative-agreement/isolated-MTL/unexplained-error bucket split
(see its docstring for why that split exists, and why no per-scan SUVR cutoff is used
anywhere in this analysis - the same reasoning applies here).

Produces two figures per run, saved next to list_discordant_cases.py's own output
(<run_dir>/figures/, or <run_dir>/validation/figures/ for a zero-shot external
validation run):

1. visual_read_discordant_suvr_percentiles.png - for each of the 5 CenTauR regions, a
   small-multiple showing each discordant case's COHORT-RELATIVE PERCENTILE (0-100,
   computed across the full run cohort, not just the discordant subset), split by
   False Positive / False Negative. No SUVR cutoff line - there is no validated
   per-scan cutoff (see list_discordant_cases.py's docstring).
2. visual_read_discordant_demographics_table.png - a descriptive (no p-values; groups
   can be single digits) characteristics table comparing the False Positive group, the
   False Negative group, and the full run cohort as a reference column, across age,
   sex, site, diagnosis, APOE, amyloid status, CDR, and CDR-SB.
3. visual_read_discordant_composition.png - the same variables as the table, as bars:
   % of each discordant group vs the full cohort in each category (counts above the
   bars), to show over/under-representation.

Usage:
    python results_plotting/plot_discordant_demographics.py /path/to/run_dir --dataset Gothenburg
"""

import argparse
from pathlib import Path

import pandas as pd

from list_discordant_cases import (
    DEFAULT_DEMO_COLS,
    DEFAULT_SUVR_COLS,
    find_dataset_demo_csv,
    find_zeroshot_results_csv,
)
from plot_utils import (
    confusion_display_label,
    find_demo_csv,
    make_discordant_composition_panel,
    make_discordant_demographics_table,
    make_discordant_percentile_panel,
    resolve_existing_path,
    resolve_path,
)

CONTINUOUS_COLS = [("age", "Age"), ("CDR_SB", "CDR-SB")]
CATEGORICAL_COLS = [
    ("gender", "Sex"),
    ("site", "Site"),
    ("dx", "Diagnosis"),
    ("apoe", "APOE"),
    ("amyloid_status", "Amyloid status"),
    ("CDR", "CDR"),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="Path to a results run directory trained with --targets visual_read.")
    parser.add_argument("--dataset", default="Gothenburg", help="Dataset name used in the result filenames.")
    parser.add_argument(
        "--demo_csv", default=None,
        help="Path to the demographics csv (defaults to data/demo_<dataset>.csv, then data/demo.csv).",
    )
    parser.add_argument(
        "--discordant_csv", default=None,
        help="Path to list_discordant_cases.py's output csv (defaults next to the run's other "
             "figures, same convention as that script - run it first if this isn't found yet).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()

    # Same preds/out_dir resolution as list_discordant_cases.py and plot_visual_read_results.py:
    # a zero-shot external validation result takes priority if present.
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

    discordant_csv = Path(args.discordant_csv) if args.discordant_csv else out_dir / "visual_read_discordant_cases.csv"
    if not discordant_csv.exists():
        raise FileNotFoundError(
            f"Could not find {discordant_csv}. Run list_discordant_cases.py for this same "
            "run_dir/--dataset first - this script reads its output rather than re-deriving it."
        )
    df_discordant = pd.read_csv(discordant_csv)
    if df_discordant.empty or not df_discordant["category"].isin(["False Positive", "False Negative"]).any():
        print(f"[done] No discordant (Model/Visual read disagree) cases in {discordant_csv}; nothing to plot.")
        return

    demo_path = resolve_path(args.demo_csv, lambda: find_dataset_demo_csv(args.dataset) or find_demo_csv())
    if demo_path is None:
        raise FileNotFoundError("Could not find a demographics csv - pass --demo_csv explicitly.")
    df_demo = pd.read_csv(demo_path)

    # Full run cohort (every subject with a prediction in this run, not just the discordant
    # subset) - needed as the percentile denominator and as the "Overall cohort" reference
    # column in the table.
    df_preds_ids = pd.read_csv(preds_path)[["ID_ind"]]
    demo_cols = [c for c in DEFAULT_DEMO_COLS + DEFAULT_SUVR_COLS if c in df_demo.columns]
    df_cohort = pd.merge(
        df_preds_ids, df_demo[["ID"] + demo_cols],
        left_on="ID_ind", right_on="ID", how="left",
    )

    for col in DEFAULT_SUVR_COLS:
        if col in df_cohort.columns:
            df_cohort[f"{col}_percentile"] = pd.to_numeric(df_cohort[col], errors="coerce").rank(pct=True) * 100

    out_dir.mkdir(parents=True, exist_ok=True)
    make_discordant_percentile_panel(df_cohort, df_discordant, out_dir, DEFAULT_SUVR_COLS)
    make_discordant_demographics_table(df_cohort, df_discordant, out_dir, CONTINUOUS_COLS, CATEGORICAL_COLS)
    make_discordant_composition_panel(df_cohort, df_discordant, out_dir)

    n_fp = (df_discordant["category"] == "False Positive").sum()
    n_fn = (df_discordant["category"] == "False Negative").sum()
    print(
        f"[done] {n_fp} x [{confusion_display_label('False Positive')}], "
        f"{n_fn} x [{confusion_display_label('False Negative')}] discordant cases; figures saved to {out_dir}"
    )


if __name__ == "__main__":
    main()
