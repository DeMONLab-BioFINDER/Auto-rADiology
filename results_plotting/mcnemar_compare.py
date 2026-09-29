#!/usr/bin/env python3
"""Paired significance test (exact McNemar) for whether two models' predictions
differ on the same subjects - e.g. weighted vs unweighted visual_read, evaluated
on the same AVID external-validation set.

Unlike comparing two independent accuracy/sensitivity numbers (which ignores that
both models were scored on the exact same subjects), McNemar's test only looks at
the subjects where the two models *disagree* (one got it right, the other didn't)
and asks whether that disagreement leans significantly toward one model - which is
the right test for "is this difference real, or could it easily happen by chance
with this few subjects."

Reports three views:
  - overall: correct/incorrect (accuracy-level agreement)
  - positives only (y == 1): pred == 1 counts as "correct" - this is the
    sensitivity-focused view (relevant to the AVID sensitivity comparison)
  - negatives only (y == 0): pred == 0 counts as "correct" - the specificity-focused view

Usage:
    python results_plotting/mcnemar_compare.py \\
        /path/to/unweighted_External_validation_AVID_results.csv \\
        /path/to/weighted_External_validation_AVID_results.csv \\
        --label_a unweighted --label_b weighted
"""

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_csv_a", help="First results csv (needs ID_ind, y, pred columns).")
    parser.add_argument("results_csv_b", help="Second results csv, evaluated on the same subjects as results_csv_a.")
    parser.add_argument("--label_a", default="a", help="Short label for the first model in the printed report.")
    parser.add_argument("--label_b", default="b", help="Short label for the second model in the printed report.")
    return parser.parse_args()


def exact_mcnemar(correct_a, correct_b, n_subjects_label):
    """Given two boolean arrays of per-subject correctness (same subjects, same order),
    print the 2x2 agreement table and the exact (binomial) McNemar p-value on the
    discordant pairs. Returns (n10, n01, p_value)."""
    both_right = int((correct_a & correct_b).sum())
    both_wrong = int((~correct_a & ~correct_b).sum())
    a_right_b_wrong = int((correct_a & ~correct_b).sum())
    a_wrong_b_right = int((~correct_a & correct_b).sum())
    n_discordant = a_right_b_wrong + a_wrong_b_right

    print(f"  [{n_subjects_label}] n={len(correct_a)}")
    print(f"    both right: {both_right}   both wrong: {both_wrong}")
    print(f"    a right / b wrong: {a_right_b_wrong}   a wrong / b right: {a_wrong_b_right}   (discordant: {n_discordant})")

    if n_discordant == 0:
        print("    No discordant pairs - models agreed on every subject in this subset. p=1.0 (no evidence of a difference).")
        return a_right_b_wrong, a_wrong_b_right, 1.0

    result = binomtest(min(a_right_b_wrong, a_wrong_b_right), n_discordant, 0.5, alternative="two-sided")
    p_value = result.pvalue
    print(f"    exact McNemar p-value: {p_value:.4f}" + ("  <-- significant at alpha=0.05" if p_value < 0.05 else ""))
    if n_discordant < 10:
        print(f"    NOTE: only {n_discordant} discordant pairs - this test has very low power here; "
              "a non-significant result doesn't mean the models are equivalent, just that this sample can't tell them apart.")
    return a_right_b_wrong, a_wrong_b_right, p_value


def main():
    args = parse_args()
    df_a = pd.read_csv(args.results_csv_a)
    df_b = pd.read_csv(args.results_csv_b)

    required_cols = {"ID_ind", "y", "pred"}
    for name, df in [(args.results_csv_a, df_a), (args.results_csv_b, df_b)]:
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{name} is missing columns {sorted(missing)}.")

    merged = df_a[["ID_ind", "y", "pred"]].merge(
        df_b[["ID_ind", "y", "pred"]], on="ID_ind", suffixes=("_a", "_b")
    )
    if len(merged) != len(df_a) or len(merged) != len(df_b):
        print(f"[WARNING] {args.results_csv_a} has {len(df_a)} rows, {args.results_csv_b} has {len(df_b)}, "
              f"only {len(merged)} ID_ind values matched between the two - are these really the same evaluation subjects?")
    mismatched_labels = merged["y_a"] != merged["y_b"]
    if mismatched_labels.any():
        raise ValueError(f"{mismatched_labels.sum()} subjects have a different true label 'y' between the two files - "
                          "these results csvs don't appear to be from the same dataset/labels.")

    y = merged["y_a"].to_numpy()
    pred_a = merged["pred_a"].to_numpy()
    pred_b = merged["pred_b"].to_numpy()

    print(f"Comparing {args.label_a} vs {args.label_b} on {len(merged)} matched subjects\n")

    print("Overall (accuracy-level: pred == y counts as correct):")
    exact_mcnemar(pred_a == y, pred_b == y, "overall")

    pos_mask = y == 1
    if pos_mask.sum() > 0:
        print(f"\nPositives only, y==1 (sensitivity-focused: pred == 1 counts as correct):")
        exact_mcnemar((pred_a == 1)[pos_mask], (pred_b == 1)[pos_mask], "positives")
    else:
        print("\nNo positive subjects in this matched set - skipping sensitivity-focused test.")

    neg_mask = y == 0
    if neg_mask.sum() > 0:
        print(f"\nNegatives only, y==0 (specificity-focused: pred == 0 counts as correct):")
        exact_mcnemar((pred_a == 0)[neg_mask], (pred_b == 0)[neg_mask], "negatives")
    else:
        print("\nNo negative subjects in this matched set - skipping specificity-focused test.")


if __name__ == "__main__":
    main()
