#!/usr/bin/env python3
"""Does the visual-read model work equally well in every subgroup? (SUVR-model counterpart:
analyze_subgroup_performance_suvr.py.)

What is compared. Sensitivity and specificity, each within its own visual-read group:
  - sensitivity: of the scans the visual read called POSITIVE, how many the model also called positive
  - specificity: of the scans the visual read called NEGATIVE, how many the model also called negative
Accuracy and PPV are not compared: both move with how many positive scans a subgroup happens to
contain, so e.g. CU would look better than AD dementia mostly because CU scans are mostly clearly
negative. Splitting by the visual read first removes that.

Variables (fixed in advance, see add_subgroup_columns in plot_utils.py): diagnosis, sex, APOE e4,
amyloid status, site, and age as a continuous number.

Tests, each run once for sensitivity and once for specificity:
  - 2 levels: Fisher's exact test
  - 3+ levels: permutation chi-square test (exact enough for the small cells here, e.g. a site with
    3 visual-read positives, where the ordinary chi-square approximation is unreliable)
  - age: Mann-Whitney test of age between the scans the model got right and the ones it got wrong
All p-values are Benjamini-Hochberg corrected together, as one family ("primary").

Secondary question, corrected as its own family so it cannot dilute the primary one:
  - CDR-SB beyond diagnosis: within each visual-read group, is CDR-SB higher in the scans the model
    got wrong, comparing people only against others with the same diagnosis? (within-diagnosis
    permutation test, see within_strata_permutation_test). Scans without CDR-SB - a whole cohort,
    if it never collected it - drop out of this test only; the script prints which.

Described but not tested: cognitive status x amyloid (CU Aβ−, CU Aβ+, impaired Aβ−, impaired Aβ+),
for the trial-screening question - in the performance csv with tested = False, no figure.

Outputs, next to the run's other figures:
    visual_read_subgroup_performance.csv  - every level of every variable: n, sensitivity and
                                            specificity (Wilson 95% CI), PPV, NPV, AUC (bootstrap CI)
    visual_read_subgroup_tests.csv        - every test: family, raw and FDR-corrected p
    visual_read_subgroup_<variable>.png   - ONLY for variables with an FDR-significant difference,
                                            showing only the measure(s) that differed
    visual_read_subgroup_cdrsb.png        - ONLY if CDR-SB beyond diagnosis is significant: CDR-SB of
                                            scans the model agreed vs disagreed with, per diagnosis

A variable without a figure has not been shown to make no difference - with small groups the
honest reading is "no difference detected", and the csv shows how wide the intervals are.

Usage:
    python results_plotting/analyze_subgroup_performance_visual_read.py /path/to/run_dir \
        --dataset Gothenburg
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analyze_bias_significance import add_fdr_columns
from list_discordant_cases import find_dataset_demo_csv
from plot_utils import (
    SUBGROUP_METRICS,
    SUBGROUP_MIN_LEVEL_N,
    add_subgroup_columns,
    bootstrap_auc_ci,
    confusion_display_label,
    find_demo_csv,
    make_subgroup_performance_panel,
    make_value_by_group_figure,
    report_missing_by_site,
    resolve_path,
    resolve_predictions_csv,
    wilson_interval,
    within_strata_permutation_test,
)

ALPHA = 0.05
# Visual-read value of the scans each metric is defined on.
METRIC_READ = {"sensitivity": 1, "specificity": 0}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="Path to a results run directory trained with --targets visual_read.")
    parser.add_argument("--dataset", default="Gothenburg", help="Dataset name used in the result filenames.")
    parser.add_argument("--demo_csv", default=None, help="Path to the demographics csv.")
    parser.add_argument("--female_code", type=float, default=2, help="Value of `gender` meaning female.")
    parser.add_argument("--min_level_n", type=int, default=SUBGROUP_MIN_LEVEL_N,
                        help="Levels with fewer scans than this are kept in the csv and the tests "
                             "but left off the figures.")
    parser.add_argument("--n_perm", type=int, default=10000, help="Permutations for the 3+ level test.")
    return parser.parse_args()


def load_scans(args, run_dir: Path):
    """One row per scan: visual read (y), model call (pred), probability and demographics."""
    preds_path, out_dir = resolve_predictions_csv(run_dir, args.dataset)
    demo_path = resolve_path(args.demo_csv, lambda: find_dataset_demo_csv(args.dataset) or find_demo_csv())
    if demo_path is None:
        raise FileNotFoundError("Could not find a demographics csv - pass --demo_csv explicitly.")

    df_preds = pd.read_csv(preds_path)
    keep = [c for c in ["ID_ind", "y", "pred", "prob"] if c in df_preds.columns]
    df = pd.merge(df_preds[keep], pd.read_csv(demo_path), left_on="ID_ind", right_on="ID", how="left")
    unmatched = df["ID"].isna()
    if unmatched.any():
        print(f"[WARNING] {int(unmatched.sum())}/{len(df)} scans have no row in {demo_path.name} and are "
              "excluded - check --demo_csv matches this run.")
        df = df[~unmatched]
    for col in ["y", "pred", "prob"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["y", "pred"]).reset_index(drop=True)
    df["_correct"] = (df["y"] == df["pred"]).astype(int)
    print(f"[load] {len(df)} scans from {preds_path}")
    return df, out_dir


def level_stats(g: pd.DataFrame) -> dict:
    """Sensitivity / specificity (Wilson 95% CI), PPV, NPV and AUC (bootstrap 95% CI) for one level."""
    y, pred = g["y"].astype(int).to_numpy(), g["pred"].astype(int).to_numpy()
    tp, fn = int(((y == 1) & (pred == 1)).sum()), int(((y == 1) & (pred == 0)).sum())
    tn, fp = int(((y == 0) & (pred == 0)).sum()), int(((y == 0) & (pred == 1)).sum())
    row = {"n": len(g), "n_read_pos": tp + fn, "n_read_neg": tn + fp, "tp": tp, "fp": fp, "tn": tn, "fn": fn}
    for metric, k, n in [("sensitivity", tp, tp + fn), ("specificity", tn, tn + fp)]:
        lo, hi = wilson_interval(k, n)
        row |= {metric: k / n if n else np.nan, f"{metric}_lo": lo, f"{metric}_hi": hi,
                f"{metric}_k": k, f"{metric}_n": n}
    row["ppv"] = tp / (tp + fp) if (tp + fp) else np.nan
    row["npv"] = tn / (tn + fn) if (tn + fn) else np.nan
    if "prob" in g.columns and g["prob"].notna().all():
        row["auc"], row["auc_lo"], row["auc_hi"] = bootstrap_auc_ci(y, g["prob"].to_numpy())
    return row


def permutation_chi2_test(codes: np.ndarray, outcome: np.ndarray, n_perm: int, seed: int = 0):
    """Chi-square statistic for a k x 2 table (level x right/wrong) and its permutation p-value:
    how often shuffling right/wrong across the scans gives a table at least this uneven."""
    k = codes.max() + 1
    onehot = np.eye(k)[codes]
    n_level = onehot.sum(axis=0)
    p = outcome.mean()
    expected = n_level * p
    weight = 1 / expected + 1 / (n_level - expected)

    def chi2(o):
        return (((o @ onehot) - expected) ** 2 * weight).sum(axis=-1)

    observed = chi2(outcome)
    perms = np.random.default_rng(seed).permuted(np.tile(outcome, (n_perm, 1)), axis=1)
    p_value = (1 + np.sum(chi2(perms) >= observed - 1e-12)) / (n_perm + 1)
    return float(observed), float(p_value)


def test_categorical(df: pd.DataFrame, col: str, name: str, levels: list, metric: str, n_perm: int) -> dict:
    """Does `metric` differ between the levels? Uses only the scans the metric is defined on."""
    sub = df[(df["y"] == METRIC_READ[metric]) & df[col].isin(levels)]
    used = [level for level in levels if (sub[col] == level).any()]
    out = {"variable": name, "metric": metric, "n_levels": len(used), "n_scans": len(sub),
           "test": None, "statistic": np.nan, "p_value": np.nan}
    if len(used) < 2 or sub["_correct"].nunique() < 2:
        out["note"] = "not testable (fewer than 2 levels, or every call right / every call wrong)"
        return out
    if len(used) == 2:
        table = [[int((g := sub[sub[col] == level])["_correct"].sum()), int(len(g) - g["_correct"].sum())]
                 for level in used]
        out["test"] = "fisher_exact"
        out["statistic"], out["p_value"] = stats.fisher_exact(table)
    else:
        codes = sub[col].map({level: i for i, level in enumerate(used)}).to_numpy(dtype=int)
        out["test"] = "permutation_chi2"
        out["statistic"], out["p_value"] = permutation_chi2_test(codes, sub["_correct"].to_numpy(float), n_perm)
    return out


def test_age(df: pd.DataFrame, metric: str) -> dict:
    """Is age different between the scans the model got right and wrong (within one visual-read group)?"""
    sub = df[(df["y"] == METRIC_READ[metric])].copy()
    sub["age"] = pd.to_numeric(sub.get("age"), errors="coerce")
    sub = sub.dropna(subset=["age"])
    right, wrong = sub.loc[sub["_correct"] == 1, "age"], sub.loc[sub["_correct"] == 0, "age"]
    out = {"variable": "Age", "metric": metric, "n_levels": np.nan, "n_scans": len(sub),
           "test": None, "statistic": np.nan, "p_value": np.nan,
           "median_age_model_right": right.median(), "median_age_model_wrong": wrong.median()}
    if len(right) < 2 or len(wrong) < 2:
        out["note"] = "not testable (fewer than 2 scans the model got right or wrong)"
        return out
    out["test"] = "mann_whitney"
    out["statistic"], out["p_value"] = stats.mannwhitneyu(right, wrong, alternative="two-sided")
    return out


def test_cdrsb_beyond_dx(df: pd.DataFrame, metric: str, n_perm: int) -> dict:
    """Is CDR-SB higher in the scans the model got wrong, among people with the same diagnosis?"""
    sub = df[df["y"] == METRIC_READ[metric]].copy()
    sub["_cdrsb"] = pd.to_numeric(sub["CDR_SB"], errors="coerce")
    sub = sub.dropna(subset=["_cdrsb", "_g_dx"])
    out = {"family": "secondary", "variable": "CDR-SB (within diagnosis)", "metric": metric,
           "n_levels": np.nan, "n_scans": len(sub), "test": None, "statistic": np.nan, "p_value": np.nan,
           "median_cdrsb_model_right": sub.loc[sub["_correct"] == 1, "_cdrsb"].median(),
           "median_cdrsb_model_wrong": sub.loc[sub["_correct"] == 0, "_cdrsb"].median()}
    if sub["_correct"].nunique() < 2 or sub["_cdrsb"].nunique() < 2:
        out["note"] = "not testable (every call right / every call wrong, or CDR-SB constant)"
        return out
    out["test"] = "within_diagnosis_permutation"
    out["statistic"], out["p_value"] = within_strata_permutation_test(
        1 - sub["_correct"].to_numpy(), sub["_cdrsb"].to_numpy(), sub["_g_dx"], n_perm=n_perm)
    return out


def make_cdrsb_figure(df: pd.DataFrame, metrics: list[str], dx_levels: list, min_level_n: int, out_dir: Path):
    """CDR-SB of the scans the model agreed vs disagreed with, one panel per diagnosis (and per
    significant metric) - the comparison the within-diagnosis test made."""
    panels = []
    for metric in metrics:
        read = METRIC_READ[metric]
        sub = df[df["y"] == read]
        cdrsb = pd.to_numeric(sub["CDR_SB"], errors="coerce")
        for level in dx_levels:
            in_dx = sub["_g_dx"] == level
            if (in_dx & cdrsb.notna()).sum() < min_level_n:
                continue
            panels.append((f"{level}, visual read {'+' if read else '−'}", {
                confusion_display_label("True Positive" if read else "True Negative", multiline=True):
                    cdrsb[in_dx & (sub["_correct"] == 1)].dropna().to_numpy(),
                confusion_display_label("False Negative" if read else "False Positive", multiline=True):
                    cdrsb[in_dx & (sub["_correct"] == 0)].dropna().to_numpy(),
            }))
    make_value_by_group_figure(panels, "CDR-SB", "CDR-SB of scans the model agreed vs disagreed with, by diagnosis",
                               out_dir / "visual_read_subgroup_cdrsb.png")


def add_cog_amyloid_column(df: pd.DataFrame) -> list:
    """CU / impaired x amyloid negative / positive, described only (trial-screening question)."""
    if "_g_dx" not in df.columns or "amyloid_status" not in df.columns:
        return []
    cog = df["_g_dx"].map(lambda v: pd.NA if pd.isna(v) else ("CU" if v == "CU" else "Impaired"))
    amy = pd.to_numeric(df["amyloid_status"], errors="coerce").map({0: "Aβ−", 1: "Aβ+"})
    df["_g_cog_amy"] = (cog.astype("string") + " " + amy.astype("string"))
    return [level for level in ["CU Aβ−", "CU Aβ+", "Impaired Aβ−", "Impaired Aβ+"]
            if (df["_g_cog_amy"] == level).any()]


def make_age_figure(df: pd.DataFrame, metrics: list[str], out_dir: Path):
    """Age of the scans the model got right vs wrong, one panel per significant metric."""
    panels = []
    for metric in metrics:
        read = METRIC_READ[metric]
        sub = df[df["y"] == read]
        age = pd.to_numeric(sub["age"], errors="coerce")
        panels.append((f"Visual read {'+' if read else '−'} scans ({metric})", {
            confusion_display_label("True Positive" if read else "True Negative", multiline=True):
                age[sub["_correct"] == 1].dropna().to_numpy(),
            confusion_display_label("False Negative" if read else "False Positive", multiline=True):
                age[sub["_correct"] == 0].dropna().to_numpy(),
        }))
    make_value_by_group_figure(panels, "Age (years)", "Age of scans the model agreed vs disagreed with",
                               out_dir / "visual_read_subgroup_age.png")


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    df, out_dir = load_scans(args, run_dir)
    if df.empty:
        print("[done] No scans with predictions; nothing to analyze.")
        return

    subgroups = add_subgroup_columns(df, female_code=args.female_code)
    stat_rows, test_rows = [], []
    for col, name, levels in subgroups:
        for level in levels:
            stat_rows.append({"variable": name, "level": level, "column": col, "tested": True}
                             | level_stats(df[df[col] == level]))
        for metric in METRIC_READ:
            test_rows.append({"family": "primary"} | test_categorical(df, col, name, levels, metric, args.n_perm))
    if "age" in df.columns:
        for metric in METRIC_READ:
            test_rows.append({"family": "primary"} | test_age(df, metric))
    has_cdrsb = "CDR_SB" in df.columns and "_g_dx" in df.columns
    if has_cdrsb:
        report_missing_by_site(df, "CDR_SB", "CDR-SB")
        for metric in METRIC_READ:
            test_rows.append(test_cdrsb_beyond_dx(df, metric, args.n_perm))
    for level in add_cog_amyloid_column(df):
        stat_rows.append({"variable": "Cognitive status × amyloid", "level": level, "column": "_g_cog_amy",
                          "tested": False} | level_stats(df[df["_g_cog_amy"] == level]))
    if not test_rows:
        print("[done] No subgroup variables available in this demographics table.")
        return

    stats_df = pd.DataFrame(stat_rows)
    # Primary and secondary are corrected separately: the CDR-SB question was added on top and
    # must not change the primary results.
    tests_df = add_fdr_columns(pd.DataFrame(test_rows), ["family"])

    out_dir.mkdir(parents=True, exist_ok=True)
    overall = level_stats(df)
    stats_df.to_csv(out_dir / "visual_read_subgroup_performance.csv", index=False)
    tests_df.to_csv(out_dir / "visual_read_subgroup_tests.csv", index=False)

    # Figures only for what differs, and only the measure that differed.
    significant = tests_df[tests_df["p_value_fdr_bh"] < ALPHA]
    metric_spec = {m[0]: m for m in SUBGROUP_METRICS}
    for name, rows in significant.groupby("variable", sort=False):
        metrics = list(rows["metric"])
        if name == "Age":
            make_age_figure(df, metrics, out_dir)
            continue
        if name == "CDR-SB (within diagnosis)":
            dx_levels = next(levels for c, _, levels in subgroups if c == "_g_dx")
            make_cdrsb_figure(df, metrics, dx_levels, args.min_level_n, out_dir)
            continue
        col = next(c for c, n, _ in subgroups if n == name)
        sub = stats_df[(stats_df["variable"] == name) & (stats_df["n"] >= args.min_level_n)]
        make_subgroup_performance_panel(
            sub, {m: overall[m] for m in metrics}, out_dir, metrics=[metric_spec[m] for m in metrics],
            filename=f"visual_read_subgroup_{col.removeprefix('_g_')}.png",
        )

    print(f"[done] {len(df)} scans; whole cohort sensitivity {overall['sensitivity']:.1%}, "
          f"specificity {overall['specificity']:.1%}")
    small = stats_df[(stats_df["n"] < args.min_level_n) & stats_df["tested"]]
    if len(small):
        print(f"[note] in the csv and tests but left off figures (n < {args.min_level_n}): "
              + ", ".join(f"{r.variable}={r.level} (n={r.n})" for r in small.itertuples()))
    untestable = tests_df[tests_df["test"].isna()]
    if len(untestable):
        print("[note] not testable: " + ", ".join(f"{r.variable} {r.metric}" for r in untestable.itertuples()))
    if significant.empty:
        print("[result] No subgroup difference survives FDR correction - no figures made.")
    else:
        print("[result] Survives FDR correction (figure made for each):")
        print(significant[["family", "variable", "metric", "test", "p_value", "p_value_fdr_bh"]].to_string(index=False))
    described = stats_df[~stats_df["tested"]]
    if len(described):
        print("[described, not tested] Cognitive status × amyloid:")
        print(described[["level", "n", "sensitivity_k", "sensitivity_n", "specificity_k", "specificity_n"]]
              .to_string(index=False))
    print(f"[done] saved to {out_dir}")


if __name__ == "__main__":
    main()
