#!/usr/bin/env python3
"""Does the regional-SUVR model work equally well in every subgroup? (Visual-read counterpart:
analyze_subgroup_performance_visual_read.py.)

What is compared. Not the raw error: the model under-predicts high SUVR (regression to the mean),
so any group with more high-tau scans - AD dementia, amyloid positive, APOE e4 carriers - has a
larger raw error simply because its scans sit higher on the SUVR scale. To ask whether the group
itself matters, each region's error is first adjusted for the true SUVR:

    signed error = predicted - true SUVR
    fit one straight line, signed error ~ true SUVR, over all scans in that region
    adjusted error = |signed error - line|   (how far off, beyond what the SUVR level explains)

The adjusted error is what is tested and plotted. Raw MAE per group is still in the csv, next to
the group's median true SUVR, so the case-mix effect stays visible.

Variables (fixed in advance, see add_subgroup_columns in plot_utils.py): diagnosis, sex, APOE e4,
amyloid status, site, and age as a continuous number.

Tests, per region:
  - 2 levels: Mann-Whitney test;  3+ levels: Kruskal-Wallis test
    (levels with fewer than --min_level_n scans are left out of the test and listed)
  - age: Spearman correlation between age and adjusted error
All p-values (every region x variable) are Benjamini-Hochberg corrected together, as one family
("primary").

Secondary question, corrected as its own family so it cannot dilute the primary one:
  - CDR-SB beyond diagnosis, per region: does the adjusted error rise with CDR-SB when people are
    compared only against others with the same diagnosis? (within-diagnosis permutation test, see
    within_strata_permutation_test). Scans without CDR-SB - a whole cohort, if it never collected
    it - drop out of this test only; the script prints which.

Outputs, next to the run's other figures:
    suvr_subgroup_performance.csv  - every level x region: n, median true SUVR, raw MAE,
                                     median adjusted error, mean signed residual (+ = over-predicted)
    suvr_subgroup_tests.csv        - every test: raw and FDR-corrected p
    suvr_subgroup_<variable>.png   - ONLY for variables with an FDR-significant difference, one
                                     panel per region where it was significant
    suvr_subgroup_cdrsb_<region>.png - ONLY if CDR-SB beyond diagnosis is significant in that region:
                                     adjusted error vs CDR-SB, one panel per diagnosis

A variable without a figure has not been shown to make no difference - only that none was detected.

Supersedes analyze_bias_significance.py for this question (that script tests raw error, so it
mostly rediscovers the SUVR-level effect, and splits age into bands); it is kept for now because
this script imports its FDR helper.

Usage:
    python results_plotting/analyze_subgroup_performance_suvr.py /path/to/run_dir --dataset Gothenburg
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analyze_bias_significance import add_fdr_columns
from list_discordant_cases import find_dataset_demo_csv
from plot_utils import (
    SUBGROUP_MIN_LEVEL_N,
    add_subgroup_columns,
    find_demo_csv,
    infer_targets,
    make_value_by_group_figure,
    make_value_vs_continuous_figure,
    order_targets,
    pretty_region_name,
    report_missing_by_site,
    resolve_path,
    resolve_predictions_csv,
    within_strata_permutation_test,
)

ALPHA = 0.05
ADJ_LABEL = "Prediction error, adjusted\nfor true SUVR (SUVR units)"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="Path to a results run directory trained on regional SUVR targets.")
    parser.add_argument("--dataset", default="Gothenburg", help="Dataset name used in the result filenames.")
    parser.add_argument("--demo_csv", default=None, help="Path to the demographics csv.")
    parser.add_argument("--female_code", type=float, default=2, help="Value of `gender` meaning female.")
    parser.add_argument("--min_level_n", type=int, default=SUBGROUP_MIN_LEVEL_N,
                        help="Levels with fewer scans than this are left out of the tests and figures "
                             "(still in the csv).")
    return parser.parse_args()


def load_scans(args, run_dir: Path):
    """One row per scan: <region>_y / <region>_pred for every region, plus demographics."""
    preds_path, out_dir = resolve_predictions_csv(run_dir, args.dataset)
    demo_path = resolve_path(args.demo_csv, lambda: find_dataset_demo_csv(args.dataset) or find_demo_csv())
    if demo_path is None:
        raise FileNotFoundError("Could not find a demographics csv - pass --demo_csv explicitly.")

    df_preds = pd.read_csv(preds_path)
    df_preds = df_preds.loc[:, ~df_preds.columns.astype(str).str.startswith("Unnamed")]
    targets = order_targets(infer_targets(df_preds))
    if not targets:
        raise ValueError(f"No <region>_y / <region>_pred columns in {preds_path} - is this an SUVR run?")
    keep = ["ID_ind"] + [c for t in targets for c in (f"{t}_y", f"{t}_pred")]
    # Demographics carry the true SUVR columns too (same region names); keep only the predictions' copy.
    df_demo = pd.read_csv(demo_path)
    df_demo = df_demo.drop(columns=[c for c in targets if c in df_demo.columns])
    df = pd.merge(df_preds[keep], df_demo, left_on="ID_ind", right_on="ID", how="left")
    unmatched = df["ID"].isna()
    if unmatched.any():
        print(f"[WARNING] {int(unmatched.sum())}/{len(df)} scans have no row in {demo_path.name} and are "
              "excluded - check --demo_csv matches this run.")
        df = df[~unmatched].reset_index(drop=True)
    print(f"[load] {len(df)} scans, regions {', '.join(targets)}, from {preds_path}")
    return df, targets, out_dir


def add_adjusted_error(df: pd.DataFrame, target: str) -> None:
    """Adds _raw_<t> (|pred - true|), _res_<t> (signed error minus its straight-line trend on true
    SUVR) and _adj_<t> (|_res_<t>|)."""
    y = pd.to_numeric(df[f"{target}_y"], errors="coerce")
    pred = pd.to_numeric(df[f"{target}_pred"], errors="coerce")
    err = pred - y
    ok = err.notna() & y.notna()
    slope, intercept = np.polyfit(y[ok], err[ok], 1)
    df[f"_raw_{target}"] = err.abs()
    df[f"_res_{target}"] = err - (intercept + slope * y)
    df[f"_adj_{target}"] = df[f"_res_{target}"].abs()


def level_stats(g: pd.DataFrame, target: str) -> dict:
    return {
        "n": int(g[f"_adj_{target}"].notna().sum()),
        "median_true_suvr": g[f"{target}_y"].median(),
        "raw_mae": g[f"_raw_{target}"].mean(),
        "median_adjusted_error": g[f"_adj_{target}"].median(),
        "mean_signed_residual": g[f"_res_{target}"].mean(),
    }


def test_categorical(df, col, name, levels, target, min_level_n) -> dict:
    values = {level: df.loc[df[col] == level, f"_adj_{target}"].dropna().to_numpy() for level in levels}
    used = {level: v for level, v in values.items() if len(v) >= min_level_n}
    out = {"region": target, "variable": name, "n_levels": len(used),
           "n_scans": int(sum(len(v) for v in used.values())), "test": None,
           "statistic": np.nan, "p_value": np.nan,
           "levels_left_out": ", ".join(f"{l} (n={len(v)})" for l, v in values.items() if l not in used)}
    if len(used) < 2:
        return out
    if len(used) == 2:
        out["test"] = "mann_whitney"
        out["statistic"], out["p_value"] = stats.mannwhitneyu(*used.values(), alternative="two-sided")
    else:
        out["test"] = "kruskal_wallis"
        out["statistic"], out["p_value"] = stats.kruskal(*used.values())
    return out


def test_age(df, target) -> dict:
    age = pd.to_numeric(df["age"], errors="coerce")
    ok = age.notna() & df[f"_adj_{target}"].notna()
    out = {"region": target, "variable": "Age", "n_levels": np.nan, "n_scans": int(ok.sum()),
           "test": None, "statistic": np.nan, "p_value": np.nan}
    if ok.sum() >= 3:
        out["test"] = "spearman"
        out["statistic"], out["p_value"] = stats.spearmanr(age[ok], df.loc[ok, f"_adj_{target}"])
    return out


def test_cdrsb_beyond_dx(df, target) -> dict:
    """Does adjusted error rise with CDR-SB, among people with the same diagnosis?"""
    sub = df.assign(_cdrsb=pd.to_numeric(df["CDR_SB"], errors="coerce"))
    sub = sub.dropna(subset=["_cdrsb", "_g_dx", f"_adj_{target}"])
    out = {"family": "secondary", "region": target, "variable": "CDR-SB (within diagnosis)",
           "n_levels": np.nan, "n_scans": len(sub), "test": None, "statistic": np.nan, "p_value": np.nan}
    if len(sub) >= 3 and sub["_cdrsb"].nunique() > 1:
        out["test"] = "within_diagnosis_permutation"
        out["statistic"], out["p_value"] = within_strata_permutation_test(
            sub[f"_adj_{target}"].to_numpy(), sub["_cdrsb"].to_numpy(), sub["_g_dx"])
    return out


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    df, targets, out_dir = load_scans(args, run_dir)
    for target in targets:
        add_adjusted_error(df, target)

    subgroups = add_subgroup_columns(df, female_code=args.female_code)
    stat_rows, test_rows = [], []
    for target in targets:
        for col, name, levels in subgroups:
            for level in levels:
                stat_rows.append({"region": target, "variable": name, "level": level, "column": col}
                                 | level_stats(df[df[col] == level], target))
            test_rows.append({"family": "primary"} | test_categorical(df, col, name, levels, target, args.min_level_n))
        if "age" in df.columns:
            test_rows.append({"family": "primary"} | test_age(df, target))
        if "CDR_SB" in df.columns and "_g_dx" in df.columns:
            test_rows.append(test_cdrsb_beyond_dx(df, target))
    if not test_rows:
        print("[done] No subgroup variables available in this demographics table.")
        return
    if "CDR_SB" in df.columns:
        report_missing_by_site(df, "CDR_SB", "CDR-SB")

    stats_df = pd.DataFrame(stat_rows)
    # Primary and secondary are corrected separately: the CDR-SB question was added on top and
    # must not change the primary results.
    tests_df = add_fdr_columns(pd.DataFrame(test_rows), ["family"])

    out_dir.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(out_dir / "suvr_subgroup_performance.csv", index=False)
    tests_df.to_csv(out_dir / "suvr_subgroup_tests.csv", index=False)

    # Figures only for what differs: one per variable, one panel per region where it differed.
    significant = tests_df[tests_df["p_value_fdr_bh"] < ALPHA]
    for name, rows in significant.groupby("variable", sort=False):
        regions = [t for t in targets if t in set(rows["region"])]
        if name == "CDR-SB (within diagnosis)":
            dx_levels = next(levels for c, _, levels in subgroups if c == "_g_dx")
            cdrsb = pd.to_numeric(df["CDR_SB"], errors="coerce")
            for t in regions:
                panels = [(level, cdrsb[df["_g_dx"] == level], df.loc[df["_g_dx"] == level, f"_adj_{t}"])
                          for level in dx_levels
                          if ((df["_g_dx"] == level) & cdrsb.notna()).sum() >= args.min_level_n]
                make_value_vs_continuous_figure(
                    panels, "CDR-SB", ADJ_LABEL, f"{pretty_region_name(t)} SUVR prediction error by CDR-SB, per diagnosis",
                    out_dir / f"suvr_subgroup_cdrsb_{t}.png")
            continue
        if name == "Age":
            make_value_vs_continuous_figure(
                [(pretty_region_name(t), df["age"], df[f"_adj_{t}"]) for t in regions],
                "Age (years)", ADJ_LABEL, "SUVR prediction error by age",
                out_dir / "suvr_subgroup_age.png")
            continue
        col, _, levels = next(s for s in subgroups if s[1] == name)
        panels = []
        for t in regions:
            groups = {level: df.loc[df[col] == level, f"_adj_{t}"].dropna().to_numpy() for level in levels}
            panels.append((pretty_region_name(t),
                           {level: v for level, v in groups.items() if len(v) >= args.min_level_n}))
        make_value_by_group_figure(panels, ADJ_LABEL, f"SUVR prediction error by {name.lower()}",
                                   out_dir / f"suvr_subgroup_{col.removeprefix('_g_')}.png")

    left_out = tests_df[tests_df.get("levels_left_out", pd.Series(dtype=str)).fillna("") != ""]
    if len(left_out):
        notes = sorted(set(f"{r.variable}: {r.levels_left_out}" for r in left_out.itertuples()))
        print(f"[note] left out of tests and figures (n < {args.min_level_n}): " + "; ".join(notes))
    if significant.empty:
        print("[result] No subgroup difference survives FDR correction - no figures made.")
    else:
        print("[result] Survives FDR correction (figure made per variable):")
        print(significant[["family", "region", "variable", "test", "p_value", "p_value_fdr_bh"]].to_string(index=False))
    print(f"[done] saved to {out_dir}")


if __name__ == "__main__":
    main()
