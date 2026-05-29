#!/usr/bin/env python3

"""Bias analysis for regression results.

This script merges the saved prediction table with the repository demographics,
computes absolute error per target, and tests whether error differs by age,
site, dx, gender, or apoe.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import plot_metatemporal_results as mod


GROUP_COLUMNS = ["age", "site", "dx_grouped", "gender", "apoe"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze absolute error bias by demographic group.")
    parser.add_argument("--run_dir", required=True, help="Path to the result directory to analyze.")
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Optional output directory. Defaults to <run_dir>/bias_analysis.",
    )
    parser.add_argument(
        "--min_group_n",
        type=int,
        default=5,
        help="Minimum samples per group for significance testing.",
    )
    return parser.parse_args()


def find_results_csv(run_dir: Path) -> Path:
    preferred = [
        run_dir / "evaluation" / "Gothenburg" / "Eval_Gothenburg_results.csv",
        run_dir / "evaluation" / "Gothenburg" / "Test_Gothenburg_results.csv",
    ]
    for path in preferred:
        if path.exists():
            return path

    candidates = sorted(run_dir.rglob("*_results.csv"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Could not find a prediction CSV under {run_dir}")


def find_demo_csv() -> Path:
    script_path = Path(__file__).resolve()
    candidates = [
        script_path.parents[2] / "data" / "demo.csv",
        script_path.parents[1] / "data" / "demo.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find demo.csv. Checked:\n{searched}")


def infer_targets(df: pd.DataFrame) -> list[str]:
    targets = []
    for col in df.columns:
        if col.endswith("_y") and f"{col[:-2]}_pred" in df.columns:
            targets.append(col[:-2])
    return targets


def load_prediction_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed")].copy()
    if "ID_ind" in df.columns:
        df["ID_ind"] = pd.to_numeric(df["ID_ind"], errors="coerce")
    return df


def load_demo_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed")].copy()
    if "ID" in df.columns:
        df["ID"] = pd.to_numeric(df["ID"], errors="coerce")
    return df


def normalize_demo_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "dx" in out.columns and "dx_grouped" not in out.columns:
        out["dx_grouped"] = mod.combine_dx_groups(out["dx"])
    if "sex" in out.columns and "gender" not in out.columns:
        out["gender"] = out["sex"]
    if "gender" in out.columns:
        out["gender"] = out["gender"].astype("string").fillna("NA")
    if "site" in out.columns:
        out["site"] = out["site"].astype("string").fillna("NA")
    if "apoe" in out.columns:
        out["apoe"] = out["apoe"].astype("string").fillna("NA")
    if "age" in out.columns:
        out["age"] = pd.to_numeric(out["age"], errors="coerce")
    return out


def get_target_frame(df_preds: pd.DataFrame, df_demo: pd.DataFrame, target: str) -> pd.DataFrame:
    target_df = df_preds[["ID_ind", f"{target}_y", f"{target}_pred"]].copy()
    target_df = target_df.rename(columns={f"{target}_y": "y", f"{target}_pred": "pred"})
    target_df["y"] = pd.to_numeric(target_df["y"], errors="coerce")
    target_df["pred"] = pd.to_numeric(target_df["pred"], errors="coerce")
    target_df["abs_error"] = (target_df["pred"] - target_df["y"]).abs()
    target_df["target"] = target

    if "ID_ind" in target_df.columns and "ID" in df_demo.columns:
        merged = pd.merge(target_df, df_demo, left_on="ID_ind", right_on="ID", how="left")
    else:
        merged = target_df.copy()

    merged = normalize_demo_columns(merged)
    if "dx" in merged.columns and "dx_grouped" not in merged.columns:
        merged["dx_grouped"] = mod.combine_dx_groups(merged["dx"])
    return merged


def make_age_bins(series: pd.Series) -> pd.Series:
    age = pd.to_numeric(series, errors="coerce")
    return pd.cut(
        age,
        bins=[50, 60, 70, 80, 90, 110],
        labels=["50-60", "60-70", "70-80", "80-90", "90+"],
        include_lowest=True,
        right=False,
    )


def prepare_group_series(df: pd.DataFrame, group_col: str) -> pd.Series:
    if group_col == "age":
        return make_age_bins(df["age"])
    if group_col == "dx_grouped" and "dx_grouped" not in df.columns and "dx" in df.columns:
        return mod.combine_dx_groups(df["dx"])
    if group_col not in df.columns:
        return pd.Series(dtype="string")
    return df[group_col].astype("string").fillna("NA")


def summarize_groups(df: pd.DataFrame, group_col: str, min_group_n: int) -> pd.DataFrame:
    groups = prepare_group_series(df, group_col)
    valid = df.loc[df["abs_error"].notna()].copy()
    valid[group_col] = groups.loc[valid.index]
    valid[group_col] = valid[group_col].astype("string").fillna("NA")

    rows = []
    for group_name, group_df in valid.groupby(group_col, dropna=False):
        values = pd.to_numeric(group_df["abs_error"], errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        rows.append(
            {
                "target": group_df["target"].iloc[0],
                "group_column": group_col,
                "group_name": str(group_name),
                "n": int(len(values)),
                "mean_abs_error": float(np.mean(values)),
                "median_abs_error": float(np.median(values)),
                "q1_abs_error": float(np.nanpercentile(values, 25)),
                "q3_abs_error": float(np.nanpercentile(values, 75)),
                "iqr_abs_error": float(np.nanpercentile(values, 75) - np.nanpercentile(values, 25)),
                "std_abs_error": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                "min_group_n": min_group_n,
            }
        )

    return pd.DataFrame(rows)


def pairwise_mann_whitney(values_by_group: dict[str, np.ndarray]) -> list[dict[str, object]]:
    rows = []
    group_names = sorted(values_by_group)
    for group_a, group_b in combinations(group_names, 2):
        values_a = values_by_group[group_a]
        values_b = values_by_group[group_b]
        stat, p_value = stats.mannwhitneyu(values_a, values_b, alternative="two-sided")
        rows.append(
            {
                "group_a": group_a,
                "group_b": group_b,
                "n_a": int(len(values_a)),
                "n_b": int(len(values_b)),
                "mean_a": float(np.mean(values_a)),
                "mean_b": float(np.mean(values_b)),
                "delta_mean": float(np.mean(values_a) - np.mean(values_b)),
                "statistic": float(stat),
                "p_value": float(p_value),
            }
        )
    return rows


def analyze_one_target(df: pd.DataFrame, target: str, min_group_n: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    omnibus_rows = []
    pairwise_rows = []

    for group_col in GROUP_COLUMNS:
        if group_col not in df.columns and group_col != "age":
            continue

        if group_col == "age" and "age" not in df.columns:
            continue

        age_continuous = None
        if group_col == "age" and "age" in df.columns:
            age_continuous = pd.to_numeric(df["age"], errors="coerce")
        group_series = prepare_group_series(df, group_col)
        valid = df.loc[df["abs_error"].notna()].copy()
        valid[group_col] = group_series.loc[valid.index]
        valid = valid.loc[valid[group_col].notna()].copy()
        valid[group_col] = valid[group_col].astype("string").fillna("NA")

        summary = summarize_groups(valid, group_col, min_group_n)
        if not summary.empty:
            summary_rows.append(summary)

        groups = {
            group_name: pd.to_numeric(group_df["abs_error"], errors="coerce").dropna().to_numpy(dtype=float)
            for group_name, group_df in valid.groupby(group_col, dropna=False)
        }
        groups = {name: values for name, values in groups.items() if len(values) >= min_group_n}
        if len(groups) < 2:
            continue

        arrays = list(groups.values())
        stat, p_value = stats.kruskal(*arrays)
        omnibus_rows.append(
            {
                "target": target,
                "group_column": group_col,
                "test": "kruskal_wallis",
                "n_groups": int(len(groups)),
                "min_group_n": int(min(len(values) for values in arrays)),
                "statistic": float(stat),
                "p_value": float(p_value),
            }
        )

        pairwise_rows.extend(
            {
                "target": target,
                "group_column": group_col,
                **row,
            }
            for row in pairwise_mann_whitney(groups)
        )

        if group_col == "age" and age_continuous is not None:
            valid_age = age_continuous.loc[valid.index]
            corr_mask = valid_age.notna() & valid["abs_error"].notna()
            if corr_mask.sum() >= 2:
                rho, p_age = stats.spearmanr(valid_age.loc[corr_mask], valid.loc[corr_mask, "abs_error"])
                omnibus_rows.append(
                    {
                        "target": target,
                        "group_column": "age_continuous",
                        "test": "spearmanr",
                        "n_groups": 1,
                        "min_group_n": int(corr_mask.sum()),
                        "statistic": float(rho),
                        "p_value": float(p_age),
                    }
                )

    summary_df = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()
    omnibus_df = pd.DataFrame(omnibus_rows)
    pairwise_df = pd.DataFrame(pairwise_rows)
    return summary_df, omnibus_df, pairwise_df


def compute_subject_mae(df_preds: pd.DataFrame) -> pd.DataFrame:
    """Compute per-subject mean absolute error across available targets."""
    targets = infer_targets(df_preds)
    if not targets:
        return pd.DataFrame()

    rows = []
    id_col = 'ID_ind' if 'ID_ind' in df_preds.columns else 'ID'
    for sid, group in df_preds.groupby(id_col):
        errors = []
        for t in targets:
            y_col = f"{t}_y"
            p_col = f"{t}_pred"
            if y_col in group.columns and p_col in group.columns:
                y = pd.to_numeric(group[y_col].iloc[0], errors='coerce')
                p = pd.to_numeric(group[p_col].iloc[0], errors='coerce')
                if np.isfinite(y) and np.isfinite(p):
                    errors.append(abs(p - y))
        if errors:
            rows.append({id_col: sid, 'subject_mae': float(np.mean(errors)), 'n_regions': len(errors)})

    return pd.DataFrame(rows)


def analyze_subject_level(df_preds: pd.DataFrame, df_demo: pd.DataFrame, out_dir: Path, min_group_n: int):
    subj = compute_subject_mae(df_preds)
    if subj.empty:
        return
    id_col = 'ID_ind' if 'ID_ind' in df_preds.columns else 'ID'
    if 'ID' in df_demo.columns:
        merged = pd.merge(subj, df_demo, left_on=id_col, right_on='ID', how='left')
    else:
        merged = subj.copy()
    merged = normalize_demo_columns(merged)

    subject_out = out_dir / 'subject_level'
    subject_out.mkdir(parents=True, exist_ok=True)

    # debug summary to explain empty outputs
    debug_lines = []
    debug_lines.append(f"n_subjects={len(merged)}")
    debug_lines.append(f"columns={','.join(merged.columns.astype(str))}")
    if 'ID' in merged.columns:
        debug_lines.append(f"n_subjects_with_demo_id={merged['ID'].notna().sum()}")
    for group_col in GROUP_COLUMNS:
        if group_col == 'age' and 'age' in merged.columns:
            group_series = prepare_group_series(merged, 'age')
        elif group_col in merged.columns:
            group_series = prepare_group_series(merged, group_col)
        else:
            debug_lines.append(f"{group_col}_missing=True")
            continue
        counts = group_series.astype('string').fillna('NA').value_counts()
        passing = counts[counts >= min_group_n]
        debug_lines.append(f"{group_col}_groups_total={len(counts)}")
        debug_lines.append(f"{group_col}_groups_ge_{min_group_n}={len(passing)}")
    with open(subject_out / 'subject_level_debug.txt', 'w', encoding='utf-8') as handle:
        handle.write("\n".join(debug_lines) + "\n")

    # prepare dataframe compatible with analyze_one_target: use 'abs_error' and 'target'
    df_for_test = merged.rename(columns={'subject_mae': 'abs_error'})
    df_for_test['target'] = 'subject_mean'

    summary_df, omnibus_df, pairwise_df = analyze_one_target(df_for_test, 'subject_mean', min_group_n)

    # apply FDR
    omnibus_df = add_fdr_columns(omnibus_df, ['target'])
    pairwise_df = add_fdr_columns(pairwise_df, ['target', 'group_column'])

    if not summary_df.empty:
        summary_df.to_csv(subject_out / 'subject_group_summary.csv', index=False)
    if not omnibus_df.empty:
        omnibus_df.to_csv(subject_out / 'subject_omnibus_tests.csv', index=False)
    if not pairwise_df.empty:
        pairwise_df.to_csv(subject_out / 'subject_pairwise_tests.csv', index=False)

    return merged, summary_df, omnibus_df, pairwise_df


def add_fdr_columns(df: pd.DataFrame, group_keys: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["p_value_fdr_bh"] = np.nan
    out["significant_fdr_0_05"] = False

    for _, idx in out.groupby(group_keys, dropna=False).groups.items():
        pvals = pd.to_numeric(out.loc[idx, "p_value"], errors="coerce")
        if pvals.notna().sum() == 0:
            continue
        adjusted = mod.fdr_bh(pvals)
        out.loc[idx, "p_value_fdr_bh"] = adjusted
        out.loc[idx, "significant_fdr_0_05"] = adjusted < 0.05

    return out


def write_outputs(out_dir: Path, summary: pd.DataFrame, omnibus: pd.DataFrame, pairwise: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not summary.empty:
        summary.to_csv(out_dir / "group_summary.csv", index=False)
    if not omnibus.empty:
        omnibus.to_csv(out_dir / "omnibus_tests.csv", index=False)
    if not pairwise.empty:
        pairwise.to_csv(out_dir / "pairwise_tests.csv", index=False)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir / "bias_analysis"
    results_csv = find_results_csv(run_dir)
    demo_csv = find_demo_csv()

    df_preds = load_prediction_table(results_csv)
    df_demo = normalize_demo_columns(load_demo_table(demo_csv))
    targets = infer_targets(df_preds)
    if not targets:
        raise ValueError(f"No target columns were found in {results_csv}")

    all_summary = []
    all_omnibus = []
    all_pairwise = []

    for target in targets:
        target_df = get_target_frame(df_preds, df_demo, target)
        summary_df, omnibus_df, pairwise_df = analyze_one_target(target_df, target, args.min_group_n)

        if not summary_df.empty:
            all_summary.append(summary_df)
        if not omnibus_df.empty:
            all_omnibus.append(omnibus_df)
        if not pairwise_df.empty:
            all_pairwise.append(pairwise_df)

    summary = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    omnibus = pd.concat(all_omnibus, ignore_index=True) if all_omnibus else pd.DataFrame()
    pairwise = pd.concat(all_pairwise, ignore_index=True) if all_pairwise else pd.DataFrame()

    omnibus = add_fdr_columns(omnibus, ["target"])
    pairwise = add_fdr_columns(pairwise, ["target", "group_column"])

    write_outputs(out_dir, summary, omnibus, pairwise)

    if not summary.empty:
        summary.to_csv(out_dir / "group_summary.csv", index=False)
    if not omnibus.empty:
        omnibus.to_csv(out_dir / "omnibus_tests.csv", index=False)
    if not pairwise.empty:
        pairwise.to_csv(out_dir / "pairwise_tests.csv", index=False)

    print(f"Analyzed {len(targets)} targets from {results_csv.name}")
    print(f"Wrote outputs to {out_dir}")

    if not omnibus.empty:
        sig_omnibus = omnibus.loc[omnibus["p_value_fdr_bh"] < 0.05].copy()
        if not sig_omnibus.empty:
            print("\nSignificant omnibus tests after FDR:")
            print(sig_omnibus[["target", "group_column", "test", "statistic", "p_value", "p_value_fdr_bh"]].to_string(index=False))
        else:
            print("\nNo omnibus tests remained significant after FDR.")

    if not pairwise.empty:
        sig_pairwise = pairwise.loc[pairwise["p_value_fdr_bh"] < 0.05].copy()
        if not sig_pairwise.empty:
            print("\nSignificant pairwise tests after FDR:")
            cols = ["target", "group_column", "group_a", "group_b", "delta_mean", "statistic", "p_value", "p_value_fdr_bh"]
            print(sig_pairwise[cols].sort_values(["target", "group_column", "p_value_fdr_bh"]).to_string(index=False))
        else:
            print("\nNo pairwise comparisons remained significant after FDR.")

    # Subject-level (per-subject mean absolute error across regions)
    print("\nRunning subject-level MAE analysis...")
    subj_res = analyze_subject_level(df_preds, df_demo, out_dir, args.min_group_n)
    if subj_res is not None:
        merged_subj, subj_summary, subj_omnibus, subj_pairwise = subj_res
        print(f"Wrote subject-level outputs to: {out_dir / 'subject_level'}")


if __name__ == "__main__":
    main()