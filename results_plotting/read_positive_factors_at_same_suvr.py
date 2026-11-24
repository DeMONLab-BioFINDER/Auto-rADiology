#!/usr/bin/env python3
"""Which patient characteristics go with Model − among Read + scans, at the same meta-temporal SUVR?

One question, one figure: for each factor, a logistic regression within Read + scans,
    Model − ~ factor + meta-temporal SUVR
likelihood-ratio test for the factor, FDR (Benjamini-Hochberg) across the factors of one data set.
Factors (fixed in advance): amyloid status, diagnosis (MCI / AD dementia vs CU), APOE ε4, age (per 10 years),
sex. Each factor separately (no model with all of them: 29 / 10 disagreeing scans are too few).
Same coding and fitting as analyze_errors_at_same_suvr.py, which keeps its own (amyloid, diagnosis) table.

Usage:
    python results_plotting/read_positive_factors_at_same_suvr.py <run_dir> --demo_csv <demo.csv> [--dataset AVID_unseen]
Outputs (no IDs; next to the run's other figures, validation/figures for an external set):
    read_positive_factors_at_same_suvr.csv   factor, n, n Model −, OR, 95% CI, p, FDR p
    read_positive_factors_at_same_suvr.png   forest plot, one row per factor
"""
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from scipy import stats

from analyze_errors_at_same_suvr import fit, stratum
from plot_utils import (CUSTOM_PALETTE, add_subgroup_columns, fdr_bh, find_demo_csv, resolve_path,
                        resolve_predictions_csv, set_figure_set, with_set)

# (column made by stratum() or below, row label: the group the odds ratio is for)
FACTORS = [("amy", "Amyloid positive (vs negative)"),
           ("impaired", "MCI or AD dementia (vs CU)"),
           ("apoe", "APOE ε4 carrier"),
           ("age10", "Age (per 10 years)"),
           ("male", "Male (vs female)")]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--dataset", default="Gothenburg")
    p.add_argument("--demo_csv", default=None)
    p.add_argument("--female_code", type=float, default=2)
    return p.parse_args()


def load(args):
    run_dir = Path(args.run_dir).resolve()
    preds_path, out_dir = resolve_predictions_csv(run_dir, args.dataset)
    preds = pd.read_csv(preds_path)[["ID_ind", "y", "pred"]].apply(pd.to_numeric, errors="coerce").dropna()
    demo_path = resolve_path(args.demo_csv, find_demo_csv)
    if demo_path is None:
        raise FileNotFoundError("No demographics csv found - pass --demo_csv.")
    demo = pd.read_csv(demo_path)
    demo = demo.loc[:, ~demo.columns.astype(str).str.startswith("Unnamed")]
    df = preds.merge(demo, left_on="ID_ind", right_on="ID", how="inner")
    add_subgroup_columns(df, female_code=args.female_code)
    sub = stratum(df, "sensitivity")  # Read + scans; "wrong" = Model −
    sub["male"] = pd.to_numeric(sub.get("gender"), errors="coerce").map(
        lambda v: np.nan if pd.isna(v) else float(v != args.female_code))
    return sub, out_dir


def test_factors(sub: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label in FACTORS:
        d = sub.dropna(subset=[col, "wrong", "meta"])
        row = {"factor": label, "n": len(d), "n_model_neg": int(d["wrong"].sum())}
        if d["wrong"].sum() < 2 or (len(d) - d["wrong"].sum()) < 2 or d[col].nunique() < 2:
            rows.append({**row, "note": "too few Model − scans / one group only"}); continue
        try:
            full, reduced = fit(d, [col, "meta"]), fit(d, ["meta"])
        except Exception as err:
            rows.append({**row, "note": f"could not fit ({type(err).__name__})"}); continue
        lo, hi = full.conf_int().loc[col]
        if not np.isfinite(np.exp(min(hi, 700))) or hi > np.log(1e4) or abs(full.params[col]) > np.log(1000):
            rows.append({**row, "note": "could not fit (one group (almost) never Model −)"}); continue
        rows.append({**row, "or": np.exp(full.params[col]), "or_lo": np.exp(lo), "or_hi": np.exp(hi),
                     "p": stats.chi2.sf(2 * (full.llf - reduced.llf), 1)})
    t = pd.DataFrame(rows)
    ok = t["p"].notna() if "p" in t else pd.Series(False, index=t.index)
    t.loc[ok, "p_fdr"] = fdr_bh(t.loc[ok, "p"])
    return t


def forest(t: pd.DataFrame, n_read_pos: int, out: Path):
    fig, ax = plt.subplots(figsize=(10, 0.75 * len(t) + 1.8))
    y = np.arange(len(t))[::-1]
    ax.axvline(1, color="0.55", ls="--", lw=1.2)
    for yi, (_, r) in zip(y, t.iterrows()):
        if not np.isfinite(r.get("or", np.nan)):
            ax.text(0.5, yi, "not estimable: " + r.get("note", ""), transform=ax.get_yaxis_transform(), ha="center",
                    va="center", fontsize=11, color="0.5"); continue
        sig = r["p_fdr"] < 0.05
        c = CUSTOM_PALETTE[1] if sig else "0.45"
        ax.hlines(yi, r["or_lo"], r["or_hi"], color=c, lw=3)
        ax.scatter([r["or"]], [yi], s=90, color=c, zorder=3, edgecolor="white", linewidth=1)
        p_txt = "< .001" if r["p_fdr"] < 0.001 else f"= {r['p_fdr']:.2f}".replace("0.", ".")
        ax.annotate(f"OR {r['or']:.2f}, FDR p {p_txt}", (r["or"], yi), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=12, color=c)
    ax.set_xscale("log")
    lo_x, hi_x = ax.get_xlim()
    ax.xaxis.set_major_locator(FixedLocator([v for v in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100)
                                             if lo_x <= v <= hi_x]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks(y, [f"{f} (n = {n})" for f, n in zip(t["factor"], t["n"])], fontsize=13)
    ax.tick_params(axis="x", labelsize=12)
    ax.set_ylim(-0.7, len(t) - 0.2)
    ax.set_xlabel("Odds of Model − at the same meta-temporal SUVR\n< 1 called negative less often · > 1 more often",
                  fontsize=13)
    ax.set_title(with_set(f"Patient characteristics, Read + scans (n = {n_read_pos})"), fontsize=15, pad=12, loc="left",
                 x=-0.45)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="0.92")
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.1); plt.close(fig)


def main():
    args = parse_args()
    sub, out_dir = load(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_figure_set(out_dir, args.dataset)
    t = test_factors(sub)
    t.round(4).to_csv(out_dir / "read_positive_factors_at_same_suvr.csv", index=False)
    forest(t, len(sub), out_dir / "read_positive_factors_at_same_suvr.png")
    with pd.option_context("display.width", 200):
        print(t.round(3).to_string(index=False))
    print(f"\nSaved read_positive_factors_at_same_suvr.csv / .png to {out_dir}")


if __name__ == "__main__":
    main()
