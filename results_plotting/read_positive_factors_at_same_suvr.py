#!/usr/bin/env python3
"""Which patient characteristics go with Model − among Read + scans, at the same meta-temporal SUVR?

One question, one figure: for each factor, the same logistic regression within Read + scans, on the same scans,
once without tau and once at the same meta-temporal SUVR:
    Model − ~ factor                    (who are the disagreements?)
    Model − ~ factor + meta-temporal SUVR   (anything beyond the tau level?)
likelihood-ratio test for the factor, FDR (Benjamini-Hochberg) across the factors of one data set, per column.
Factors (fixed in advance): amyloid status, diagnosis (MCI / AD dementia vs CU), APOE ε4, age (per 10 years),
sex. Each factor separately (no model with all of them: 29 / 10 disagreeing scans are too few).
Same coding and fitting as analyze_errors_at_same_suvr.py, which keeps its own (amyloid, diagnosis) table.

Usage:
    python results_plotting/read_positive_factors_at_same_suvr.py <run_dir> --demo_csv <demo.csv> [--dataset AVID_unseen]
Outputs (no IDs; next to the run's other figures, validation/figures for an external set):
    read_positive_factors_at_same_suvr.csv   factor, n, n Model −; plain_or / plain_p / plain_p_fdr (without tau);
                                             or / or_lo / or_hi / p / p_fdr (same meta-temporal SUVR)
    read_positive_factors_at_same_suvr.png   forest plot, two dots per factor (open = without tau, filled = same tau)
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


def _fit_one(d, col, adjust):
    """OR of Model − for `col` (with `adjust` terms), LR test against the model without `col`."""
    full, reduced = fit(d, [col, *adjust]), fit(d, adjust)
    lo, hi = full.conf_int().loc[col]
    if not np.isfinite(np.exp(min(hi, 700))) or hi > np.log(1e4) or abs(full.params[col]) > np.log(1000):
        raise ValueError("one group (almost) never Model −")
    return np.exp(full.params[col]), np.exp(lo), np.exp(hi), stats.chi2.sf(2 * (full.llf - reduced.llf), 1)


def test_factors(sub: pd.DataFrame) -> pd.DataFrame:
    """Per characteristic, the same logistic regression twice on the same scans: without tau (plain_*) and at the
    same meta-temporal SUVR (or, or_lo, or_hi, p). FDR across the five characteristics, separately per column."""
    rows = []
    for col, label in FACTORS:
        d = sub.dropna(subset=[col, "wrong", "meta"])
        row = {"factor": label, "n": len(d), "n_model_neg": int(d["wrong"].sum())}
        if d["wrong"].sum() < 2 or (len(d) - d["wrong"].sum()) < 2 or d[col].nunique() < 2:
            rows.append({**row, "note": "too few Model − scans / one group only"}); continue
        for prefix, adjust in (("plain_", []), ("", ["meta"])):
            try:
                o, lo, hi, pv = _fit_one(d, col, adjust)
                row.update({f"{prefix}or": o, f"{prefix}or_lo": lo, f"{prefix}or_hi": hi, f"{prefix}p": pv})
            except Exception as err:
                row[f"{prefix}note"] = f"could not fit ({err})"
        rows.append(row)
    t = pd.DataFrame(rows)
    for prefix in ("plain_", ""):
        if f"{prefix}p" in t:
            ok = t[f"{prefix}p"].notna()
            t.loc[ok, f"{prefix}p_fdr"] = fdr_bh(t.loc[ok, f"{prefix}p"])
    return t


def _p(v):
    if not np.isfinite(v):
        return "–"
    return "< .001" if v < 0.001 else (f"{v:.3f}" if v < 0.01 else f"{v:.2f}").replace("0.", ".", 1)


def forest(t: pd.DataFrame, n_read_pos: int, out: Path):
    """Two dots per characteristic: open = without tau, filled = at the same meta-temporal SUVR (same scans, same
    test). Rose = FDR p < .05, grey = not. FDR p of both on the right."""
    fig, ax = plt.subplots(figsize=(11, 0.85 * len(t) + 2.0))
    y = np.arange(len(t))[::-1]
    ax.axvline(1, color="0.55", ls="--", lw=1.2)
    for yi, (_, r) in zip(y, t.iterrows()):
        for prefix, dy, filled in (("plain_", 0.17, False), ("", -0.17, True)):
            o = r.get(f"{prefix}or", np.nan)
            if not np.isfinite(o):
                continue
            c = CUSTOM_PALETTE[1] if r.get(f"{prefix}p_fdr", 1) < 0.05 else "0.45"
            ax.hlines(yi + dy, r[f"{prefix}or_lo"], r[f"{prefix}or_hi"], color=c, lw=2.5, alpha=1 if filled else 0.7)
            ax.scatter([o], [yi + dy], s=95, zorder=3, linewidth=2, edgecolor=c, facecolor=c if filled else "white")
        ax.text(1.02, yi, f"{_p(r.get('plain_p_fdr', np.nan))}  →  {_p(r.get('p_fdr', np.nan))}",
                transform=ax.get_yaxis_transform(), va="center", fontsize=12.5, color="0.25")
    ax.text(1.02, len(t) - 0.35, "FDR p\nwithout → same tau", transform=ax.get_yaxis_transform(), va="bottom",
            fontsize=11.5, color="0.35")
    ax.set_xscale("log")
    lo_x, hi_x = ax.get_xlim()
    ax.xaxis.set_major_locator(FixedLocator([v for v in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100)
                                             if lo_x <= v <= hi_x]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks(y, [f"{f} (n = {n})" for f, n in zip(t["factor"], t["n"])], fontsize=13)
    ax.tick_params(axis="x", labelsize=12)
    ax.set_ylim(-0.7, len(t) - 0.2)
    ax.set_xlabel("Odds of Model − among Read + scans\n< 1 called negative less often, > 1 more often", fontsize=13)
    ax.set_title(with_set(f"Patient characteristics, Read + scans (n = {n_read_pos})"), fontsize=15, pad=30, loc="left",
                 x=-0.45)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=9, mfc="white", mec="0.35", mew=2, label="Without tau"),
                       Line2D([], [], marker="o", ls="", ms=9, mfc="0.35", mec="0.35", label="Same meta-temporal SUVR")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=12)
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
