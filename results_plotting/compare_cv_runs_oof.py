#!/usr/bin/env python3
"""Paired comparison of two visual_read 5-fold CV runs on their out-of-fold predictions.

Is model B (e.g. attention pooling) really better than model A (average pooling)? Every scan in the
CV pool is predicted once by a fold model that did not train on it, so the two runs can be compared
scan by scan on ~4x more scans than the test set. Fair only when both runs used the same folds (same
seed, split fractions and stratification): checked here, the script stops otherwise.

Fixed in advance (2026-11-04):
    Main      - scans where the call at 0.5 disagrees with the read, A vs B, exact McNemar test
                (only the scans where exactly one of the two models agrees with the read count).
    Secondary - the same in the borderline band (--col, default MetaTemporal SUVR 1.25-1.45), and
                the paired AUC difference with a bootstrap 95% CI.
    Spread    - per fold, read + scans B calls + and A calls - (and the reverse): a gain that shows up
                in every fold does not depend on one set of training data.
Out-of-fold predictions come from the fold also used for early stopping, so both runs are slightly
optimistic in the same way; the paired comparison stays fair.

Usage:
    python results_plotting/compare_cv_runs_oof.py <cv_run_A> <cv_run_B> --labels avg attention
Output: <proj_path>/results/cv_oof_comparison/<A>_vs_<B>.csv (outside the repo; no IDs).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from compare_model_vs_suvr_threshold import mcnemar_p, paired_auc
from plot_utils import find_demo_csv, resolve_path


def parse_args():
    repo = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.add_argument("--labels", nargs=2, default=["A", "B"])
    p.add_argument("--demo_csv", default=None, help="Demographics csv with the SUVR column (default data/demo.csv).")
    p.add_argument("--col", default="MetaTemporal")
    p.add_argument("--band", default="1.25,1.45")
    p.add_argument("--seed", type=int, default=0, help="Bootstrap seed.")
    p.add_argument("--out_dir", default=str(repo.parent / "results" / "cv_oof_comparison"))
    return p.parse_args()


def load_oof(run: str) -> pd.DataFrame:
    f = Path(run) / "oof_predictions.csv"
    if not f.exists():
        raise SystemExit(f"{f} not found (not a finished --run_kfold_cv run?)")
    d = pd.read_csv(f)[["ID_ind", "fold", "y", "pred", "prob"]]
    if d["ID_ind"].duplicated().any():
        raise SystemExit(f"{f} has a scan predicted more than once")
    return d


def compare(d: pd.DataFrame, la: str, lb: str, rng) -> dict:
    y = d["y"].to_numpy(int)
    ok_a, ok_b = d["pred_a"].to_numpy(int) == y, d["pred_b"].to_numpy(int) == y
    out = {"n": len(d), "read +": int(y.sum())}
    for lab, s in ((la, "a"), (lb, "b")):
        pred = d[f"pred_{s}"].to_numpy(int)
        out[f"{lab}: Model + / Read -"] = int(((pred == 1) & (y == 0)).sum())
        out[f"{lab}: Model - / Read +"] = int(((pred == 0) & (y == 1)).sum())
        out[f"{lab}: disagreements"] = int((pred != y).sum())
    out[f"only {lb} agrees with read"] = int((ok_b & ~ok_a).sum())
    out[f"only {la} agrees with read"] = int((ok_a & ~ok_b).sum())
    out["McNemar p"] = mcnemar_p(ok_a, ok_b)
    if len(np.unique(y)) == 2:
        a = paired_auc(y, d["prob_b"].to_numpy(float), d["prob_a"].to_numpy(float), rng)
        out[f"AUC {la}"], out[f"AUC {lb}"] = round(a["auc_suvr"], 3), round(a["auc_model"], 3)
        out[f"AUC {lb} - {la} [95% CI]"] = f"{a['auc_diff']:+.3f} [{a['auc_diff_lo']:+.3f} to {a['auc_diff_hi']:+.3f}]"
    return out


def main():
    args = parse_args()
    la, lb = args.labels
    lo, hi = (float(v) for v in args.band.split(","))
    a, b = load_oof(args.run_a), load_oof(args.run_b)

    # Same folds: same scans, each in the same fold, with the same read.
    m = a.merge(b, on="ID_ind", how="outer", suffixes=("_a", "_b"), indicator=True)
    if (m["_merge"] != "both").any():
        raise SystemExit(f"Not the same CV pool: {int((m['_merge'] != 'both').sum())} scans only in one run")
    if (m["fold_a"] != m["fold_b"]).any():
        raise SystemExit(f"Not the same folds: {int((m['fold_a'] != m['fold_b']).sum())} scans in a different fold")
    if (m["y_a"] != m["y_b"]).any():
        raise SystemExit("Same scans but different reads - check the runs")
    m = m.rename(columns={"y_a": "y", "fold_a": "fold"})

    demo_path = resolve_path(args.demo_csv, find_demo_csv)
    if demo_path is None:
        raise FileNotFoundError("No demographics csv found - pass --demo_csv.")
    demo = pd.read_csv(demo_path)[["ID", args.col]]
    m = m.merge(demo, left_on="ID_ind", right_on="ID", how="left")

    rng = np.random.default_rng(args.seed)
    border = m[m[args.col].between(lo, hi)]
    table = pd.DataFrame({"all (main)": compare(m, la, lb, rng),
                          f"borderline {args.col} {lo}-{hi}": compare(border, la, lb, rng)})

    pos = m[m["y"] == 1]
    spread = pd.DataFrame({
        f"read + fixed by {lb}": pos[(pos["pred_b"] == 1) & (pos["pred_a"] == 0)].groupby("fold").size(),
        f"read + lost by {lb}": pos[(pos["pred_b"] == 0) & (pos["pred_a"] == 1)].groupby("fold").size(),
    }).reindex(sorted(m["fold"].unique()), fill_value=0).fillna(0).astype(int)

    print(f"Same {m['fold'].nunique()} folds, {len(m)} out-of-fold scans: {la} vs {lb}\n")
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(table.to_string())
        print(f"\nPer fold ({lb} vs {la}, read + scans):")
        print(spread.to_string())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{la}_vs_{lb}.csv"
    table.to_csv(path)
    spread.to_csv(out / f"{la}_vs_{lb}_per_fold.csv")
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
