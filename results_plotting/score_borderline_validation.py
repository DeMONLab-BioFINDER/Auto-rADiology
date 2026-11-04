#!/usr/bin/env python3
"""Compare visual_read runs on their VALIDATION scans, overall and on borderline scans.

For choosing between model variants (pooling, input scaling, ...) without using the test set.
Reads <run>/evaluation/<dataset>/Val_<dataset>_results.csv (saved by runs with a validation split since
the borderline-weighting change; for older runs make it with run_predict_split.py). All runs must share
the same validation scans (same split seed / fractions).

Per run, all validation scans and the borderline ones (--band of --col, default MetaTemporal SUVR
1.25-1.45, fixed in advance):
    AUC                      - threshold-free; the main comparison (0.5 itself moves between seeds)
    Model + / Read -         - count and % of read - scans
    Model - / Read +         - count and % of read + scans
Plus, for read + scans below the band (low tau), the % the model calls + - the cases where models
have followed tau amount instead of the read.
Descriptive: with ~100 borderline scans, differences of a few scans are within seed noise.

Usage:
    python results_plotting/score_borderline_validation.py <run_dir> [<run_dir> ...] --labels att42 att7 blw42 blw7
Outputs (default <proj_path>/results/borderline_validation/<labels>.csv, outside the repo; no IDs).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from plot_utils import find_demo_csv, resolve_path


def parse_args():
    repo = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+")
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--dataset", default="Gothenburg")
    p.add_argument("--demo_csv", default=None, help="Demographics csv with the SUVR column (default data/demo.csv).")
    p.add_argument("--col", default="MetaTemporal")
    p.add_argument("--band", default="1.25,1.45")
    p.add_argument("--out_dir", default=str(repo.parent / "results" / "borderline_validation"))
    args = p.parse_args()
    args.labels = args.labels or [f"run{i + 1}" for i in range(len(args.runs))]
    if len(args.labels) != len(args.runs):
        p.error("--labels needs one name per run")
    return args


def auc(y, prob):
    return roc_auc_score(y, prob) if len(np.unique(y)) == 2 else np.nan


def score(d: pd.DataFrame) -> dict:
    y, pred, prob = d["y"].to_numpy(int), d["pred"].to_numpy(int), d["prob"].to_numpy(float)
    neg, pos = y == 0, y == 1
    fp, fn = int((pred[neg] == 1).sum()), int((pred[pos] == 0).sum())
    return {"n": len(d), "read +": int(pos.sum()), "AUC": round(auc(y, prob), 3),
            "Model + / Read -": f"{fp} ({fp / max(neg.sum(), 1):.0%})",
            "Model - / Read +": f"{fn} ({fn / max(pos.sum(), 1):.0%})"}


def main():
    args = parse_args()
    lo, hi = (float(v) for v in args.band.split(","))
    demo_path = resolve_path(args.demo_csv, find_demo_csv)
    if demo_path is None:
        raise FileNotFoundError("No demographics csv found - pass --demo_csv.")
    demo = pd.read_csv(demo_path)[["ID", args.col]]

    runs, ids = [], None
    for run, lab in zip(args.runs, args.labels):
        f = Path(run) / "evaluation" / args.dataset / f"Val_{args.dataset}_results.csv"
        if not f.exists():
            raise SystemExit(f"{f} not found - run run_predict_split.py on {run} first.")
        d = pd.read_csv(f).merge(demo, left_on="ID_ind", right_on="ID", how="left")
        runs.append((lab, d))
        ids = set(d["ID_ind"]) if ids is None else ids
        if set(d["ID_ind"]) != ids:
            raise SystemExit(f"{lab} has different validation scans from {args.labels[0]} - not the same split.")

    rows = []
    for lab, d in runs:
        x = d[args.col]
        border = d[x.between(lo, hi)]
        low_pos = d[(x < lo) & (d["y"] == 1)]
        r = {"run": lab}
        r.update({f"all: {k}": v for k, v in score(d).items()})
        r.update({f"borderline: {k}": v for k, v in score(border).items()})
        r[f"read + below {lo}: Model +"] = (f"{int(low_pos['pred'].sum())} of {len(low_pos)}" if len(low_pos) else "-")
        rows.append(r)
    table = pd.DataFrame(rows).set_index("run")

    print(f"Validation scans ({len(ids)}); borderline = {args.col} {lo}-{hi}")
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(table.T.to_string())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{'_vs_'.join(args.labels)}.csv"
    table.to_csv(path)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
