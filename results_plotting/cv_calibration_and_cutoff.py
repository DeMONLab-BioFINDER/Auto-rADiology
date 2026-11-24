#!/usr/bin/env python3
"""Compare CV runs on their out-of-fold predictions only (never the test set): calibration, disagreements at 0.5,
and the cut-off where the CV calibration says "50% chance of Read +".

Why: on the held-out test set the frozen MNI attention model under-scored borderline scans (model probability
0.1-0.5, mean 0.25, but 59% Read +). Choosing a cut-off or a training change on the test set would be tuning on
test data, so the decision is made here, on the training pool's out-of-fold predictions (every scan predicted once
by the fold model that did not train on it).

Per run:
    AUC; Model + / Read - and Model - / Read + at cut-off 0.5
    borderline calibration: scans with probability 0.1-0.5 -> mean probability vs share actually Read +
    98%-specificity cut-off: the lowest cut-off at which 98% of Read - scans are called negative (the trial-screening
    rule; in October 0.5 sat at about this point on CV); disagreements at that cut-off
    CV cut-off: the probability at which a logistic recalibration (Read + ~ logit(probability), fitted on the
    out-of-fold predictions) gives P(Read +) = 0.5; disagreements at that cut-off
    calibration bins (0-0.1, ..., 0.9-1)
One figure: calibration curves of all runs on one plot (dashed diagonal = honest probabilities).

Decision criteria (fixed 24 Nov 2026, before looking at the class-balanced run):
    adopt class-balanced sampling only if, compared with the plain run, (1) Model + / Read - at 0.5 is not higher
    by more than the plain-vs-plain rerun difference, (2) borderline calibration is clearly closer to the
    diagonal, and (3) AUC is not lower. A CV cut-off for the plain model is the no-retraining alternative.

Usage:
    python results_plotting/cv_calibration_and_cutoff.py <cv_run_dir> [<cv_run_dir> ...] --labels plain plain-rerun classbal \
        [--out_dir <folder>]
Run dirs may be relative to <proj_path>/results. Each needs oof_predictions.csv (ID_ind, y, prob, pred).
Outputs (no IDs): cv_comparison.csv, cv_calibration_bins.csv, cv_calibration.png (default <proj_path>/results/cv_calibration/).
"""
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

from plot_utils import CUSTOM_PALETTE

EPS = 1e-4
BINS = np.round(np.arange(0, 1.01, 0.1), 2)


def parse_args():
    repo = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="CV run folders with oof_predictions.csv.")
    p.add_argument("--labels", nargs="+", default=None, help="Short names, one per run (default: folder names).")
    p.add_argument("--results_root", default=str(repo.parent / "results"))
    p.add_argument("--out_dir", default=str(repo.parent / "results" / "cv_calibration"))
    return p.parse_args()


def load(run: str, root: str) -> pd.DataFrame:
    d = Path(run)
    if not (d.is_absolute() or d.exists()):
        d = Path(root) / d
    f = d / "oof_predictions.csv"
    if not f.exists():
        raise SystemExit(f"No oof_predictions.csv in {d}")
    return pd.read_csv(f)[["y", "prob"]].dropna()


def cv_cutoff(df: pd.DataFrame) -> float:
    """Probability where the logistic recalibration Read + ~ a + b*logit(p) crosses 0.5, i.e. logit(p) = -a/b."""
    x = np.log(df["prob"].clip(EPS, 1 - EPS) / (1 - df["prob"].clip(EPS, 1 - EPS)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sm.Logit(df["y"].to_numpy(), sm.add_constant(x.to_numpy())).fit(disp=0)
    a, b = fit.params
    return float(1 / (1 + np.exp(a / b)))


def counts(df: pd.DataFrame, cut: float) -> tuple[int, int]:
    call = df["prob"] >= cut
    return int(((df["y"] == 0) & call).sum()), int(((df["y"] == 1) & ~call).sum())


def main():
    args = parse_args()
    labels = args.labels or [Path(r).name for r in args.runs]
    if len(labels) != len(args.runs):
        raise SystemExit("--labels needs one name per run")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    rows, bins_rows = [], []
    fig, ax = plt.subplots(figsize=(5.4, 5.2), dpi=180)
    ax.plot([0, 1], [0, 1], ls="--", color="0.4", lw=1)
    for i, (run, lab) in enumerate(zip(args.runs, labels)):
        d = load(run, args.results_root)
        fp05, fn05 = counts(d, 0.5)
        mid = d[(d["prob"] > 0.1) & (d["prob"] < 0.5)]
        cut = cv_cutoff(d)
        fpc, fnc = counts(d, cut)
        cut98 = float(np.quantile(d.loc[d["y"] == 0, "prob"], 0.98))
        fp98, fn98 = counts(d, cut98)
        rows.append({"run": lab, "n": len(d), "Read +": int(d["y"].sum()), "AUC": round(roc_auc_score(d["y"], d["prob"]), 4),
                     "Model + / Read - at 0.5": fp05, "Model - / Read + at 0.5": fn05,
                     "P 0.1-0.5: n": len(mid), "P 0.1-0.5: mean probability": round(mid["prob"].mean(), 3) if len(mid) else np.nan,
                     "P 0.1-0.5: share Read +": round(mid["y"].mean(), 3) if len(mid) else np.nan,
                     "CV cut-off (calibrated P = 0.5)": round(cut, 3),
                     "Model + / Read - at CV cut-off": fpc, "Model - / Read + at CV cut-off": fnc,
                     "98%-specificity cut-off": round(cut98, 3),
                     "Model + / Read - at 98% cut-off": fp98, "Model - / Read + at 98% cut-off": fn98})
        b = pd.cut(d["prob"], BINS, include_lowest=True)
        g = d.groupby(b, observed=True).agg(n=("y", "size"), mean_probability=("prob", "mean"), share_read_pos=("y", "mean")).reset_index()
        g.insert(0, "run", lab); g["prob"] = g["prob"].astype(str)
        bins_rows.append(g)
        colour = CUSTOM_PALETTE[i % len(CUSTOM_PALETTE)]
        ax.plot(g["mean_probability"], g["share_read_pos"], marker="o", color=colour, lw=1.6, ms=5, label=f"{lab} (AUC {rows[-1]['AUC']:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Model probability (mean per bin)")
    ax.set_ylabel("Share Read +")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(color="0.92")
    fig.tight_layout(); fig.savefig(out / "cv_calibration.png"); plt.close(fig)

    table = pd.DataFrame(rows)
    table.to_csv(out / "cv_comparison.csv", index=False)
    pd.concat(bins_rows).to_csv(out / "cv_calibration_bins.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(table.set_index("run").T.to_string())
    print(f"\nSaved cv_comparison.csv, cv_calibration_bins.csv, cv_calibration.png to {out}")


if __name__ == "__main__":
    main()
