#!/usr/bin/env python3
"""Do training seeds learn the same strategy and only move the cut-off? (same test scans, final models
that differ only in --train_seed)

If two seeds learned the same strategy they rank the scans the same way (which scan looks more
positive than which), and only the probability at which a scan crosses 0.5 differs: one seed is more
liberal, the other more cautious. Three checks, all on the held-out test set (or AVID):

  1. Ranking       - Spearman correlation of the seeds' probabilities, on all scans and on the
                     borderline ones (MetaTemporal SUVR 1.25-1.45, where the seeds differ).
  2. Cut-off       - the MetaTemporal SUVR at which each seed's call (and the visual read) becomes more
                     likely positive than not (logistic 0.5 crossing, bootstrap 95% CI on the same
                     resampled scans for every seed). A pure cut-off shift moves this point and nothing else.
  3. Crossings     - move seed A's cut-off so it calls exactly as many scans positive as seed B. If the
                     seeds rank the scans the same way, A then makes the same call as B on every scan;
                     the scans that still differ ("crossings") are where the seeds really disagree on
                     which scan is more positive. Also: plain flips at 0.5 in each direction (a pure
                     shift flips scans in one direction only).

Descriptive; no tests. Usage:
    python results_plotting/compare_seed_strategy.py <seed42_run> <seed7_run> <seed123_run> --labels s42 s7 s123
    (add --dataset AVID_unseen --demo_csv data/demo_AVID_unseen.csv for AVID)
Outputs (default <proj_path>/results/seed_strategy/<labels>_<dataset>, outside the repo, no IDs):
    seed_summary.csv, seed_pairs.csv, seed_strategy.png (switch-point curves | probabilities vs the first seed)
"""
import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from analyze_model_confidence_visual_read import _logistic_crossing
from plot_utils import find_demo_csv, resolve_path, resolve_predictions_csv

SUVR = "MetaTemporal"
BAND = (1.25, 1.45)
N_BOOT = 1000
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]


def parse_args():
    repo = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="Two or more run dirs (same data type and test set, different train seeds).")
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--dataset", default="Gothenburg")
    p.add_argument("--demo_csv", default=None, help="Demographics csv with MetaTemporal (default data/demo.csv).")
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()
    if len(args.runs) < 2:
        p.error("give at least two runs")
    args.labels = args.labels or [f"seed{i + 1}" for i in range(len(args.runs))]
    if len(args.labels) != len(args.runs):
        p.error("--labels needs one name per run")
    args.out_dir = Path(args.out_dir) if args.out_dir else \
        repo.parent / "results" / "seed_strategy" / f"{'_'.join(args.labels)}_{args.dataset}"
    return args


def load(run: str, dataset: str) -> pd.DataFrame:
    path, _ = resolve_predictions_csv(Path(run), dataset)
    d = pd.read_csv(path)[["ID_ind", "y", "prob", "pred"]]
    print(f"[load] {len(d)} predictions from {path}")
    return d


def crossings(pa: np.ndarray, pb: np.ndarray, yb_calls: np.ndarray) -> tuple[int, float]:
    """Give A as many positives as B has (A's top-k scans), count scans where the calls still differ."""
    k = int(yb_calls.sum())
    order = np.argsort(-pa, kind="stable")
    a_calls = np.zeros(len(pa), int)
    a_calls[order[:k]] = 1
    thr = pa[order[k - 1]] if k > 0 else np.inf
    return int((a_calls != yb_calls).sum()), float(thr)


def main():
    args = parse_args()
    m = None
    for lab, run in zip(args.labels, args.runs):
        d = load(run, args.dataset).rename(columns={"prob": f"prob_{lab}", "pred": f"pred_{lab}"})
        if m is None:
            m = d
        else:
            n0 = len(m)
            m = m.merge(d, on="ID_ind", how="inner", suffixes=("", "_dup"))
            if len(m) != n0 or len(m) != len(d) or (m["y"] != m["y_dup"]).any():
                raise SystemExit("The runs do not have the same scans / reads")
            m = m.drop(columns="y_dup")

    demo_path = resolve_path(args.demo_csv, find_demo_csv)
    if demo_path is None:
        raise FileNotFoundError("No demographics csv found - pass --demo_csv.")
    demo = pd.read_csv(demo_path)[["ID", SUVR]]
    m = m.merge(demo, left_on="ID_ind", right_on="ID", how="left").drop(columns="ID")
    border = m[SUVR].between(*BAND)
    ref = args.labels[0]

    # switch points: same bootstrap resamples for every judge
    s = m.dropna(subset=[SUVR])
    x = s[SUVR].to_numpy(float)
    judges = {"Visual read": s["y"].to_numpy(float), **{lab: s[f"pred_{lab}"].to_numpy(float) for lab in args.labels}}
    fits = {k: _logistic_crossing(x, v) for k, v in judges.items()}
    rng = np.random.default_rng(0)
    boot = {k: [] for k in judges}
    for idx in rng.integers(0, len(x), (N_BOOT, len(x))):
        for k, v in judges.items():
            boot[k].append(_logistic_crossing(x[idx], v[idx])[2])
    boot = {k: np.array(v) for k, v in boot.items()}

    def ci(arr):
        arr = arr[np.isfinite(arr)]
        return np.percentile(arr, [2.5, 97.5]) if len(arr) else (np.nan, np.nan)

    rows = []
    for k in judges:
        lo, hi = ci(boot[k])
        row = {"judge": k, "switch point (SUVR)": round(fits[k][2], 3), "95% CI": f"{lo:.2f}-{hi:.2f}"}
        if k != "Visual read":
            y, p, c = m["y"], m[f"prob_{k}"], m[f"pred_{k}"]
            dlo, dhi = ci(boot[k] - boot["Visual read"])
            row.update({"vs read (SUVR)": f"{fits[k][2] - fits['Visual read'][2]:+.2f} [{dlo:+.2f} to {dhi:+.2f}]",
                        "AUC": round(roc_auc_score(y, p), 3), "Model +": int(c.sum()),
                        "Model + / Read -": int(((c == 1) & (y == 0)).sum()),
                        "Model - / Read +": int(((c == 0) & (y == 1)).sum())})
        else:
            row["Model +"] = int(m["y"].sum())
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("judge")

    pairs = []
    for a, b in combinations(args.labels, 2):
        pa, pb = m[f"prob_{a}"].to_numpy(), m[f"prob_{b}"].to_numpy()
        ca, cb = m[f"pred_{a}"].to_numpy(int), m[f"pred_{b}"].to_numpy(int)
        n_cross, thr = crossings(pa, pb, cb)
        pairs.append({
            "pair": f"{a} vs {b}",
            "Spearman, all": round(spearmanr(pa, pb)[0], 3),
            f"Spearman, borderline (n={int(border.sum())})": round(spearmanr(pa[border], pb[border])[0], 3),
            "first + / second -": int(((ca == 1) & (cb == 0)).sum()),
            "first - / second +": int(((ca == 0) & (cb == 1)).sum()),
            "crossings (first's cut-off moved to match second's positive count)": n_cross,
            "first's cut-off that matches": round(thr, 3),
        })
    pairs = pd.DataFrame(pairs).set_index("pair")

    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(f"\n{len(m)} {args.dataset} scans; borderline = {SUVR} {BAND[0]}-{BAND[1]} (n={int(border.sum())})\n")
        print(summary.to_string())
        print("\nPairs (a pure cut-off shift: high Spearman, flips in one direction only, ~0 crossings):")
        print(pairs.T.to_string())

    # figure: switch-point curves | probabilities of each seed vs the first
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    grid = np.linspace(*np.percentile(x, [1, 97]), 300)
    ax1.axhline(0.5, color="#999999", ls=":", lw=1)
    for i, k in enumerate(judges):
        b0, b1, c = fits[k]
        if not np.isfinite(c):
            continue
        col = "black" if k == "Visual read" else COLORS[(i - 1) % len(COLORS)]
        ax1.plot(grid, 1 / (1 + np.exp(-(b0 + b1 * grid))), color=col, lw=2.4 if k == "Visual read" else 2,
                 ls="--" if k == "Visual read" else "-", label=f"{k} ({c:.2f})")
        ax1.scatter([c], [0.5], color=col, s=50, zorder=5, edgecolor="white")
    ax1.set_xlabel("Meta-temporal SUVR")
    ax1.set_ylabel("Probability of a positive call")
    ax1.set_title("Where each seed switches to positive")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(alpha=0.25)

    def logit(p):
        p = np.clip(p, 1e-5, 1 - 1e-5)
        return np.log(p / (1 - p))

    lr = logit(m[f"prob_{ref}"].to_numpy())
    for i, b in enumerate(args.labels[1:], start=1):
        lb = logit(m[f"prob_{b}"].to_numpy())
        ax2.scatter(lr, lb, s=10, alpha=0.5, color=COLORS[i % len(COLORS)],
                    label=f"{b} (Spearman {spearmanr(lr, lb)[0]:.2f})")
    lim = [min(ax2.get_xlim()[0], ax2.get_ylim()[0]), max(ax2.get_xlim()[1], ax2.get_ylim()[1])]
    ax2.plot(lim, lim, color="#999999", lw=1)
    ax2.axhline(0, color="#999999", ls=":", lw=1)
    ax2.axvline(0, color="#999999", ls=":", lw=1)
    ax2.set_xlim(lim)
    ax2.set_ylim(lim)
    ax2.set_xlabel(f"{ref}: log-odds of positive")
    ax2.set_ylabel("Other seed: log-odds of positive")
    ax2.set_title("Same order of scans, shifted up or down")
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(alpha=0.25)
    fig.tight_layout()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_dir / "seed_summary.csv")
    pairs.to_csv(args.out_dir / "seed_pairs.csv")
    fig.savefig(args.out_dir / "seed_strategy.png", dpi=200)
    print(f"\nSaved seed_summary.csv, seed_pairs.csv, seed_strategy.png to {args.out_dir}")


if __name__ == "__main__":
    main()
