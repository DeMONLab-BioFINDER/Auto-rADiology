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

  4. Noise or strategy? - for the crossing scans: how far apart they sit in the two seeds' rankings
                     (percentile rank, 0-100; "gap"), next to the usual gap over all scans. Near-ties
                     (small gap, scans next to each other around the cut-off) = same strategy, random
                     wobble; large gaps (bigger than 90% of all scans' gaps) = the seeds really see that
                     scan differently - look at those (inspect_scans.py), with their regional SUVRs,
                     amyloid, site, to see if they share something.

Descriptive; no tests. Usage:
    python results_plotting/compare_seed_strategy.py <seed42_run> <seed7_run> <seed123_run> --labels s42 s7 s123
    (add --dataset AVID_unseen --demo_csv data/demo_AVID_unseen.csv for AVID)
Outputs (default <proj_path>/results/seed_strategy/<labels>_<dataset>, outside the repo, no IDs):
    seed_summary.csv, seed_pairs.csv, seed_strategy.png (switch-point curves | probabilities vs the first seed),
    seed_crossings.csv (one row per crossing scan and pair, has IDs - not for slides),
    seed_swaps_<a>_vs_<b>.png (slope chart: one line per scan from its distance to seed a's cut-off (matched)
    to its distance to seed b's; red = crossing scans - short red lines close to the dashed cut-off = near-ties),
    seed_ranking_slide.png (the right panel alone, for a slide; score = log-odds, dotted lines = the 0.5 cut-off)
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
EXTRA = ["TemporoParietal", "MesialTemporal", "Frontal", "amyloid_status", "CL", "site", "dx"]
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


def crossings(pa: np.ndarray, pb: np.ndarray, yb_calls: np.ndarray) -> tuple[np.ndarray, float]:
    """Give A as many positives as B has (A's top-k scans); returns the mask of scans where the calls
    still differ, and A's matching cut-off."""
    k = int(yb_calls.sum())
    order = np.argsort(-pa, kind="stable")
    a_calls = np.zeros(len(pa), int)
    a_calls[order[:k]] = 1
    thr = pa[order[k - 1]] if k > 0 else np.inf
    return a_calls != yb_calls, float(thr)


def slope_chart(pa: np.ndarray, pb: np.ndarray, thr_a: float, cross: np.ndarray, a: str, b: str, out: Path,
                ylim: float = 4.0):
    """Each scan's distance from its seed's own cut-off, in log-odds: A's cut-off matched to B's positive
    count, B's own 0.5. 0 = on the cut-off line, so a near-tie swap is a short red line close to 0."""
    def logit(p):
        p = np.clip(p, 1e-5, 1 - 1e-5)
        return np.log(p / (1 - p))
    da, db = logit(pa) - logit(thr_a), logit(pb)
    fig, ax = plt.subplots(figsize=(5.2, 7))
    ax.axhspan(0, ylim, color="#fbe3e3", zorder=0)
    ax.axhspan(-ylim, 0, color="#e3ecfb", zorder=0)
    show = (np.abs(da) < ylim) | (np.abs(db) < ylim)
    for i in np.where(show & ~cross)[0]:
        ax.plot([0, 1], [da[i], db[i]], color="#9a9a9a", lw=0.8, alpha=0.6, zorder=1)
    for i in np.where(cross)[0]:
        ax.plot([0, 1], [da[i], db[i]], color="#d62728", lw=2.2, alpha=0.95, zorder=3)
    ax.axhline(0, color="black", ls="--", lw=1.3, zorder=2)
    for xpos in (0, 1):
        ax.axvline(xpos, color="#555555", lw=1.5, zorder=2)
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-ylim, ylim)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([a, b], fontsize=13)
    ax.set_ylabel("Distance from the cut-off (log-odds)", fontsize=12)
    ax.text(0.5, ylim * 0.93, "Model +", ha="center", va="top", fontsize=12, color="#8b1a1a")
    ax.text(0.5, -ylim * 0.93, "Model -", ha="center", va="bottom", fontsize=12, color="#1a3f8b")
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title(f"{int(cross.sum())} scans swap sides (red)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


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
    demo = pd.read_csv(demo_path)
    demo = demo[["ID", SUVR] + [c for c in EXTRA if c in demo.columns]]
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

    pct = {lab: m[f"prob_{lab}"].rank(pct=True).to_numpy() * 100 for lab in args.labels}
    pairs, cross_rows = [], []
    for a, b in combinations(args.labels, 2):
        pa, pb = m[f"prob_{a}"].to_numpy(), m[f"prob_{b}"].to_numpy()
        ca, cb = m[f"pred_{a}"].to_numpy(int), m[f"pred_{b}"].to_numpy(int)
        cross, thr = crossings(pa, pb, cb)
        n_cross = int(cross.sum())
        gap = np.abs(pct[a] - pct[b])
        big = np.percentile(gap, 90)
        c = m[cross].copy()
        c["pair"], c["gap (percentile points)"] = f"{a} vs {b}", gap[cross].round(1)
        c[f"rank {a}"], c[f"rank {b}"] = pct[a][cross].round(1), pct[b][cross].round(1)
        c["ranked higher by"] = np.where(pct[b][cross] > pct[a][cross], b, a)
        c["large gap"] = gap[cross] > big
        cross_rows.append(c)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        slope_chart(pa, pb, thr, cross, a, b, args.out_dir / f"seed_swaps_{a}_vs_{b}.png")
        pairs.append({
            "pair": f"{a} vs {b}",
            "Spearman, all": round(spearmanr(pa, pb)[0], 3),
            f"Spearman, borderline (n={int(border.sum())})": round(spearmanr(pa[border], pb[border])[0], 3),
            "first + / second -": int(((ca == 1) & (cb == 0)).sum()),
            "first - / second +": int(((ca == 0) & (cb == 1)).sum()),
            "crossings (first's cut-off moved to match second's positive count)": n_cross,
            "first's cut-off that matches": round(thr, 3),
            "usual gap, all scans (median, pct points)": round(float(np.median(gap)), 1),
            "crossing scans' gap (median)": round(float(np.median(gap[cross])), 1) if n_cross else np.nan,
            "large-gap crossings (> 90% of all scans' gaps)": int((gap[cross] > big).sum()),
        })
    pairs = pd.DataFrame(pairs).set_index("pair")

    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(f"\n{len(m)} {args.dataset} scans; borderline = {SUVR} {BAND[0]}-{BAND[1]} (n={int(border.sum())})\n")
        print(summary.to_string())
        print("\nPairs (a pure cut-off shift: high Spearman, flips in one direction only, ~0 crossings):")
        print(pairs.T.to_string())

        cr = pd.concat(cross_rows) if cross_rows else pd.DataFrame()
        if len(cr):
            print("\nCrossing scans - near-ties vs large gaps (medians; same strategy if large gaps are rare and"
                  " look like ordinary borderline scans):")
            num = [c for c in [SUVR] + EXTRA if c in cr and c not in ("site", "dx", "amyloid_status")]
            agg = cr.groupby(["pair", "large gap"]).agg(
                n=("ID_ind", "size"), read_pos=("y", "mean"),
                **{f"{c}": (c, "median") for c in num},
                **({"amyloid_pos": ("amyloid_status", "mean")} if "amyloid_status" in cr else {}))
            agg["read_pos"] = (100 * agg["read_pos"]).round(0)
            if "amyloid_pos" in agg:
                agg["amyloid_pos"] = (100 * agg["amyloid_pos"]).round(0)
            print(agg.rename(columns={"read_pos": "% read +", "amyloid_pos": "% amyloid +"}).round(2).to_string())
            big_cr = cr[cr["large gap"]]
            if len(big_cr):
                print("\nLarge-gap crossing scans (for inspect_scans.py):")
                cols = ["pair", "ID_ind", "y"] + [f"prob_{l}" for l in args.labels] + \
                       ["gap (percentile points)", "ranked higher by"] + [c for c in [SUVR] + EXTRA if c in cr]
                print(big_cr.sort_values(["pair", "gap (percentile points)"], ascending=[True, False])[cols]
                      .round(3).to_string(index=False))

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
    if cross_rows:
        pd.concat(cross_rows).to_csv(args.out_dir / "seed_crossings.csv", index=False)
    pairs.to_csv(args.out_dir / "seed_pairs.csv")
    fig.savefig(args.out_dir / "seed_strategy.png", dpi=200)

    # slide version: the ranking panel alone, larger text
    fig2, ax = plt.subplots(figsize=(6.2, 6))
    for i, b in enumerate(args.labels[1:], start=1):
        lb = logit(m[f"prob_{b}"].to_numpy())
        ax.scatter(lr, lb, s=14, alpha=0.55, color=COLORS[i % len(COLORS)], label=b)
    ax.plot(lim, lim, color="#999999", lw=1)
    ax.axhline(0, color="#999999", ls=":", lw=1)
    ax.axvline(0, color="#999999", ls=":", lw=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(f"Score, {ref}", fontsize=13)
    ax.set_ylabel("Score, other seed", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.legend(frameon=False, fontsize=12, markerscale=1.6)
    ax.grid(alpha=0.25)
    fig2.tight_layout()
    fig2.savefig(args.out_dir / "seed_ranking_slide.png", dpi=200)
    print(f"\nSaved seed_summary.csv, seed_pairs.csv, seed_crossings.csv (IDs), seed_strategy.png, seed_ranking_slide.png to {args.out_dir}")


if __name__ == "__main__":
    main()
