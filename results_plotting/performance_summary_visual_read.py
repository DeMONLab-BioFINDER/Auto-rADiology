#!/usr/bin/env python3
"""Headline performance of one final visual-read model, held-out test set and external set side by side.

For each set (test = the run's held-out 20%, avid = its zero-shot external result), at the fixed 0.5 cut-off:
  AUC (bootstrap 95% CI), AUPRC, agreement with read + (sensitivity) and read - (specificity), PPV, NPV
  (Wilson 95% CIs), Cohen's kappa vs the read (bootstrap 95% CI), Model + / Read - and Model - / Read + counts,
  and the AUC of MetaTemporal SUVR alone with the paired bootstrap 95% CI of the difference (model - SUVR).
Per cohort (test sites, plus the external set as its own row): agreement with read + and read - with Wilson CIs.

Kappa is the number to compare with published agreement between human readers; agreement percentages and
counts are the plain-language version. Everything is descriptive with CIs; sets are never pooled.

Usage:
    python results_plotting/performance_summary_visual_read.py <final_run_dir> \\
        [--sets test avid] [--demo_csv data/demo.csv] [--ext_demo_csv data/demo_AVID_unseen.csv]
Outputs (<run_dir>/figures/performance_summary/, outside the repo; no IDs):
    performance_summary.csv   one row per set
    per_cohort.csv            one row per cohort (and the external set)
    roc_test_avid.png         ROC curves of both sets, 0.5 cut-off marked, MetaTemporal SUVR as dashed baseline
    forest_per_cohort.png     agreement with read + / read - per cohort, 95% CIs
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score, roc_curve

from plot_utils import external_name, bootstrap_auc_ci, find_demo_csv, resolve_path, wilson_interval

SUVR_COL = "MetaTemporal"
SET_COLOURS = {"test": "#1f4e79", "avid": "#c0504d"}
SET_LABELS = {"test": "Held-out test", "avid": "External (AVID)"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", help="Final visual-read run directory.")
    p.add_argument("--sets", nargs="+", default=["test", "avid"], choices=["test", "avid"])
    p.add_argument("--dataset", default="Gothenburg")
    p.add_argument("--ext_dataset", default="AVID_unseen")
    p.add_argument("--demo_csv", default=None, help="Demographics of the test set (default data/demo.csv).")
    p.add_argument("--ext_demo_csv", default=None, help="Demographics of the external set (default data/demo_<ext_dataset>.csv).")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


def find_ext_demo(name: str) -> Path | None:
    here = Path(__file__).resolve()
    return next((c for c in [here.parents[1] / "data" / f"demo_{name}.csv", here.parents[2] / "data" / f"demo_{name}.csv"]
                 if c.exists()), None)


def load_set(run: Path, which: str, args) -> pd.DataFrame:
    if which == "test":
        f = run / "evaluation" / args.dataset / f"Eval_{args.dataset}_results.csv"
        demo_path = resolve_path(args.demo_csv, find_demo_csv)
    else:
        hits = sorted((run / "validation").glob(f"External_validation_{args.ext_dataset}__*_zeroshot_results.csv"))
        if not hits:
            raise SystemExit(f"No {args.ext_dataset} zero-shot result in {run}/validation")
        f = hits[0]
        demo_path = resolve_path(args.ext_demo_csv, lambda: find_ext_demo(args.ext_dataset))
    if not f.exists():
        raise SystemExit(f"Missing {f}")
    d = pd.read_csv(f)[["ID_ind", "y", "pred", "prob"]].apply(pd.to_numeric, errors="coerce").dropna()
    if demo_path is None:
        raise SystemExit(f"No demographics csv for the {which} set - pass --{'ext_' if which == 'avid' else ''}demo_csv")
    demo = pd.read_csv(demo_path)
    demo = demo.loc[:, ~demo.columns.astype(str).str.startswith("Unnamed")][["ID", "site", SUVR_COL]]
    out = d.merge(demo, left_on="ID_ind", right_on="ID", how="left")
    if out["site"].isna().any():
        print(f"[WARNING] {which}: {int(out['site'].isna().sum())} scans have no row in {demo_path.name}")
    return out


def rate(k: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return np.nan, np.nan, np.nan
    lo, hi = wilson_interval(k, n)
    return k / n, lo, hi


def summarise(d: pd.DataFrame, n_boot: int, seed: int = 0) -> dict:
    y, pred, prob = d["y"].to_numpy(int), d["pred"].to_numpy(int), d["prob"].to_numpy(float)
    tp, tn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    fp, fn = int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    row = {"n": len(d), "read +": int(y.sum()), "read -": int((y == 0).sum())}
    row["auc"], row["auc_lo"], row["auc_hi"] = bootstrap_auc_ci(y, prob, n_boot=n_boot, seed=seed)
    row["auprc"] = average_precision_score(y, prob)
    for name, k, n in [("agreement read +", tp, tp + fn), ("agreement read -", tn, tn + fp),
                       ("ppv", tp, tp + fp), ("npv", tn, tn + fn), ("accuracy", tp + tn, len(d))]:
        row[name], row[f"{name}_lo"], row[f"{name}_hi"] = rate(k, n)
    rng = np.random.default_rng(seed)
    row["kappa"] = cohen_kappa_score(y, pred)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            boots.append(cohen_kappa_score(y[idx], pred[idx]))
    row["kappa_lo"], row["kappa_hi"] = np.percentile(boots, [2.5, 97.5])
    row["Model + / Read -"], row["Model - / Read +"] = fp, fn
    # MetaTemporal SUVR alone: AUC and the paired difference
    s = d.dropna(subset=[SUVR_COL])
    ys, ps, xs = s["y"].to_numpy(int), s["prob"].to_numpy(float), s[SUVR_COL].to_numpy(float)
    if len(np.unique(ys)) == 2:
        row["auc_suvr"] = roc_auc_score(ys, xs)
        diffs = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(ys), len(ys))
            if len(np.unique(ys[idx])) == 2:
                diffs.append(roc_auc_score(ys[idx], ps[idx]) - roc_auc_score(ys[idx], xs[idx]))
        row["auc_model_minus_suvr"] = roc_auc_score(ys, ps) - row["auc_suvr"]
        row["auc_diff_lo"], row["auc_diff_hi"] = np.percentile(diffs, [2.5, 97.5])
        row["n_with_suvr"] = len(s)
    return row


def per_cohort(sets: dict) -> pd.DataFrame:
    rows = []
    for which, d in sets.items():
        groups = d.groupby("site") if which == "test" else [(SET_LABELS["avid"], d)]
        for site, g in groups:
            y, pred = g["y"].to_numpy(int), g["pred"].to_numpy(int)
            r = {"set": which, "cohort": site, "n": len(g), "read +": int(y.sum()), "read -": int((y == 0).sum())}
            for name, k, n in [("agreement read +", int(((pred == 1) & (y == 1)).sum()), int(y.sum())),
                               ("agreement read -", int(((pred == 0) & (y == 0)).sum()), int((y == 0).sum()))]:
                r[name], r[f"{name}_lo"], r[f"{name}_hi"] = rate(k, n)
            rows.append(r)
    return pd.DataFrame(rows)


def roc_figure(sets: dict, summary: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=180)
    for which, d in sets.items():
        c, row = SET_COLOURS[which], summary.loc[which]
        fpr, tpr, _ = roc_curve(d["y"], d["prob"])
        short = "Test" if which == "test" else SET_LABELS[which].replace("External (", "").rstrip(")")
        ax.plot(fpr, tpr, color=c, lw=2, label=f"Model, {short}   {row['auc']:.3f}")
        s = d.dropna(subset=[SUVR_COL])
        if "auc_suvr" in row and not pd.isna(row["auc_suvr"]):
            f2, t2, _ = roc_curve(s["y"], s[SUVR_COL])
            ax.plot(f2, t2, color=c, lw=1.2, ls="--", label=f"Meta-temporal SUVR, {short}   {row['auc_suvr']:.3f}")
        ax.plot(1 - row["agreement read -"], row["agreement read +"], "o", color=c, ms=7, mec="white", mew=1.2)
    ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=0.8)
    ax.set_xlabel("Disagreement with Read −", fontsize=12)
    ax.set_ylabel("Agreement with Read +", fontsize=12)
    ax.tick_params(labelsize=10.5)
    ax.set_xlim(-0.01, 1); ax.set_ylim(0, 1.01); ax.set_aspect("equal")
    ax.set_title("ROC curves, model and meta-temporal SUVR", fontsize=13)
    ax.legend(loc="lower right", fontsize=9.5, frameon=True, title="AUC", title_fontsize=10, alignment="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def forest_figure(pc: pd.DataFrame, out: Path):
    pc = pc.iloc[::-1].reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 0.5 * len(pc) + 1.6), dpi=180, sharey=True)
    for ax, name in zip(axes, ["agreement read +", "agreement read -"]):
        for i, r in pc.iterrows():
            c = SET_COLOURS[r["set"]]
            # clipped at 0: at 100% (or 0%) agreement the Wilson bound can land a rounding error inside the point
            lo, hi = max(r[name] - r[f"{name}_lo"], 0), max(r[f"{name}_hi"] - r[name], 0)
            ax.errorbar(100 * r[name], i, xerr=[[100 * lo], [100 * hi]],
                        fmt="o", color=c, ms=5, capsize=2.5, lw=1.2)
        ax.set_title(f"Read {name.split()[-1].replace('-', '−')} scans", fontsize=12)
        ax.set_xlabel("Agreement with the read (%)", fontsize=11.5)
        ax.tick_params(axis="x", labelsize=10.5)
        ax.set_xlim(0, 101); ax.grid(axis="x", color="#eeeeee"); ax.spines[["top", "right"]].set_visible(False)
    labels = [f"{r['cohort']} (n={r['n']}; {r['read +']} read +)" for _, r in pc.iterrows()]
    axes[0].set_yticks(range(len(pc))); axes[0].set_yticklabels(labels, fontsize=10.5)
    fig.suptitle("Agreement per cohort", fontsize=14)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def main():
    args = parse_args()
    run = Path(args.run_dir).resolve()
    SET_LABELS["avid"] = f"External ({external_name(args.ext_dataset)})"  # the external set, whichever it is
    out = Path(args.out_dir) if args.out_dir else run / "figures" / "performance_summary"
    out.mkdir(parents=True, exist_ok=True)
    sets = {w: load_set(run, w, args) for w in args.sets}
    summary = pd.DataFrame({w: summarise(d, args.n_boot) for w, d in sets.items()}).T
    summary.index.name = "set"
    pc = per_cohort(sets)
    summary.to_csv(out / "performance_summary.csv")
    pc.to_csv(out / "per_cohort.csv", index=False)
    roc_figure(sets, summary, out / "roc_test_avid.png")
    forest_figure(pc, out / "forest_per_cohort.png")

    for w, r in summary.iterrows():
        print(f"\n{SET_LABELS[w]} (n = {int(r['n'])}: {int(r['read +'])} read +, {int(r['read -'])} read -)")
        print(f"  AUC {r['auc']:.3f} [{r['auc_lo']:.3f}-{r['auc_hi']:.3f}]   AUPRC {r['auprc']:.3f}   "
              f"kappa {r['kappa']:.3f} [{r['kappa_lo']:.3f}-{r['kappa_hi']:.3f}]")
        for name in ["agreement read +", "agreement read -", "ppv", "npv", "accuracy"]:
            print(f"  {name:17s} {100 * r[name]:5.1f}% [{100 * r[f'{name}_lo']:.1f}-{100 * r[f'{name}_hi']:.1f}]")
        print(f"  Model + / Read - {int(r['Model + / Read -'])}   Model - / Read + {int(r['Model - / Read +'])}")
        if "auc_suvr" in r and not pd.isna(r["auc_suvr"]):
            print(f"  MetaTemporal SUVR alone: AUC {r['auc_suvr']:.3f}; model - SUVR {r['auc_model_minus_suvr']:+.3f} "
                  f"[{r['auc_diff_lo']:+.3f} to {r['auc_diff_hi']:+.3f}] (n = {int(r['n_with_suvr'])})")
    print("\nPer cohort:")
    with pd.option_context("display.width", 200):
        print(pc.assign(**{c: (100 * pc[c]).round(1) for c in pc.columns if c.startswith("agreement")})
              [["set", "cohort", "n", "read +", "agreement read +", "agreement read +_lo", "agreement read +_hi",
                "agreement read -", "agreement read -_lo", "agreement read -_hi"]].to_string(index=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
