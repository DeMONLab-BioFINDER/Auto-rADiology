#!/usr/bin/env python3
"""How much tau is in each CenTauR region, per model / read group? (Goes next to the occlusion table.)

Per group (Model + / Read +, Model − / Read −, Model − / Read +, Model + / Read −): median SUVR in the CenTauR
meta-temporal, temporoparietal, mesial temporal and frontal regions (the spreadsheet SUVRs). Model − / Read + vs
Model + / Read +: Mann-Whitney per region, FDR across the four regions. Model + / Read − is shown, not tested.

Figure: boxplots per region (Model - / Read -, Model - / Read +, Model + / Read +; Model - / Read + scans also as dots;
Model + / Read - as single points, too few for a box). No numbers on the plot (medians + FDR p in the csv).

Usage:
    python results_plotting/regional_suvr_by_group.py <run_dir> --demo_csv <demo.csv> [--dataset AVID_unseen]
    # both sets side by side, from the data csvs (local is fine, no IDs):
    python results_plotting/regional_suvr_by_group.py --plot_from "held-out test=<test data csv>" "AVID=<avid data csv>" --out fig.png
Outputs (no IDs), next to the run's other figures (validation/figures for an external set): regional_suvr_by_group.csv,
    regional_suvr_by_group.png, regional_suvr_by_group_data.csv (group + the four region SUVRs per scan, for --plot_from)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from plot_utils import CONFUSION_CATEGORY_PALETTE, fdr_bh, find_demo_csv, resolve_path, resolve_predictions_csv

REGIONS = [("MetaTemporal", "Meta-temporal"), ("TemporoParietal", "Temporoparietal"),
           ("MesialTemporal", "Mesial temporal"), ("Frontal", "Frontal")]
GROUPS = [("Model + / Read +", 1, 1), ("Model − / Read −", 0, 0), ("Model − / Read +", 1, 0), ("Model + / Read −", 0, 1)]
COLOUR = {"Model − / Read −": CONFUSION_CATEGORY_PALETTE["True Negative"],
          "Model − / Read +": CONFUSION_CATEGORY_PALETTE["False Negative"],
          "Model + / Read +": CONFUSION_CATEGORY_PALETTE["True Positive"],
          "Model + / Read −": CONFUSION_CATEGORY_PALETTE["False Positive"]}
TICK = {"Temporoparietal": "Temporo-\nparietal", "Mesial temporal": "Mesial\ntemporal"}
BOXES = ["Model − / Read −", "Model − / Read +", "Model + / Read +"]  # low -> high tau; Model + / Read − = points only


def plot(sets: list, out: Path):
    """sets = [(title, data frame with 'group' + region columns)]: one panel per set, boxes per region and group."""
    labs = [lab for _, lab in REGIONS]
    fig, axes = plt.subplots(1, len(sets), figsize=(6.4 * len(sets), 5.2), sharey=True, squeeze=False)
    rng = np.random.default_rng(0)
    w = 0.24
    for ax, (title, d) in zip(axes[0], sets):
        for i, lab in enumerate(labs):
            for j, g in enumerate(BOXES):
                v = pd.to_numeric(d.loc[d["group"] == g, lab], errors="coerce").dropna()
                if v.empty:
                    continue
                x = i + (j - 1) * w
                c = COLOUR[g]
                ax.boxplot(v, positions=[x], widths=w * 0.8, showfliers=g != "Model − / Read +", patch_artist=True,
                           boxprops=dict(facecolor=c, alpha=0.35, edgecolor=c), medianprops=dict(color=c, lw=2.2),
                           whiskerprops=dict(color=c), capprops=dict(color=c),
                           flierprops=dict(marker="o", ms=2.5, mfc=c, mec="none", alpha=0.5))
                if g == "Model − / Read +":
                    ax.scatter(x + rng.uniform(-w * 0.25, w * 0.25, len(v)), v, s=14, color=c, edgecolor="white",
                               linewidth=0.4, zorder=3)
            fp = pd.to_numeric(d.loc[d["group"] == "Model + / Read −", lab], errors="coerce").dropna()
            ax.scatter(np.full(len(fp), i + 1.75 * w), fp, marker="D", s=30, color=COLOUR["Model + / Read −"],
                       edgecolor="white", linewidth=0.5, zorder=4)
        ax.set_xticks(range(len(labs)), [TICK.get(l, l) for l in labs], fontsize=11.5)
        ax.set_xlim(-0.55, len(labs) - 0.35)
        ax.set_title(title, fontsize=13.5)
        ax.yaxis.grid(True, color="#E5E5E5", lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("SUVR (CenTauR regions)", fontsize=12)
    first = sets[0][1]
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLOUR[g], alpha=0.6, edgecolor=COLOUR[g]) for g in BOXES]
    handles.append(plt.Line2D([], [], marker="D", ls="", color=COLOUR["Model + / Read −"], ms=6))
    fig.legend(handles, BOXES + ["Model + / Read − (single scans)"], loc="lower center", ncol=4, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Tau in each CenTauR region", fontsize=15)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", nargs="?"); p.add_argument("--dataset", default="Gothenburg"); p.add_argument("--demo_csv", default=None)
    p.add_argument("--plot_from", nargs="+", default=None, help='"title=data.csv" per panel (regional_suvr_by_group_data.csv)')
    p.add_argument("--out", default="regional_suvr_by_group_sets.png")
    args = p.parse_args()
    if args.plot_from:
        plot([(t, pd.read_csv(f)) for t, f in (x.split("=", 1) for x in args.plot_from)], Path(args.out))
        return
    if not args.run_dir:
        p.error("run_dir is required unless --plot_from is given")
    preds_path, out_dir = resolve_predictions_csv(Path(args.run_dir).resolve(), args.dataset)
    preds = pd.read_csv(preds_path)[["ID_ind", "y", "pred"]].apply(pd.to_numeric, errors="coerce").dropna()
    demo_path = resolve_path(args.demo_csv, find_demo_csv)
    if demo_path is None:
        raise FileNotFoundError("No demographics csv found - pass --demo_csv.")
    demo = pd.read_csv(demo_path)
    df = preds.merge(demo, left_on="ID_ind", right_on="ID", how="inner")

    rows = []
    for name, y, pred in GROUPS:
        g = df[(df["y"] == y) & (df["pred"] == pred)]
        rows.append({"group": name, "n": len(g),
                     **{lab: round(pd.to_numeric(g[col], errors="coerce").median(), 2) for col, lab in REGIONS if col in g}})
    table = pd.DataFrame(rows)
    tp, fn = (df[(df["y"] == 1) & (df["pred"] == k)] for k in (1, 0))
    tests = []
    for col, lab in REGIONS:
        a, b = (pd.to_numeric(x[col], errors="coerce").dropna() for x in (fn, tp))
        tests.append({"region": lab, "p": stats.mannwhitneyu(a, b).pvalue if len(a) > 1 and len(b) > 1 else float("nan")})
    tests = pd.DataFrame(tests)
    tests["p_fdr"] = fdr_bh(tests["p"])
    test_row = {"group": "Model − / Read + vs Model + / Read +: FDR p", "n": ""}
    test_row.update({r["region"]: (f"{r['p_fdr']:.3f}" if pd.notna(r["p_fdr"]) else "") for _, r in tests.iterrows()})
    table = pd.concat([table, pd.DataFrame([test_row])], ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "regional_suvr_by_group.csv", index=False)
    df["group"] = None
    for name, y, pred in GROUPS:
        df.loc[(df["y"] == y) & (df["pred"] == pred), "group"] = name
    data = df[["group", *[c for c, _ in REGIONS]]].rename(columns=dict(REGIONS))
    data.to_csv(out_dir / "regional_suvr_by_group_data.csv", index=False)  # no IDs
    plot([("Held-out test" if args.dataset == "Gothenburg" else args.dataset.removesuffix("_unseen"), data)],
         out_dir / "regional_suvr_by_group.png")
    print(table.to_string(index=False))
    print(f"\nSaved {out_dir / 'regional_suvr_by_group.csv'}")


if __name__ == "__main__":
    main()
