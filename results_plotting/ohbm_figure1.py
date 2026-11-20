#!/usr/bin/env python3
"""OHBM abstract, Figure 1: how well the model reproduces the visual read, and where it differs from the
meta-temporal SUVR cut-off.

  A  ROC curve of the model on the held-out test set, with the operating point at 0.5 (AVID AUC in the label).
  B  Confusion matrices at 0.5, test and AVID.
  C, D  Agreement by meta-temporal SUVR level, test (C) and AVID (D): three lines, as in the lab's amyloid
     visual-read figure (Vogel lab, PALS application Fig. 1C):
        model vs read          (blue)
        cut-off vs read        (orange; cut-off = CenTauR CTRz 2 = meta-temporal SUVR 1.34)
        model vs cut-off       (grey, dashed)
     Where the tau level is clear, all three agree. Around the cut-off, does the model side with the read or with
     the SUVR number? Wilson 95% CIs on the two lines that compare with the read.
     Bins: < 1.2, 1.2-1.3, 1.3-1.4 (contains the cut-off), 1.4-1.6, >= 1.6. Chosen so every bin has enough
     scans in both sets (the top bins would hold only a handful each) while keeping the two bins around the
     cut-off separate. The test of model vs cut-off is the pre-specified one, on the 1.3-1.5 range (exact
     McNemar), printed in the panel and saved in figure1_numbers.csv.

OHBM limits: at most 1000 px per side and 2 MB. Drawn at 3x and shrunk to 1000 px wide with a Lanczos filter,
so lines and text stay sharp.

Usage:
    python results_plotting/ohbm_figure1.py <final_run_dir> --demo_csv ../data/demo.csv \\
        --ext_demo_csv ../data/demo_AVID_unseen.csv
Outputs (<run_dir>/figures/ohbm/, no IDs): figure1.png, figure1_numbers.csv
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.stats.contingency_tables import mcnemar

from plot_utils import CENTAUR_FTP_CTRZ2_SUVR, find_demo_csv, resolve_path, wilson_interval
from regress_model_output_visual_read import find_ext_demo

CUTOFF = CENTAUR_FTP_CTRZ2_SUVR["MetaTemporal"]
BINS = [(-np.inf, 1.2, "< 1.2"), (1.2, 1.3, "1.2–1.3"), (1.3, 1.4, "1.3–1.4"), (1.4, 1.6, "1.4–1.6"),
        (1.6, np.inf, "≥ 1.6")]
BORDERLINE = (1.3, 1.5)  # pre-specified range for the McNemar test
SETS = {"test": "Held-out test", "avid": "External (AVID)"}
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8a8a"  # validated pair (dataviz validator) + neutral grey
INK, MUTED = "#222222", "#666666"
WIDTH_PX, SCALE = 1000, 3


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--dataset", default="Gothenburg")
    p.add_argument("--ext_dataset", default="AVID_unseen")
    p.add_argument("--demo_csv", default=None)
    p.add_argument("--ext_demo_csv", default=None)
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


def load(run: Path, which: str, args) -> pd.DataFrame:
    if which == "test":
        f = run / "evaluation" / args.dataset / f"Eval_{args.dataset}_results.csv"
        demo_path = resolve_path(args.demo_csv, find_demo_csv)
    else:
        hits = sorted((run / "validation").glob(f"External_validation_{args.ext_dataset}__*_zeroshot_results.csv"))
        f = hits[0] if hits else run / "validation" / "missing"
        demo_path = resolve_path(args.ext_demo_csv, lambda: find_ext_demo(args.ext_dataset))
    if demo_path is None or not f.exists():
        raise SystemExit(f"Missing predictions ({f}) or demographics for the {which} set")
    d = pd.read_csv(f)[["ID_ind", "y", "pred", "prob"]]
    demo = pd.read_csv(demo_path)[["ID", "MetaTemporal"]]
    d = d.merge(demo, left_on="ID_ind", right_on="ID", how="left")
    d["MetaTemporal"] = pd.to_numeric(d["MetaTemporal"], errors="coerce")
    d = d.dropna(subset=["MetaTemporal"])
    d["rule"] = (d["MetaTemporal"] > CUTOFF).astype(int)
    return d


def mcnemar_p(b: pd.DataFrame) -> float:
    """Exact McNemar test, model vs cut-off on the same scans (only scans where exactly one agrees with the read count)."""
    m, r = b["pred"] == b["y"], b["rule"] == b["y"]
    table = [[int((m & r).sum()), int((m & ~r).sum())], [int((~m & r).sum()), int((~m & ~r).sum())]]
    return 1.0 if table[0][1] + table[1][0] == 0 else float(mcnemar(table, exact=True).pvalue)


def agreement(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lo, hi, label in BINS:
        b = d[(d["MetaTemporal"] >= lo) & (d["MetaTemporal"] < hi)]
        for pair, a, c in [("model vs read", b["pred"], b["y"]), ("cut-off vs read", b["rule"], b["y"]),
                           ("model vs cut-off", b["pred"], b["rule"])]:
            k, n = int((a == c).sum()), len(b)
            lo_ci, hi_ci = wilson_interval(k, n)
            rows.append({"bin": label, "pair": pair, "n": n, "n_read_pos": int(b["y"].sum()),
                         "agree": k / n if n else np.nan, "lo": lo_ci, "hi": hi_ci})
    return pd.DataFrame(rows)


def panel_roc(ax, data: dict):
    d = data["test"]
    fpr, tpr, _ = roc_curve(d["y"], d["prob"])
    ax.plot([0, 1], [0, 1], color="#d0d0d0", lw=1, ls=(0, (2, 2)))
    ax.plot(fpr, tpr, color=BLUE, lw=2.2, solid_capstyle="round")
    tn = ((d["pred"] == 0) & (d["y"] == 0)).sum(); fp = ((d["pred"] == 1) & (d["y"] == 0)).sum()
    tp = ((d["pred"] == 1) & (d["y"] == 1)).sum(); fn = ((d["pred"] == 0) & (d["y"] == 1)).sum()
    ax.plot(fp / (fp + tn), tp / (tp + fn), "o", ms=8, color=BLUE, mec="white", mew=2, zorder=3)
    ax.annotate("cut-off 0.5", (fp / (fp + tn), tp / (tp + fn)), xytext=(14, -16), textcoords="offset points",
                fontsize=9, color=MUTED)
    auc_t = roc_auc_score(d["y"], d["prob"])
    auc_a = roc_auc_score(data["avid"]["y"], data["avid"]["prob"])
    ax.text(0.97, 0.06, f"AUC {auc_t:.3f} (test)\nAUC {auc_a:.3f} (AVID, external)", ha="right", va="bottom",
            fontsize=10, color=INK, transform=ax.transAxes, linespacing=1.5)
    ax.set(xlim=(-0.02, 1), ylim=(0, 1.02), xlabel="False positive rate (1 − specificity)", ylabel="Sensitivity")
    ax.set_title("A   ROC, held-out test", loc="left", fontweight="bold")


def panel_confusion(ax, d: pd.DataFrame, title: str):
    m = pd.crosstab(d["y"], d["pred"]).reindex(index=[0, 1], columns=[0, 1], fill_value=0).to_numpy()
    agree = np.array([[1, 0], [0, 1]], bool)
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, color="#e3eefb" if agree[i, j] else "#fbe7df",
                                       ec="none"))
            ax.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=15, color=INK, fontweight="bold")
    ax.set(xlim=(-0.5, 1.5), ylim=(1.5, -0.5))
    ax.set_xticks([0, 1], ["Model −", "Model +"])
    ax.set_yticks([0, 1], ["Read −", "Read +"])
    ax.tick_params(length=0)
    ax.set_title(title, fontsize=10.5, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)


def panel_agreement(ax, rows: pd.DataFrame, d: pd.DataFrame, title: str):
    x = np.arange(len(BINS))
    ax.axvspan(1.5, 3.5, color="#f4f4f2", zorder=0)  # the two bins around the cut-off
    ax.text(2.5, 101.5, "around the cut-off", ha="center", va="bottom", fontsize=8.5, color=MUTED)
    for pair, colour, ls, ci, dx in [("model vs cut-off", GREY, (0, (4, 2.5)), False, 0),
                                     ("cut-off vs read", ORANGE, "-", True, 0.07), ("model vs read", BLUE, "-", True, -0.07)]:
        r = rows[rows["pair"] == pair].set_index("bin").reindex([b[2] for b in BINS])
        y = 100 * r["agree"].to_numpy()
        if ci:  # CI whiskers nudged sideways so the two read lines' intervals don't sit on top of each other
            ax.vlines(x + dx, 100 * r["lo"], 100 * r["hi"], color=colour, lw=1.4, alpha=0.5, zorder=2)
        ax.plot(x + dx, y, color=colour, lw=2, ls=ls, zorder=3, solid_capstyle="round")
        ax.plot(x + dx, y, "o", ms=6.5, color=colour, mec="white", mew=1.5, zorder=4)
    n = rows[rows["pair"] == "model vs read"].set_index("bin").reindex([b[2] for b in BINS])["n"]
    ax.set_xticks(x, [f"{b}\n{int(k)}" for (_, _, b), k in zip(BINS, n)])
    ax.set(xlim=(-0.4, len(BINS) - 0.6), xlabel="Meta-temporal SUVR  (scans per bin below)")
    b = d[(d["MetaTemporal"] >= BORDERLINE[0]) & (d["MetaTemporal"] < BORDERLINE[1])]
    am, ar, p = (b["pred"] == b["y"]).mean(), (b["rule"] == b["y"]).mean(), mcnemar_p(b)
    ax.text(0.02, 0.04, f"SUVR {BORDERLINE[0]}–{BORDERLINE[1]} (n = {len(b)}): model {am:.0%} vs cut-off {ar:.0%}, "
            f"p = {p:.2g}", transform=ax.transAxes, fontsize=8.5, color=MUTED)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.yaxis.grid(True, color="#ececec", lw=0.8)
    ax.set_axisbelow(True)
    return {"range": f"{BORDERLINE[0]}-{BORDERLINE[1]}", "n": len(b), "model": am, "cutoff": ar, "mcnemar_p": p}


def main():
    args = parse_args()
    run = Path(args.run_dir).resolve()
    out = Path(args.out_dir) if args.out_dir else run / "figures" / "ohbm"
    out.mkdir(parents=True, exist_ok=True)
    data = {which: load(run, which, args) for which in SETS}

    plt.rcParams.update({"font.family": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 10, "axes.titlesize": 11.5, "axes.labelsize": 10, "axes.labelcolor": INK,
                         "xtick.labelsize": 9, "ytick.labelsize": 9, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.edgecolor": "#bbbbbb", "axes.linewidth": 0.8})
    fig = plt.figure(figsize=(10, 7.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.05], hspace=0.55, left=0.07, right=0.985, top=0.93, bottom=0.1)
    top = gs[0].subgridspec(1, 3, width_ratios=[1.25, 0.62, 0.62], wspace=0.45)
    bottom = gs[1].subgridspec(1, 2, width_ratios=[1, 1.18], wspace=0.12)

    panel_roc(fig.add_subplot(top[0]), data)
    for k, which in enumerate(SETS):
        ax = fig.add_subplot(top[k + 1])
        panel_confusion(ax, data[which], f"{'Test' if which == 'test' else 'AVID'} (n = {len(data[which])})")
        pos = ax.get_position()  # a little lower, so the matrix titles clear the panel label
        ax.set_position([pos.x0, pos.y0 - 0.01, pos.width, pos.height * 0.88])
        if k == 0:
            pos = ax.get_position()
            fig.text(pos.x0 - 0.06, 0.94, "B   At cut-off 0.5", fontweight="bold", fontsize=11.5, va="bottom")
    tests, numbers, bottom_axes = [], [], []
    for k, (which, letter) in enumerate(zip(SETS, "CD")):
        ax = fig.add_subplot(bottom[k])
        rows = agreement(data[which])
        tests.append({"set": which, **panel_agreement(ax, rows, data[which], f"{letter}   Agreement by tau level, "
                                                      f"{'test' if which == 'test' else 'AVID'}")})
        numbers.append(rows.assign(set=which))
        bottom_axes.append(ax)
    # one y range for both panels, from the lowest CI rounded down to 10
    low = min(100 * pd.concat(numbers)[["lo", "agree"]].min().min(), 90)
    for k, ax in enumerate(bottom_axes):
        ax.set_ylim(np.floor(low / 10) * 10 - 2, 106)
        ax.set_yticks(np.arange(np.ceil(low / 20) * 20, 101, 20))
        if k == 0:
            ax.set_ylabel("Agreement (%)")
        else:
            ax.set_yticklabels([])
    handles = [plt.Line2D([], [], color=c, lw=2, ls=ls, marker="o", ms=6, mec="white", mew=1.2)
               for c, ls in [(BLUE, "-"), (ORANGE, "-"), (GREY, (0, (4, 2.5)))]]
    fig.legend(handles, ["Model vs read", f"Cut-off (SUVR > {CUTOFF:.2f}, CTRz 2) vs read", "Model vs cut-off"],
               loc="center", bbox_to_anchor=(0.53, 0.505), ncol=3, frameon=False, fontsize=9.5, handlelength=2.4)
    for ax in fig.axes:
        ax.spines[["top", "right"]].set_visible(False)

    big = out / "_figure1_3x.png"
    fig.savefig(big, dpi=100 * SCALE, facecolor="white")
    plt.close(fig)
    img = Image.open(big)
    img = img.resize((WIDTH_PX, round(img.height * WIDTH_PX / img.width)), Image.LANCZOS)
    img.save(out / "figure1.png", optimize=True)
    big.unlink()

    pd.concat(numbers).to_csv(out / "figure1_numbers.csv", index=False)
    pd.DataFrame(tests).to_csv(out / "figure1_borderline_test.csv", index=False)
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print(pd.concat(numbers).to_string(index=False))
        print("\nPre-specified borderline test (exact McNemar):")
        print(pd.DataFrame(tests).to_string(index=False))
    size = (out / "figure1.png").stat().st_size / 1e6
    print(f"\nSaved {out / 'figure1.png'} ({img.width} x {img.height} px, {size:.2f} MB)")


if __name__ == "__main__":
    main()
