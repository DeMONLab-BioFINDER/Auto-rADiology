#!/usr/bin/env python3
"""OHBM abstract Figure 1: agreement with visual reads, per cohort, and errors of the model vs the meta temporal
SUVR cut-off. Built from tables the analysis step already wrote (no images or demographics needed):

  A  Negative and positive agreement with the visual read (Wilson 95% CI) for every cohort of the held-out test
     set, and for the left-out cohort (e.g. OASIS-3), which was withheld from training.
     figures/performance_summary/per_cohort.csv and figures/performance_summary_<ext>/per_cohort.csv
  B  Disagreements with the visual read at the fixed cut-offs, for the model (0.5) and the meta temporal SUVR
     cut-off (CTRz 2 = 1.34), stacked: false positives (read - scans called +) and missed positive reads.
     Counts from the sensitivity / specificity in visual_read_model_vs_suvr_threshold.csv (test set:
     figures/, external set: validation/figures_<ext>/).

Colours follow the read: blue = read - scans (negative agreement, false positives), rose = read + scans (positive
agreement, missed reads). Same style as the other results_plotting figures; drawn large and shrunk to 1000 px
wide (OHBM limit), saved as <run>/figures/ohbm/figure_agreement.png with the numbers in figure_agreement_numbers.csv.

Usage:
    python results_plotting/ohbm_figure_agreement.py <final_run_dir> [--ext OASIS]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from PIL import Image

from plot_utils import CUSTOM_PALETTE

TITLE, LABEL, TICK, LEGEND = 12, 11, 9.5, 9.5  # as ohbm_figure1.py
WIDTH_PX = 1000
NEG, POS = CUSTOM_PALETTE[0], CUSTOM_PALETTE[1]  # read - (blue), read + (rose)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--ext", default="OASIS", help="Left-out cohort name as in the folder names (OASIS -> _oasis).")
    p.add_argument("--ext_label", default=None, help="Name shown in the figure (default: OASIS -> OASIS-3).")
    return p.parse_args()


def cohort_rows(run: Path, ext: str, ext_label: str) -> pd.DataFrame:
    test = pd.read_csv(run / "figures/performance_summary/per_cohort.csv")
    test = test[test["set"] == "test"].copy()
    test["label"] = test["cohort"]
    other = pd.read_csv(run / f"figures/performance_summary_{ext.lower()}/per_cohort.csv")
    other = other[other["set"] != "test"].copy()
    other["label"] = f"{ext_label} (unseen)"
    return pd.concat([test, other], ignore_index=True)


def counts(table: Path, set_name: str) -> list[dict]:
    t = pd.read_csv(table)
    t = t[(t["region"] == "MetaTemporal") & (t["group"] == "All")]
    rows = []
    for rule, name in [("Model", "Model"), ("CTRz 2", "SUVR cut-off")]:
        r = t[t["rule"] == rule].iloc[0]
        n_pos = int(r["n_read_positive"])
        n_neg = int(r["n"]) - n_pos
        rows.append({"set": set_name, "method": name, "n": int(r["n"]), "n_read_positive": n_pos,
                     "false_positives": int(round(n_neg * (1 - r["specificity"]))),
                     "missed": int(round(n_pos * (1 - r["sensitivity"]))),
                     "sensitivity": r["sensitivity"], "specificity": r["specificity"]})
    return rows


def panel_cohorts(ax, rows: pd.DataFrame):
    ext_row = len(rows) - 1
    # the unseen cohort sits a little apart, on a light band
    y = np.array([len(rows) - i + (0 if i < ext_row else -0.6) for i in range(len(rows))], dtype=float)
    ax.axhspan(y[ext_row] - 0.5, y[ext_row] + 0.5, color="0.93", zorder=0)
    for col, colour, dy, label in [("agreement read -", NEG, 0.14, "Read − scans (negative agreement)"),
                                   ("agreement read +", POS, -0.14, "Read + scans (positive agreement)")]:
        v = 100 * rows[col].to_numpy()
        lo = 100 * rows[f"{col}_lo"].to_numpy()
        hi = 100 * rows[f"{col}_hi"].to_numpy()
        ax.errorbar(v, y + dy, xerr=[v - lo, hi - v], fmt="o", color=colour, ms=6, capsize=2.5, lw=1.4,
                    label=label, zorder=3)
    labels = [f"{l}\nn = {n} ({p} read +)" for l, n, p in zip(rows["label"], rows["n"], rows["read +"])]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=TICK)
    for t in ax.get_yticklabels()[ext_row:]:
        t.set_fontweight("bold")
    ax.set_xlim(0, 104)
    ax.set_xticks(range(0, 101, 20))
    ax.tick_params(axis="x", labelsize=TICK)
    ax.set_xlabel("Agreement with visual read (%)", fontsize=LABEL)
    ax.set_title("Agreement per cohort", fontsize=TITLE, pad=10)
    ax.set_ylim(y[ext_row] - 0.6, y[0] + 0.9)
    ax.text(2, y[0] + 0.55, "Held-out test set", ha="left", va="center", fontsize=TICK, color="0.45", style="italic")
    ax.text(2, y[ext_row] + 0.38, "Withheld from training", ha="left", va="center", fontsize=TICK, color="0.45",
            style="italic")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=2, fontsize=LEGEND - 0.5, frameon=False,
              handletextpad=0.3, columnspacing=1.0)
    ax.grid(axis="y", visible=False)


def panel_errors(ax, c: pd.DataFrame, ext_label: str):
    sets = list(dict.fromkeys(c["set"]))
    width, gap = 0.34, 0.1
    xt, xl = [], []
    for i, s in enumerate(sets):
        for j, m in enumerate(["Model", "SUVR cut-off"]):
            r = c[(c["set"] == s) & (c["method"] == m)].iloc[0]
            x = i + (j - 0.5) * (width + gap)
            fp, fn = r["false_positives"], r["missed"]
            ax.bar(x, fp, width, color=NEG, edgecolor="white", linewidth=1.5, zorder=3)
            ax.bar(x, fn, width, bottom=fp, color=POS, edgecolor="white", linewidth=1.5, zorder=3)
            for val, base in [(fp, 0), (fn, fp)]:
                if val >= 3:
                    ax.text(x, base + val / 2, str(val), ha="center", va="center", fontsize=TICK, color="white",
                            fontweight="bold", zorder=4)
                elif val > 0:
                    ax.text(x + width / 2 + 0.03, base + val / 2, str(val), ha="left", va="center", fontsize=TICK,
                            color=NEG if base == 0 else POS, fontweight="bold", zorder=4)
            ax.text(x, fp + fn + 0.8, str(fp + fn), ha="center", va="bottom", fontsize=TICK, color="0.3", zorder=4)
            xt.append(x)
            xl.append("Model" if m == "Model" else "SUVR\ncut-off")
        n = c[c["set"] == s]["n"].iloc[0]
        ax.text(i, -0.135, f"{s}\nn = {n}", transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=TICK + 0.5, linespacing=1.15, fontweight="bold" if s != "Held-out test" else "normal")
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=TICK)
    ax.set_xlim(-0.6, len(sets) - 0.4)
    ax.set_ylim(0, (c["false_positives"] + c["missed"]).max() * 1.3)
    ax.tick_params(axis="y", labelsize=TICK)
    ax.set_ylabel("Disagreements with visual read (scans)", fontsize=LABEL)
    ax.set_title("Model vs meta-temporal SUVR cut-off", fontsize=TITLE, pad=10)
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[Patch(color=NEG, label="False positives (read −)"), Patch(color=POS, label="Missed (read +)")],
              loc="upper left", fontsize=LEGEND, frameon=True, fancybox=True, framealpha=0.95, edgecolor="0.75")


def main():
    args = parse_args()
    run = Path(args.run_dir)
    ext_label = args.ext_label or {"OASIS": "OASIS-3"}.get(args.ext, args.ext)
    out = run / "figures" / "ohbm"
    out.mkdir(parents=True, exist_ok=True)

    rows = cohort_rows(run, args.ext, ext_label)
    c = pd.DataFrame(counts(run / "figures/visual_read_model_vs_suvr_threshold.csv", "Held-out test") +
                     counts(run / f"validation/figures_{args.ext.lower()}/visual_read_model_vs_suvr_threshold.csv",
                            ext_label))

    sns.set_theme(style="whitegrid", palette=CUSTOM_PALETTE)
    fig = plt.figure(figsize=(10, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.32, left=0.2, right=0.98, top=0.91, bottom=0.21)
    axa, axb = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    panel_cohorts(axa, rows)
    panel_errors(axb, c, ext_label)
    for ax, letter, dx in [(axa, "A", -122), (axb, "B", -46)]:
        ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(dx, 8), textcoords="offset points",
                    ha="right", va="bottom", fontsize=LABEL + 2, fontweight="bold")

    big = out / "_figure_agreement_large.png"
    fig.savefig(big, dpi=300, facecolor="white")
    plt.close(fig)
    img = Image.open(big)
    img = img.resize((WIDTH_PX, round(img.height * WIDTH_PX / img.width)), Image.LANCZOS)
    img.save(out / "figure_agreement.png", optimize=True)
    big.unlink()

    c.to_csv(out / "figure_agreement_numbers.csv", index=False)
    with pd.option_context("display.width", 160):
        print(rows[["label", "n", "read +", "agreement read -", "agreement read +"]].to_string(index=False))
        print(c.to_string(index=False))
    print(f"Saved {out / 'figure_agreement.png'} ({img.width} x {img.height} px)")


if __name__ == "__main__":
    main()
