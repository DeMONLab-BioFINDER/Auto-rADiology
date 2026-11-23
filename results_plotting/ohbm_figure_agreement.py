#!/usr/bin/env python3
"""OHBM abstract Figure 1: the model and the meta temporal SUVR cut-off against the visual read, as plain
confusion matrices (counts), for the held-out test set and the left-out cohort (e.g. OASIS-3).

  rows     held-out test set, left-out cohort (withheld from training)
  columns  model (threshold 0.5), meta temporal SUVR cut-off (CTRz 2 = 1.34)

Counts come from the sensitivity / specificity in visual_read_model_vs_suvr_threshold.csv (test set: figures/,
external set: validation/figures_<ext>/), which the analysis step already wrote: no images or demographics needed.
Same style as ohbm_figure1.py's confusion matrix; drawn large and shrunk to 1000 px wide (OHBM limit). Saves
<run>/figures/ohbm/figure_agreement.png and figure_agreement_numbers.csv.

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
from PIL import Image

from plot_utils import CUSTOM_PALETTE

TITLE, LABEL, TICK = 12, 11, 10
WIDTH_PX = 1000


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--ext", default="OASIS", help="Left-out cohort as in the folder names (OASIS -> _oasis).")
    p.add_argument("--ext_label", default=None, help="Name shown in the figure (default: OASIS -> OASIS-3).")
    return p.parse_args()


def matrices(table: Path) -> dict:
    """{method: 2x2 counts [[read - called -, read - called +], [read + called -, read + called +]]}."""
    t = pd.read_csv(table)
    t = t[(t["region"] == "MetaTemporal") & (t["group"] == "All")]
    out = {}
    for rule, name in [("Model", "Model"), ("CTRz 2", "Meta-temporal SUVR")]:
        r = t[t["rule"] == rule].iloc[0]
        n_pos = int(r["n_read_positive"])
        n_neg = int(r["n"]) - n_pos
        fp = int(round(n_neg * (1 - r["specificity"])))
        fn = int(round(n_pos * (1 - r["sensitivity"])))
        out[name] = np.array([[n_neg - fp, fp], [fn, n_pos - fn]])
    return out


def panel(ax, cm: np.ndarray, title: str, xlabel: str, show_y: bool):
    shade = cm / cm.sum(axis=1, keepdims=True)  # colour by row share, so both rows are readable
    ax.imshow(shade, cmap="Blues", vmin=0, vmax=1.15)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=LABEL + 4,
                    color="white" if shade[i, j] > 0.6 else "0.15", fontweight="bold" if i != j else "normal")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Negative", "Positive"], fontsize=TICK)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Negative", "Positive"] if show_y else [], fontsize=TICK)
    ax.set_xlabel(xlabel, fontsize=LABEL)
    if show_y:
        ax.set_ylabel("Visual read", fontsize=LABEL)
    ax.set_title(title, fontsize=TITLE, pad=8)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    args = parse_args()
    run = Path(args.run_dir)
    ext_label = args.ext_label or {"OASIS": "OASIS-3"}.get(args.ext, args.ext)
    out = run / "figures" / "ohbm"
    out.mkdir(parents=True, exist_ok=True)
    sets = [("Held-out test set", run / "figures/visual_read_model_vs_suvr_threshold.csv"),
            (f"{ext_label} (unseen)", run / f"validation/figures_{args.ext.lower()}/visual_read_model_vs_suvr_threshold.csv")]

    sns.set_theme(style="white", palette=CUSTOM_PALETTE)
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.6))
    fig.subplots_adjust(left=0.27, right=0.97, top=0.9, bottom=0.09, wspace=0.12, hspace=0.5)
    rows = []
    for i, (set_name, table) in enumerate(sets):
        m = matrices(table)
        n = int(next(iter(m.values())).sum())
        for j, (method, cm) in enumerate(m.items()):
            xlabel = "Model" if method == "Model" else "Meta-temporal SUVR (cut-off 1.34)"
            panel(axes[i, j], cm, method if i == 0 else "", xlabel, show_y=(j == 0))
            rows.append({"set": set_name, "method": method, "read-_called-": cm[0, 0], "read-_called+": cm[0, 1],
                         "read+_called-": cm[1, 0], "read+_called+": cm[1, 1]})
        axes[i, 0].text(-0.62, 0.5, f"{set_name}\nn = {n}", transform=axes[i, 0].transAxes, ha="center",
                        va="center", rotation=90, fontsize=LABEL, fontweight="bold")

    big = out / "_figure_agreement_large.png"
    fig.savefig(big, dpi=300, facecolor="white")
    plt.close(fig)
    img = Image.open(big)
    img = img.resize((WIDTH_PX, round(img.height * WIDTH_PX / img.width)), Image.LANCZOS)
    img.save(out / "figure_agreement.png", optimize=True)
    big.unlink()
    pd.DataFrame(rows).to_csv(out / "figure_agreement_numbers.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved {out / 'figure_agreement.png'} ({img.width} x {img.height} px)")


if __name__ == "__main__":
    main()
