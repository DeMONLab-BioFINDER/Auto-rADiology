#!/usr/bin/env python3
"""OHBM abstract, Figure 1: how well the model reproduces the visual read, and where it differs from the
meta-temporal SUVR cut-off.

Held-out test set only by default (AVID optional, --sets test avid, until it is clear AVID can go in the abstract).
Same style as the other results_plotting figures (seaborn whitegrid, CUSTOM_PALETTE, the ROC / confusion-matrix
panel and the agreement-vs-cut-off plot with its checkmark legend), at panel-sized fonts.
  A  ROC curve of the model on the held-out test set.
  B  Confusion matrix (the model's calls).
  C  Agreement by meta-temporal SUVR level (with AVID: C test, D AVID): three lines, as in the lab's amyloid
     visual-read figure (Vogel lab, PALS application Fig. 1C): model vs cut-off, read vs cut-off, model vs read
     (cut-off = CenTauR CTRz 2 = meta-temporal SUVR 1.34), over grey bars of the share of scans per bin.
     Bins: < 1.2, 1.2-1.3, 1.3-1.4 (contains the cut-off), 1.4-1.6, >= 1.6. Chosen so every bin has enough
     scans in both sets (the top bins would hold only a handful each) while keeping the two bins around the
     cut-off separate. The test of model vs cut-off is the pre-specified one, on the 1.3-1.5 range (exact
     McNemar), saved in figure1_borderline_test.csv (for the text, not drawn).

OHBM limits: at most 1000 px per side and 2 MB. Drawn large and shrunk to 1000 px wide with a Lanczos filter,
so lines and text stay sharp.

Usage:
    python results_plotting/ohbm_figure1.py <final_run_dir> --demo_csv ../data/demo.csv \\
        --ext_demo_csv ../data/demo_AVID_unseen.csv
    (add --sets test avid for the version with AVID: figure1_with_avid.png)
Outputs (<run_dir>/figures/ohbm/, no IDs): figure1.png, figure1_numbers.csv, figure1_borderline_test.csv
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.stats.contingency_tables import mcnemar

from plot_utils import CENTAUR_FTP_CTRZ2_SUVR, CUSTOM_PALETTE, find_demo_csv, resolve_path, wilson_interval
from regress_model_output_visual_read import find_ext_demo

CUTOFF = CENTAUR_FTP_CTRZ2_SUVR["MetaTemporal"]
BINS = [(-np.inf, 1.2, "< 1.2"), (1.2, 1.3, "1.2–1.3"), (1.3, 1.4, "1.3–1.4"), (1.4, 1.6, "1.4–1.6"),
        (1.6, np.inf, "≥ 1.6")]
BORDERLINE = (1.3, 1.5)  # pre-specified range for the McNemar test
SETS = {"test": "Held-out test", "avid": "External (AVID)"}
TITLE, LABEL, TICK, LEGEND = 12, 11, 9.5, 9.5  # the plot_utils sizes, scaled down to fit three panels
WIDTH_PX = 1000


def style(ax, title: str, xlabel: str, ylabel: str):
    """plot_utils.style_axes at panel-sized fonts."""
    ax.set_title(title, fontsize=TITLE, pad=10)
    ax.set_xlabel(xlabel, fontsize=LABEL)
    ax.set_ylabel(ylabel, fontsize=LABEL)
    ax.tick_params(labelsize=TICK)
    ax.grid(alpha=0.28)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--dataset", default="Gothenburg")
    p.add_argument("--ext_dataset", default="AVID_unseen")
    p.add_argument("--demo_csv", default=None)
    p.add_argument("--ext_demo_csv", default=None)
    p.add_argument("--sets", nargs="+", default=["test"], choices=list(SETS),
                   help="test only (default; figure1.png) or test avid (figure1_with_avid.png).")
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
    label = f"AUC = {roc_auc_score(d['y'], d['prob']):.3f}"
    if "avid" in data:
        label += f" (test)\nAUC = {roc_auc_score(data['avid']['y'], data['avid']['prob']):.3f} (AVID)"
    ax.plot(fpr, tpr, color=CUSTOM_PALETTE[0], linewidth=2.2, label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.2, alpha=0.6)
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    style(ax, "ROC curve", "False positive rate", "True positive rate")
    ax.legend(loc="lower right", fontsize=LEGEND, frameon=True, fancybox=True, framealpha=0.9, edgecolor="0.75")


def panel_confusion(ax, d: pd.DataFrame, title: str):
    cm = pd.crosstab(d["y"], d["pred"]).reindex(index=[0, 1], columns=[0, 1], fill_value=0).to_numpy()
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Negative", "Positive"])
    ax.set_yticks([0, 1], ["Negative", "Positive"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=LABEL + 2,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    style(ax, title, "Predicted (model)", "Reference (clinical read)")
    ax.grid(False)


def checkmark_table(ax):
    """The checkmark legend of the agreement-vs-cut-off plots (plot_utils._draw_cutoff_agreement_table),
    drawn inside the panel (axes coordinates) so it fits in a multi-panel figure."""
    rows = [(CUSTOM_PALETTE[5], (True, True, False)), (CUSTOM_PALETTE[1], (False, True, True)),
            (CUSTOM_PALETTE[3], (True, False, True))]
    x0, y0, w, h = 0.5, 0.04, 0.48, 0.4
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.02", transform=ax.transAxes,
                                facecolor="white", edgecolor="0.75", lw=1, zorder=10, clip_on=False))
    slot = h / 5
    yc = lambda k: y0 + h - slot * (k + 0.5)
    cols = [x0 + w * f for f in (0.38, 0.62, 0.87)]
    ax.text(x0 + w / 2, yc(0), "Agreement between", transform=ax.transAxes, ha="center", va="center",
            fontsize=TICK, fontweight="bold", zorder=11)
    for cx, lab in zip(cols, ["Model", "SUVR", "Human"]):
        ax.text(cx, yc(1), lab, transform=ax.transAxes, ha="center", va="center", fontsize=TICK - 1.5, zorder=11)
    for k in range(1, 4):
        yb = y0 + h - slot * (k + 1)
        ax.add_line(Line2D([x0 + 0.01, x0 + w - 0.01], [yb + slot * 0.0, yb], transform=ax.transAxes, color="0.8",
                           lw=0.9, zorder=11)) if k < 4 else None
    for (colour, flags), k in zip(rows, (2, 3, 4)):
        y = yc(k)
        ax.add_line(Line2D([x0 + 0.03, x0 + 0.13], [y, y], transform=ax.transAxes, color=colour, lw=2.2, zorder=11))
        ax.add_line(Line2D([x0 + 0.08], [y], transform=ax.transAxes, color=colour, marker="o", ms=5, ls="None", zorder=11))
        for cx, flag in zip(cols, flags):
            if flag:
                sx, sy = 0.012, 0.022
                ax.add_line(Line2D([cx - sx, cx - sx * 0.25, cx + sx * 1.2], [y, y - sy * 0.9, y + sy],
                                   transform=ax.transAxes, color="#2E8B45", lw=2, solid_capstyle="round", zorder=11))


def panel_agreement(ax, rows: pd.DataFrame, d: pd.DataFrame, title: str, legend: bool = True):
    labels = [b[2] for b in BINS]
    x = np.arange(len(BINS))
    n = rows[rows["pair"] == "model vs read"].set_index("bin").reindex(labels)["n"]
    ax2 = ax.twinx()  # background: share of scans per bin, as in the original agreement-vs-cut-off plots
    ax2.bar(x, (n / n.sum() * 100).to_numpy(), color="0.85", width=0.65, zorder=1)
    ax2.set_ylabel("Percentage of scans (%)", fontsize=LABEL, color="0.6")
    ax2.tick_params(axis="y", labelsize=TICK, color="0.6", labelcolor="0.6")
    ax2.spines["right"].set_color("0.6")
    ax2.grid(False)
    for pair, colour in [("model vs cut-off", CUSTOM_PALETTE[5]), ("cut-off vs read", CUSTOM_PALETTE[1]),
                         ("model vs read", CUSTOM_PALETTE[3])]:
        r = rows[rows["pair"] == pair].set_index("bin").reindex(labels)
        ax.plot(x, 100 * r["agree"].to_numpy(), marker="o", markersize=6, linewidth=2.2, color=colour, zorder=3)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 105)
    b = d[(d["MetaTemporal"] >= BORDERLINE[0]) & (d["MetaTemporal"] < BORDERLINE[1])]
    am, ar, p = (b["pred"] == b["y"]).mean(), (b["rule"] == b["y"]).mean(), mcnemar_p(b)
    style(ax, title, f"Meta-temporal tau SUVR (cut-off {CUTOFF:.2f} = CTRz 2)", "Agreement (%)")
    if legend:
        checkmark_table(ax)
    return {"range": f"{BORDERLINE[0]}-{BORDERLINE[1]}", "n": len(b), "model": am, "cutoff": ar, "mcnemar_p": p}


def main():
    args = parse_args()
    run = Path(args.run_dir).resolve()
    out = Path(args.out_dir) if args.out_dir else run / "figures" / "ohbm"
    out.mkdir(parents=True, exist_ok=True)
    data = {which: load(run, which, args) for which in SETS if which in args.sets}
    name = "figure1.png" if list(data) == ["test"] else "figure1_with_avid.png"

    sns.set_theme(style="whitegrid", palette=CUSTOM_PALETTE)  # the style of the other results_plotting figures
    if list(data) == ["test"]:  # one row: ROC, confusion matrix, agreement
        fig = plt.figure(figsize=(10.5, 4.2))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.9, 1.6], wspace=0.45, left=0.065, right=0.935,
                              top=0.88, bottom=0.15)
        top, bottom = [gs[0], gs[1]], [gs[2]]
    else:
        fig = plt.figure(figsize=(10.5, 8.4))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.38, left=0.05, right=0.955, top=0.94, bottom=0.08)
        top = gs[0].subgridspec(1, 3, width_ratios=[1.2, 0.85, 0.85], wspace=0.4)
        bottom = gs[1].subgridspec(1, 2, wspace=0.35)

    panel_roc(fig.add_subplot(top[0]), data)
    for k, which in enumerate(data):
        title = "Confusion matrix" + ("" if len(data) == 1 else f", {'test' if which == 'test' else 'AVID'}")
        panel_confusion(fig.add_subplot(top[k + 1]), data[which], title)
    tests, numbers = [], []
    for k, which in enumerate(data):
        ax = fig.add_subplot(bottom[k])
        rows = agreement(data[which])
        title = "Agreement with SUVR cut-off" + ("" if len(data) == 1 else f", {'test' if which == 'test' else 'AVID'}")
        tests.append({"set": which, **panel_agreement(ax, rows, data[which], title, legend=(k == len(data) - 1))})
        numbers.append(rows.assign(set=which))
    axes = [a for a in fig.axes if a.get_label() != "<colorbar>"]
    for ax, letter in zip([a for a in axes if not a.get_ylabel().startswith("Percentage")], "ABCDE"):
        ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-14, 8), textcoords="offset points",
                    ha="right", va="bottom", fontsize=LABEL + 2, fontweight="bold")

    big = out / "_figure1_large.png"
    fig.savefig(big, dpi=300, facecolor="white")
    plt.close(fig)
    img = Image.open(big)
    img = img.resize((WIDTH_PX, round(img.height * WIDTH_PX / img.width)), Image.LANCZOS)
    img.save(out / name, optimize=True)
    big.unlink()

    pd.concat(numbers).to_csv(out / "figure1_numbers.csv", index=False)
    pd.DataFrame(tests).to_csv(out / "figure1_borderline_test.csv", index=False)
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print(pd.concat(numbers).to_string(index=False))
        print("\nPre-specified borderline test (exact McNemar):")
        print(pd.DataFrame(tests).to_string(index=False))
    print(f"\nSaved {out / name} ({img.width} x {img.height} px, {(out / name).stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
