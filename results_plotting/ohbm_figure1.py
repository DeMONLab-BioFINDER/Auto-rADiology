#!/usr/bin/env python3
"""OHBM abstract, Figure 1: how well the model reproduces the visual read, and where it differs from the
meta-temporal SUVR cut-off.

Held-out test set only by default (AVID optional, --sets test avid, until it is clear AVID can go in the abstract).
Same style as the other results_plotting figures (seaborn whitegrid, CUSTOM_PALETTE, the ROC / confusion-matrix
panel and the agreement-vs-cut-off plot with its checkmark legend), at panel-sized fonts.
Laid out like that figure: A confusion matrix (the model's calls), B ROC curves of the model and of the region's
SUVR (threshold-free comparison), C agreement panel underneath.
  C  Agreement by meta-temporal SUVR level (with AVID: C test, D AVID): three lines, as in the lab's amyloid
     visual-read figure (Vogel lab, PALS application Fig. 1C): model vs cut-off, read vs cut-off, model vs read
     (cut-off = CenTauR CTRz 2 = meta-temporal SUVR 1.34), over grey bars of the share of scans per bin.
     Bins: < 1.2, 1.2-1.3, 1.3-1.4 (contains the cut-off), 1.4-1.6, >= 1.6, every one with >= 29 test scans
     (0.1-wide bins in the high-tau tail hold 15-25 scans, where one scan moves a point by 4-6 points). --region Universal (or another CenTauR
     region) draws the same figure with that region's SUVR and CTRz-2 cut-off. The test of model vs cut-off is the pre-specified one, on the 1.3-1.5 range (exact
     McNemar), saved in figure1_borderline_test.csv (for the text, not drawn).

OHBM limits: at most 1000 px per side and 2 MB. Drawn large and shrunk to 1000 px wide with a Lanczos filter,
so lines and text stay sharp.

Usage:
    python results_plotting/ohbm_figure1.py <final_run_dir> --demo_csv ../data/demo.csv \\
        --ext_demo_csv ../data/demo_AVID_unseen.csv
    (--sets avid: the same figure for the external set --ext_dataset, e.g. figure1_avid.png / figure1_oasis.png;
     --sets test avid: both, figure1_with_<ext>.png)
Outputs (<run_dir>/figures/ohbm/, no IDs): figure1[_avid].png, <figure>_numbers.csv, <figure>_borderline_test.csv
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

from plot_utils import CENTAUR_FTP_CTRZ2_SUVR, CUSTOM_PALETTE, external_name, find_demo_csv, resolve_path, wilson_interval
from regress_model_output_visual_read import find_ext_demo

# set in main() from --region: the SUVR column, its CTRz-2 cut-off and the bins (from the test set's distribution)
REGION, CUTOFF = "MetaTemporal", CENTAUR_FTP_CTRZ2_SUVR["MetaTemporal"]
# fixed bins, every one with >= 29 test scans; the cut-off bin (1.3-1.4) on its own
BINS = [(-np.inf, 1.2, "< 1.2"), (1.2, 1.3, "1.2–1.3"), (1.3, 1.4, "1.3–1.4"), (1.4, 1.6, "1.4–1.6"),
        (1.6, np.inf, "≥ 1.6")]
REGION_NAMES = {"MetaTemporal": "Meta-temporal", "Universal": "Universal", "TemporoParietal": "Temporoparietal",
                "MesialTemporal": "Mesial temporal", "Frontal": "Frontal"}
BORDERLINE = (1.3, 1.5)  # pre-specified meta-temporal range for the McNemar test (meta-temporal only)
SETS = {"test": "Held-out test", "avid": "External"}  # "avid" = the external set (--ext_dataset), whichever it is
EXT = "AVID"
TITLE, LABEL, TICK, LEGEND = 12, 11, 9.5, 9.5  # the plot_utils sizes, scaled down to fit three panels
WIDTH_PX = 1000


def cutoff_name() -> str:
    """One name for the SUVR cut-off everywhere in the figure (ROC legend, panel C legend). Just the region's SUVR:
    the x-axis of panel C gives the cut-off value (1.34 = CTRz 2)."""
    return f"{REGION_NAMES.get(REGION, REGION)} SUVR dichotomy"


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
                   help="test (default; figure1.png), avid (the same figure for AVID: figure1_avid.png), "
                        "or test avid (both: figure1_with_avid.png).")
    p.add_argument("--region", default="MetaTemporal", choices=list(REGION_NAMES),
                   help="SUVR region for panel C (default meta-temporal; e.g. Universal -> figure1_universal.png).")
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
    demo = pd.read_csv(demo_path)[["ID", REGION]].rename(columns={REGION: "suvr"})
    d = d.merge(demo, left_on="ID_ind", right_on="ID", how="left")
    d["suvr"] = pd.to_numeric(d["suvr"], errors="coerce")
    d = d.dropna(subset=["suvr"])
    d["rule"] = (d["suvr"] > CUTOFF).astype(int)
    return d


def mcnemar_p(b: pd.DataFrame) -> float:
    """Exact McNemar test, model vs cut-off on the same scans (only scans where exactly one agrees with the read count)."""
    m, r = b["pred"] == b["y"], b["rule"] == b["y"]
    table = [[int((m & r).sum()), int((m & ~r).sum())], [int((~m & r).sum()), int((~m & ~r).sum())]]
    return 1.0 if table[0][1] + table[1][0] == 0 else float(mcnemar(table, exact=True).pvalue)


def agreement(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lo, hi, label in BINS:
        b = d[(d["suvr"] >= lo) & (d["suvr"] < hi)]
        for pair, a, c in [("model vs read", b["pred"], b["y"]), ("cut-off vs read", b["rule"], b["y"]),
                           ("model vs cut-off", b["pred"], b["rule"])]:
            k, n = int((a == c).sum()), len(b)
            lo_ci, hi_ci = wilson_interval(k, n)
            rows.append({"bin": label, "pair": pair, "n": n, "n_read_pos": int(b["y"].sum()),
                         "agree": k / n if n else np.nan, "lo": lo_ci, "hi": hi_ci})
    return pd.DataFrame(rows)


def panel_roc(ax, data: dict):
    d = data[next(iter(data))]  # the first set (test, or AVID when drawn on its own)
    # the model, and classifying by a cut-off on the region's SUVR, swept over every cut-off value
    # (threshold-free comparison; pink like the cut-off line in C)
    for score, colour, name in [("prob", CUSTOM_PALETTE[0], "Model"),
                                ("suvr", CUSTOM_PALETTE[1], cutoff_name())]:
        fpr, tpr, _ = roc_curve(d["y"], d[score])
        ax.plot(fpr, tpr, color=colour, linewidth=2.2, label=f"{name} (AUC = {roc_auc_score(d['y'], d[score]):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.2, alpha=0.6)
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    style(ax, "ROC curves", "False positive rate", "True positive rate")
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
    style(ax, title, "Model", "Visual read")
    ax.grid(False)


def checkmark_table(ax):
    """Legend as in the lab's amyloid figure (PALS application Fig. 1C): one column per line (its swatch on top),
    one row per rater, a checkmark where that line compares that rater, dashed dividers between the columns.
    Drawn inside the panel, bottom left: at low tau every line sits near 100%, so nothing is covered."""
    lines = [CUSTOM_PALETTE[5], CUSTOM_PALETTE[1], CUSTOM_PALETTE[3]]  # model vs cut-off, read vs cut-off, model vs read
    raters = [("Model", (True, False, True)), (cutoff_name(), (True, True, False)), ("Visual read", (False, True, True))]
    x0, y0, w, h = 0.03, 0.05, 0.44, 0.44  # right edge stays left of the cut-off bin's points
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.015", transform=ax.transAxes,
                                facecolor="white", edgecolor="0.75", lw=1, zorder=10, clip_on=False))
    T = dict(transform=ax.transAxes, zorder=11)
    row_h = h / 5.2
    ax.text(x0 + 0.02, y0 + h - row_h * 0.6, "Agreement between", va="center", fontsize=TICK, **T)
    cols = [x0 + w * f for f in (0.68, 0.8, 0.92)]
    y_sw = y0 + h - row_h * 1.5
    for cx, colour in zip(cols, lines):
        ax.add_line(Line2D([cx - 0.022, cx + 0.022], [y_sw, y_sw], color=colour, lw=2, **T))
        ax.add_line(Line2D([cx], [y_sw], color=colour, marker="o", ms=4.5, ls="None", **T))
    for xd in [(cols[0] + cols[1]) / 2, (cols[1] + cols[2]) / 2]:  # dashed column dividers
        ax.add_line(Line2D([xd, xd], [y0 + row_h * 0.3, y_sw - row_h * 0.45], color="0.2", lw=0.9, ls=(0, (3, 2)), **T))
    for k, (name, flags) in enumerate(raters):
        y = y_sw - row_h * (k + 1.1)
        ax.text(x0 + 0.02, y, name, va="center", fontsize=TICK, **T)
        for cx, flag in zip(cols, flags):
            if flag:
                sx, sy = 0.008, 0.024  # same shape and colour as plot_utils._draw_checkmark
                ax.add_line(Line2D([cx - sx, cx - sx * 0.25, cx + sx * 1.2], [y, y - sy * 0.9, y + sy], color="#2E8B45",
                                   lw=2.0, solid_capstyle="round", solid_joinstyle="round", **T))


def panel_agreement(ax, rows: pd.DataFrame, d: pd.DataFrame, title: str, legend: bool = True):
    labels = [b[2] for b in BINS]
    x = np.arange(len(BINS))
    n = rows[rows["pair"] == "model vs read"].set_index("bin").reindex(labels)["n"]
    ax2 = ax.twinx()  # background: share of scans per bin, as in the original agreement-vs-cut-off plots
    ax2.bar(x, (n / n.sum() * 100).to_numpy(), color="0.85", width=0.65, zorder=1)
    ax2.set_ylim(0, max(50, float(np.ceil((n / n.sum() * 100).max() / 10) * 10)))  # fixed 0-50% unless a bin is bigger
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
    ax.set_ylim(0, 102)  # full 0-100% axis (not zoomed); the legend box sits bottom left, below every line
    b = d[(d["suvr"] >= BORDERLINE[0]) & (d["suvr"] < BORDERLINE[1])]
    am, ar, p = (b["pred"] == b["y"]).mean(), (b["rule"] == b["y"]).mean(), mcnemar_p(b)
    style(ax, title, f"{REGION_NAMES.get(REGION, REGION)} SUVR (cut-off {CUTOFF:.2f} = CTRz 2)", "Agreement (%)")
    if legend:
        checkmark_table(ax)
    if REGION != "MetaTemporal":
        return {}
    return {"range": f"{BORDERLINE[0]}-{BORDERLINE[1]}", "n": len(b), "model": am, "cutoff": ar, "mcnemar_p": p}


def main():
    args = parse_args()
    run = Path(args.run_dir).resolve()
    out = Path(args.out_dir) if args.out_dir else run / "figures" / "ohbm"
    out.mkdir(parents=True, exist_ok=True)
    global REGION, CUTOFF, EXT
    EXT = external_name(args.ext_dataset)  # AVID, OASIS, ...
    REGION, CUTOFF = args.region, CENTAUR_FTP_CTRZ2_SUVR[args.region]
    data = {which: load(run, which, args) for which in SETS if which in args.sets}
    suffix = "" if REGION == "MetaTemporal" else f"_{REGION.lower()}"
    ext = EXT.lower()
    name = {("test",): f"figure1{suffix}.png", ("avid",): f"figure1{suffix}_{ext}.png"}.get(
        tuple(data), f"figure1{suffix}_with_{ext}.png")

    sns.set_theme(style="whitegrid", palette=CUSTOM_PALETTE)  # the style of the other results_plotting figures
    # layout of the amyloid figure: confusion matrix and ROC on top, the agreement panel wide underneath
    if len(data) == 1:
        fig = plt.figure(figsize=(10, 7.6))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.05], hspace=0.42, left=0.1, right=0.92, top=0.95, bottom=0.08)
        top = gs[0].subgridspec(1, 2, width_ratios=[0.8, 1.2], wspace=0.35)
        bottom = [gs[1]]
    else:
        fig = plt.figure(figsize=(10, 8.4))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.38, left=0.06, right=0.94, top=0.94, bottom=0.08)
        top = gs[0].subgridspec(1, 3, width_ratios=[0.85, 0.85, 1.2], wspace=0.4)
        bottom = gs[1].subgridspec(1, 2, wspace=0.38)

    for k, which in enumerate(data):
        title = "Confusion matrix" + ("" if len(data) == 1 else f", {'test' if which == 'test' else EXT}")
        panel_confusion(fig.add_subplot(top[k]), data[which], title)
    panel_roc(fig.add_subplot(top[len(data)]), data)
    tests, numbers = [], []
    for k, which in enumerate(data):
        ax = fig.add_subplot(bottom[k])
        rows = agreement(data[which])
        title = f"Agreement by {REGION_NAMES.get(REGION, REGION).lower()} SUVR" + \
            ("" if len(data) == 1 else f", {'test' if which == 'test' else EXT}")
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

    stem = Path(name).stem  # numbers saved next to their figure, so the test and AVID runs don't overwrite each other
    pd.concat(numbers).to_csv(out / f"{stem}_numbers.csv", index=False)
    if REGION == "MetaTemporal":
        pd.DataFrame(tests).to_csv(out / f"{stem}_borderline_test.csv", index=False)
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print(pd.concat(numbers).to_string(index=False))
        print("\nPre-specified borderline test (exact McNemar):")
        print(pd.DataFrame(tests).to_string(index=False))
    print(f"\nSaved {out / name} ({img.width} x {img.height} px, {(out / name).stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
