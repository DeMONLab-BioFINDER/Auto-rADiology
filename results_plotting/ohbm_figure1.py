#!/usr/bin/env python3
"""OHBM abstract, Figure 1: how well the model reproduces the visual read, next to meta-temporal SUVR.

  A  ROC curves, held-out test and AVID: the model (solid) and meta-temporal SUVR (dashed).
  B  Confusion matrices at the fixed cut-off 0.5 (test, AVID).
  C  Agreement with the read by meta-temporal SUVR level (below 1.3 / 1.3-1.5 / above 1.5), the model vs the
     published CenTauR cut-off (CTRz = 2, i.e. meta-temporal SUVR 1.34 in this pipeline), with Wilson 95% CIs.
     Borderline SUVR is where the cut-off breaks down; does the model hold up there?

OHBM limits: at most 1000 px per side and 2 MB, so the figure is drawn at exactly 1000 px wide.

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
from sklearn.metrics import roc_auc_score, roc_curve

from plot_utils import CENTAUR_FTP_CTRZ2_SUVR, find_demo_csv, resolve_path, wilson_interval
from regress_model_output_visual_read import find_ext_demo

CUTOFF = CENTAUR_FTP_CTRZ2_SUVR["MetaTemporal"]
BINS = [(-np.inf, 1.3, "< 1.3"), (1.3, 1.5, "1.3–1.5"), (1.5, np.inf, "> 1.5")]
SETS = {"test": "Held-out test", "avid": "External (AVID)"}
COLOURS = {("model", "test"): "#1f4e79", ("model", "avid"): "#6fa8dc",
           ("suvr", "test"): "#555555", ("suvr", "avid"): "#b0b0b0"}


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
    return d.dropna(subset=["MetaTemporal"])


def panel_roc(ax, data: dict):
    for which, d in data.items():
        for kind, score, ls in [("model", d["prob"], "-"), ("suvr", d["MetaTemporal"], "--")]:
            fpr, tpr, _ = roc_curve(d["y"], score)
            auc = roc_auc_score(d["y"], score)
            name = "Model" if kind == "model" else "Meta-temporal SUVR"
            ax.plot(fpr, tpr, ls, color=COLOURS[(kind, which)], lw=1.6,
                    label=f"{name}, {'test' if which == 'test' else 'AVID'}: {auc:.3f}")
    ax.plot([0, 1], [0, 1], ":", color="#cccccc", lw=0.8)
    ax.set(xlim=(-0.01, 1), ylim=(0, 1.01), xlabel="1 − specificity", ylabel="Sensitivity")
    ax.set_title("A  ROC (AUC)", loc="left", fontweight="bold")
    ax.legend(fontsize=6.5, loc="lower right", frameon=False, handlelength=2.2)


def panel_confusion(ax, d: pd.DataFrame, title: str):
    m = pd.crosstab(d["y"], d["pred"]).reindex(index=[0, 1], columns=[0, 1], fill_value=0).to_numpy()
    shade = m / m.sum(axis=1, keepdims=True)  # colour by share of each read row
    ax.imshow(shade, cmap="Blues", vmin=0, vmax=1.2)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=10,
                    color="white" if shade[i, j] > 0.6 else "black", fontweight="bold")
    ax.set_xticks([0, 1], ["Model −", "Model +"], fontsize=7)
    ax.set_yticks([0, 1], ["Read −", "Read +"], fontsize=7)
    ax.set_title(title, fontsize=8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def agreement_rows(data: dict) -> pd.DataFrame:
    rows = []
    for which, d in data.items():
        for lo, hi, label in BINS:
            b = d[(d["MetaTemporal"] >= lo) & (d["MetaTemporal"] < hi)]
            for kind, call in [("model", b["pred"]), ("suvr", (b["MetaTemporal"] > CUTOFF).astype(int))]:
                k, n = int((call == b["y"]).sum()), len(b)
                ci = wilson_interval(k, n) if n else (np.nan, np.nan)
                rows.append({"set": which, "suvr_bin": label, "rule": kind, "n": n, "n_read_pos": int(b["y"].sum()),
                             "agree": k / n if n else np.nan, "lo": ci[0], "hi": ci[1]})
    return pd.DataFrame(rows)


def panel_agreement(ax, rows: pd.DataFrame):
    order = [("model", "test"), ("suvr", "test"), ("model", "avid"), ("suvr", "avid")]
    width = 0.19
    for k, (kind, which) in enumerate(order):
        r = rows[(rows["rule"] == kind) & (rows["set"] == which)].set_index("suvr_bin").reindex([b[2] for b in BINS])
        x = np.arange(len(BINS)) + (k - 1.5) * width
        y = 100 * r["agree"].to_numpy()
        err = 100 * np.vstack([np.clip(r["agree"] - r["lo"], 0, None), np.clip(r["hi"] - r["agree"], 0, None)])
        name = ("Model" if kind == "model" else f"SUVR > {CUTOFF:.2f} (CTRz 2)") + (", test" if which == "test" else ", AVID")
        ax.bar(x, y, width, color=COLOURS[(kind, which)], label=name, yerr=err, capsize=1.5,
               error_kw={"lw": 0.7, "ecolor": "#333333"})
    n = rows[rows["rule"] == "model"].pivot(index="suvr_bin", columns="set", values="n").reindex([b[2] for b in BINS])
    ax.set_xticks(np.arange(len(BINS)),
                  [f"{b}\nn = {int(n.loc[b, 'test'])} / {int(n.loc[b, 'avid'])}" for _, _, b in BINS], fontsize=7)
    ax.set(ylim=(0, 128), ylabel="Agreement with read (%)", xlabel="Meta-temporal SUVR (n test / AVID)")
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_title("C  Agreement by tau level", loc="left", fontweight="bold")
    ax.legend(fontsize=6.3, loc="upper center", frameon=False, ncol=2, columnspacing=0.8)  # in the space above 100%


def main():
    args = parse_args()
    run = Path(args.run_dir).resolve()
    out = Path(args.out_dir) if args.out_dir else run / "figures" / "ohbm"
    out.mkdir(parents=True, exist_ok=True)
    data = {which: load(run, which, args) for which in SETS}

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7})
    fig = plt.figure(figsize=(10, 3.6), dpi=100)  # 1000 x 360 px
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.6, 1.5], wspace=0.35,
                          left=0.06, right=0.99, top=0.9, bottom=0.2)
    panel_roc(fig.add_subplot(gs[0]), data)
    # the two confusion matrices sit a little lower, so their titles clear the panel label
    gs_b = gs[1].subgridspec(2, 1, hspace=0.5)
    for i, which in enumerate(SETS):
        ax = fig.add_subplot(gs_b[i])
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 - 0.05 * (1 - i) - 0.02, pos.width, pos.height * 0.92])
        panel_confusion(ax, data[which], f"{'Test' if which == 'test' else 'AVID'} (n = {len(data[which])})")
    fig.text(fig.axes[1].get_position().x0 - 0.03, 0.93, "B  Cut-off 0.5", fontweight="bold", fontsize=9)
    rows = agreement_rows(data)
    panel_agreement(fig.add_subplot(gs[2]), rows)
    for ax in fig.axes:
        ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(out / "figure1.png", dpi=100, facecolor="white")
    plt.close(fig)
    rows.to_csv(out / "figure1_numbers.csv", index=False)
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print(rows.to_string(index=False))
    size = (out / "figure1.png").stat().st_size / 1e6
    print(f"\nSaved {out / 'figure1.png'} (1000 x 360 px, {size:.2f} MB)")


if __name__ == "__main__":
    main()
