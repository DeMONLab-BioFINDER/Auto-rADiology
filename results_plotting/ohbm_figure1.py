#!/usr/bin/env python3
"""OHBM abstract, Figure 1: how well the model reproduces the visual read, next to meta-temporal SUVR.

  A  ROC curves, held-out test and AVID: the model (solid) and meta-temporal SUVR (dashed).
  B  Confusion matrices at the fixed cut-off 0.5 (test, AVID).
  C  Agreement with the read by meta-temporal SUVR level (below 1.3 / 1.3-1.5 / above 1.5), the model vs the
     published CenTauR cut-off (CTRz = 2, i.e. meta-temporal SUVR 1.34 in this pipeline), with Wilson 95% CIs.
     Borderline SUVR is where the cut-off breaks down; does the model hold up there?

OHBM limits: at most 1000 px per side and 2 MB; drawn at 1000 x 520 px. The borderline bin also gets an exact
McNemar test of model vs cut-off on the same scans (p in the figure and in figure1_numbers.csv).

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
from statsmodels.stats.contingency_tables import mcnemar

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
    ax.set_title("A  ROC (AUC)", loc="left", fontweight="bold", pad=58)
    ax.legend(fontsize=8.5, loc="lower right", frameon=False, handlelength=2.0)


def panel_confusion(ax, d: pd.DataFrame, title: str):
    m = pd.crosstab(d["y"], d["pred"]).reindex(index=[0, 1], columns=[0, 1], fill_value=0).to_numpy()
    shade = m / m.sum(axis=1, keepdims=True)  # colour by share of each read row
    ax.imshow(shade, cmap="Blues", vmin=0, vmax=1.2)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=12,
                    color="white" if shade[i, j] > 0.6 else "black", fontweight="bold")
    ax.set_xticks([0, 1], ["Model −", "Model +"], fontsize=9)
    ax.set_yticks([0, 1], ["Read −", "Read +"], fontsize=9)
    ax.set_title(title, fontsize=10)
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
                             "agree": k / n if n else np.nan, "lo": ci[0], "hi": ci[1],
                             "mcnemar_p": mcnemar_p(b) if kind == "model" else np.nan})
    return pd.DataFrame(rows)


def mcnemar_p(b: pd.DataFrame) -> float:
    """Exact McNemar test, model vs cut-off on the same scans: only scans where exactly one of the two agrees
    with the read count."""
    model_ok = b["pred"] == b["y"]
    rule_ok = (b["MetaTemporal"] > CUTOFF).astype(int) == b["y"]
    table = [[int((model_ok & rule_ok).sum()), int((model_ok & ~rule_ok).sum())],
             [int((~model_ok & rule_ok).sum()), int((~model_ok & ~rule_ok).sum())]]
    if table[0][1] + table[1][0] == 0:
        return 1.0
    return float(mcnemar(table, exact=True).pvalue)


def panel_agreement(ax, rows: pd.DataFrame):
    order = [("model", "test"), ("suvr", "test"), ("model", "avid"), ("suvr", "avid")]
    width = 0.19
    mid = [b[2] for b in BINS].index("1.3–1.5")
    ax.axvspan(mid - 0.48, mid + 0.48, color="#fff2cc", zorder=0)  # the borderline range
    for k, (kind, which) in enumerate(order):
        r = rows[(rows["rule"] == kind) & (rows["set"] == which)].set_index("suvr_bin").reindex([b[2] for b in BINS])
        x = np.arange(len(BINS)) + (k - 1.5) * width
        y = 100 * r["agree"].to_numpy()
        err = 100 * np.vstack([np.clip(r["agree"] - r["lo"], 0, None), np.clip(r["hi"] - r["agree"], 0, None)])
        name = ("Model" if kind == "model" else "Cut-off") + (", test" if which == "test" else ", AVID")
        ax.bar(x, y, width, color=COLOURS[(kind, which)], label=name, yerr=err, capsize=2,
               error_kw={"lw": 0.8, "ecolor": "#333333"}, zorder=2)
    # McNemar p (model vs cut-off, same scans) over each pair of bars in the borderline bin
    tops = 100 * rows[rows["suvr_bin"] == "1.3–1.5"]["hi"].max()
    for k0, which, lift in [(0, "test", 0), (2, "avid", 7)]:  # AVID's p a little higher so the two don't overlap
        r = rows[(rows["rule"] == "model") & (rows["set"] == which) & (rows["suvr_bin"] == "1.3–1.5")].iloc[0]
        top = tops + lift
        x0 = mid + (k0 - 1.5) * width
        ax.plot([x0, x0, x0 + width, x0 + width], [top + 2, top + 4, top + 4, top + 2], color="#333333", lw=0.8)
        p = r["mcnemar_p"]
        ax.text(x0 + width / 2, top + 4.5, f"p = {p:.2g}" if p >= 0.001 else "p < .001", ha="center", va="bottom", fontsize=7.5)
    n = rows[rows["rule"] == "model"].pivot(index="suvr_bin", columns="set", values="n").reindex([b[2] for b in BINS])
    ax.set_xticks(np.arange(len(BINS)),
                  [f"{b}{' (borderline)' if b == '1.3–1.5' else ''}\nn = {int(n.loc[b, 'test'])} / {int(n.loc[b, 'avid'])}"
                   for _, _, b in BINS])
    ax.set(ylim=(0, 122), xlim=(-0.5, len(BINS) - 0.5), ylabel="Agreement with read (%)",
           xlabel="Meta-temporal SUVR (n test / AVID)")
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_title("C  Agreement by tau level", loc="left", fontweight="bold", pad=58)
    ax.legend(fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 0.99), frameon=False, ncol=2,
              columnspacing=1.2, handlelength=1.1, handletextpad=0.4, title_fontsize=8.5,
              title=f"Cut-off: SUVR > {CUTOFF:.2f} (CTRz 2) · p: McNemar")


def main():
    args = parse_args()
    run = Path(args.run_dir).resolve()
    out = Path(args.out_dir) if args.out_dir else run / "figures" / "ohbm"
    out.mkdir(parents=True, exist_ok=True)
    data = {which: load(run, which, args) for which in SETS}

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11.5, "axes.labelsize": 10.5,
                         "xtick.labelsize": 9.5, "ytick.labelsize": 9.5})
    fig = plt.figure(figsize=(10, 5.2), dpi=100)  # 1000 x 520 px
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.62, 1.5], wspace=0.38,
                          left=0.065, right=0.99, top=0.79, bottom=0.15)
    panel_roc(fig.add_subplot(gs[0]), data)
    # the two confusion matrices sit a little lower, so their titles clear the panel label
    gs_b = gs[1].subgridspec(2, 1, hspace=0.5)
    for i, which in enumerate(SETS):
        ax = fig.add_subplot(gs_b[i])
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 - 0.04 * (1 - i) - 0.02, pos.width, pos.height * 0.9])
        panel_confusion(ax, data[which], f"{'Test' if which == 'test' else 'AVID'} (n = {len(data[which])})")
    fig.text(fig.axes[1].get_position().x0 - 0.035, 0.915, "B  Cut-off 0.5", fontweight="bold", fontsize=11.5)
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
    print(f"\nSaved {out / 'figure1.png'} (1000 x 520 px, {size:.2f} MB)")


if __name__ == "__main__":
    main()
