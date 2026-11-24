#!/usr/bin/env python3
"""What do amyloid-negative Read + scans have in common, and why does the model call most of them negative?

At the same meta-temporal SUVR, amyloid-negative Read + scans are still called negative more often than amyloid-positive
ones (analyze_errors_at_same_suvr.py; test and AVID). This script lays out those scans one by one, next to the
amyloid-positive Read + scans, to look for what the "something beyond the tau level" is. Descriptive case series only:
there are about a dozen such scans per set, so there are no tests. Read the counts, then look at the images.

Candidate explanations and the column that speaks to each:
    borderline amyloid          Centiloid just below the 24 CL cut-off (i.e. early Alzheimer's after all)
    non-Alzheimer's tau          regional SUVR profile: mesial-predominant, little posterior spread; older age
    off-target signal            % of the model's evidence outside the cortex (occlusion), images
    unusual spatial pattern      left-right asymmetry |LI|, CenTauR evidence % (occlusion)
    uncertain read               reader votes (unanimous / split / single) and tau score (moderate / frequent)

Groups (Read + scans only): amyloid-negative called negative, amyloid-negative called positive, and amyloid-positive
called negative / positive as the comparison.

Usage:
    python results_plotting/amyloid_negative_read_positive.py \
        --confidence <run>/figures/visual_read_confidence_groups.csv --demo_csv <demo.csv> \
        [--votes IDs_reads.csv] [--asymmetry <run>/figures/asymmetry_test/asymmetry_per_scan.csv] \
        [--occlusion <occlusion folder>/group_maps/centaur_shares_per_scan.csv] [--label "Held-out test"] [--out_dir ...]
Outputs (default: next to the confidence csv, amyloid_negative_read_positive/):
    group_summary.csv           per group: n and medians / percentages (no IDs)
    fig_centiloid.png           Centiloid of every Read + scan by amyloid status and model call, 24 CL cut-off marked
    fig_regional_profile.png    the five CenTauR SUVRs of each amyloid-negative Read + scan (one line per scan,
                                coloured by model call) over the amyloid-positive Read + median as reference
    cases.csv                   one row per amyloid-negative Read + scan, everything joined (has IDs - keep off slides)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_reader_votes import load_votes
from discordant_by_confidence import band
from plot_utils import CUSTOM_PALETTE

CL_CUT = 24  # amyloid positive = Centiloid > 24 in every cohort
REGIONS = ["MesialTemporal", "MetaTemporal", "TemporoParietal", "Frontal", "Universal"]
REGION_LABELS = ["Mesial temporal", "Meta-temporal", "Temporoparietal", "Frontal", "Universal"]
GROUPS = [("Aβ− / Model −", 0, 0), ("Aβ− / Model +", 0, 1), ("Aβ+ / Model −", 1, 0), ("Aβ+ / Model +", 1, 1)]
CALL_COLOURS = {0: CUSTOM_PALETTE[1], 1: CUSTOM_PALETTE[0]}  # Model − rose, Model + blue


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--confidence", required=True, help="visual_read_confidence_groups.csv of the set (ID_ind, y, prob, pred).")
    p.add_argument("--demo_csv", required=True, help="Demographics csv of the set (ID, amyloid_status, CL, SUVRs, age, dx, site).")
    p.add_argument("--votes", default=None, help="Per-reader vote csv (optional).")
    p.add_argument("--asymmetry", default=None, help="asymmetry_per_scan.csv (optional).")
    p.add_argument("--occlusion", default=None, help="centaur_shares_per_scan.csv (optional).")
    p.add_argument("--label", default="", help="Name of the set for titles, e.g. 'Held-out test'.")
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


def load(args) -> pd.DataFrame:
    conf = pd.read_csv(args.confidence)
    d = conf[conf["y"] == 1].rename(columns={"ID_ind": "ID"})[["ID", "prob", "pred"]].copy()
    d["band"] = band(d["prob"])
    demo = pd.read_csv(args.demo_csv)
    demo = demo.loc[:, ~demo.columns.astype(str).str.startswith("Unnamed")]
    cols = [c for c in ["site", "dx", "age", "amyloid_status", "CL", "apoe", "CDR_SB", *REGIONS] if c in demo.columns]
    d = d.merge(demo[["ID", *cols]], on="ID", how="left")
    d["amyloid"] = pd.to_numeric(d.get("amyloid_status"), errors="coerce")
    if args.votes:
        v = load_votes(Path(args.votes).expanduser())
        v["tau score"] = v["visual_read_3_cat"].map({1: "moderate", 2: "frequent"})
        d = d.merge(v[["ID", "agreement", "positive_votes", "n_readers", "tau score"]], on="ID", how="left")
    if args.asymmetry:
        a = pd.read_csv(args.asymmetry)
        d = d.merge(a[["ID"] + [c for c in ["MetaTemporal_absAI", "TemporoParietal_absAI"] if c in a]], on="ID", how="left")
    if args.occlusion:
        o = pd.read_csv(args.occlusion)
        keep = {c: f"evidence % {c}" for c in o.columns if c in ("Meta-temporal", "Temporoparietal", "Mesial temporal", "Frontal")
                or c.startswith("Outside cortex")}
        d = d.merge(o[["ID", *keep]].rename(columns=keep), on="ID", how="left")
    n_missing = int(d["amyloid"].isna().sum())
    if n_missing:
        print(f"[data] {n_missing} Read + scans without amyloid status - left out of the groups")
    return d


def med(s, fmt="{:.2f}"):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return fmt.format(s.median()) if len(s) else ""


def pct(mask):
    mask = mask.dropna()
    return f"{100 * mask.mean():.0f}% ({int(mask.sum())}/{len(mask)})" if len(mask) else ""


def summary(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, amy, call in GROUPS:
        s = d[(d["amyloid"] == amy) & (d["pred"] == call)]
        row = {"group": name, "n": len(s), "median probability": med(s["prob"], "{:.3f}")}
        if "CL" in s:
            cl = pd.to_numeric(s["CL"], errors="coerce")
            row["median Centiloid"] = med(cl, "{:.1f}")
            if amy == 0:
                row["Centiloid 15-24 (just below cut-off)"] = pct(cl.between(15, CL_CUT).where(cl.notna()))
        for r, lab in zip(REGIONS, REGION_LABELS):
            if r in s:
                row[f"median {lab} SUVR"] = med(s[r])
        if "age" in s:
            row["median age"] = med(s["age"], "{:.0f}")
        if "dx" in s:
            row["cognitively unimpaired"] = pct(s["dx"].eq("CU").where(s["dx"].notna()))
        if "agreement" in s:
            row["unanimous (3-reader scans)"] = pct((s.loc[s["n_readers"] == 3, "agreement"] == "unanimous").astype(float))
            row["graded moderate"] = pct(s["tau score"].map({"moderate": 1.0, "frequent": 0.0}))
        if "MetaTemporal_absAI" in s:
            row["median |LI| (%) meta-temporal"] = med(s["MetaTemporal_absAI"], "{:.1f}")
        off = [c for c in s.columns if c.startswith("evidence % Outside cortex")]
        if off:
            row["median % evidence outside cortex"] = med(100 * s[off[0]], "{:.0f}")
        if "evidence % Mesial temporal" in s:
            row["median % evidence mesial temporal"] = med(100 * s["evidence % Mesial temporal"], "{:.0f}")
        rows.append(row)
    return pd.DataFrame(rows)


def fig_centiloid(d: pd.DataFrame, label: str, out: Path):
    if "CL" not in d:
        return
    fig, ax = plt.subplots(figsize=(5.6, 4.2), dpi=180)
    rng = np.random.default_rng(0)
    for i, (name, amy, call) in enumerate(GROUPS):
        s = pd.to_numeric(d.loc[(d["amyloid"] == amy) & (d["pred"] == call), "CL"], errors="coerce").dropna()
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(s)), s, s=22, color=CALL_COLOURS[call], alpha=0.8, edgecolor="white", lw=0.4)
        if len(s):
            ax.text(i, s.max() + 8, f"n = {len(s)}", ha="center", fontsize=8, color="0.3")
    ax.axhline(CL_CUT, color="0.4", ls="--", lw=1)
    ax.text(3.45, CL_CUT + 2, f"{CL_CUT} CL", ha="right", fontsize=8, color="0.4")
    ax.set_xticks(range(len(GROUPS)), [g[0] for g in GROUPS], fontsize=9)
    ax.set_ylabel("Centiloid")
    ax.set_title(f"Read + scans: amyloid level by amyloid status and model call\n{label}", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_profile(d: pd.DataFrame, label: str, out: Path):
    regions = [r for r in REGIONS if r in d]
    if not regions:
        return
    labels = [REGION_LABELS[REGIONS.index(r)] for r in regions]
    neg = d[d["amyloid"] == 0]
    ref = d[d["amyloid"] == 1]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=180)
    x = np.arange(len(regions))
    for call, lab in [(1, "Aβ− / Model +"), (0, "Aβ− / Model −")]:
        s = neg[neg["pred"] == call]
        for k, (_, r) in enumerate(s.iterrows()):
            ax.plot(x, pd.to_numeric(r[regions], errors="coerce").to_numpy(float), color=CALL_COLOURS[call], lw=1.2,
                    alpha=0.75, marker="o", ms=3, label=f"{lab} (n = {len(s)})" if k == 0 else None)
    for call, ls, lab in [(1, "-", "Aβ+ / Model + median"), (0, ":", "Aβ+ / Model − median")]:
        s = ref[ref["pred"] == call]
        if len(s):
            ax.plot(x, s[regions].apply(pd.to_numeric, errors="coerce").median().to_numpy(float), color="0.25", lw=2.2,
                    ls=ls, label=f"{lab} (n = {len(s)})")
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylabel("SUVR (CenTauR region)")
    ax.set_title(f"Amyloid-negative Read + scans: regional tau profile\n{label}", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.92")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def main():
    args = parse_args()
    d = load(args)
    out = Path(args.out_dir) if args.out_dir else Path(args.confidence).parent / "amyloid_negative_read_positive"
    out.mkdir(parents=True, exist_ok=True)
    label = args.label or Path(args.confidence).parent.name

    table = summary(d)
    table.to_csv(out / "group_summary.csv", index=False)
    cases = d[d["amyloid"] == 0].sort_values(["pred", "prob"])
    cases.to_csv(out / "cases.csv", index=False)
    fig_centiloid(d, label, out / "fig_centiloid.png")
    fig_profile(d, label, out / "fig_regional_profile.png")

    print(f"{label}: {len(d)} Read + scans, {len(cases)} amyloid negative "
          f"({int((cases['pred'] == 0).sum())} called negative by the model)")
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.max_colwidth", 40):
        print(table.set_index("group").T.to_string())
    print(f"\nSaved group_summary.csv, fig_centiloid.png, fig_regional_profile.png (no IDs) and cases.csv (has IDs) to {out}")


if __name__ == "__main__":
    main()
