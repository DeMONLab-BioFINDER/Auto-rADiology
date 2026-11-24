#!/usr/bin/env python3
"""Does the model put its evidence where tau differs between Read + and Read - scans? CenTauR regions only.

One descriptive figure, two panels with one row per CenTauR region (meta-temporal, temporoparietal, mesial
temporal, frontal):
  left   median share of each scan's cortical positive evidence in the region, per outcome group
         (Model + / Read +, Model - / Read +, Model - / Read -), from occlusion_group_maps.py; a black tick
         marks the region's share of the cortex volume (evidence spread evenly would land on the tick)
  right  how much higher the region's SUVR is in Read + than in Read - scans of the same set, as a
         standardised difference (Cohen's d: difference in means / pooled SD), from the demographics csv
Reading it: if the regions with the largest tau difference also get the most evidence, the model looks where
the read's tau signal is. Four overlapping regions are too few for a correlation, so there is no test.

Usage:
    python results_plotting/occlusion_vs_tau.py <occlusion folder>/group_maps --demo_csv <demo.csv> \
        [--label "Held-out test"]
Inputs: centaur_shares_per_scan.csv and centaur_shares_by_group.csv (occlusion_group_maps.py), the demo csv
(ID + CenTauR SUVR columns). Outputs (same folder, no IDs): occlusion_vs_tau.png, occlusion_vs_tau.csv.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_utils import CUSTOM_PALETTE

REGIONS = [("Meta-temporal", "MetaTemporal"), ("Temporoparietal", "TemporoParietal"),
           ("Mesial temporal", "MesialTemporal"), ("Frontal", "Frontal")]
GROUPS = [("TP", "Model + / Read +"), ("FN", "Model − / Read +"), ("TN", "Model − / Read −")]
READ_POS = {"TP", "FN"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("group_maps", help="Output folder of occlusion_group_maps.py.")
    p.add_argument("--demo_csv", required=True, help="Demographics csv with ID and the CenTauR SUVR columns.")
    p.add_argument("--label", default="", help="Name of the set for the title, e.g. 'Held-out test'.")
    return p.parse_args()


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    pooled = np.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2))
    return (x.mean() - y.mean()) / pooled


def main():
    args = parse_args()
    folder = Path(args.group_maps)
    per_scan = pd.read_csv(folder / "centaur_shares_per_scan.csv")
    by_group = pd.read_csv(folder / "centaur_shares_by_group.csv", index_col=0)
    demo = pd.read_csv(args.demo_csv)
    suvr_cols = {c: f"SUVR_{c}" for _, c in REGIONS if c in demo.columns}  # prefixed: per_scan has a "Frontal" share column
    d = per_scan.merge(demo[["ID", *suvr_cols]].rename(columns=suvr_cols), on="ID", how="left")
    d["read_pos"] = d["group"].isin(READ_POS)
    print(f"{args.label or folder.parent.name}: {len(d)} scans ({int(d['read_pos'].sum())} Read +); "
          f"{int(d[list(suvr_cols.values())].isna().all(axis=1).sum())} without SUVRs")

    rows = []
    for name, col in REGIONS:
        row = {"region": name, "volume share of cortex": by_group.loc["volume share", name]}
        for g, _ in GROUPS:
            row[f"median evidence share, {g}"] = by_group.loc[g, name] if g in by_group.index else np.nan
        if f"SUVR_{col}" in d:
            suvr = pd.to_numeric(d[f"SUVR_{col}"], errors="coerce")
            row["SUVR Cohen's d, Read + vs Read -"] = cohens_d(suvr[d["read_pos"]].to_numpy(float),
                                                               suvr[~d["read_pos"]].to_numpy(float))
        rows.append(row)
    table = pd.DataFrame(rows)
    table.round(3).to_csv(folder / "occlusion_vs_tau.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        print(table.to_string(index=False))

    # figure: two panels sharing the region rows (two measures, two x-axes, never one dual axis)
    present = [(g, lab) for g, lab in GROUPS if g in by_group.index]
    y = np.arange(len(REGIONS))[::-1]
    h = 0.8 / max(len(present), 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True, gridspec_kw={"width_ratios": [1.4, 1]})
    for i, (g, lab) in enumerate(present):
        n = int(by_group.loc[g, "n"])
        ax1.barh(y + 0.4 - h * (i + 0.5), table[f"median evidence share, {g}"], height=h * 0.9,
                 color=CUSTOM_PALETTE[i], label=f"{lab} (n = {n})")
    ax1.scatter(table["volume share of cortex"], y, marker="|", s=400, color="black", linewidths=2,
                label="Region's share of the cortex", zorder=5)
    ax1.set_yticks(y, [r for r, _ in REGIONS])
    ax1.set_xlabel("Median share of the scan's positive evidence")
    ax1.set_xlim(0, max(0.05, float(np.nanmax(table.filter(like="median evidence").to_numpy()) * 1.15)))
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8, frameon=False)
    ax1.set_title("Where the model finds evidence", fontsize=11)

    ax2.barh(y, table["SUVR Cohen's d, Read + vs Read -"], height=0.5, color="0.55")
    ax2.set_xlabel("SUVR difference, Read + vs Read − (Cohen's d)")
    dvals = table["SUVR Cohen's d, Read + vs Read -"]
    ax2.set_xlim(min(0, float(np.nanmin(dvals)) * 1.15), max(0.5, float(np.nanmax(dvals)) * 1.15))
    ax2.set_title("Where tau differs between reads", fontsize=11)
    for ax in (ax1, ax2):
        ax.grid(axis="x", color="0.9"); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    if args.label:
        fig.suptitle(args.label, fontsize=12)
    fig.tight_layout()
    fig.savefig(folder / "occlusion_vs_tau.png", dpi=200)
    print(f"Saved occlusion_vs_tau.png / .csv to {folder}")


if __name__ == "__main__":
    main()
