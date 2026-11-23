#!/usr/bin/env python3
"""OHBM abstract Figure 2: where the model finds its evidence (group-level occlusion).

  A  Group-average occlusion maps (log-odds; red = supports a positive call, blue = supports a negative call) for
     true positives, missed positive reads and true negatives of the held-out test set, axial slices, with the
     CenTauR meta temporal region outlined. From <occlusion folder>/group_maps/group_mean_{TP,FN,TN}.nii.gz
     (occlusion_group_maps.py); one colour scale for all three rows.
  B  Share of each scan's positive evidence in the cortex that lies in each CenTauR region (median per group),
     next to the region's share of the cortex volume (black line = what its size alone would give), for the
     held-out test set and the left-out cohort. From group_maps/centaur_shares_by_group.csv. The meta temporal
     and temporoparietal masks overlap, so the shares do not add up to 100%.

Usage:
    python results_plotting/ohbm_figure_occlusion.py <final_run_dir> --meta_mask <data>/masks/Meta_CenTauR.nii.gz [--ext OASIS]
Saves <run>/figures/ohbm/figure_occlusion.png (1000 px wide).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from nilearn import image, plotting
from PIL import Image

from plot_utils import CUSTOM_PALETTE

TITLE, LABEL, TICK, LEGEND = 12, 11, 9.5, 9.5  # as ohbm_figure1.py
WIDTH_PX = 1000
CUTS = [-30, -20, -10, 0, 10, 20]  # axial slices (MNI z, mm): meta temporal region and above
GROUPS = [("TP", "True positives"), ("FN", "Missed (read +)"), ("TN", "True negatives")]
GROUP_COLOURS = {"TP": CUSTOM_PALETTE[1], "FN": CUSTOM_PALETTE[4], "TN": CUSTOM_PALETTE[0]}
REGIONS = [("Meta-temporal", "Meta-\ntemporal"), ("Temporoparietal", "Temporo-\nparietal"),
           ("Mesial temporal", "Mesial\ntemporal"), ("Outside CenTauR regions", "Rest of\ncortex")]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--meta_mask", required=True, help="CenTauR meta temporal mask (Meta_CenTauR.nii.gz).")
    p.add_argument("--ext", default="OASIS", help="Left-out cohort as in the folder names.")
    p.add_argument("--ext_label", default=None, help="Name shown (default: OASIS -> OASIS-3).")
    return p.parse_args()


def occlusion_dir(run: Path, ext: str | None) -> Path:
    tag = "" if ext is None else f"{ext}_unseen_"
    return run / f"validation/visualization/occlusion_all_{tag}logit_median_brain_mask4_ov0.5/group_maps"


def load_mask(path: Path, target) -> nib.Nifti1Image:
    m = nib.load(path)
    m = nib.Nifti1Image(np.squeeze(np.asarray(m.dataobj)).astype(np.float32), m.affine)  # CenTauR files are 4D
    return image.resample_to_img(m, target, interpolation="nearest", force_resample=True, copy_header=True)


def panel_maps(fig, gs_cell, maps: dict, mask_img, n: dict):
    vals = np.concatenate([np.abs(m.get_fdata()[m.get_fdata() != 0]) for m in maps.values()])
    vmax = float(np.percentile(vals, 99.5))
    sub = gs_cell.subgridspec(len(maps), 1, hspace=0.08)
    axes = []
    for k, (g, label) in enumerate(GROUPS):
        ax = fig.add_subplot(sub[k])
        disp = plotting.plot_stat_map(maps[g], display_mode="z", cut_coords=CUTS, axes=ax, cmap="RdBu_r",
                                      vmax=vmax, symmetric_cbar=True, threshold=vmax * 0.08,
                                      colorbar=False, black_bg=False, draw_cross=False, annotate=(k == len(maps) - 1),
                                      dim=-0.3)
        disp.add_contours(mask_img, levels=[0.5], colors="0.1", linewidths=0.8)
        ax.text(-0.01, 0.5, f"{label}\nn = {n[g]}", transform=ax.transAxes, ha="right", va="center",
                fontsize=TICK, color=GROUP_COLOURS[g], fontweight="bold")
        axes.append(ax)
    axes[0].set_title("Group-average occlusion, held-out test set (black outline: meta-temporal region)",
                      fontsize=TITLE, pad=6)
    # one colour bar for the three rows, labelled at -max / 0 / +max only
    pos = axes[-1].get_position(), axes[0].get_position()
    cax = fig.add_axes([0.925, pos[0].y0 + 0.02, 0.012, pos[1].y1 - pos[0].y0 - 0.04])
    sm = matplotlib.cm.ScalarMappable(cmap="RdBu_r", norm=matplotlib.colors.Normalize(-vmax, vmax))
    cb = fig.colorbar(sm, cax=cax, ticks=[-vmax, 0, vmax])
    cb.ax.set_yticklabels(["supports\nnegative", "0", "supports\npositive"], fontsize=TICK - 1)
    cb.outline.set_visible(False)
    return axes


def panel_shares(ax, table: pd.DataFrame, title: str):
    vol = table.loc["volume share"]
    x = np.arange(len(REGIONS))
    w = 0.26
    for j, (g, label) in enumerate(GROUPS):
        if g not in table.index:
            continue
        vals = [100 * table.loc[g, r] for r, _ in REGIONS]
        ax.bar(x + (j - 1) * w, vals, w, color=GROUP_COLOURS[g], label=label, edgecolor="white", linewidth=1, zorder=3)
    for i, (r, _) in enumerate(REGIONS):
        ax.plot([i - 1.6 * w, i + 1.6 * w], [100 * vol[r]] * 2, color="0.1", lw=1.8, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in REGIONS], fontsize=TICK)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="y", labelsize=TICK)
    ax.set_title(title, fontsize=TITLE - 1, pad=6)
    ax.grid(axis="x", visible=False)


def main():
    args = parse_args()
    run = Path(args.run_dir)
    ext_label = args.ext_label or {"OASIS": "OASIS-3"}.get(args.ext, args.ext)
    out = run / "figures" / "ohbm"
    out.mkdir(parents=True, exist_ok=True)

    test_dir = occlusion_dir(run, None)
    maps = {g: image.load_img(str(test_dir / f"group_mean_{g}.nii.gz")) for g, _ in GROUPS}
    mask_img = load_mask(Path(args.meta_mask), maps["TP"])
    t_test = pd.read_csv(test_dir / "centaur_shares_by_group.csv", index_col=0)
    t_ext = pd.read_csv(occlusion_dir(run, args.ext) / "centaur_shares_by_group.csv", index_col=0)
    n = t_test["n"].dropna().astype(int).to_dict()

    sns.set_theme(style="whitegrid", palette=CUSTOM_PALETTE)
    fig = plt.figure(figsize=(10, 8.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 1], hspace=0.32, left=0.13, right=0.9, top=0.95, bottom=0.12)
    map_axes = panel_maps(fig, gs[0], maps, mask_img, n)
    bottom = gs[1].subgridspec(1, 2, wspace=0.12)
    axl, axr = fig.add_subplot(bottom[0]), fig.add_subplot(bottom[1], sharey=None)
    panel_shares(axl, t_test, "Held-out test set")
    panel_shares(axr, t_ext, f"{ext_label} (unseen)")
    axl.set_ylabel("Share of cortical evidence (%)", fontsize=LABEL)
    axr.set_yticklabels([])
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOURS[g]) for g, _ in GROUPS] + \
              [Line2D([0], [0], color="0.1", lw=1.8)]
    fig.legend(handles, [lab for _, lab in GROUPS] + ["Share of cortex volume"], loc="lower center", ncol=4,
               fontsize=LEGEND, frameon=False, bbox_to_anchor=(0.55, 0.0))
    for ax, letter in [(map_axes[0], "A"), (axl, "B")]:
        ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-62 if ax is axl else -80, 6),
                    textcoords="offset points", ha="right", va="bottom", fontsize=LABEL + 2, fontweight="bold")

    big = out / "_figure_occlusion_large.png"
    fig.savefig(big, dpi=300, facecolor="white")
    plt.close(fig)
    img = Image.open(big)
    img = img.resize((WIDTH_PX, round(img.height * WIDTH_PX / img.width)), Image.LANCZOS)
    img.save(out / "figure_occlusion.png", optimize=True)
    big.unlink()
    print(f"Saved {out / 'figure_occlusion.png'} ({img.width} x {img.height} px)")


if __name__ == "__main__":
    main()
