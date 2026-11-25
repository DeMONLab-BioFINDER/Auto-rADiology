#!/usr/bin/env python3
"""Off-target signal per scan: do Model − / Read + scans have little cortical tau but the usual off-target signal?

Question (planned before looking at the voxelwise maps): readers may have called some scans positive on off-target
signal (choroid plexus, basal ganglia, thalamus, meninges / skull) while the model, rightly, said negative.
If so, Model − / Read + scans should have cortical tau close to Model − / Read − scans but off-target signal as high
as (or higher than) the other groups.

Per scan, from the original MNI images (SUVR = image / reference-region mean, the same SUVR images as the voxelwise
analysis, no smoothing), the mean SUVR in:
    cortex          CenTauR meta-temporal and temporoparietal masks (the tau the read is about)
    off-target      choroid plexus (the lateral ventricles stand in: the template's FreeSurfer aseg has no choroid
                    plexus label, and fluid does not bind the tracer), striatum (caudate + putamen + pallidum),
                    thalamus, brainstem (full FreeSurfer aseg of the same template as the other atlas files), and
                    with --rim_mm a rim just outside the brain (meninges / skull). One template atlas for everyone
                    (no MRI): fine for the large structures, approximate for the small ones.
Only voxels inside the scan's field of view count; a region less than half covered is left empty for that scan.

Groups: Model + / Read +, Model − / Read −, Model − / Read + (split into confident, P ≤ .1, and the rest: the
pre-set confidence cut-off), Model + / Read − (shown only, too few).
Tests (Mann-Whitney, FDR across the regions, per comparison): Model − / Read + vs Model − / Read − and vs
Model + / Read +; the same for the confident Model − / Read + scans.

Usage (Berzelius, CPU; see sbatch_scripts/submit_offtarget-regions.sh):
    python results_plotting/offtarget_by_group.py <run_dir> --set test --pkl_dir <tau_T1MNI> \\
        --ref_mask <masks>/voi_CerebGry_tau_2mm.nii --demo_csv <demo.csv> [--rim_mm 8] [--mark_ids 12,34]
Outputs (<run>/figures/offtarget_<set>/):
    offtarget_by_group.csv     median SUVR per group and region, n (no IDs)
    offtarget_tests.csv        the comparisons above, FDR within each comparison (no IDs)
    offtarget_scatter.png      one dot per scan: cortical tau (x) vs off-target signal (y), one panel per off-target
                               region; --mark_ids draws a ring around those scans (no IDs on the figure)
    offtarget_per_scan.csv     every scan with its values - HAS IDs, keep it local / off slides
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image
from scipy import stats
from scipy.ndimage import binary_dilation

from atlas_utils import DEFAULT_ATLAS_DIR, centaur_masks
from plot_fda_regions import DK_FILE, FS_ASEG, fetch
from plot_utils import CUSTOM_PALETTE, fdr_bh
from voxelwise_stats_visual_read import load_table
from voxelwise_visual_read import brain_mask_on_grid, suvr_images

CORTEX = ["Meta-temporal", "Temporoparietal"]
# FreeSurfer aseg label numbers (the template's full aseg). It has no choroid plexus label, so the lateral ventricles
# stand in for it: fluid does not bind the tracer, so signal inside the ventricles is the choroid plexus.
OFF = {"Choroid plexus (lateral ventricles)": [4, 5, 43, 44], "Striatum": [11, 12, 13, 50, 51, 52],
       "Thalamus": [10, 49], "Brainstem": [16]}
RIM = "Outside the brain (meninges, skull)"
CONFIDENT = 0.1
GROUPS = ["Model + / Read +", "Model − / Read −", "Model − / Read + (confident)", "Model − / Read + (other)",
          "Model + / Read −"]
COLOURS = {"Model + / Read +": CUSTOM_PALETTE[3], "Model − / Read −": CUSTOM_PALETTE[0],
           "Model − / Read + (confident)": CUSTOM_PALETTE[1], "Model − / Read + (other)": "#E9A9C2",
           "Model + / Read −": CUSTOM_PALETTE[4]}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--set", default="test", choices=["test", "avid"])
    p.add_argument("--pkl_dir", required=True)
    p.add_argument("--ref_mask", required=True)
    p.add_argument("--centaur_dir", default=None, help="CenTauR masks (default: the --ref_mask folder).")
    p.add_argument("--atlas_dir", default=None)
    p.add_argument("--rim_mm", type=float, default=8.0, help="Rim outside the brain (0 = none).")
    p.add_argument("--dataset", default="Gothenburg")
    p.add_argument("--ext_dataset", default="AVID_unseen")
    p.add_argument("--demo_csv", default=None)
    p.add_argument("--ext_demo_csv", default=None)
    p.add_argument("--female_code", type=float, default=2)
    p.add_argument("--mark_ids", default=None, help="Comma-separated scan IDs to ring on the scatter (e.g. the ones "
                                                    "you judged 'off-target only' by eye).")
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


def aseg_masks(shape, affine, atlas_dir) -> dict:
    """{name: mask} for the off-target structures on the scan grid: a 2 mm voxel counts when at least half of it is in
    the structure (keeps the edges, where neighbouring tissue would mix in, out)."""
    atlas_dir = Path(atlas_dir) if atlas_dir else DEFAULT_ATLAS_DIR
    if not (atlas_dir / FS_ASEG).exists() or not (atlas_dir / DK_FILE).exists():
        fetch(atlas_dir)
    fs = nib.load(atlas_dir / FS_ASEG)
    K = np.asarray(fs.dataobj).astype(int)
    target = nib.Nifti1Image(np.zeros(shape, np.int8), affine)
    out = {}
    for region, ids in OFF.items():
        if not np.isin(K, ids).any():
            print(f"[atlas] no labels for {region} - left out")
            continue
        frac = image.resample_to_img(nib.Nifti1Image(np.isin(K, ids).astype(np.float32), fs.affine), target,
                                     interpolation="linear", force_resample=True, copy_header=True).get_fdata()
        out[region] = frac >= 0.5
        print(f"[atlas] {region}: labels {ids}, {int(out[region].sum())} voxels on the scan grid")
    return out


def region_means(x: np.ndarray, masks: dict) -> dict:
    cover = x != 0
    out = {}
    for name, m in masks.items():
        inside = m & cover
        out[name] = float(x[inside].mean()) if m.sum() and inside.sum() >= 0.5 * m.sum() else np.nan
    return out


def group_of(r) -> str:
    if r["read"] == 1 and r["call"] == 1:
        return GROUPS[0]
    if r["read"] == 0 and r["call"] == 0:
        return GROUPS[1]
    if r["read"] == 1:
        return GROUPS[2] if r["prob"] <= CONFIDENT else GROUPS[3]
    return GROUPS[4]


def tests(d: pd.DataFrame, regions: list) -> pd.DataFrame:
    neg_pos = d["group"].str.startswith("Model − / Read +")
    conf = d["group"] == GROUPS[2]
    comparisons = [("All Model − / Read + vs Model − / Read −", neg_pos, d["group"] == GROUPS[1]),
                   ("All Model − / Read + vs Model + / Read +", neg_pos, d["group"] == GROUPS[0]),
                   ("Confident Model − / Read + vs Model − / Read −", conf, d["group"] == GROUPS[1]),
                   ("Confident Model − / Read + vs Model + / Read +", conf, d["group"] == GROUPS[0])]
    rows = []
    for name, a, b in comparisons:
        part = []
        for reg in regions:
            x, y = d.loc[a, reg].dropna(), d.loc[b, reg].dropna()
            p = stats.mannwhitneyu(x, y).pvalue if len(x) >= 3 and len(y) >= 3 else np.nan
            part.append({"comparison": name, "region": reg, "n_a": len(x), "n_b": len(y),
                         "median_a": x.median(), "median_b": y.median(), "p": p})
        part = pd.DataFrame(part)
        ok = part["p"].notna()
        part.loc[ok, "p_fdr"] = fdr_bh(part.loc[ok, "p"])
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def scatter(d: pd.DataFrame, off_regions: list, marked: set, set_label: str, out: Path):
    """Cortical tau (meta-temporal) against each off-target region, one dot per scan."""
    n = len(off_regions)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.9), squeeze=False)
    for ax, reg in zip(axes[0], off_regions):
        for g in GROUPS:
            s = d[d["group"] == g]
            big = g.startswith("Model − / Read +") or g == GROUPS[4]
            ax.scatter(s["Meta-temporal"], s[reg], s=46 if big else 14, alpha=0.9 if big else 0.35,
                       color=COLOURS[g], edgecolor="white" if big else "none", linewidth=0.5, zorder=3 if big else 2,
                       label=f"{g} (n = {len(s)})")
        m = d[d["ID"].isin(marked)]
        if len(m):
            ax.scatter(m["Meta-temporal"], m[reg], s=170, facecolor="none", edgecolor="0.1", linewidth=1.4, zorder=4,
                       label="Judged off-target only by eye")
        ax.set_xlabel("Meta-temporal SUVR (cortical tau)", fontsize=12)
        ax.set_ylabel(f"{reg} SUVR", fontsize=12)
        ax.set_title(reg, fontsize=13)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=10.5)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Cortical tau and off-target signal, {set_label}", fontsize=15)
    fig.tight_layout(rect=(0, 0.12, 1, 0.97))
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    run = Path(args.run_dir)
    out = Path(args.out_dir) if args.out_dir else run / "figures" / f"offtarget_{args.set}"
    out.mkdir(parents=True, exist_ok=True)
    table = load_table(run, args).set_index("ID")
    table["prob"] = 1 / (1 + np.exp(-table["logodds"]))

    masks, skipped, ref_values, rows = None, [], {}, []
    for scan_id, x, aff in suvr_images(Path(args.pkl_dir), table.index, Path(args.ref_mask), skipped, ref_values):
        if masks is None:
            cdir = Path(args.centaur_dir) if args.centaur_dir else Path(args.ref_mask).parent
            c = centaur_masks(x.shape, aff, cdir)
            masks = {k: c[k] for k in CORTEX}
            masks.update(aseg_masks(x.shape, aff, args.atlas_dir))
            if args.rim_mm > 0:
                brain = brain_mask_on_grid(x.shape, aff)
                vox = np.sqrt((aff[:3, :3] ** 2).sum(0)).min()
                masks[RIM] = binary_dilation(brain, iterations=int(np.ceil(args.rim_mm / vox))) & ~brain
        rows.append({"ID": scan_id, **region_means(x, masks)})
    if skipped:
        print(f"[data] {len(skipped)} scans left out: reference region not covered")
    vals = pd.DataFrame(rows)
    d = table.reset_index()[["ID", "read", "call", "prob"]].merge(vals, on="ID")
    d["group"] = d.apply(group_of, axis=1)
    regions = [k for k in masks]
    off_regions = [k for k in masks if k not in CORTEX]

    d.round(4).to_csv(out / "offtarget_per_scan.csv", index=False)
    summ = d.groupby("group")[regions].median().reindex(GROUPS).round(3)
    summ.insert(0, "n", d["group"].value_counts().reindex(GROUPS).fillna(0).astype(int))
    summ.to_csv(out / "offtarget_by_group.csv")
    t = tests(d, regions)
    t.round(4).to_csv(out / "offtarget_tests.csv", index=False)
    marked = {int(i) for i in args.mark_ids.split(",")} if args.mark_ids else set()
    set_label = "held-out test" if args.set == "test" else args.ext_dataset.removesuffix("_unseen")
    scatter(d, off_regions, marked, set_label, out / "offtarget_scatter.png")
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print("\nMedian SUVR per group:\n" + summ.to_string())
        print("\nTests (Mann-Whitney, FDR across regions within each comparison):\n"
              + t.round(3).to_string(index=False))
        if marked:
            print("\nScans marked by eye:\n" + d[d["ID"].isin(marked)].round(3).to_string(index=False))
    print(f"\nSaved to {out} (offtarget_per_scan.csv has IDs - keep it local)")


if __name__ == "__main__":
    main()
