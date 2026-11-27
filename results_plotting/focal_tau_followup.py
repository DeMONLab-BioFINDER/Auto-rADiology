#!/usr/bin/env python3
"""Focal tau: could a Model - / Read + scan have a small hot spot that the voxelwise group map (which averages across
people) or a regional mean (which averages across the region) hides?

Per scan, over the whole cortex (DK outline only, as in the voxelwise test; same SUVR images, cerebellar grey
reference and 6 mm smoothing within the field of view), three numbers:
  peak SUVR                 hottest bit of cortex: 99th percentile of the cortex voxels
  raised voxels %           share of cortex clearly above normal: z > 2 against the normal scans of the same set at that
                            voxel (normal = Read -, amyloid negative, cognitively unimpaired; falls back to Read - and
                            amyloid negative, then Read -, if fewer than 20 scans)
  % raised in Universal     share of those raised voxels inside Universal CenTauR (where a read can turn positive);
                            scattered noise would give Universal's share of the cortex (printed)
A hidden focal spot = high peak with few raised voxels (top left of the left panel).

Tests (test and AVID separately): misses vs Model - / Read - and vs Model + / Read + (Mann-Whitney, BH-FDR over the
table); within the misses, medians per confidence band and confident vs the rest (Mann-Whitney, descriptive).

Usage (Berzelius, CPU):
    python results_plotting/focal_tau_followup.py <run_dir> --set test \\
        --summary_csv <run>/validation/visualization/occlusion_all_logit_median_brain_mask4_ov0.5/summary.csv \\
        --pkl_dir /proj/berzelius-2024-156/dataset/tau-PET/tau_T1MNI --ref_mask <masks>/voi_CerebGry_tau_2mm.nii \\
        --universal_mask <masks>/CenTauR.nii.gz --atlas_dir <data>/atlases/lotter2025_mni_freesurfer --demo_csv <data>/demo.csv
Outputs (<run>/figures/focal_tau_<set>/; no IDs except focal_per_scan.csv):
    focal_by_group.csv, focal_tests.csv, focal_misses_by_band.csv, focal_tau.png;
    focal_per_scan.csv (has IDs - stays on Berzelius)
"""
import argparse
import warnings
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image
from scipy import ndimage, stats
from statsmodels.stats.multitest import multipletests

from atlas_utils import DKCortex
from plot_utils import CONFUSION_CATEGORY_PALETTE, add_subgroup_columns
from voxelwise_visual_read import suvr_images

MISS, NEG, POS, FP = "Model − / Read +", "Model − / Read −", "Model + / Read +", "Model + / Read −"
GROUPS = {(1, 1): POS, (1, 0): MISS, (0, 0): NEG, (0, 1): FP}
COLOUR = {POS: CONFUSION_CATEGORY_PALETTE["True Positive"], NEG: CONFUSION_CATEGORY_PALETTE["True Negative"],
          MISS: CONFUSION_CATEGORY_PALETTE["False Negative"], FP: CONFUSION_CATEGORY_PALETTE["False Positive"]}
MEASURES = {"peak_SUVR": "peak SUVR (99th pct)", "extent_pct": "raised voxels (%)",
            "pct_raised_in_universal": "% of raised voxels in Universal"}
BANDS = ["Confident", "Moderately confident", "Uncertain"]
SHADE = {"Confident": "#8c5a00", "Moderately confident": COLOUR[MISS], "Uncertain": "#f3d39a"}
MIN_REF = 20


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--set", default="test", choices=["test", "avid"])
    p.add_argument("--summary_csv", required=True, help="Per-scan read, model call and P_positive (occlusion summary.csv).")
    p.add_argument("--pkl_dir", required=True)
    p.add_argument("--ref_mask", required=True)
    p.add_argument("--universal_mask", required=True, help="Universal CenTauR mask (CenTauR.nii.gz).")
    p.add_argument("--atlas_dir", default=None, help="DK atlas folder, cortex outline only (as the voxelwise test).")
    p.add_argument("--demo_csv", required=True)
    p.add_argument("--female_code", type=float, default=2)
    p.add_argument("--fwhm", type=float, default=6.0, help="Smoothing in mm, as in the voxelwise test.")
    p.add_argument("--z", type=float, default=2.0, help="z above the normal scans that counts as raised.")
    p.add_argument("--pct", type=float, default=99.0, help="Percentile used as the 'peak'.")
    p.add_argument("--min_coverage", type=float, default=0.9, help="Leave out scans covering less of the cortex.")
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


def band(prob: float) -> str:
    c = min(prob, 1 - prob)
    return "Confident" if c <= 0.1 else ("Uncertain" if c >= 0.3 else "Moderately confident")


def med(s):
    s = s.dropna()
    return round(float(s.median()), 3) if len(s) else np.nan


def load_table(args) -> pd.DataFrame:
    s = pd.read_csv(args.summary_csv).dropna(subset=["visual_read", "pred"])
    d = pd.DataFrame({"ID": s["ID"].astype(int), "read": s["visual_read"].astype(int), "call": s["pred"].astype(int),
                      "prob": s["P_positive"].astype(float)})
    d["group"] = [GROUPS[(r, c)] for r, c in zip(d["read"], d["call"])]
    d["band"] = d["prob"].map(band)
    demo = pd.read_csv(args.demo_csv)
    demo = demo.loc[:, ~demo.columns.str.startswith("Unnamed")].drop_duplicates("ID")
    add_subgroup_columns(demo, args.female_code)
    return d.merge(demo[[c for c in ("ID", "_g_amy", "_g_dx") if c in demo.columns]], on="ID", how="left")


def cortex_values(d: pd.DataFrame, args):
    """Smoothed SUVR of every scan in the cortex (scans x voxels, NaN outside the field of view), the cortex mask and
    the Universal mask on the same grid."""
    ids, vals, skipped, refs = [], [], [], {}
    cortex = universal = sigma = None
    n_fov = 0
    for sid, x, aff in suvr_images(Path(args.pkl_dir), pd.Index(d["ID"]), Path(args.ref_mask), skipped, refs):
        if cortex is None:
            cortex = DKCortex(x.shape, aff, args.atlas_dir).mask
            u = nib.load(args.universal_mask)
            u = nib.Nifti1Image(np.squeeze(np.asarray(u.dataobj)), u.affine)  # some CenTauR files are 4D
            universal = image.resample_to_img(u, nib.Nifti1Image(np.zeros(x.shape, np.int8), aff),
                                              interpolation="nearest", force_resample=True,
                                              copy_header=True).get_fdata() > 0
            universal &= cortex
            vox = np.sqrt((aff[:3, :3] ** 2).sum(0))
            sigma = args.fwhm / np.sqrt(8 * np.log(2)) / vox if args.fwhm else None
            print(f"[cortex] {int(cortex.sum())} voxels; Universal CenTauR covers {universal.sum() / cortex.sum():.0%}")
        cover = x != 0
        if cover[cortex].mean() < args.min_coverage:
            n_fov += 1
            continue
        if sigma is not None:  # smooth within the field of view, as the voxelwise test
            w = ndimage.gaussian_filter(cover.astype(np.float64), sigma)
            x = np.divide(ndimage.gaussian_filter(x, sigma), w, out=np.zeros_like(x), where=w > 1e-3)
        v = x[cortex].astype(np.float32)
        v[~cover[cortex]] = np.nan
        ids.append(sid)
        vals.append(v)
    if skipped or n_fov:
        print(f"[data] left out: {len(skipped)} reference region not covered, {n_fov} cover < "
              f"{args.min_coverage:.0%} of the cortex (field of view)")
    return np.array(ids), np.vstack(vals), universal[cortex]


def normal_scans(d: pd.DataFrame) -> tuple[pd.Series, str]:
    neg = d["read"] == 0
    amy = d["_g_amy"] if "_g_amy" in d else pd.Series(pd.NA, index=d.index)
    dx = d["_g_dx"] if "_g_dx" in d else pd.Series(pd.NA, index=d.index)
    for name, sel in (("Read −, Aβ negative, CU", neg & (amy == "Aβ negative") & (dx == "CU")),
                      ("Read −, Aβ negative", neg & (amy == "Aβ negative")), ("Read −", neg)):
        sel = sel.fillna(False).astype(bool)
        if sel.sum() >= MIN_REF:
            return sel, name
    raise SystemExit("Fewer than 20 Read - scans: cannot build the normal reference")


def measure(d: pd.DataFrame, args) -> tuple[pd.DataFrame, str, float]:
    ids, v, in_u = cortex_values(d, args)
    d = d.set_index("ID").loc[ids].reset_index()
    ref, ref_name = normal_scans(d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu, sd = np.nanmean(v[ref.to_numpy()], 0), np.nanstd(v[ref.to_numpy()], 0, ddof=1)
    raised = np.nan_to_num((v - mu) / np.where(sd > 0, sd, np.nan)) > args.z
    print(f"[normal] {int(ref.sum())} scans ({ref_name})")
    n_raised = raised.sum(1)
    d["peak_SUVR"] = np.nanpercentile(v, args.pct, axis=1)
    d["extent_pct"] = 100 * raised.mean(1)
    d["pct_raised_in_universal"] = np.where(n_raised > 0, 100 * raised[:, in_u].sum(1) / np.maximum(n_raised, 1), np.nan)
    d["normal_ref"] = ref.to_numpy()
    return d, ref_name, float(in_u.mean())


def mwu(a, b):
    a, b = a.dropna(), b.dropna()
    return stats.mannwhitneyu(a, b).pvalue if len(a) >= 2 and len(b) >= 2 else np.nan


def tests(d: pd.DataFrame) -> pd.DataFrame:
    m = d[d["group"] == MISS]
    rows = [{"measure": name, "comparison": f"{MISS} vs {other}", "median misses": med(m[col]),
             "median other": med(d.loc[d["group"] == other, col]), "p": mwu(m[col], d.loc[d["group"] == other, col])}
            for col, name in MEASURES.items() for other in (NEG, POS)]
    t = pd.DataFrame(rows)
    ok = t["p"].notna()
    t.loc[ok, "p_fdr"] = multipletests(t.loc[ok, "p"], method="fdr_bh")[1]
    return t


def by_band(m: pd.DataFrame) -> pd.DataFrame:
    rows = [{"band": lvl, "n": int((m["band"] == lvl).sum()),
             **{name: med(m.loc[m["band"] == lvl, col]) for col, name in MEASURES.items()}} for lvl in BANDS]
    conf = m["band"] == "Confident"
    rows.append({"band": "p: confident vs the rest",
                 **{name: mwu(m.loc[conf, col], m.loc[~conf, col]) for col, name in MEASURES.items()}})
    return pd.DataFrame(rows)


def scatter(ax, d, x, y, rng=None):
    order = [NEG, POS, FP]
    for j, g in enumerate(order):
        s = d[d["group"] == g]
        xs = s[x] if rng is None else j + rng.uniform(-0.25, 0.25, len(s))
        ax.scatter(xs, s[y], s=14, alpha=0.45, color=COLOUR[g], edgecolor="white", linewidth=0.4,
                   label=f"{g} (n = {len(s)})", zorder=2)
    m = d[d["group"] == MISS]
    for lvl in BANDS:
        s = m[m["band"] == lvl]
        if len(s):
            xs = s[x] if rng is None else len(order) + rng.uniform(-0.25, 0.25, len(s))
            ax.scatter(xs, s[y], s=46, color=SHADE[lvl], edgecolor="black", linewidth=0.5, zorder=3,
                       label=f"{MISS}, {lvl.lower()} (n = {len(s)})")
    ax.spines[["top", "right"]].set_visible(False)


def figure(d: pd.DataFrame, title: str, ref_name: str, z: float, u_share: float, out: Path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 6.2))
    scatter(a, d, "extent_pct", "peak_SUVR")
    a.set_xscale("symlog", linthresh=1)
    a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_xlabel(f"Raised cortex voxels (%, z > {z:g}, log scale)", fontsize=12)
    a.set_ylabel("Peak tau in the cortex (99th percentile SUVR)", fontsize=12)
    a.set_title("Hidden focal spot? It would sit top left", fontsize=13)
    scatter(b, d, "group", "pct_raised_in_universal", rng=np.random.default_rng(0))
    b.axhline(100 * u_share, color="0.4", ls="--", lw=1)
    b.text(3.45, 100 * u_share, "scattered\nnoise", fontsize=8, color="0.4", va="center")
    b.set_xticks(range(4), ["Model −\nRead −", "Model +\nRead +", "Model +\nRead −", "Model −\nRead +"])
    b.set_ylim(-3, 103)
    b.set_ylabel("Raised voxels inside Universal CenTauR (%)", fontsize=12)
    b.set_title("Is the raised tau where a read counts?", fontsize=13)
    fig.suptitle(title, fontsize=14)
    h, lab = a.get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.04))
    fig.text(0.5, 0.005, f"Raised = z > {z:g} against the normal scans ({ref_name}) at that voxel; cortex = DK outline; "
             f"scans without raised voxels are not in the right panel", ha="center", fontsize=9, color="0.4")
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    run = Path(args.run_dir)
    out = Path(args.out_dir) if args.out_dir else run / "figures" / f"focal_tau_{args.set}"
    out.mkdir(parents=True, exist_ok=True)
    d, ref_name, u_share = measure(load_table(args), args)
    d.to_csv(out / "focal_per_scan.csv", index=False)
    m = d[d["group"] == MISS]
    print(f"[data] {len(d)} scans, {len(m)} misses")
    groups = pd.DataFrame([{"group": g, "n": int((d["group"] == g).sum()),
                            **{name: med(d.loc[d["group"] == g, col]) for col, name in MEASURES.items()}}
                           for g in COLOUR]
                          + [{"group": f"normal reference ({ref_name})", "n": int(d["normal_ref"].sum()),
                              **{name: med(d.loc[d["normal_ref"], col]) for col, name in MEASURES.items()}},
                             {"group": "Universal share of the cortex (scattered noise)",
                              "% of raised voxels in Universal": round(100 * u_share, 1)}])
    t, bb = tests(d), by_band(m)
    for name, tab in (("focal_by_group", groups), ("focal_tests", t), ("focal_misses_by_band", bb)):
        tab.to_csv(out / f"{name}.csv", index=False)
    label = "held-out test" if args.set == "test" else "AVID"
    figure(d, f"Focal tau in the cortex, {label}", ref_name, args.z, u_share, out / "focal_tau.png")
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.float_format", "{:.3g}".format):
        for name, tab in (("By group", groups), ("Misses vs the others", t), ("Misses by confidence band", bb)):
            print(f"\n{name}:\n{tab.to_string(index=False)}")
    print(f"\nSaved to {out} (focal_per_scan.csv has IDs - keep it on Berzelius)")


if __name__ == "__main__":
    main()
