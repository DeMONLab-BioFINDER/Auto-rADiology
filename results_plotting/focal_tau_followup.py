#!/usr/bin/env python3
"""Focal tau: do the Model - / Read + scans have a small hot spot that a regional mean or a group map would average away?

The voxelwise group test only finds tau that sits in the same place in many misses, and a mean SUVR over a region
dilutes a small spot. These measures do not care where the spot is. Per scan, inside the Universal CenTauR region
(same SUVR images, cerebellar grey reference and 6 mm smoothing as the voxelwise test):
  mean SUVR      the usual regional mean (for reference)
  peak SUVR      99th percentile of the voxels (the hottest 1% of the region)
  extent %       share of voxels that are clearly raised: z > 2 against the 'normal' scans of the same set at that
                 voxel (normal = Read - and amyloid negative and cognitively unimpaired; falls back to Read - and amyloid
                 negative, then Read -, if fewer than 20 scans). The normal scans score ~2-3% by construction.
  largest spot   volume (ml) of the largest connected cluster of raised voxels: one spot vs scattered noise
  peak evidence  99th percentile of the scan's occlusion map ("pos vs neg", log-odds) inside the region: does the model
                 react strongly to any one place? (per-patch value, not a total push)
Reading it: misses with a high peak but small extent = a focal spot (too small for the model?); misses with a low peak
as well = little tau anywhere.

Tests (one set at a time; test and AVID separately): misses vs Model - / Read - and misses vs Model + / Read + for
each measure (Mann-Whitney, BH-FDR over the table); within the misses: confident vs the rest (Mann-Whitney) and across
the three bands (Kruskal-Wallis), descriptive (small groups). Did the model react to the peak? OLS (HC3), all scans:
    peak evidence ~ peak SUVR + miss
(miss term near 0 = the misses get as much evidence as their peak predicts).

Usage (Berzelius, CPU):
    python results_plotting/focal_tau_followup.py <run_dir> --set test \\
        --occlusion_dir <run>/validation/visualization/occlusion_all_logit_median_brain_mask4_ov0.5 \\
        --pkl_dir /proj/berzelius-2024-156/dataset/tau-PET/tau_T1MNI --ref_mask <masks>/voi_CerebGry_tau_2mm.nii \\
        --region_mask <masks>/CenTauR.nii.gz --demo_csv <data>/demo.csv
Outputs (<run>/figures/focal_tau_<set>/; no IDs except focal_per_scan.csv):
    focal_by_group.csv, focal_tests.csv, focal_misses_by_band.csv, evidence_vs_peak.csv, focal_tau.png;
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
import statsmodels.formula.api as smf
from nilearn import image
from scipy import ndimage, stats
from statsmodels.stats.multitest import multipletests

from elevated_area_followup import COLOUR, MISS, load_table, med
from make_case_viewer import MNI_2MM_RAS, MNI_SHAPE, on_mni_grid
from voxelwise_visual_read import suvr_images

MEASURES = {"mean_SUVR": "mean SUVR", "peak_SUVR": "peak SUVR (99th pct)", "extent_pct": "raised voxels (%)",
            "largest_spot_ml": "largest raised spot (ml)", "peak_evidence": "peak model evidence (99th pct)",
            "mean_evidence": "mean model evidence"}
NEG, POS = "Model − / Read −", "Model + / Read +"
BANDS = ["Confident", "Moderately confident", "Uncertain"]
SHADE = {"Confident": "#8c5a00", "Moderately confident": COLOUR[MISS], "Uncertain": "#f3d39a"}
MIN_REF = 20


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--set", default="test", choices=["test", "avid"])
    p.add_argument("--occlusion_dir", required=True, help="Merged occlusion folder of this set (maps + summary.csv).")
    p.add_argument("--pkl_dir", required=True)
    p.add_argument("--ref_mask", required=True)
    p.add_argument("--region_mask", required=True, help="Universal CenTauR mask (CenTauR.nii.gz).")
    p.add_argument("--demo_csv", required=True)
    p.add_argument("--votes", default=None, help="Not used for the tests; only so the shared table loader works.")
    p.add_argument("--female_code", type=float, default=2)
    p.add_argument("--fwhm", type=float, default=6.0, help="Smoothing in mm, as in the voxelwise test.")
    p.add_argument("--z", type=float, default=2.0, help="z above the normal scans that counts as raised.")
    p.add_argument("--pct", type=float, default=99.0, help="Percentile used as the 'peak'.")
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


def load_mask(path: Path, target: nib.Nifti1Image) -> np.ndarray:
    img = nib.load(path)
    img = nib.Nifti1Image(np.squeeze(np.asarray(img.dataobj)), img.affine)  # some CenTauR files are 4D
    return image.resample_to_img(img, target, interpolation="nearest", force_resample=True,
                                 copy_header=True).get_fdata() > 0


def region_values(d: pd.DataFrame, args):
    """Smoothed SUVR of every scan inside the region (scans x voxels), the mask, and the voxel volume in ml."""
    ids, vals, skipped, refs, mask, vox_ml = [], [], [], {}, None, None
    for sid, x, aff in suvr_images(Path(args.pkl_dir), pd.Index(d["ID"]), Path(args.ref_mask), skipped, refs):
        img = nib.Nifti1Image(x.astype(np.float32), aff)
        if mask is None:
            mask = load_mask(Path(args.region_mask), img)
            vox_ml = abs(np.linalg.det(aff[:3, :3])) / 1000
            print(f"[region] {int(mask.sum())} voxels ({mask.sum() * vox_ml:.0f} ml)")
        raw = x[mask]
        if (raw != 0).mean() < 0.9:  # region not covered (field of view)
            skipped.append(sid)
            continue
        if args.fwhm:
            x = image.smooth_img(img, args.fwhm).get_fdata()
        ids.append(sid)
        vals.append(x[mask].astype(np.float32))
    if skipped:
        print(f"[data] {len(skipped)} scans left out: reference region or Universal region not covered")
    return np.array(ids), np.vstack(vals), mask, vox_ml


def normal_scans(d: pd.DataFrame) -> tuple[pd.Series, str]:
    neg = d["read"] == 0
    for name, sel in (("Read −, Aβ negative, CU", neg & (d.get("_g_amy") == "Aβ negative") & (d.get("_g_dx") == "CU")),
                      ("Read −, Aβ negative", neg & (d.get("_g_amy") == "Aβ negative")), ("Read −", neg)):
        sel = sel.fillna(False).astype(bool)
        if sel.sum() >= MIN_REF:
            return sel, name
    raise SystemExit("Fewer than 20 Read - scans: cannot build the normal reference")


def measure(d: pd.DataFrame, args) -> tuple[pd.DataFrame, str]:
    ids, v, mask, vox_ml = region_values(d, args)
    d = d.set_index("ID").loc[ids].reset_index()
    ref, ref_name = normal_scans(d)
    mu, sd = v[ref.to_numpy()].mean(0), v[ref.to_numpy()].std(0, ddof=1)
    z = (v - mu) / np.where(sd > 0, sd, np.nan)
    print(f"[normal] {int(ref.sum())} scans ({ref_name})")
    occ_mask = load_mask(Path(args.region_mask), nib.Nifti1Image(np.zeros(MNI_SHAPE, np.int8), MNI_2MM_RAS))
    occ_dir = Path(args.occlusion_dir)
    rows = []
    for i, sid in enumerate(ids):
        raised = np.zeros(mask.shape, bool)
        raised[mask] = np.nan_to_num(z[i]) > args.z
        lab, n = ndimage.label(raised)
        big = np.bincount(lab.ravel())[1:].max() if n else 0
        f = occ_dir / f"{sid}_occlusion_pos_vs_neg.nii.gz"
        e = on_mni_grid(f)[occ_mask] if f.exists() else None
        rows.append({"ID": sid, "mean_SUVR": float(v[i].mean()), "peak_SUVR": float(np.percentile(v[i], args.pct)),
                     "extent_pct": 100 * float(raised[mask].mean()), "largest_spot_ml": big * vox_ml,
                     "peak_evidence": float(np.percentile(e, args.pct)) if e is not None else np.nan,
                     "mean_evidence": float(e.mean()) if e is not None else np.nan, "normal_ref": bool(ref.iloc[i])})
    return d.merge(pd.DataFrame(rows), on="ID"), ref_name


def mwu(a, b):
    a, b = a.dropna(), b.dropna()
    return stats.mannwhitneyu(a, b).pvalue if len(a) >= 2 and len(b) >= 2 else np.nan


def tests(d: pd.DataFrame) -> pd.DataFrame:
    m = d[d["group"] == MISS]
    rows = []
    for col, name in MEASURES.items():
        for other in (NEG, POS):
            o = d.loc[d["group"] == other, col]
            rows.append({"measure": name, "comparison": f"{MISS} vs {other}", "median misses": med(m[col]),
                         "median other": med(o), "p": mwu(m[col], o)})
    t = pd.DataFrame(rows)
    ok = t["p"].notna()
    t.loc[ok, "p_fdr"] = multipletests(t.loc[ok, "p"], method="fdr_bh")[1]
    return t


def by_band(m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lvl in BANDS:
        g = m[m["band"] == lvl]
        rows.append({"band": lvl, "n": len(g), **{name: med(g[col]) for col, name in MEASURES.items()}})
    conf = m["band"] == "Confident"
    p_two = {"band": "p: confident vs the rest", "n": np.nan}
    p_three = {"band": "p: across the three bands", "n": np.nan}
    for col, name in MEASURES.items():
        p_two[name] = mwu(m.loc[conf, col], m.loc[~conf, col])
        groups = [m.loc[m["band"] == lvl, col].dropna() for lvl in BANDS]
        groups = [g for g in groups if len(g) >= 2]
        p_three[name] = stats.kruskal(*groups).pvalue if len(groups) >= 2 else np.nan
    return pd.DataFrame(rows + [p_two, p_three])


def evidence_vs_peak(d: pd.DataFrame) -> pd.DataFrame:
    e = d.dropna(subset=["peak_SUVR", "peak_evidence"]).copy()
    e["miss"] = (e["group"] == MISS).astype(int)
    rows = []
    for f, term in (("peak_evidence ~ peak_SUVR", "peak_SUVR"), ("peak_evidence ~ peak_SUVR + miss", "peak_SUVR"),
                    ("peak_evidence ~ peak_SUVR + miss", "miss")):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = smf.ols(f, e).fit(cov_type="HC3")
        lo, hi = r.conf_int().loc[term]
        rows.append({"model": f, "term": term, "estimate": r.params[term], "lo": lo, "hi": hi, "p": r.pvalues[term],
                     "n": int(r.nobs), "r2": r.rsquared})
    rho, p = stats.spearmanr(e["peak_SUVR"], e["peak_evidence"])
    rows.append({"model": "Spearman, all scans", "term": "peak_SUVR vs peak_evidence", "estimate": rho, "p": p, "n": len(e)})
    return pd.DataFrame(rows)


def scatter(ax, d, x, y):
    for g in [NEG, POS, "Model + / Read −"]:
        s = d[d["group"] == g]
        ax.scatter(s[x], s[y], s=14, alpha=0.45, color=COLOUR[g], edgecolor="white", linewidth=0.4,
                   label=f"{g} (n = {len(s)})", zorder=2)
    m = d[d["group"] == MISS]
    for lvl in BANDS:
        s = m[m["band"] == lvl]
        if len(s):
            ax.scatter(s[x], s[y], s=46, color=SHADE[lvl], edgecolor="black", linewidth=0.5, zorder=3,
                       label=f"{MISS}, {lvl.lower()} (n = {len(s)})")
    ax.spines[["top", "right"]].set_visible(False)


def figure(d: pd.DataFrame, title: str, ref_name: str, z: float, out: Path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 6.2))
    scatter(a, d, "extent_pct", "peak_SUVR")
    a.set_xscale("symlog", linthresh=1)
    a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_xlabel(f"Raised voxels in the Universal region (%, z > {z:g}, log scale)", fontsize=12)
    a.set_ylabel("Peak tau (99th percentile SUVR)", fontsize=12)
    a.set_title("Focal spot? High peak, small extent = top left", fontsize=13)
    scatter(b, d, "peak_SUVR", "peak_evidence")
    b.axhline(0, color="0.6", lw=0.8)
    b.set_xlabel("Peak tau (99th percentile SUVR)", fontsize=12)
    b.set_ylabel("Peak model evidence\n(99th percentile occlusion drop, log-odds)", fontsize=12)
    b.set_title("Does the model react to the hottest spot?", fontsize=13)
    fig.suptitle(title, fontsize=14)
    h, lab = a.get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.04))
    fig.text(0.5, 0.005, f"Raised = z > {z:g} against the normal scans ({ref_name}) at that voxel", ha="center",
             fontsize=9, color="0.4")
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    run = Path(args.run_dir)
    out = Path(args.out_dir) if args.out_dir else run / "figures" / f"focal_tau_{args.set}"
    out.mkdir(parents=True, exist_ok=True)
    d, ref_name = measure(load_table(args), args)
    d.to_csv(out / "focal_per_scan.csv", index=False)
    m = d[d["group"] == MISS]
    print(f"[data] {len(d)} scans, {len(m)} misses")
    groups = pd.DataFrame([{"group": g, "n": int((d["group"] == g).sum()),
                            **{name: med(d.loc[d["group"] == g, col]) for col, name in MEASURES.items()}}
                           for g in COLOUR] + [{"group": f"normal reference ({ref_name})", "n": int(d["normal_ref"].sum()),
                                                **{name: med(d.loc[d["normal_ref"], col]) for col, name in MEASURES.items()}}])
    t, bb, fit = tests(d), by_band(m), evidence_vs_peak(d)
    for name, tab in (("focal_by_group", groups), ("focal_tests", t), ("focal_misses_by_band", bb),
                      ("evidence_vs_peak", fit)):
        tab.to_csv(out / f"{name}.csv", index=False)
    label = "held-out test" if args.set == "test" else "AVID"
    figure(d, f"Focal tau in the Universal CenTauR region, {label}", ref_name, args.z, out / "focal_tau.png")
    with pd.option_context("display.width", 220, "display.max_columns", None, "display.float_format", "{:.3g}".format):
        for name, tab in (("By group", groups), ("Misses vs the others", t), ("Misses by confidence band", bb),
                          ("Peak evidence vs peak tau (OLS, HC3)", fit)):
            print(f"\n{name}:\n{tab.to_string(index=False)}")
    print(f"\nSaved to {out} (focal_per_scan.csv has IDs - keep it on Berzelius)")


if __name__ == "__main__":
    main()
