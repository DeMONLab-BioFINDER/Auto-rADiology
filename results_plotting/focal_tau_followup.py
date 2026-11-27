#!/usr/bin/env python3
"""Focal tau: do the Model - / Read + scans have a small hot spot that a regional mean or a group map would average
away, and is it inside the regions where a read can turn positive?

The voxelwise group test only finds tau that sits in the same place in many misses, and a mean SUVR over a region
dilutes a small spot. These measures do not care where the spot is. Per scan, over the whole cortex (DK outline only,
as in the voxelwise test; same SUVR images, cerebellar grey reference and 6 mm smoothing within the field of view):
  peak SUVR       99th percentile of the cortex voxels (the hottest 1%)
  extent %        share of cortex voxels that are clearly raised: z > 2 against the 'normal' scans of the same set at
                  that voxel (normal = Read - and amyloid negative and cognitively unimpaired; falls back to Read - and
                  amyloid negative, then Read -, if fewer than 20 scans). Normal scans score ~2-3% by construction.
  largest spot    volume (ml) of the largest connected cluster of raised voxels: one spot vs scattered noise
Then where it is, against the Universal CenTauR region (roughly the FDA read regions):
  % raised in Universal   share of the raised voxels inside Universal CenTauR (compare with the share of the cortex
                          that Universal covers: what scattered noise would give)
  largest spot location   inside (>= 75% of its voxels in Universal), partly, outside (<= 25%), or no spot
And the model: peak evidence = 99th percentile of the occlusion map ("pos vs neg", log-odds) over the cortex, and the
share of the positive evidence that falls inside Universal (per-patch values, not a total push).
Reading it: misses with a high peak but small extent = a focal spot; inside Universal = tau where a read counts,
outside = tau the read rule would not count (check on ThinLinc whether it is tau or skull / meninges signal).

Tests (one set at a time; test and AVID separately): misses vs Model - / Read - and vs Model + / Read + for each
measure (Mann-Whitney, BH-FDR over the table); within the misses: confident vs the rest (Mann-Whitney) and across the
three bands (Kruskal-Wallis), descriptive (small groups). Did the model react to the peak? OLS (HC3), all scans:
    peak evidence ~ peak SUVR + miss

Usage (Berzelius, CPU):
    python results_plotting/focal_tau_followup.py <run_dir> --set test \\
        --occlusion_dir <run>/validation/visualization/occlusion_all_logit_median_brain_mask4_ov0.5 \\
        --pkl_dir /proj/berzelius-2024-156/dataset/tau-PET/tau_T1MNI --ref_mask <masks>/voi_CerebGry_tau_2mm.nii \\
        --universal_mask <masks>/CenTauR.nii.gz --atlas_dir <data>/atlases/lotter2025_mni_freesurfer --demo_csv <data>/demo.csv
Outputs (<run>/figures/focal_tau_<set>/; no IDs except focal_per_scan.csv):
    focal_by_group.csv, focal_tests.csv, focal_misses_by_band.csv, spot_location.csv, evidence_vs_peak.csv,
    focal_tau.png; focal_per_scan.csv (has IDs - stays on Berzelius)
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

from atlas_utils import DKCortex
from elevated_area_followup import COLOUR, MISS, load_table, med
from make_case_viewer import MNI_2MM_RAS, MNI_SHAPE, on_mni_grid
from voxelwise_visual_read import suvr_images

MEASURES = {"peak_SUVR": "peak SUVR (99th pct)", "extent_pct": "raised voxels (%)",
            "largest_spot_ml": "largest raised spot (ml)", "pct_raised_in_universal": "% of raised voxels in Universal",
            "mean_SUVR_universal": "mean SUVR in Universal", "peak_evidence": "peak model evidence (99th pct)",
            "pct_evidence_in_universal": "% of positive evidence in Universal"}
NEG, POS = "Model − / Read −", "Model + / Read +"
BANDS = ["Confident", "Moderately confident", "Uncertain"]
SHADE = {"Confident": "#8c5a00", "Moderately confident": COLOUR[MISS], "Uncertain": "#f3d39a"}
LOCATIONS = ["inside", "partly", "outside", "no spot"]
MIN_REF = 20


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--set", default="test", choices=["test", "avid"])
    p.add_argument("--occlusion_dir", required=True, help="Merged occlusion folder of this set (maps + summary.csv).")
    p.add_argument("--pkl_dir", required=True)
    p.add_argument("--ref_mask", required=True)
    p.add_argument("--universal_mask", required=True, help="Universal CenTauR mask (CenTauR.nii.gz).")
    p.add_argument("--atlas_dir", default=None, help="DK atlas folder, cortex outline only (as the voxelwise test).")
    p.add_argument("--demo_csv", required=True)
    p.add_argument("--votes", default=None, help="Not used for the tests; only so the shared table loader works.")
    p.add_argument("--female_code", type=float, default=2)
    p.add_argument("--fwhm", type=float, default=6.0, help="Smoothing in mm, as in the voxelwise test.")
    p.add_argument("--z", type=float, default=2.0, help="z above the normal scans that counts as raised.")
    p.add_argument("--pct", type=float, default=99.0, help="Percentile used as the 'peak'.")
    p.add_argument("--min_coverage", type=float, default=0.9, help="Leave out scans covering less of the cortex.")
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


def load_mask(path: Path, shape, affine) -> np.ndarray:
    img = nib.load(path)
    img = nib.Nifti1Image(np.squeeze(np.asarray(img.dataobj)), img.affine)  # some CenTauR files are 4D
    return image.resample_to_img(img, nib.Nifti1Image(np.zeros(shape, np.int8), affine), interpolation="nearest",
                                 force_resample=True, copy_header=True).get_fdata() > 0


def cortex_values(d: pd.DataFrame, args):
    """Smoothed SUVR of every scan in the cortex (scans x voxels, NaN outside the field of view), the cortex mask,
    the Universal mask on the same grid and the voxel volume in ml."""
    ids, vals, skipped, refs = [], [], [], {}
    cortex = universal = vox_ml = sigma = None
    n_fov = 0
    for sid, x, aff in suvr_images(Path(args.pkl_dir), pd.Index(d["ID"]), Path(args.ref_mask), skipped, refs):
        if cortex is None:
            cortex = DKCortex(x.shape, aff, args.atlas_dir).mask
            universal = load_mask(Path(args.universal_mask), x.shape, aff) & cortex
            vox = np.sqrt((aff[:3, :3] ** 2).sum(0))
            vox_ml = float(np.prod(vox)) / 1000
            sigma = args.fwhm / np.sqrt(8 * np.log(2)) / vox if args.fwhm else None
            print(f"[cortex] {int(cortex.sum())} voxels ({cortex.sum() * vox_ml:.0f} ml); Universal CenTauR covers "
                  f"{universal.sum() / cortex.sum():.0%} of it")
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
    return np.array(ids), np.vstack(vals), cortex, universal, vox_ml


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


def location(share: float) -> str:
    if np.isnan(share):
        return "no spot"
    return "inside" if share >= 0.75 else ("outside" if share <= 0.25 else "partly")


def measure(d: pd.DataFrame, args) -> tuple[pd.DataFrame, str, float]:
    ids, v, cortex, universal, vox_ml = cortex_values(d, args)
    d = d.set_index("ID").loc[ids].reset_index()
    ref, ref_name = normal_scans(d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu, sd = np.nanmean(v[ref.to_numpy()], 0), np.nanstd(v[ref.to_numpy()], 0, ddof=1)
    z = (v - mu) / np.where(sd > 0, sd, np.nan)
    print(f"[normal] {int(ref.sum())} scans ({ref_name})")
    in_u = universal[cortex]
    u_share = float(in_u.mean())
    occ_cortex = DKCortex(MNI_SHAPE, MNI_2MM_RAS, args.atlas_dir).mask
    occ_u = load_mask(Path(args.universal_mask), MNI_SHAPE, MNI_2MM_RAS)[occ_cortex]
    occ_dir = Path(args.occlusion_dir)
    rows = []
    for i, sid in enumerate(ids):
        r = np.nan_to_num(z[i]) > args.z
        raised = np.zeros(cortex.shape, bool)
        raised[cortex] = r
        lab, n = ndimage.label(raised)
        spot_share = np.nan
        big = 0
        if n:
            sizes = np.bincount(lab.ravel())[1:]
            k = int(sizes.argmax()) + 1
            big = int(sizes[k - 1])
            spot_share = float(universal[lab == k].mean())
        f = occ_dir / f"{sid}_occlusion_pos_vs_neg.nii.gz"
        pk_e = pct_e = np.nan
        if f.exists():
            e = on_mni_grid(f)[occ_cortex]
            pk_e = float(np.percentile(e, args.pct))
            pos = np.clip(e, 0, None)
            pct_e = 100 * float(pos[occ_u].sum() / pos.sum()) if pos.sum() > 0 else np.nan
        rows.append({"ID": sid, "peak_SUVR": float(np.nanpercentile(v[i], args.pct)),
                     "extent_pct": 100 * float(r.mean()), "largest_spot_ml": big * vox_ml,
                     "pct_raised_in_universal": 100 * float(r[in_u].sum() / r.sum()) if r.any() else np.nan,
                     "largest_spot_location": location(spot_share),
                     "mean_SUVR_universal": float(np.nanmean(v[i][in_u])), "peak_evidence": pk_e,
                     "pct_evidence_in_universal": pct_e, "normal_ref": bool(ref.iloc[i])})
    return d.merge(pd.DataFrame(rows), on="ID"), ref_name, u_share


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
        rows.append({"band": lvl, "n": len(g), **{name: med(g[col]) for col, name in MEASURES.items()},
                     **{f"spot {loc}": int((g["largest_spot_location"] == loc).sum()) for loc in LOCATIONS}})
    conf = m["band"] == "Confident"
    p_two = {"band": "p: confident vs the rest"}
    p_three = {"band": "p: across the three bands"}
    for col, name in MEASURES.items():
        p_two[name] = mwu(m.loc[conf, col], m.loc[~conf, col])
        groups = [g for g in (m.loc[m["band"] == lvl, col].dropna() for lvl in BANDS) if len(g) >= 2]
        p_three[name] = stats.kruskal(*groups).pvalue if len(groups) >= 2 else np.nan
    return pd.DataFrame(rows + [p_two, p_three])


def spot_location(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g in COLOUR:
        s = d.loc[d["group"] == g, "largest_spot_location"]
        rows.append({"group": g, "n": len(s), **{loc: int((s == loc).sum()) for loc in LOCATIONS}})
    t = pd.DataFrame(rows)
    sub = t[t["group"].isin([MISS, NEG])].set_index("group")[["inside", "partly", "outside"]]
    if sub.to_numpy().sum() and (sub.sum(0) > 0).sum() >= 2:  # do misses' spots sit elsewhere than negatives'?
        t["p (misses vs Model − / Read −, chi-square)"] = stats.chi2_contingency(sub.loc[:, sub.sum(0) > 0])[1]
    return t


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


def scatter(ax, d, x, y, rng=None):
    order = [NEG, POS, "Model + / Read −"]
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
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(18, 6.2))
    scatter(a, d, "extent_pct", "peak_SUVR")
    a.set_xscale("symlog", linthresh=1)
    a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_xlabel(f"Raised cortex voxels (%, z > {z:g}, log scale)", fontsize=12)
    a.set_ylabel("Peak tau in the cortex (99th percentile SUVR)", fontsize=12)
    a.set_title("Focal spot? High peak, small extent = top left", fontsize=13)
    scatter(b, d, "group", "pct_raised_in_universal", rng=np.random.default_rng(0))
    b.axhline(100 * u_share, color="0.4", ls="--", lw=1)
    b.text(3.45, 100 * u_share, "scattered\nnoise", fontsize=8, color="0.4", va="center")
    b.set_xticks(range(4), ["Model −\nRead −", "Model +\nRead +", "Model +\nRead −", "Model −\nRead +"])
    b.set_ylim(-3, 103)
    b.set_ylabel("Raised voxels inside Universal CenTauR (%)", fontsize=12)
    b.set_title("Is the raised tau where a read counts?", fontsize=13)
    scatter(c, d, "peak_SUVR", "peak_evidence")
    c.axhline(0, color="0.6", lw=0.8)
    c.set_xlabel("Peak tau in the cortex (99th percentile SUVR)", fontsize=12)
    c.set_ylabel("Peak model evidence\n(99th percentile occlusion drop, log-odds)", fontsize=12)
    c.set_title("Does the model react to the hottest spot?", fontsize=13)
    fig.suptitle(title, fontsize=14)
    h, lab = a.get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.04))
    fig.text(0.5, 0.005, f"Raised = z > {z:g} against the normal scans ({ref_name}) at that voxel; cortex = DK outline; "
             f"scans without raised voxels are not in the middle panel", ha="center", fontsize=9, color="0.4")
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
                           for g in COLOUR] + [{"group": f"normal reference ({ref_name})", "n": int(d["normal_ref"].sum()),
                                                **{name: med(d.loc[d["normal_ref"], col]) for col, name in MEASURES.items()}},
                                               {"group": "Universal share of the cortex (scattered noise)",
                                                "% of raised voxels in Universal": round(100 * u_share, 1)}])
    t, bb, loc, fit = tests(d), by_band(m), spot_location(d), evidence_vs_peak(d)
    for name, tab in (("focal_by_group", groups), ("focal_tests", t), ("focal_misses_by_band", bb),
                      ("spot_location", loc), ("evidence_vs_peak", fit)):
        tab.to_csv(out / f"{name}.csv", index=False)
    label = "held-out test" if args.set == "test" else "AVID"
    figure(d, f"Focal tau in the cortex, {label}", ref_name, args.z, u_share, out / "focal_tau.png")
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.float_format", "{:.3g}".format):
        for name, tab in (("By group", groups), ("Misses vs the others", t), ("Misses by confidence band", bb),
                          ("Largest raised spot: inside Universal CenTauR?", loc),
                          ("Peak evidence vs peak tau (OLS, HC3)", fit)):
            print(f"\n{name}:\n{tab.to_string(index=False)}")
    print(f"\nSaved to {out} (focal_per_scan.csv has IDs - keep it on Berzelius)")


if __name__ == "__main__":
    main()
