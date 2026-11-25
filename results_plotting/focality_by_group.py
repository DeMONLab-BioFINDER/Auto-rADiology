#!/usr/bin/env python3
"""Why does the model say negative on Model - / Read + scans? Off-target signal or small focal tau?

Two explanations for Model - / Read + scans whose cortex looks normal by SUVR:
  A  off-target   readers saw off-target signal (striatum / thalamus, edges) and called it positive; the model discounted it
  B  focal tau    real tau in a small spot that regional averages hide and the model ignores
Per scan, this measures what tells A and B apart.

Images: the original MNI images as SUVR on the readers' scale (divided by the whole-cerebellum mean, like the case viewer;
the readers' display used the cerebellum at its largest cross-section, grey + white matter). Cortex = DK cortex outline.
Per scan:
    mean          mean cortical SUVR (the amount; held equal in the tests)
    peak          mean SUVR of the hot spot: the hottest point of the cortex (found after 6 mm smoothing) and the 2 ml
                  of cortex closest to it (about a 1.5 cm spot)                         B: high for the amount
    extent_ml     cortex above 1.65 x cerebellum (the readers' grey -> colour threshold), ml; a spatial-extent measure
    hot_region    where the hot spot is: CenTauR region at its hottest point, within 4 mm (mesial temporal /
                  meta-temporal / temporoparietal / frontal / other cortex); hot_x/y/z = that point in MNI mm        B: reading regions; A: next to off-target areas (mesial)
    hot_share     share of the model's positive occlusion evidence on the hot spot (chance = 2 ml of ~1,060 ml cortex,
                  about 0.2%; a share, so independent of how strong the output was; needs --occ_dir)   B: low
    deep          mean SUVR in striatum + thalamus (FreeSurfer aseg of the template; classic flortaucipir off-target)
                                                                                           A: higher than normal scans
    chp_peak      hottest 2 ml spot in the choroid plexus stand-in (lateral ventricle + temporal horn behind the foramen
                  of Monro, grown 2 mm into white matter; atlas_utils.choroid_plexus_frac). A spot, because the region is
                  mostly fluid and white matter around the small hot plexus                A: readers took spill-over
                  next to the hippocampus for mesial-temporal tau -> higher in confident Model - / Read +
    chp           mean SUVR of that whole region (check; diluted by fluid, lower with bigger ventricles)
    peak_1ml, peak_4ml, hot_share_1ml, hot_share_4ml   the same with a 1 or 4 ml hot spot (does the size matter?)
    meta          CenTauR meta-temporal SUVR on the same images (links to the rest of the analysis)
    grey_vs_whole mean of the inferior cerebellar grey matter (CenTauR reference) on this scale: the spreadsheet SUVR is
                  about this scale's SUVR / grey_vs_whole (how much the two references differ)
Group map (no statistics): per voxel, the share of each group's scans above 1.65 (where a reader would see colour).

Tests:
  within Read + scans, Model - vs Model + : logistic regression, each measure alone and at the same mean cortical SUVR,
      likelihood-ratio test, FDR across the main measures (peak, extent, hot_share, deep); the choroid plexus measures
      (chp_peak, chp) are their own family, added later (so the main results do not change)
  Mann-Whitney: Model - / Read + (all; confident P <= .1) vs Model + / Read + and vs Model - / Read -, FDR across measures
  hot-spot region: counts per group (descriptive)

Usage (Berzelius, CPU; see sbatch_scripts/submit_focality.sh):
    python results_plotting/focality_by_group.py <run_dir> --set test --pkl_dir <tau_T1MNI> \\
        --ref_mask <masks>/voi_WhlCbl_2mm.nii --grey_ref <masks>/voi_CerebGry_tau_2mm.nii --demo_csv <demo.csv> \\
        [--occ_dir <occlusion_all_... folder>]
Outputs (<run>/figures/focality_<set>/): focality_by_group.csv, focality_tests.csv, focality_hot_regions.csv,
    focality_scatter.png, focality_maps.png (no IDs); focality_per_scan.csv (HAS IDs - keep local)
"""
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from nilearn import image
from scipy import stats
from scipy.ndimage import gaussian_filter

from atlas_utils import DEFAULT_ATLAS_DIR, DKCortex, centaur_masks, choroid_plexus_frac
from plot_fda_regions import FS_ASEG, fetch
from plot_utils import CUSTOM_PALETTE, fdr_bh
from voxelwise_stats_visual_read import load_table
from voxelwise_visual_read import suvr_images

THRESH = 1.65  # readers' grey -> colour transition (x cerebellum)
HOT_ML = (2.0, 1.0, 4.0)  # main hot-spot size first, then the size checks
SMOOTH_MM = 6.0  # only to find the hottest point (the measures use the unsmoothed SUVR)
DEEP_LABELS = [10, 11, 12, 13, 49, 50, 51, 52]  # FreeSurfer aseg: thalamus, caudate, putamen, pallidum (both sides)
CONFIDENT = 0.1
GROUPS = ["Model + / Read +", "Model − / Read −", "Model − / Read + (confident)", "Model − / Read + (other)",
          "Model + / Read −"]
COLOURS = {GROUPS[0]: CUSTOM_PALETTE[3], GROUPS[1]: CUSTOM_PALETTE[0], GROUPS[2]: CUSTOM_PALETTE[1],
           GROUPS[3]: "#E9A9C2", GROUPS[4]: CUSTOM_PALETTE[4]}
MAIN = ["peak", "extent_ml", "hot_share", "deep"]
CHP = ["chp_peak", "chp"]
FAMILY = {**{m: "main" for m in MAIN}, **{m: "choroid plexus" for m in CHP}}
CHECKS = ["peak_1ml", "peak_4ml", "hot_share_1ml", "hot_share_4ml"]
LABELS = {"peak": "Peak (hottest 2 ml of cortex)", "extent_ml": "Cortex above 1.65 (ml)",
          "hot_share": "Model evidence on the hot spot (share)", "deep": "Striatum + thalamus SUVR",
          "chp_peak": "Choroid plexus: hottest 2 ml", "chp": "Choroid plexus region: mean SUVR (check)",
          "peak_1ml": "Peak, hottest 1 ml (check)", "peak_4ml": "Peak, hottest 4 ml (check)",
          "hot_share_1ml": "Model evidence on hottest 1 ml (check)", "hot_share_4ml": "Model evidence on hottest 4 ml (check)",
          "mean": "Mean cortical SUVR", "meta": "Meta-temporal SUVR", "grey_vs_whole": "Cerebellar grey / whole cerebellum"}
REGION_ORDER = ["Mesial temporal", "Meta-temporal (not mesial)", "Temporoparietal (outside meta-temporal)", "Frontal",
                "Other cortex"]
CUTS = [-24, -12, 0, 12, 24, 36, 48]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--set", default="test", choices=["test", "avid"])
    p.add_argument("--pkl_dir", required=True)
    p.add_argument("--ref_mask", required=True, help="Whole-cerebellum mask (voi_WhlCbl_2mm.nii): the readers' scale.")
    p.add_argument("--grey_ref", default=None, help="Inferior cerebellar grey mask (voi_CerebGry_tau_2mm.nii, the CenTauR "
                                                    "reference): reports how much the two references differ.")
    p.add_argument("--centaur_dir", default=None, help="CenTauR masks (default: the --ref_mask folder).")
    p.add_argument("--atlas_dir", default=None)
    p.add_argument("--occ_dir", default=None, help="Occlusion maps of every scan (model evidence).")
    p.add_argument("--dataset", default="Gothenburg")
    p.add_argument("--ext_dataset", default="AVID_unseen")
    p.add_argument("--demo_csv", default=None)
    p.add_argument("--ext_demo_csv", default=None)
    p.add_argument("--female_code", type=float, default=2)
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


def on_grid(img, shape, affine, interp="nearest") -> np.ndarray:
    img = nib.Nifti1Image(np.squeeze(np.asarray(img.dataobj)).astype(np.float32), img.affine)
    with warnings.catch_warnings():  # linear resampling of a 0/1 mask is intended (fraction of the voxel inside)
        warnings.simplefilter("ignore", UserWarning)
        return image.resample_to_img(img, nib.Nifti1Image(np.zeros(shape, np.int8), affine), interpolation=interp,
                                     force_resample=True, copy_header=True).get_fdata()


def deep_mask(shape, affine, atlas_dir) -> np.ndarray:
    """Striatum + thalamus: a voxel counts when at least half of it is in these structures."""
    atlas_dir = Path(atlas_dir) if atlas_dir else DEFAULT_ATLAS_DIR
    if not (atlas_dir / FS_ASEG).exists():
        fetch(atlas_dir)
    fs = nib.load(atlas_dir / FS_ASEG)
    K = np.asarray(fs.dataobj).astype(int)
    return on_grid(nib.Nifti1Image(np.isin(K, DEEP_LABELS).astype(np.float32), fs.affine), shape, affine, "linear") >= 0.5


def hot_region(centre, c: dict, near_mm: float = 4.0, vox_mm=None) -> str:
    """CenTauR region at the hottest point (within near_mm, since the masks are patchy within the cortex ribbon). The
    regions overlap (mesial inside meta-temporal inside temporoparietal); the most specific one wins."""
    r = int(np.ceil(near_mm / (vox_mm.min() if vox_mm is not None else 2.0)))
    sl = tuple(slice(max(0, k - r), k + r + 1) for k in centre)
    near = lambda name: bool(c[name][sl].any())
    for name, label in (("Mesial temporal", "Mesial temporal"), ("Meta-temporal", "Meta-temporal (not mesial)"),
                        ("Temporoparietal", "Temporoparietal (outside meta-temporal)"), ("Frontal", "Frontal")):
        if near(name):
            return label
    return "Other cortex"


def spot(x, region, vox_mm):
    """Hottest point of a region (found after 6 mm smoothing within the region, so one noisy voxel cannot win); returns
    it and the region's voxels sorted by distance to it."""
    sigma = SMOOTH_MM / np.sqrt(8 * np.log(2)) / vox_mm
    w = gaussian_filter(region.astype(np.float32), sigma)
    xs = np.divide(gaussian_filter(np.where(region, x, 0), sigma), w, out=np.zeros_like(x), where=w > 1e-3)
    ijk = np.argwhere(region)
    centre = ijk[np.argmax(xs[region])]
    return centre, ijk[np.argsort(np.sqrt((((ijk - centre) * vox_mm) ** 2).sum(1)))]


def per_scan(x, cortex, masks, vox_ml, occ=None):
    """Measures of one SUVR volume (RAS grid), and its above-1.65 mask; None if the cortex is not covered."""
    cov = cortex & (x != 0)
    v = x[cov]
    if v.size < 0.5 * cortex.sum():
        return None, None
    above = cov & (x >= THRESH)
    m = {"mean": float(v.mean()), "extent_ml": float(above.sum() * vox_ml)}
    pos = np.clip(occ[cov], 0, None) if occ is not None else None
    # the hot spot is a spot: the hottest point of the cortex (lightly smoothed within the cortex, so one noisy voxel
    # cannot win), then the cortical voxels closest to it, 1 / 2 / 4 ml
    vox_mm = np.sqrt((masks["affine"][:3, :3] ** 2).sum(0))
    centre, near = spot(x, cov, vox_mm)
    m["hot_x"], m["hot_y"], m["hot_z"] = (masks["affine"] @ np.r_[centre, 1])[:3].round(1)
    m["hot_region"] = hot_region(centre, masks["centaur"], vox_mm=vox_mm)
    for ml in HOT_ML:
        n_hot = max(1, int(round(ml / vox_ml)))
        hot = np.zeros_like(cov)
        hot[tuple(near[:n_hot].T)] = True
        sfx = "" if ml == HOT_ML[0] else f"_{ml:g}ml"
        m["peak" + sfx] = float(x[hot].mean())
        if pos is not None:
            m["hot_share" + sfx] = float(np.clip(occ[hot], 0, None).sum() / pos.sum()) if pos.sum() > 0 else np.nan
    for key, msk in (("meta", masks["centaur"]["Meta-temporal"]), ("deep", masks["deep"])):
        ok = msk & (x != 0)
        m[key] = float(x[ok].mean()) if ok.sum() >= 0.5 * msk.sum() else np.nan
    chp = masks["chp"] & (x != 0)
    if chp.sum() >= 0.5 * masks["chp"].sum():
        m["chp"] = float(x[chp].mean())
        _, near = spot(x, chp, vox_mm)
        m["chp_peak"] = float(x[tuple(near[:max(1, int(round(HOT_ML[0] / vox_ml)))].T)].mean())
    if masks.get("grey") is not None:
        ok = masks["grey"] & (x != 0)
        m["grey_vs_whole"] = float(x[ok].mean()) if ok.sum() >= 0.5 * masks["grey"].sum() else np.nan
    return m, above


def group_of(read, call, prob) -> str:
    if read == 1 and call == 1:
        return GROUPS[0]
    if read == 0 and call == 0:
        return GROUPS[1]
    if read == 1:
        return GROUPS[2] if prob <= CONFIDENT else GROUPS[3]
    return GROUPS[4]


def logit_tests(d: pd.DataFrame, measures: list) -> pd.DataFrame:
    """Within Read + scans: Model - ~ measure, and ~ measure + mean cortical SUVR (LR test). FDR over the main measures."""
    rp = d[d["read"] == 1].copy()
    rp["wrong"] = (rp["call"] == 0).astype(int)
    rows = []
    for m in measures:
        dd = rp.dropna(subset=[m, "mean"]).copy()
        if dd[m].std() == 0 or dd["wrong"].sum() < 3:
            continue
        dd["z"] = (dd[m] - dd[m].mean()) / dd[m].std()
        dd["amount"] = (dd["mean"] - dd["mean"].mean()) / dd["mean"].std()
        row = {"measure": LABELS[m], "family": FAMILY.get(m, "check"), "n": len(dd), "n_model_neg": int(dd["wrong"].sum())}
        for prefix, adj in (("plain_", []), ("", ["amount"])):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    full = smf.logit("wrong ~ " + " + ".join(["z", *adj]), data=dd).fit(disp=0)
                    red = smf.logit("wrong ~ " + (" + ".join(adj) or "1"), data=dd).fit(disp=0)
                lo, hi = full.conf_int().loc["z"]
                row.update({f"{prefix}or_per_sd": np.exp(full.params["z"]), f"{prefix}or_lo": np.exp(lo),
                            f"{prefix}or_hi": np.exp(hi), f"{prefix}p": stats.chi2.sf(2 * (full.llf - red.llf), 1)})
            except Exception as err:
                row[f"{prefix}note"] = f"could not fit ({err})"
        rows.append(row)
    t = pd.DataFrame(rows)
    for prefix in ("plain_", ""):
        if f"{prefix}p" in t:
            for fam in ("main", "choroid plexus"):
                ok = t[f"{prefix}p"].notna() & (t["family"] == fam)
                if ok.any():
                    t.loc[ok, f"{prefix}p_fdr"] = fdr_bh(t.loc[ok, f"{prefix}p"])
    return t


def group_tests(d: pd.DataFrame, measures: list) -> pd.DataFrame:
    neg_pos = d["group"].isin(GROUPS[2:4])
    conf = d["group"] == GROUPS[2]
    comps = [("Model − / Read + vs Model + / Read +", neg_pos, d["group"] == GROUPS[0]),
             ("Model − / Read + vs Model − / Read −", neg_pos, d["group"] == GROUPS[1]),
             ("Confident Model − / Read + vs Model + / Read +", conf, d["group"] == GROUPS[0]),
             ("Confident Model − / Read + vs Model − / Read −", conf, d["group"] == GROUPS[1]),
             ("Other Model − / Read + vs Model + / Read +", d["group"] == GROUPS[3], d["group"] == GROUPS[0]),
             ("Other Model − / Read + vs Model − / Read −", d["group"] == GROUPS[3], d["group"] == GROUPS[1])]
    rows = []
    for name, a, b in comps:
        part = []
        for m in measures:
            x, y = d.loc[a, m].dropna(), d.loc[b, m].dropna()
            p = stats.mannwhitneyu(x, y).pvalue if len(x) >= 3 and len(y) >= 3 else np.nan
            part.append({"comparison": name, "measure": LABELS[m], "family": FAMILY.get(m, "check"), "n_a": len(x), "n_b": len(y),
                         "median_a": x.median(), "median_b": y.median(), "p": p})
        part = pd.DataFrame(part)
        for fam in ("main", "choroid plexus"):
            ok = part["p"].notna() & (part["family"] == fam)
            if ok.any():
                part.loc[ok, "p_fdr"] = fdr_bh(part.loc[ok, "p"])
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def scatter(d: pd.DataFrame, measures: list, set_label: str, out: Path):
    """Each main measure against the mean cortical SUVR (the amount), one dot per scan."""
    fig, axes = plt.subplots(1, len(measures), figsize=(4.4 * len(measures), 4.6), squeeze=False)
    for ax, m in zip(axes[0], measures):
        for g in GROUPS:
            s = d[d["group"] == g]
            big = g in GROUPS[2:]
            ax.scatter(s["mean"], s[m], s=44 if big else 13, alpha=0.9 if big else 0.35, color=COLOURS[g],
                       edgecolor="white" if big else "none", linewidth=0.5, zorder=3 if big else 2,
                       label=f"{g} (n = {len(s)})")
        ax.set_xlabel("Mean cortical SUVR (× whole cerebellum)", fontsize=11)
        ax.set_title(LABELS[m], fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Focal tau or off-target signal? {set_label}", fontsize=15)
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def maps_figure(freq: dict, affine, set_label: str, out: Path):
    """Rows = groups: per voxel, the share of the group's scans above 1.65 x cerebellum."""
    from nilearn import datasets, plotting
    bg = datasets.load_mni152_template(resolution=1)
    groups = [g for g in GROUPS[:4] if g in freq]
    fig, axes = plt.subplots(len(groups), 1, figsize=(13, 2.3 * len(groups)), facecolor="white", squeeze=False)
    for i, g in enumerate(groups):
        img = nib.Nifti1Image((freq[g]["above"] / freq[g]["n"]).astype(np.float32), affine)
        plotting.plot_stat_map(img, bg_img=bg, display_mode="z", cut_coords=CUTS, axes=axes[i][0], cmap="hot_r",
                               vmax=1.0, threshold=0.02, colorbar=(i == 0), black_bg=False, draw_cross=False,
                               annotate=False, dim=-0.3, figure=fig)
        axes[i][0].text(-0.01, 0.5, f"{g}\n(n = {freq[g]['n']})", transform=axes[i][0].transAxes, ha="right",
                        va="center", fontsize=12)
    axes[0][0].set_title("Share of each group's scans above 1.65 × cerebellum (where a reader sees colour)", fontsize=13)
    fig.suptitle(f"Where the readers' colour would appear, {set_label}", fontsize=15, x=0.08, ha="left")
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    run = Path(args.run_dir)
    out = Path(args.out_dir) if args.out_dir else run / "figures" / f"focality_{args.set}"
    out.mkdir(parents=True, exist_ok=True)
    table = load_table(run, args).set_index("ID")
    table["prob"] = 1 / (1 + np.exp(-table["logodds"]))
    occ_dir = Path(args.occ_dir) if args.occ_dir else None
    cdir = Path(args.centaur_dir) if args.centaur_dir else Path(args.ref_mask).parent

    masks, cortex = None, None
    rows, skipped, refs, freq, no_occ = [], [], {}, {}, 0
    for scan_id, x, aff in suvr_images(Path(args.pkl_dir), table.index, Path(args.ref_mask), skipped, refs):
        img = nib.as_closest_canonical(nib.Nifti1Image(x.astype(np.float32), aff))  # RAS, like the occlusion maps
        x, aff = np.asarray(img.dataobj, dtype=np.float32), img.affine
        if masks is None:
            cortex = DKCortex(x.shape, aff, args.atlas_dir).mask
            masks = {"affine": aff, "centaur": centaur_masks(x.shape, aff, cdir),
                     "deep": deep_mask(x.shape, aff, args.atlas_dir),
                     "chp": choroid_plexus_frac(x.shape, aff, args.atlas_dir) >= 0.5,
                     "grey": on_grid(nib.load(args.grey_ref), x.shape, aff) > 0 if args.grey_ref else None}
            vox_ml = float(abs(np.linalg.det(aff[:3, :3]))) / 1000
            print(f"[grid] {x.shape}, cortex {cortex.sum() * vox_ml:.0f} ml, striatum + thalamus "
                  f"{masks['deep'].sum() * vox_ml:.0f} ml, choroid plexus region {masks['chp'].sum() * vox_ml:.0f} ml")
        occ = None
        if occ_dir is not None:
            f = occ_dir / f"{scan_id}_occlusion_pos_vs_neg.nii.gz"
            if f.exists():
                occ = np.asarray(nib.load(f).dataobj, dtype=np.float32)
                if occ.shape != x.shape:
                    raise SystemExit(f"{f.name} is {occ.shape}, the image {x.shape}: needs the MNI model's occlusion maps")
            else:
                no_occ += 1
        m, above = per_scan(x, cortex, masks, vox_ml, occ)
        if m is None:
            skipped.append(scan_id)
            continue
        r = table.loc[scan_id]
        g = group_of(r["read"], r["call"], r["prob"])
        rows.append({"ID": scan_id, "read": int(r["read"]), "call": int(r["call"]), "prob": float(r["prob"]),
                     "group": g, **m})
        f_ = freq.setdefault(g, {"above": np.zeros(x.shape), "n": 0})
        f_["above"] += above
        f_["n"] += 1
    if skipped:
        print(f"[data] {len(skipped)} scans left out (reference region or cortex not covered)")
    if occ_dir is not None:
        print(f"[occlusion] maps found for {len(rows) - no_occ} of {len(rows)} scans")
    d = pd.DataFrame(rows)
    main_m = [m for m in MAIN + CHP if m in d and d[m].notna().any()]
    check_m = [m for m in CHECKS if m in d and d[m].notna().any()]

    d.round(4).to_csv(out / "focality_per_scan.csv", index=False)
    cols = ["mean", "meta", *main_m, *check_m] + (["grey_vs_whole"] if "grey_vs_whole" in d else [])
    summ = d.groupby("group")[cols].median().reindex(GROUPS).round(3)
    summ.insert(0, "n", d["group"].value_counts().reindex(GROUPS).fillna(0).astype(int))
    summ.to_csv(out / "focality_by_group.csv")
    regions = pd.crosstab(d["group"], d["hot_region"]).reindex(index=GROUPS, columns=REGION_ORDER).fillna(0).astype(int)
    regions.to_csv(out / "focality_hot_regions.csv")
    lt, gt = logit_tests(d, main_m + check_m), group_tests(d, main_m + check_m)
    pd.concat([lt.assign(test="Model − within Read + (logistic; plain and at the same mean cortical SUVR)"),
               gt.assign(test="Mann-Whitney")], ignore_index=True).round(4).to_csv(out / "focality_tests.csv", index=False)
    set_label = "held-out test" if args.set == "test" else args.ext_dataset.removesuffix("_unseen")
    scatter(d, [m for m in main_m if m != "chp"], set_label, out / "focality_scatter.png")
    maps_figure(freq, masks["affine"], set_label, out / "focality_maps.png")
    with pd.option_context("display.width", 230, "display.max_columns", None):
        print("\nMedian per group:\n" + summ.to_string())
        print("\nWhere the hottest 2 ml of cortex is (number of scans):\n" + regions.to_string())
        print("\nWithin Read + scans, odds of Model − per SD (plain / at the same mean cortical SUVR):\n"
              + lt.round(3).to_string(index=False))
        print("\nGroup comparisons (Mann-Whitney, FDR over the main measures):\n" + gt.round(3).to_string(index=False))
        if "grey_vs_whole" in d:
            print(f"\nReference check: inferior cerebellar grey / whole cerebellum, median {d['grey_vs_whole'].median():.3f} "
                  f"(spreadsheet-style SUVR ≈ this scale's SUVR / that)")
    print(f"\nSaved to {out} (focality_per_scan.csv has IDs - keep it local)")


if __name__ == "__main__":
    main()
