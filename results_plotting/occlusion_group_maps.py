#!/usr/bin/env python3
"""Where does the model find its evidence? Group occlusion maps and CenTauR-region shares, from the occlusion
maps of every scan in a set (submit_vis-occlusion-all.sh + merge_occlusion_chunks.py).

The maps are 'pos vs neg' occlusion in log-odds: how much hiding a patch lowers the model's log-odds of
"positive" (above 0 = the patch supports positive, below 0 = it supports negative). Log-odds are on the same
scale in every scan, so the maps can be averaged without rescaling. MNI model only (all scans on one grid).

Outcome groups (cut-off 0.5): Model + / Read + (true positive), Model - / Read + (missed), Model - / Read -
(true negative), Model + / Read - (only shown when there are at least 5).

  1. Group means: one map per group; saved as NIfTI, an axial slice montage with the CenTauR meta-temporal
     region outlined, and a cortical surface view (the surface only samples the cortex - deep spots such as the
     choroid plexus show on the slices only).
  2. CenTauR region shares (main): per scan, the share of its positive evidence (sum of the map above 0) in
     the cortex that falls in each published CenTauR mask (meta-temporal, temporoparietal, mesial temporal,
     frontal; the masks overlap, so the shares do not add up to 1) and outside all of them. Cortex = the
     Desikan-Killiany cortical labels on the template (atlas_utils.py) plus the CenTauR masks themselves, so
     white matter and deep off-target spots do not count. Big regions get big shares by size alone, so each
     share is also given as an enrichment: share / the region's share of the cortex volume (1 = no more than
     its size; 3 = three times more evidence than its size).
     Off-target check: "Outside cortex" = the share of the scan's positive evidence in the whole brain that
     falls outside the cortex (white matter, deep structures such as the striatum / choroid plexus, brainstem,
     cerebellum). A different total from the CenTauR shares above (whole brain, not cortex).
  3. Test: meta-temporal (pre-specified, the comparator region) and temporoparietal share, true positives vs
     missed scans and true positives vs true negatives (permutation test on the difference in medians,
     10,000 shuffles; FDR over the four).
  3b. Amount, not share: the same comparisons on the AMOUNT of positive evidence inside each region (sum of the
     map above 0, in log-odds drop x voxels; same occlusion settings in every scan, so comparable across scans).
     A share can look the same when the total is small; the amount shows whether a group puts less evidence on
     the region. Own FDR family (the four share tests above are unchanged).
  4. FDA regions (optional, --regions; supplementary): the old shares in the atlas-based approximation of the
     TAUVID reading regions (whole brain, each region grown 6 mm), region_shares_fda_*.csv.

Usage:
    python results_plotting/occlusion_group_maps.py <occlusion_all folder> --centaur_dir <data>/masks
        [--atlas_dir <DK atlas folder>] [--regions <fda_regions_MNI152NLin2009cAsym.nii.gz>] [--label "Held-out test"]
Outputs (<folder>/group_maps/, no IDs): group_mean_<group>.nii.gz, group_montage.png, group_surface.png,
centaur_shares_by_group.csv (median share and enrichment per group, plus each region's volume share),
centaur_share_tests.csv, centaur_amount_by_group.csv, centaur_amount_tests.csv (and centaur_shares_per_scan.csv,
which has IDs); with --regions also
region_shares_fda_by_group.csv, region_share_fda_tests.csv, region_shares_fda_per_scan.csv.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image, plotting
from statsmodels.stats.multitest import multipletests

from atlas_utils import CENTAUR_FILES as CENTAUR, DKCortex
from make_case_viewer import MNI_2MM_RAS, MNI_SHAPE, brain_mask, on_mni_grid, region_zones

ZONES = {1: "Posterolateral temporal", 2: "Occipital", 3: "Parietal / precuneus", 4: "Frontal",
         5: "Anterolateral temporal", 6: "Mesial temporal", 0: "Rest of brain"}
FDA_POSITIVE = [1, 2, 3]
OUTSIDE = "Outside CenTauR regions"
OFF_CORTEX = "Outside cortex (share of whole-brain evidence)"
TESTED = ["Meta-temporal", "Temporoparietal"]
GROUPS = [("TP", "Model + / Read +", 1, 1), ("FN", "Model − / Read +", 1, 0),
          ("TN", "Model − / Read −", 0, 0), ("FP", "Model + / Read −", 0, 1)]
MIN_GROUP = 5
CUTS = [-30, -18, -6, 6, 18, 30, 42, 54]  # axial slices (MNI z, mm)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("vis_dir", help="Merged occlusion folder (one map per scan + summary.csv).")
    p.add_argument("--centaur_dir", required=True, help="Folder with the CenTauR masks (Meta_CenTauR.nii.gz, ...).")
    p.add_argument("--atlas_dir", default=None, help="DK atlas folder (default <proj>/data/atlases/lotter2025_mni_freesurfer).")
    p.add_argument("--regions", default=None, help="FDA region map from plot_fda_regions.py (optional, supplementary).")
    p.add_argument("--label", default="", help="Name of the set for figure titles, e.g. 'Held-out test'.")
    p.add_argument("--n_perm", type=int, default=10000)
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


def perm_test(a: np.ndarray, b: np.ndarray, n_perm: int, seed: int = 0) -> tuple[float, float]:
    """Difference in medians a - b, two-sided permutation p-value."""
    rng = np.random.default_rng(seed)
    obs = np.median(a) - np.median(b)
    pooled, n = np.concatenate([a, b]), len(a)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        hits += abs(np.median(pooled[:n]) - np.median(pooled[n:])) >= abs(obs) - 1e-12
    return float(obs), (hits + 1) / (n_perm + 1)


def montage(means: dict, n: dict, regions_img, vmax: float, title: str, out: Path,
            outline: str = "CenTauR meta-temporal"):
    fig = plt.figure(figsize=(16, 2.3 * len(means) + 0.6), facecolor="black")
    for i, (g, img) in enumerate(means.items()):
        ax = fig.add_axes([0, 1 - (i + 1) / len(means) * 0.9 - 0.06, 1, 0.9 / len(means)])
        name = dict((k, lab) for k, lab, _, _ in GROUPS)[g]
        disp = plotting.plot_stat_map(img, display_mode="z", cut_coords=CUTS, axes=ax, figure=fig, cmap="cold_hot",
                                      vmax=vmax, threshold=vmax * 0.05, colorbar=(i == 0), black_bg=True,
                                      annotate=(i == 0), title=f"{name} (n = {n[g]})", draw_cross=False)
        disp.add_contours(regions_img, levels=[0.5], colors="white", linewidths=0.6)
    fig.suptitle(f"{title}  ·  mean occlusion (log-odds of positive): red = supports positive, blue = supports "
                 f"negative  ·  white outline: {outline}", color="white", fontsize=11, y=0.995)
    fig.savefig(out, facecolor="black", dpi=110); plt.close(fig)


def surfaces(means: dict, n: dict, vmax: float, out: Path):
    paths = []
    for g, img in means.items():
        name = dict((k, lab) for k, lab, _, _ in GROUPS)[g]
        f = out.with_name(f"_surface_{g}.png")
        fig = plotting.plot_img_on_surf(img, views=["lateral", "medial"], hemispheres=["left", "right"], cmap="cold_hot",
                                        vmax=vmax, threshold=vmax * 0.05, colorbar=True, symmetric_cbar=True,
                                        title=f"{name} (n = {n[g]})")
        (fig[0] if isinstance(fig, tuple) else fig).savefig(f, dpi=110)
        plt.close("all")
        paths.append(f)
    # stack the per-group surface figures into one picture
    ims = [plt.imread(p) for p in paths]
    w = max(im.shape[1] for im in ims)
    stacked = np.concatenate([np.pad(im, ((0, 0), (0, w - im.shape[1]), (0, 0)), constant_values=1) for im in ims])
    plt.imsave(out, stacked)
    for p in paths:
        p.unlink()


def load_mask(path: Path) -> np.ndarray:
    """A mask (any MNI file) on the model's 2 mm MNI grid. The CenTauR files are 4D (x, y, z, 1): squeezed."""
    img = nib.load(path)
    img = nib.Nifti1Image(np.squeeze(np.asarray(img.dataobj)), img.affine)
    return image.resample_to_img(img, nib.Nifti1Image(np.zeros(MNI_SHAPE, np.int8), MNI_2MM_RAS),
                                 interpolation="nearest", force_resample=True, copy_header=True).get_fdata() > 0


def share_tests(shares: pd.DataFrame, cols: list, n_perm: int) -> pd.DataFrame:
    tests = []
    for col in cols:
        for a, b in [("TP", "FN"), ("TP", "TN")]:
            x = shares.loc[shares["group"] == a, col].dropna().to_numpy()
            y = shares.loc[shares["group"] == b, col].dropna().to_numpy()
            if len(x) >= MIN_GROUP and len(y) >= MIN_GROUP:
                d, p = perm_test(x, y, n_perm)
                tests.append({"region": col, "comparison": f"{a} vs {b}", "n": f"{len(x)} / {len(y)}",
                              "median_diff": d, "p": p})
    tests = pd.DataFrame(tests)
    if len(tests):
        tests["p_fdr"] = multipletests(tests["p"], method="fdr_bh")[1]
    return tests


def by_group(shares: pd.DataFrame, cols: list) -> pd.DataFrame:
    order = [g for g, _, _, _ in GROUPS]
    t = shares.groupby("group")[cols].median().reindex(order).dropna(how="all")
    t.insert(0, "n", shares.groupby("group").size())
    return t


def main():
    args = parse_args()
    vis = Path(args.vis_dir)
    out = Path(args.out_dir) if args.out_dir else vis / "group_maps"
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(vis / "summary.csv")
    summary = summary.dropna(subset=["visual_read", "pred"])
    in_brain = brain_mask(MNI_SHAPE, MNI_2MM_RAS, 0)

    # CenTauR masks and the cortex (DK cortex + the masks), on the model's MNI grid
    centaur = {name: load_mask(Path(args.centaur_dir) / f) & in_brain for name, f in CENTAUR.items()}
    any_centaur = np.logical_or.reduce(list(centaur.values()))
    cortex = (DKCortex(MNI_SHAPE, MNI_2MM_RAS, args.atlas_dir).mask | any_centaur) & in_brain
    regions = {**centaur, OUTSIDE: cortex & ~any_centaur}
    vol_share = {name: m.sum() / cortex.sum() for name, m in regions.items()}
    print(f"[masks] cortex {cortex.sum() * 8 / 1000:.0f} ml; volume share: "
          + ", ".join(f"{k} {v:.0%}" for k, v in vol_share.items()))
    outline_img = nib.Nifti1Image(centaur["Meta-temporal"].astype(np.int8), MNI_2MM_RAS)

    # FDA region zones (optional, supplementary)
    zones = None
    if args.regions:
        zones_1mm, z_aff = region_zones(Path(args.regions))
        zones = image.resample_to_img(nib.Nifti1Image(zones_1mm.astype(np.int16), z_aff),
                                      nib.Nifti1Image(np.zeros(MNI_SHAPE, np.int16), MNI_2MM_RAS),
                                      interpolation="nearest", force_resample=True,
                                      copy_header=True).get_fdata().astype(int)
        zones[~in_brain] = -1  # outside the brain: not counted at all

    sums, counts, shares, fda_shares = {}, {}, [], []
    for _, r in summary.iterrows():
        sid = int(r["ID"])
        f = vis / f"{sid}_occlusion_pos_vs_neg.nii.gz"
        if not f.exists():
            continue
        m = on_mni_grid(f)
        g = next(k for k, _, read, call in GROUPS if read == int(r["visual_read"]) and call == int(r["pred"]))
        sums[g] = m if g not in sums else sums[g] + m
        counts[g] = counts.get(g, 0) + 1
        pos = np.clip(m, 0, None)
        base = {"ID": sid, "group": g, "P_positive": r.get("P_positive", np.nan)}

        total = pos[cortex].sum()
        row = {**base, "cortical_positive_evidence": total}
        for name, mask in regions.items():
            row[f"{name} amount"] = pos[mask].sum()
            row[name] = row[f"{name} amount"] / total if total > 0 else np.nan
            row[f"{name} enrichment"] = row[name] / vol_share[name]
        brain_total = pos[in_brain].sum()
        row[OFF_CORTEX] = pos[in_brain & ~cortex].sum() / brain_total if brain_total > 0 else np.nan
        shares.append(row)

        if zones is not None:
            pos_b = pos * in_brain
            tot_b = pos_b.sum()
            frow = {**base, "total_positive_evidence": tot_b}
            for z, name in ZONES.items():
                frow[name] = pos_b[zones == z].sum() / tot_b if tot_b > 0 else np.nan
            frow["FDA positive regions"] = sum(frow[ZONES[z]] for z in FDA_POSITIVE) if tot_b > 0 else np.nan
            fda_shares.append(frow)
    if not shares:
        raise SystemExit(f"No <ID>_occlusion_pos_vs_neg.nii.gz for the IDs in {vis}/summary.csv")
    shares = pd.DataFrame(shares)
    shares.to_csv(out / "centaur_shares_per_scan.csv", index=False)

    # group means (groups with at least MIN_GROUP scans)
    means = {}
    for g, _, _, _ in GROUPS:
        if counts.get(g, 0) >= MIN_GROUP:
            img = nib.Nifti1Image((sums[g] / counts[g]).astype(np.float32), MNI_2MM_RAS)
            nib.save(img, out / f"group_mean_{g}.nii.gz")
            means[g] = img
    if means:
        vmax = float(np.percentile(np.abs(np.concatenate([i.get_fdata()[in_brain] for i in means.values()])), 99.5))
        montage(means, counts, outline_img, vmax, args.label, out / "group_montage.png")
        surfaces(means, counts, vmax, out / "group_surface.png")

    # CenTauR shares by group (+ each region's volume share for reference) and tests
    names = list(regions)
    cols = names + [f"{k} enrichment" for k in names] + [OFF_CORTEX]
    table = by_group(shares, cols)
    off_vol = (in_brain & ~cortex).sum() / in_brain.sum()
    table.loc["volume share"] = {"n": np.nan, **vol_share, **{f"{k} enrichment": 1.0 for k in names}, OFF_CORTEX: off_vol}
    table.to_csv(out / "centaur_shares_by_group.csv")
    tests = share_tests(shares, TESTED, args.n_perm)
    tests.to_csv(out / "centaur_share_tests.csv", index=False)
    amount_cols = ["cortical_positive_evidence"] + [f"{k} amount" for k in TESTED]
    by_group(shares, amount_cols).to_csv(out / "centaur_amount_by_group.csv")
    amount_tests = share_tests(shares, amount_cols, args.n_perm)
    amount_tests.to_csv(out / "centaur_amount_tests.csv", index=False)

    labels = dict((k, lab) for k, lab, _, _ in GROUPS)
    print(f"{args.label or vis.name}: {len(shares)} scans with maps "
          + ", ".join(f"{labels[g]} {counts[g]}" for g, _, _, _ in GROUPS if g in counts))
    with pd.option_context("display.width", 220, "display.float_format", "{:.2f}".format):
        print("\nMedian share of each scan's cortical positive evidence, by group (last row: share of the cortex volume):")
        print(table[["n"] + names].rename(index=labels).to_string())
        print("\nEnrichment (share / volume share; 1 = no more than the region's size):")
        print(table[[f"{k} enrichment" for k in names]].rename(index=labels).drop(index="volume share").to_string())
        print(f"\nOff-target check, median share of whole-brain positive evidence outside the cortex "
              f"(last row: share of the brain volume):")
        print(table[[OFF_CORTEX]].rename(index=labels).to_string())
    if len(tests):
        print("\nShare in CenTauR regions (permutation test, difference in medians, FDR over all four):")
        for _, t in tests.iterrows():
            print(f"  {t['region']}, {t['comparison']} (n {t['n']}): {t['median_diff']:+.3f}, p = {t['p']:.3g}, "
                  f"FDR {t['p_fdr']:.3g}")

    if len(amount_tests):
        print("\nAmount of positive evidence (sum of log-odds drop; median per group):")
        with pd.option_context("display.width", 220, "display.float_format", "{:.1f}".format):
            print(by_group(shares, amount_cols).rename(index=labels).to_string())
        print("Amount, permutation test (difference in medians, own FDR family):")
        for _, t in amount_tests.iterrows():
            print(f"  {t['region']}, {t['comparison']} (n {t['n']}): {t['median_diff']:+.1f}, p = {t['p']:.3g}, "
                  f"FDR {t['p_fdr']:.3g}")

    if fda_shares:  # supplementary: the FDA-region approximation, as before
        fda = pd.DataFrame(fda_shares)
        fda.to_csv(out / "region_shares_fda_per_scan.csv", index=False)
        fcols = ["FDA positive regions"] + [ZONES[z] for z in [4, 5, 6, 0]] + [ZONES[z] for z in FDA_POSITIVE]
        by_group(fda, fcols).to_csv(out / "region_shares_fda_by_group.csv")
        ft = share_tests(fda, ["FDA positive regions"], args.n_perm)
        ft.to_csv(out / "region_share_fda_tests.csv", index=False)
        print("\nSupplementary, FDA-region approximation (whole brain): share in the FDA positive regions")
        for _, t in ft.iterrows():
            print(f"  {t['comparison']} (n {t['n']}): {t['median_diff']:+.3f}, p = {t['p']:.3g}, FDR {t['p_fdr']:.3g}")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
