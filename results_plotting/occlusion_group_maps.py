#!/usr/bin/env python3
"""Where does the model find its evidence? Group occlusion maps and FDA-region shares, from the occlusion
maps of every scan in a set (submit_vis-occlusion-all.sh + merge_occlusion_chunks.py).

The maps are 'pos vs neg' occlusion in log-odds: how much hiding a patch lowers the model's log-odds of
"positive" (above 0 = the patch supports positive, below 0 = it supports negative). Log-odds are on the same
scale in every scan, so the maps can be averaged without rescaling. MNI model only (all scans on one grid).

Outcome groups (cut-off 0.5): Model + / Read + (true positive), Model - / Read + (missed), Model - / Read -
(true negative), Model + / Read - (only shown when there are at least 5).

  1. Group means: one map per group; saved as NIfTI, an axial slice montage with the FDA positive regions
     outlined, and a cortical surface view (the surface only samples the cortex - deep spots such as the
     choroid plexus show on the slices only).
  2. Region shares: per scan, the share of its positive evidence (sum of the map above 0, inside the brain)
     that falls in each FDA region: the three regions that make a read positive (posterolateral temporal,
     occipital, parietal/precuneus), frontal, anterolateral temporal, mesial temporal, and the rest of the
     brain (deep structures, white matter, cerebellum). The FDA regions are thin cortical ribbons in the atlas,
     so each is grown by 6 mm inside the brain (as in the case viewer); occlusion patches are 8 mm anyway.
  3. Test: share in the FDA positive regions, true positives vs missed scans, and true positives vs true
     negatives (permutation test on the difference in medians, 10,000 shuffles; FDR over the two).

Usage:
    python results_plotting/occlusion_group_maps.py <occlusion_all folder> --regions <fda_regions_MNI152NLin2009cAsym.nii.gz>
        [--label "Held-out test"]
Outputs (<folder>/group_maps/, no IDs): group_mean_<group>.nii.gz, group_montage.png, group_surface.png,
region_shares_by_group.csv, region_share_tests.csv (and region_shares_per_scan.csv, which has IDs).
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

from make_case_viewer import MNI_2MM_RAS, MNI_SHAPE, brain_mask, on_mni_grid, region_zones

ZONES = {1: "Posterolateral temporal", 2: "Occipital", 3: "Parietal / precuneus", 4: "Frontal",
         5: "Anterolateral temporal", 6: "Mesial temporal", 0: "Rest of brain"}
FDA_POSITIVE = [1, 2, 3]
GROUPS = [("TP", "Model + / Read +", 1, 1), ("FN", "Model − / Read +", 1, 0),
          ("TN", "Model − / Read −", 0, 0), ("FP", "Model + / Read −", 0, 1)]
MIN_GROUP = 5
CUTS = [-30, -18, -6, 6, 18, 30, 42, 54]  # axial slices (MNI z, mm)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("vis_dir", help="Merged occlusion folder (one map per scan + summary.csv).")
    p.add_argument("--regions", required=True, help="FDA region map from plot_fda_regions.py (values 1-6).")
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


def montage(means: dict, n: dict, regions_img, vmax: float, title: str, out: Path):
    fig = plt.figure(figsize=(16, 2.3 * len(means) + 0.6), facecolor="black")
    for i, (g, img) in enumerate(means.items()):
        ax = fig.add_axes([0, 1 - (i + 1) / len(means) * 0.9 - 0.06, 1, 0.9 / len(means)])
        name = dict((k, lab) for k, lab, _, _ in GROUPS)[g]
        disp = plotting.plot_stat_map(img, display_mode="z", cut_coords=CUTS, axes=ax, figure=fig, cmap="cold_hot",
                                      vmax=vmax, threshold=vmax * 0.05, colorbar=(i == 0), black_bg=True,
                                      annotate=(i == 0), title=f"{name} (n = {n[g]})", draw_cross=False)
        disp.add_contours(regions_img, levels=[0.5], colors="white", linewidths=0.6)
    fig.suptitle(f"{title}  ·  mean occlusion (log-odds of positive): red = supports positive, blue = supports "
                 "negative  ·  white outline: FDA positive regions", color="white", fontsize=11, y=0.995)
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


def main():
    args = parse_args()
    vis = Path(args.vis_dir)
    out = Path(args.out_dir) if args.out_dir else vis / "group_maps"
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(vis / "summary.csv")
    summary = summary.dropna(subset=["visual_read", "pred"])
    in_brain = brain_mask(MNI_SHAPE, MNI_2MM_RAS, 0)

    # FDA region zones on the model's MNI grid
    zones_1mm, z_aff = region_zones(Path(args.regions))
    zones = image.resample_to_img(nib.Nifti1Image(zones_1mm.astype(np.int16), z_aff),
                                  nib.Nifti1Image(np.zeros(MNI_SHAPE, np.int16), MNI_2MM_RAS),
                                  interpolation="nearest", force_resample=True, copy_header=True).get_fdata().astype(int)
    zones[~in_brain] = -1  # outside the brain: not counted at all
    fda_pos_img = nib.Nifti1Image(np.isin(zones, FDA_POSITIVE).astype(np.int8), MNI_2MM_RAS)

    sums, counts, shares = {}, {}, []
    for _, r in summary.iterrows():
        sid = int(r["ID"])
        f = vis / f"{sid}_occlusion_pos_vs_neg.nii.gz"
        if not f.exists():
            continue
        m = on_mni_grid(f)
        g = next(k for k, _, read, call in GROUPS if read == int(r["visual_read"]) and call == int(r["pred"]))
        sums[g] = m if g not in sums else sums[g] + m
        counts[g] = counts.get(g, 0) + 1
        pos = np.clip(m, 0, None) * in_brain
        total = pos.sum()
        row = {"ID": sid, "group": g, "P_positive": r.get("P_positive", np.nan), "total_positive_evidence": total}
        for z, name in ZONES.items():
            row[name] = pos[zones == z].sum() / total if total > 0 else np.nan
        row["FDA positive regions"] = sum(row[ZONES[z]] for z in FDA_POSITIVE) if total > 0 else np.nan
        shares.append(row)
    if not shares:
        raise SystemExit(f"No <ID>_occlusion_pos_vs_neg.nii.gz for the IDs in {vis}/summary.csv")
    shares = pd.DataFrame(shares)
    shares.to_csv(out / "region_shares_per_scan.csv", index=False)

    # group means (groups with at least MIN_GROUP scans)
    means = {}
    for g, _, _, _ in GROUPS:
        if counts.get(g, 0) >= MIN_GROUP:
            img = nib.Nifti1Image((sums[g] / counts[g]).astype(np.float32), MNI_2MM_RAS)
            nib.save(img, out / f"group_mean_{g}.nii.gz")
            means[g] = img
    if means:
        vmax = float(np.percentile(np.abs(np.concatenate([i.get_fdata()[in_brain] for i in means.values()])), 99.5))
        montage(means, counts, fda_pos_img, vmax, args.label, out / "group_montage.png")
        surfaces(means, counts, vmax, out / "group_surface.png")

    # region shares by group + tests
    cols = ["FDA positive regions"] + [ZONES[z] for z in [4, 5, 6, 0]] + [ZONES[z] for z in FDA_POSITIVE]
    by_group = shares.groupby("group")[cols].median().reindex([g for g, _, _, _ in GROUPS]).dropna(how="all")
    by_group.insert(0, "n", shares.groupby("group").size())
    by_group.to_csv(out / "region_shares_by_group.csv")
    tests = []
    for a, b in [("TP", "FN"), ("TP", "TN")]:
        x = shares.loc[shares["group"] == a, "FDA positive regions"].dropna().to_numpy()
        y = shares.loc[shares["group"] == b, "FDA positive regions"].dropna().to_numpy()
        if len(x) >= MIN_GROUP and len(y) >= MIN_GROUP:
            d, p = perm_test(x, y, args.n_perm)
            tests.append({"comparison": f"{a} vs {b}", "n": f"{len(x)} / {len(y)}", "median_diff": d, "p": p})
    tests = pd.DataFrame(tests)
    if len(tests):
        tests["p_fdr"] = multipletests(tests["p"], method="fdr_bh")[1]
    tests.to_csv(out / "region_share_tests.csv", index=False)

    names = dict((k, lab) for k, lab, _, _ in GROUPS)
    print(f"{args.label or vis.name}: {len(shares)} scans with maps "
          + ", ".join(f"{names[g]} {counts[g]}" for g, _, _, _ in GROUPS if g in counts))
    print("\nMedian share of each scan's positive evidence, by group:")
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        print(by_group.rename(index=names).to_string())
    if len(tests):
        print("\nShare in the FDA positive regions (permutation test, difference in medians):")
        for _, t in tests.iterrows():
            print(f"  {t['comparison']} (n {t['n']}): {t['median_diff']:+.3f}, p = {t['p']:.3g}, FDR {t['p_fdr']:.3g}")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
