#!/usr/bin/env python3
"""Voxelwise statistics on tau SUVR for one final visual-read model (MNI): where in the brain does tau go
with the model's output, beyond the visual read and the meta-temporal tau level?

One set at a time (held-out test or AVID; never pooled). Each scan's SUVR image is the original MNI image
divided by its own mean in the reference region (as in voxelwise_visual_read.py), smoothed (default 6 mm).

General linear models, fitted at every voxel (voxel SUVR is the outcome):
  model_beyond_read   SUVR ~ model log-odds + read + MetaTemporal SUVR + age + sex [+ site]
                      Where is tau higher in scans the model scores higher, when the read and the meta-temporal
                      average are the same? = the spatial pattern behind the model's output.
  read_beyond_model   the same model, tested on the read instead: where is tau higher in read-positive scans
                      when the model's output is the same? = what readers see that the model does not weigh.
  missed_vs_detected  read + scans only: SUVR ~ missed (Model - / Read +, vs Model + / Read +)
                      + MetaTemporal SUVR + age + sex [+ site]. Where do missed scans have less (or more) tau
                      than detected ones at the same meta-temporal level?
  read_pos_vs_neg     SUVR ~ read + age + sex [+ site]. Reference map: where read-positive scans have more tau.

Inference: permutation test (nilearn permuted_ols, Freedman-Lane, two-sided), family-wise error corrected
over the whole brain, with threshold-free cluster enhancement (TFCE) as the main map and voxel-level max-t
alongside. Significant = FWE p < 0.05. Model log-odds use probabilities clipped to [1e-4, 1 - 1e-4], as in
regress_model_output_visual_read.py. Continuous covariates are z-scored.

Voxels: inside nilearn's MNI152 brain mask and covered (non-zero) in every scan used. Scans covering less than
--min_scan_coverage of the brain (scanner field of view cuts off the top of the head) are left out, so a few
short scans do not remove the top of the brain for everyone; how many is printed.

Usage (Berzelius, a job: the permutations take a while, see sbatch_scripts/submit_voxelwise-stats.sh):
    python results_plotting/voxelwise_stats_visual_read.py <final_run_dir> --set test \\
        --pkl_dir /proj/berzelius-2024-156/dataset/tau-PET/tau_T1MNI \\
        --ref_mask /proj/berzelius-2024-156/users/x_nadpi/data/masks/voi_CerebGry_tau_2mm.nii \\
        [--regions <fda_regions_MNI152NLin2009cAsym.nii.gz>] [--n_perm 5000] [--n_jobs 16]
Outputs (<run_dir>/figures/voxelwise_stats_<set>/, no IDs): per analysis <name>_t.nii.gz,
<name>_logp_tfce.nii.gz, <name>_logp_maxt.nii.gz (-log10 FWE p), <name>.png; summary.csv (n, voxels
significant each way, peak), fda_region_overlap.csv (with --regions: share of significant voxels per FDA region).
"""
import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image, plotting
from nilearn.maskers import NiftiMasker
from nilearn.mass_univariate import permuted_ols
from scipy.ndimage import gaussian_filter

from make_case_viewer import region_zones
from plot_utils import add_subgroup_columns, find_demo_csv, resolve_path
from regress_model_output_visual_read import CLIP, find_ext_demo
from voxelwise_visual_read import brain_mask_on_grid, suvr_images

ZONES = {1: "Posterolateral temporal", 2: "Occipital", 3: "Parietal / precuneus", 4: "Frontal",
         5: "Anterolateral temporal", 6: "Mesial temporal", 0: "Rest of brain"}
CUTS = [-30, -18, -6, 6, 18, 30, 42, 54]  # axial slices (MNI z, mm), as in occlusion_group_maps.py
ALPHA = 0.05
TITLES = {
    "model_beyond_read": "Higher model output at the same read and meta-temporal SUVR",
    "read_beyond_model": "Read + at the same model output and meta-temporal SUVR",
    "missed_vs_detected": "Missed (Model − / Read +) vs detected (Model + / Read +), same meta-temporal SUVR",
    "read_pos_vs_neg": "Read + vs Read − (age, sex, site adjusted)",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--set", default="test", choices=["test", "avid"])
    p.add_argument("--pkl_dir", required=True, help="Folder with the original MNI tau_batch_*.pkl.")
    p.add_argument("--ref_mask", required=True, help="Reference-region mask (NIfTI, MNI space).")
    p.add_argument("--regions", default=None, help="FDA region map (plot_fda_regions.py) for the overlap table.")
    p.add_argument("--dataset", default="Gothenburg")
    p.add_argument("--ext_dataset", default="AVID_unseen")
    p.add_argument("--demo_csv", default=None, help="Demographics for the test set (default data/demo.csv).")
    p.add_argument("--ext_demo_csv", default=None, help="Demographics for AVID (default data/demo_AVID_unseen.csv).")
    p.add_argument("--female_code", type=float, default=2)
    p.add_argument("--analyses", nargs="+", default=list(TITLES), choices=list(TITLES))
    p.add_argument("--fwhm", type=float, default=6.0, help="Smoothing in mm (0 = none).")
    p.add_argument("--min_scan_coverage", type=float, default=0.95)
    p.add_argument("--n_perm", type=int, default=5000)
    p.add_argument("--n_jobs", type=int, default=1)
    p.add_argument("--no_tfce", action="store_true", help="Voxel-level max-t only (much faster).")
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


def load_table(run: Path, args) -> pd.DataFrame:
    """One row per scan in the set: model log-odds, read, call and covariates."""
    if args.set == "test":
        f = run / "evaluation" / args.dataset / f"Eval_{args.dataset}_results.csv"
        demo_path = resolve_path(args.demo_csv, find_demo_csv)
    else:
        hits = sorted((run / "validation").glob(f"External_validation_{args.ext_dataset}__*_zeroshot_results.csv"))
        f = hits[0] if hits else run / "validation" / "missing"
        demo_path = resolve_path(args.ext_demo_csv, lambda: find_ext_demo(args.ext_dataset))
    if demo_path is None or not f.exists():
        raise SystemExit(f"Missing predictions ({f}) or demographics for the {args.set} set")
    d = pd.read_csv(f)[["ID_ind", "y", "pred", "prob"]]
    demo = pd.read_csv(demo_path)
    demo = demo.loc[:, ~demo.columns.astype(str).str.startswith("Unnamed")]
    d = d.merge(demo, left_on="ID_ind", right_on="ID", how="left")
    add_subgroup_columns(d, args.female_code)
    p = d["prob"].clip(CLIP, 1 - CLIP)
    t = pd.DataFrame({
        "ID": d["ID_ind"].astype(int), "logodds": np.log(p / (1 - p)),
        "read": d["y"].astype(int), "call": d["pred"].astype(int),
        "MetaTemporal": pd.to_numeric(d["MetaTemporal"], errors="coerce"),
        "age": pd.to_numeric(d["age"], errors="coerce"), "sex": d["_g_sex"], "site": d["_g_site"],
    })
    n0 = len(t)
    t = t.dropna(subset=["MetaTemporal", "age", "sex"]).reset_index(drop=True)
    if len(t) < n0:
        print(f"[data] {n0 - len(t)} scans left out: missing meta-temporal SUVR, age or sex")
    return t


def load_images(table: pd.DataFrame, args):
    """SUVR images of the scans in the table, smoothed, as (scans x brain voxels) plus per-scan coverage."""
    skipped, ref_values = [], {}
    rows, covers, ids, affine, brain = [], [], [], None, None
    sigma = None
    for scan_id, x, aff in suvr_images(Path(args.pkl_dir), pd.Index(table["ID"]), Path(args.ref_mask),
                                       skipped, ref_values):
        if affine is None:
            affine, brain = aff, brain_mask_on_grid(x.shape, aff)
            vox = np.sqrt((aff[:3, :3] ** 2).sum(0))
            sigma = args.fwhm / np.sqrt(8 * np.log(2)) / vox if args.fwhm else None
        cover = x != 0
        if sigma is not None:
            # smooth only within the field of view: blur the image and the coverage, then divide, so the zeros
            # outside the scanner's view do not pull down the voxels near its edge
            w = gaussian_filter(cover.astype(np.float64), sigma)
            x = np.divide(gaussian_filter(x, sigma), w, out=np.zeros_like(x), where=w > 1e-3)
        rows.append(x[brain].astype(np.float32))
        covers.append(cover[brain])
        ids.append(scan_id)
    if not rows:
        raise SystemExit("No images loaded (check --pkl_dir and the IDs)")
    if skipped:
        print(f"[data] {len(skipped)} scans left out: reference region not covered")
    return np.vstack(rows), np.vstack(covers), np.array(ids), affine, brain


def design(t: pd.DataFrame, analysis: str):
    """(rows used, tested variable, confounds) for one analysis."""
    base = t
    if analysis == "missed_vs_detected":
        base = t[t["read"] == 1]
    z = lambda s: (s - s.mean()) / s.std()
    conf = {"age": z(base["age"]), "male": (base["sex"] == "Male").astype(float)}
    if base["site"].nunique() > 1:
        dummies = pd.get_dummies(base["site"], prefix="site", drop_first=True, dtype=float)
        conf.update({c: dummies[c] for c in dummies})
    if analysis == "model_beyond_read":
        tested = z(base["logodds"])
        conf.update(read=base["read"].astype(float), meta=z(base["MetaTemporal"]))
    elif analysis == "read_beyond_model":
        tested = base["read"].astype(float)
        conf.update(logodds=z(base["logodds"]), meta=z(base["MetaTemporal"]))
    elif analysis == "missed_vs_detected":
        tested = (base["call"] == 0).astype(float)
        conf.update(meta=z(base["MetaTemporal"]))
    else:
        tested = base["read"].astype(float)
    return base.index.to_numpy(), tested.to_numpy()[:, None], pd.DataFrame(conf).to_numpy()


def plot_map(t_img, sig_img, title: str, out: Path, regions_img=None):
    """t-map shown only where significant (FWE p < 0.05); red = tau higher with the tested variable."""
    t = t_img.get_fdata() * (sig_img.get_fdata() > 0)
    shown = nib.Nifti1Image(t.astype(np.float32), t_img.affine)
    vmax = float(np.abs(t_img.get_fdata()).max()) or 1.0
    fig = plt.figure(figsize=(16, 2.6), facecolor="black")
    disp = plotting.plot_stat_map(shown, display_mode="z", cut_coords=CUTS, figure=fig, cmap="cold_hot", vmax=vmax,
                                  threshold=1e-6, colorbar=True, black_bg=True, draw_cross=False, annotate=True,
                                  title=f"{title}  (t, FWE p < {ALPHA})")
    if regions_img is not None:
        disp.add_contours(regions_img, levels=[0.5], colors="white", linewidths=0.6)
    fig.savefig(out, facecolor="black", dpi=110)
    plt.close(fig)


def main():
    args = parse_args()
    run = Path(args.run_dir)
    out = Path(args.out_dir) if args.out_dir else run / "figures" / f"voxelwise_stats_{args.set}"
    out.mkdir(parents=True, exist_ok=True)

    table = load_table(run, args)
    X, cover, ids, affine, brain = load_images(table, args)
    table = table.set_index("ID").loc[ids].reset_index()  # same order as the images

    # leave out scans that miss a lot of the brain, then keep voxels every remaining scan covers
    frac = cover.mean(1)
    ok = frac >= args.min_scan_coverage
    if (~ok).any():
        print(f"[data] {int((~ok).sum())} scans left out: cover < {args.min_scan_coverage:.0%} of the brain "
              f"(field of view)")
    X, cover, table = X[ok], cover[ok], table[ok].reset_index(drop=True)
    vox = cover.all(0)
    X = X[:, vox]
    keep = brain.copy()
    keep[brain] = vox
    print(f"[data] {len(table)} scans ({int(table['read'].sum())} read +), {int(vox.sum())} of {int(brain.sum())} "
          f"brain voxels covered in every scan; smoothing {args.fwhm:g} mm")
    masker = NiftiMasker(nib.Nifti1Image(keep.astype(np.int8), affine)).fit()
    # the masker orders voxels like keep[...] boolean indexing, the same order as X's columns

    regions_img, zones = None, None
    if args.regions:
        zones_1mm, z_aff = region_zones(Path(args.regions))
        zones = image.resample_to_img(nib.Nifti1Image(zones_1mm.astype(np.int16), z_aff),
                                      nib.Nifti1Image(np.zeros(keep.shape, np.int16), affine),
                                      interpolation="nearest", force_resample=True, copy_header=True).get_fdata().astype(int)
        regions_img = nib.Nifti1Image(np.isin(zones, [1, 2, 3]).astype(np.int8), affine)

    summary, overlap = [], []
    for name in args.analyses:
        rows, tested, conf = design(table, name)
        if name == "missed_vs_detected" and min(tested.sum(), len(tested) - tested.sum()) < 5:
            print(f"[skip] {name}: fewer than 5 scans in a group")
            continue
        t0 = time.time()
        res = permuted_ols(tested, X[rows], confounding_vars=conf, model_intercept=True, n_perm=args.n_perm,
                           two_sided_test=True, random_state=0, n_jobs=args.n_jobs, masker=masker,
                           tfce=not args.no_tfce, output_type="dict")
        t = res["t"][0]
        logp = res["logp_max_tfce"][0] if not args.no_tfce else res["logp_max_t"][0]
        sig = logp > -np.log10(ALPHA)
        imgs = {"t": t, "logp_maxt": res["logp_max_t"][0]}
        if not args.no_tfce:
            imgs["logp_tfce"] = res["logp_max_tfce"][0]
        for k, v in imgs.items():
            nib.save(masker.inverse_transform(v.astype(np.float32)), out / f"{name}_{k}.nii.gz")
        t_img = masker.inverse_transform(t.astype(np.float32))
        sig_img = masker.inverse_transform(sig.astype(np.float32))
        plot_map(t_img, sig_img, TITLES[name], out / f"{name}.png", regions_img)

        peak = int(np.argmax(np.abs(t) * sig)) if sig.any() else int(np.argmax(np.abs(t)))
        ijk = np.argwhere(keep)[peak]
        xyz = nib.affines.apply_affine(affine, ijk)
        n_tested = int(tested.sum()) if set(np.unique(tested)) <= {0, 1} else len(rows)
        summary.append({"analysis": name, "n_scans": len(rows), "n_tested_group": n_tested,
                        "voxels_sig_higher": int((sig & (t > 0)).sum()), "voxels_sig_lower": int((sig & (t < 0)).sum()),
                        "voxels_sig_maxt": int((res["logp_max_t"][0] > -np.log10(ALPHA)).sum()),
                        "peak_t": float(t[peak]), "peak_x": xyz[0], "peak_y": xyz[1], "peak_z": xyz[2],
                        "min_fwe_p": float(10 ** -logp.max()), "minutes": (time.time() - t0) / 60})
        if zones is not None:
            z_in = zones[keep]
            for sign, mask in [("higher", sig & (t > 0)), ("lower", sig & (t < 0))]:
                if mask.sum():
                    overlap.append({"analysis": name, "direction": sign, "n_voxels": int(mask.sum()),
                                    **{ZONES[k]: float((z_in[mask] == k).mean()) for k in ZONES}})
        s = summary[-1]
        print(f"[{name}] n = {s['n_scans']}: {s['voxels_sig_higher']} voxels tau higher, {s['voxels_sig_lower']} lower "
              f"(TFCE FWE p < {ALPHA}); peak t = {s['peak_t']:+.2f} at ({xyz[0]:.0f}, {xyz[1]:.0f}, {xyz[2]:.0f}); "
              f"{s['minutes']:.1f} min")

    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    if overlap:
        ov = pd.DataFrame(overlap)
        ov.to_csv(out / "fda_region_overlap.csv", index=False)
        with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
            print("\nShare of significant voxels in each FDA region (grown 6 mm):")
            print(ov.to_string(index=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
