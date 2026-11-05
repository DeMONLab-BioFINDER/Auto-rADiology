#!/usr/bin/env python3
"""Voxelwise look at the data: what does an average Read + scan and an average Read - scan look like?

Descriptive only (no statistics yet). For every voxel of the MNI grid:
  mean Read -    average value over all scans read negative
  mean Read +    the same over all scans read positive
  difference     mean Read + minus mean Read -: where the two groups differ
Every scan with a visual read in the demographics csv is used (train and test alike: the read does not
depend on the model). Only MNI images line up across people (raw / 2 mm do not).

Two kinds of input:
  model input (default)  the cached inputs of the MNI model (<input_path>/<data_type>, tau_batch_*.pt), i.e.
      what the model sees. Each scan was scaled to 0-1 by its own 1st-99th percentile, so values are
      brightness relative within a scan, not SUVR: a scan with a lot of tau has a higher top percentile,
      so its tau-free areas end up darker. Blue (Read + lower) in the difference map can come from that.
  SUVR (--ref_mask)  the original MNI images (--pkl_dir, tau_batch_*.pkl), each divided by its own mean
      inside the reference-region mask (e.g. the Centiloid cerebellar grey matter VOI). Sites deliver
      different units (ADNI / NACC already SUVR, the others raw activity); a ratio has no units, so after
      this every scan is "times the reference region". A scan whose reference region is less than half
      covered by data (scanner field of view) is left out and counted per site.

Voxels outside the head (model input) or outside the scanner's field of view are 0 (or NaN), and many
scanners cut off the top of the head, so a voxel is averaged only over the scans that have data there,
and is kept only if at least --min_coverage of each group has data there, and it is inside nilearn's
MNI152 brain mask.

Outputs (default <proj_path>/results/voxelwise/<data_type or suvr_<mask>>, outside the repo; no IDs):
  mean_read_negative.nii.gz, mean_read_positive.nii.gz, difference_read_pos_minus_neg.nii.gz,
  coverage.nii.gz (share of all scans with data per voxel), group_counts.csv (n per site and read),
  axial.png (three rows: the two averages on one colour scale, then the difference on the MNI template),
  surface_<map>.png + surface_<map>.html (interactive 3D version: open in a browser, drag to rotate) (each map projected onto the fsaverage5 cortical surface; volume_to_surface only
  samples the cortex, so deep / off-target areas such as the choroid plexus do not show there),
  SUVR only: reference_by_site.csv (median reference-region value per site in the original units, and
  how many scans were left out).

Usage (Berzelius, from the repo; needs nilearn in ai-pet):
    python results_plotting/voxelwise_visual_read.py \\
        --input_path /proj/berzelius-2024-156/users/x_nadpi/data --data_type tau_T1MNI_pts
    python results_plotting/voxelwise_visual_read.py \\
        --pkl_dir /proj/berzelius-2024-156/dataset/tau-PET/tau_T1MNI \\
        --ref_mask /proj/berzelius-2024-156/users/x_nadpi/data/masks/voi_CerebGry_2mm.nii
"""
import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from nilearn import datasets, image, plotting

from inspect_scans import find_demo

# tau_T1MNI_pts grid: the SPM MNI 2 mm grid (91x109x91, all source pkls share one affine with x running
# right -> left), flipped to RAS by the conversion, so x now runs left -> right from -90 mm.
# The pkls carry their own affine, which the SUVR mode uses instead.
MNI_2MM_RAS = np.array([[2.0, 0, 0, -90], [0, 2.0, 0, -126], [0, 0, 2.0, -72], [0, 0, 0, 1]])
AXIAL_CUTS = [-36, -24, -12, 0, 12, 24, 36, 48, 60]  # mm, bottom -> top
GROUPS = {0: "read_negative", 1: "read_positive"}


def parse_args():
    repo = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_path", default=str(repo / "data"), help="Folder that holds <data_type>.")
    p.add_argument("--data_type", default="tau_T1MNI_pts", help="Cached MNI model inputs.")
    p.add_argument("--pkl_dir", default=None, help="SUVR mode: folder with the original MNI tau_batch_*.pkl.")
    p.add_argument("--ref_mask", default=None, help="SUVR mode: reference-region mask (NIfTI in MNI space).")
    p.add_argument("--dataset", default="Gothenburg", help="Dataset name (picks data/demo_<dataset>.csv if it exists).")
    p.add_argument("--demo_csv", default=None, help="Demographics csv (default data/demo_<dataset>.csv, then data/demo.csv).")
    p.add_argument("--min_coverage", type=float, default=0.9,
                   help="Keep a voxel only if at least this share of each group has data there.")
    p.add_argument("--smooth_fwhm", type=float, default=0.0, help="Gaussian smoothing of the maps in mm (0 = none).")
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()
    if bool(args.pkl_dir) != bool(args.ref_mask):
        p.error("--pkl_dir and --ref_mask go together (SUVR mode)")
    if args.out_dir is None:
        name = f"suvr_{Path(args.ref_mask).name.split('.')[0]}" if args.ref_mask else args.data_type
        suffix = f"_fwhm{args.smooth_fwhm:g}" if args.smooth_fwhm else ""
        args.out_dir = str(repo.parent / "results" / "voxelwise" / f"{name}{suffix}")
    return args


def model_inputs(data_dir: Path, ids: pd.Index):
    """(ID, volume, affine) from the cached model inputs, one batch file at a time."""
    batch_size = json.loads((data_dir / "tau_pt_manifest.json").read_text())["batch_size"]
    for f in sorted(data_dir.glob("tau_batch_*.pt")):
        b = int(f.stem.split("_")[-1])
        wanted = ids[(ids // batch_size) == b]
        if wanted.empty:
            continue
        batch = torch.load(f, map_location="cpu", weights_only=True)
        for scan_id in wanted:
            x = batch[scan_id % batch_size]
            x = (x.numpy() if torch.is_tensor(x) else np.asarray(x)).astype(np.float64)
            yield scan_id, (x[0] if x.ndim == 4 else x), MNI_2MM_RAS
        print(f"[load] {f.name}: {len(wanted)} scans")


def suvr_images(pkl_dir: Path, ids: pd.Index, ref_path: Path, skipped: list, ref_values: dict):
    """(ID, SUVR volume, affine) from the original pkls: each scan divided by its reference-region mean."""
    ref_img, ref, batch_len = nib.load(ref_path), None, None
    wanted_ids = set(ids)
    for f in sorted(pkl_dir.glob("tau_batch_*.pkl")):
        b = int(f.stem.split("_")[-1])
        with open(f, "rb") as fh:
            data = pickle.load(fh)
        batch_len = batch_len or len(data)  # IDs index batches the same way src/data.py does
        n = 0
        for idx, sample in enumerate(data):
            scan_id = b * batch_len + idx
            if scan_id not in wanted_ids:
                continue
            x = np.squeeze(np.asarray(sample["image"], dtype=np.float64))
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            affine = np.asarray(sample["meta"]["affine"], dtype=float)
            if ref is None:  # all scans share one grid; put the mask on it once
                ref = image.resample_to_img(ref_img, nib.Nifti1Image(np.zeros(x.shape, np.int8), affine),
                                            interpolation="nearest", force_resample=True,
                                            copy_header=True).get_fdata() > 0
                print(f"[ref] {ref_path.name}: {int(ref.sum())} voxels on the scan grid")
            in_ref = ref & (x != 0)
            if in_ref.sum() < 0.5 * ref.sum():
                skipped.append(scan_id)
                continue
            ref_values[scan_id] = x[in_ref].mean()
            n += 1
            yield scan_id, x / ref_values[scan_id], affine
        print(f"[load] {f.name}: {n} scans")


def group_sums(scans, reads: pd.Series) -> dict:
    """Per read group: voxelwise sum, number of scans with data (non-zero) and number of scans."""
    sums, counts, n, affine = {}, {}, {0: 0, 1: 0}, None
    for scan_id, x, aff in scans:
        if affine is None:
            affine = aff
        elif not np.allclose(aff, affine):
            raise SystemExit(f"Scan {scan_id} is on a different grid than the others; not voxel-aligned.")
        read = reads[scan_id]
        if read not in sums:
            sums[read], counts[read] = np.zeros_like(x), np.zeros(x.shape, dtype=np.int32)
        sums[read] += x
        counts[read] += x != 0
        n[read] += 1
    return {"sum": sums, "count": counts, "n": n, "affine": affine}


def brain_mask_on_grid(shape, affine) -> np.ndarray:
    target = nib.Nifti1Image(np.zeros(shape, dtype=np.int8), affine)
    mask = image.resample_to_img(datasets.load_mni152_brain_mask(resolution=2), target,
                                 interpolation="nearest", force_resample=True, copy_header=True)
    return mask.get_fdata() > 0


def plot_axial(imgs: dict, n: dict, out: Path, what: str):
    template = datasets.load_mni152_template(resolution=2)
    vals = np.concatenate([imgs[g].get_fdata()[imgs[g].get_fdata() > 0] for g in GROUPS.values()])
    vmax = float(np.percentile(vals, 99.5))
    fig, axes = plt.subplots(3, 1, figsize=(16, 7.5))
    for ax, (read, g) in zip(axes, GROUPS.items()):
        title = f"Mean {'Read +' if read else 'Read −'} (n = {n[read]})"
        plotting.plot_img(imgs[g], display_mode="z", cut_coords=AXIAL_CUTS, axes=ax, cmap="inferno",
                          vmin=0, vmax=vmax, colorbar=True, black_bg=True, title=title, annotate=False)
    plotting.plot_stat_map(imgs["difference"], bg_img=template, display_mode="z", cut_coords=AXIAL_CUTS,
                           axes=axes[2], cmap="RdBu_r", symmetric_cbar=True, colorbar=True, black_bg=True,
                           title="Difference: mean Read + − mean Read −", annotate=False)
    fig.suptitle(f"{what}, axial slices z = {AXIAL_CUTS[0]} to {AXIAL_CUTS[-1]} mm", fontsize=11)
    fig.savefig(out, dpi=130, facecolor="black")
    plt.close(fig)
    print(f"[save] {out}")


def plot_surface(img: nib.Nifti1Image, title: str, out: Path, diverging: bool):
    kw = dict(cmap="RdBu_r", symmetric_cbar=True) if diverging else dict(cmap="inferno", symmetric_cbar=False, vmin=0)
    fig, _ = plotting.plot_img_on_surf(img, surf_mesh="fsaverage5", views=["lateral", "medial"],
                                       hemispheres=["left", "right"], colorbar=True, cbar_tick_format="%.2f",
                                       title=title, **kw)
    fig.set_size_inches(9, 9)  # default is too narrow for the colour-bar labels
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[save] {out}")
    # same map as an interactive 3D page (drag to rotate, scroll to zoom); open the .html in a browser
    view_kw = dict(cmap="RdBu_r", symmetric_cmap=True) if diverging else dict(cmap="inferno", symmetric_cmap=False, vmin=0)
    try:
        view = plotting.view_img_on_surf(img, surf_mesh="fsaverage5", title=title, **view_kw)
    except ImportError:  # the interactive page needs plotly; the PNG above is already saved
        print("[skip] interactive .html needs plotly (pip install plotly in ai-pet)")
        return
    view.save_as_html(out.with_suffix(".html"))
    print(f"[save] {out.with_suffix('.html')}")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    demo = pd.read_csv(find_demo(args))
    demo = demo[demo["visual_read"].isin([0, 1])]
    if not args.ref_mask:
        data_dir = Path(args.input_path) / args.data_type
        demo = demo[demo["ID"] < json.loads((data_dir / "tau_pt_manifest.json").read_text())["n_subjects"]]
    reads = demo.set_index("ID")["visual_read"].astype(int).sort_index()

    skipped, ref_values = [], {}
    if args.ref_mask:
        scans = suvr_images(Path(args.pkl_dir), reads.index, Path(args.ref_mask), skipped, ref_values)
        what = f"SUVR (reference: {Path(args.ref_mask).name})"
    else:
        scans = model_inputs(data_dir, reads.index)
        what = "Model input (0–1, brightness relative within each scan)"
    s = group_sums(scans, reads)
    n = s["n"]
    print(f"[data] {n[0] + n[1]} scans with a visual read: {n[0]} Read −, {n[1]} Read +"
          + (f" ({len(skipped)} left out: reference region not covered)" if args.ref_mask else ""))

    used = demo[demo["ID"].isin(set(reads.index) - set(skipped))]
    if args.ref_mask:
        used = used[used["ID"].isin(ref_values)]
    counts = (used.assign(read=used["visual_read"].map({0: "Read −", 1: "Read +"}))
              .groupby(["site", "read"]).size().unstack(fill_value=0))
    counts.loc["all"] = counts.sum()
    counts.to_csv(out_dir / "group_counts.csv")
    print(counts.to_string())
    if args.ref_mask:
        site = demo.set_index("ID")["site"]
        ref = pd.DataFrame({"site": site[list(ref_values)].values, "reference": list(ref_values.values())})
        by_site = ref.groupby("site")["reference"].agg(n="size", median_reference="median")
        by_site["left_out"] = site[skipped].value_counts().reindex(by_site.index, fill_value=0)
        by_site.to_csv(out_dir / "reference_by_site.csv")
        print(by_site.round(3).to_string())

    affine = s["affine"]
    keep = brain_mask_on_grid(s["sum"][0].shape, affine)
    maps = {}
    for read, g in GROUPS.items():
        cnt = s["count"][read]
        keep &= cnt >= args.min_coverage * n[read]
        maps[g] = np.divide(s["sum"][read], cnt, out=np.zeros_like(s["sum"][read]), where=cnt > 0)
    for g in GROUPS.values():
        maps[g] = np.where(keep, maps[g], 0.0)
        if args.smooth_fwhm:
            smoothed = image.smooth_img(nib.Nifti1Image(maps[g], affine), args.smooth_fwhm).get_fdata()
            maps[g] = np.where(keep, smoothed, 0.0)
    maps["difference"] = np.where(keep, maps["read_positive"] - maps["read_negative"], 0.0)
    coverage = (s["count"][0] + s["count"][1]) / (n[0] + n[1])
    print(f"[mask] {int(keep.sum())} voxels kept (brain mask and >= {args.min_coverage:.0%} of each group with data)")

    imgs = {k: nib.Nifti1Image(v.astype(np.float32), affine) for k, v in maps.items()}
    for name, img in {**{f"mean_{g}": imgs[g] for g in GROUPS.values()},
                      "difference_read_pos_minus_neg": imgs["difference"],
                      "coverage": nib.Nifti1Image(coverage.astype(np.float32), affine)}.items():
        nib.save(img, out_dir / f"{name}.nii.gz")
    d = maps["difference"][keep]
    print(f"[difference] median {np.median(d):+.3f}, 1st-99th percentile {np.percentile(d, 1):+.3f} to "
          f"{np.percentile(d, 99):+.3f}, max {d.max():+.3f}")

    note = f", smoothed {args.smooth_fwhm:g} mm" if args.smooth_fwhm else ""
    plot_axial(imgs, n, out_dir / "axial.png", what + note)
    plot_surface(imgs["difference"], f"Mean Read + − mean Read −{note}", out_dir / "surface_difference.png",
                 diverging=True)
    for read, g in GROUPS.items():
        plot_surface(imgs[g], f"Mean {'Read +' if read else 'Read −'} (n = {n[read]}){note}",
                     out_dir / f"surface_mean_{g}.png", diverging=False)


if __name__ == "__main__":
    main()
