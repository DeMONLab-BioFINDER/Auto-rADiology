#!/usr/bin/env python3
"""Voxelwise look at the data: what does an average Read + scan and an average Read - scan look like?

Descriptive only (no statistics yet). For every voxel of the MNI grid:
  mean Read -    average model-input value over all scans read negative
  mean Read +    the same over all scans read positive
  difference     mean Read + minus mean Read -: where the two groups differ
Every scan with a visual read in the demographics csv is used (train and test alike: the read does not
depend on the model).

Input = the cached model inputs of the MNI model (<input_path>/<data_type>, tau_batch_*.pt), i.e. what
the model sees. Only MNI inputs line up across people (raw / 2 mm do not). Each scan was scaled to 0-1 by
its own 1st-99th percentile, so values are brightness relative within a scan, not SUVR: a scan with a lot
of tau has a higher top percentile, so its tau-free areas end up darker. Blue (Read + lower) in the
difference map can come from that, not from less tracer.

Voxels outside the head are 0 in the inputs, and many scanners cut off the top of the head, so a voxel
is averaged only over the scans that have data there (value > 0), and is kept only if at least
--min_coverage of each group has data there, and it is inside nilearn's MNI152 brain mask.

Outputs (default <proj_path>/results/voxelwise/<data_type>, outside the repo; no IDs):
  mean_read_negative.nii.gz, mean_read_positive.nii.gz, difference_read_pos_minus_neg.nii.gz,
  coverage.nii.gz (share of all scans with data per voxel), group_counts.csv (n per site and read),
  axial.png (three rows: the two averages on one colour scale, then the difference on the MNI template),
  surface_<map>.png (each map projected onto the fsaverage5 cortical surface; volume_to_surface only
  samples the cortex, so deep / off-target areas such as the choroid plexus do not show there).

Usage (Berzelius, from the repo; needs nilearn in ai-pet):
    python results_plotting/voxelwise_visual_read.py \\
        --input_path /proj/berzelius-2024-156/users/x_nadpi/data --data_type tau_T1MNI_pts
"""
import argparse
import json
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
MNI_2MM_RAS = np.array([[2.0, 0, 0, -90], [0, 2.0, 0, -126], [0, 0, 2.0, -72], [0, 0, 0, 1]])
AXIAL_CUTS = [-36, -24, -12, 0, 12, 24, 36, 48, 60]  # mm, bottom -> top
GROUPS = {0: "read_negative", 1: "read_positive"}


def parse_args():
    repo = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_path", default=str(repo / "data"), help="Folder that holds <data_type>.")
    p.add_argument("--data_type", default="tau_T1MNI_pts", help="Cached MNI model inputs.")
    p.add_argument("--dataset", default="Gothenburg", help="Dataset name (picks data/demo_<dataset>.csv if it exists).")
    p.add_argument("--demo_csv", default=None, help="Demographics csv (default data/demo_<dataset>.csv, then data/demo.csv).")
    p.add_argument("--min_coverage", type=float, default=0.9,
                   help="Keep a voxel only if at least this share of each group has data there.")
    p.add_argument("--smooth_fwhm", type=float, default=0.0, help="Gaussian smoothing of the maps in mm (0 = none).")
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()
    if args.out_dir is None:
        suffix = f"_fwhm{args.smooth_fwhm:g}" if args.smooth_fwhm else ""
        args.out_dir = str(repo.parent / "results" / "voxelwise" / f"{args.data_type}{suffix}")
    return args


def group_sums(data_dir: Path, reads: pd.Series) -> dict:
    """Per read group: voxelwise sum and number of scans with data (> 0). Streams one batch at a time."""
    manifest = data_dir / "tau_pt_manifest.json"
    batch_size = json.loads(manifest.read_text())["batch_size"]
    sums, counts = {}, {}
    for f in sorted(data_dir.glob("tau_batch_*.pt")):
        b = int(f.stem.split("_")[-1])
        wanted = reads[(reads.index // batch_size) == b]
        if wanted.empty:
            continue
        batch = torch.load(f, map_location="cpu", weights_only=True)
        for scan_id, read in wanted.items():
            x = batch[scan_id % batch_size]
            x = (x.numpy() if torch.is_tensor(x) else np.asarray(x)).astype(np.float64)
            x = x[0] if x.ndim == 4 else x
            if read not in sums:
                sums[read], counts[read] = np.zeros_like(x), np.zeros(x.shape, dtype=np.int32)
            sums[read] += x
            counts[read] += x > 0
        print(f"[load] {f.name}: {len(wanted)} scans")
    return {"sum": sums, "count": counts}


def brain_mask_on_grid(shape) -> np.ndarray:
    target = nib.Nifti1Image(np.zeros(shape, dtype=np.int8), MNI_2MM_RAS)
    mask = image.resample_to_img(datasets.load_mni152_brain_mask(resolution=2), target,
                                 interpolation="nearest", force_resample=True, copy_header=True)
    return mask.get_fdata() > 0


def to_img(a: np.ndarray) -> nib.Nifti1Image:
    return nib.Nifti1Image(a.astype(np.float32), MNI_2MM_RAS)


def plot_axial(maps: dict, n: dict, out: Path, note: str):
    template = datasets.load_mni152_template(resolution=2)
    vmax = float(np.percentile(np.concatenate([maps[g][maps[g] > 0] for g in GROUPS.values()]), 99.5))
    fig, axes = plt.subplots(3, 1, figsize=(16, 7.5))
    for ax, (read, g) in zip(axes, GROUPS.items()):
        title = f"Mean {'Read +' if read else 'Read −'} (n = {n[read]})"
        plotting.plot_img(to_img(maps[g]), display_mode="z", cut_coords=AXIAL_CUTS, axes=ax, cmap="inferno",
                          vmin=0, vmax=vmax, colorbar=True, black_bg=True, title=title, annotate=False)
    plotting.plot_stat_map(to_img(maps["difference"]), bg_img=template, display_mode="z", cut_coords=AXIAL_CUTS,
                           axes=axes[2], cmap="RdBu_r", symmetric_cbar=True, colorbar=True, black_bg=True,
                           title="Difference: mean Read + − mean Read −", annotate=False)
    fig.suptitle(f"Model input (0–1, brightness relative within each scan), axial slices z = "
                 f"{AXIAL_CUTS[0]} to {AXIAL_CUTS[-1]} mm{note}", fontsize=11)
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


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.input_path) / args.data_type

    demo = pd.read_csv(find_demo(args))
    n_scans = json.loads((data_dir / "tau_pt_manifest.json").read_text())["n_subjects"]
    demo = demo[demo["visual_read"].isin([0, 1]) & (demo["ID"] < n_scans)]
    reads = demo.set_index("ID")["visual_read"].astype(int).sort_index()
    n = reads.value_counts().to_dict()
    print(f"[data] {len(reads)} scans with a visual read: {n.get(0, 0)} Read −, {n.get(1, 0)} Read +")
    counts = (demo.assign(read=demo["visual_read"].map({0: "Read −", 1: "Read +"}))
              .groupby(["site", "read"]).size().unstack(fill_value=0))
    counts.loc["all"] = counts.sum()
    counts.to_csv(out_dir / "group_counts.csv")
    print(counts.to_string())

    s = group_sums(data_dir, reads)
    keep = brain_mask_on_grid(s["sum"][0].shape)
    maps = {}
    for read, g in GROUPS.items():
        cnt = s["count"][read]
        keep &= cnt >= args.min_coverage * n[read]
        maps[g] = np.divide(s["sum"][read], cnt, out=np.zeros_like(s["sum"][read]), where=cnt > 0)
    for g in GROUPS.values():
        maps[g] = np.where(keep, maps[g], 0.0)
    if args.smooth_fwhm:
        for g in GROUPS.values():
            maps[g] = np.where(keep, image.smooth_img(to_img(maps[g]), args.smooth_fwhm).get_fdata(), 0.0)
    maps["difference"] = np.where(keep, maps["read_positive"] - maps["read_negative"], 0.0)
    coverage = (s["count"][0] + s["count"][1]) / len(reads)
    print(f"[mask] {int(keep.sum())} voxels kept (brain mask and >= {args.min_coverage:.0%} of each group with data)")

    for name, a in {**{f"mean_{g}": maps[g] for g in GROUPS.values()},
                    "difference_read_pos_minus_neg": maps["difference"], "coverage": coverage}.items():
        nib.save(to_img(a), out_dir / f"{name}.nii.gz")
    d = maps["difference"][keep]
    print(f"[difference] median {np.median(d):+.3f}, 1st-99th percentile {np.percentile(d, 1):+.3f} to "
          f"{np.percentile(d, 99):+.3f}, max {d.max():+.3f}")

    note = f", smoothed {args.smooth_fwhm:g} mm" if args.smooth_fwhm else ""
    plot_axial(maps, n, out_dir / "axial.png", note)
    plot_surface(to_img(maps["difference"]), f"Mean Read + − mean Read −{note}",
                 out_dir / "surface_difference.png", diverging=True)
    for read, g in GROUPS.items():
        plot_surface(to_img(maps[g]), f"Mean {'Read +' if read else 'Read −'} (n = {n[read]}){note}",
                     out_dir / f"surface_mean_{g}.png", diverging=False)


if __name__ == "__main__":
    main()
