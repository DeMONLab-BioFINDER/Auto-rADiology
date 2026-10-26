#!/usr/bin/env python3
"""How blurry are the tau-PET images, per site? Were some sites smoothed before we got them?

ADNI's ready-processed PET is usually smoothed to a common ~8 mm resolution; other sites may
not be. This estimates each scan's smoothness as an effective FWHM in mm, the standard estimate
from how similar neighbouring voxels are (Forman et al. 1995, as in AFNI 3dFWHM):
    FWHM = voxel size * sqrt(-2 ln 2 / ln(1 - var(neighbour difference) / (2 var(image))))
per axis, inside the head (eroded, so the head-air edge does not count). It does not depend on
the units (SUVR or raw counts), but it does include the brain's own structure, so it is a
comparison between sites (best on tau_T1MNI, where every scan has the same grid and anatomy),
not the scanner resolution itself.

For reference, every scan is also measured after an extra 8 mm smoothing: if a site's original
FWHM is about another site's "+8 mm" value, it was probably smoothed already.

Reads only, no GPU needed. Outputs (default <proj_path>/results/image_sharpness/, outside the repo):
    sharpness_<data_type>.csv          - one row per scan
    sharpness_<data_type>_by_site.csv  - medians and ranges per site
    sharpness_<data_type>_slices.png   - one middle slice per site (each scan its own brightness),
                                         to see the blur by eye
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `src.*` imports

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_erosion, gaussian_filter

from src.data import brain_outer_mask
from scripts.check_image_geometry import load_sites, voxel_sizes

FWHM_TO_SIGMA = 1 / (2 * np.sqrt(2 * np.log(2)))  # sigma = FWHM / 2.355


def fwhm_mm(img, mask, vox_mm):
    """Effective FWHM in mm per axis, from neighbour differences inside the mask."""
    v = img[mask].var()
    out = []
    for ax in range(3):
        a = [slice(None)] * 3
        b = [slice(None)] * 3
        a[ax], b[ax] = slice(1, None), slice(None, -1)
        both = mask[tuple(a)] & mask[tuple(b)]
        d = (img[tuple(a)] - img[tuple(b)])[both]
        r = d.var() / (2 * v) if v > 0 and d.size else np.nan
        out.append(float(vox_mm[ax] * np.sqrt(-2 * np.log(2) / np.log(1 - r))) if 0 < r < 1 else np.nan)
    return out


def scan_row(img, meta, extra_mm, erode_mm):
    img = np.squeeze(np.clip(np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0), 0, None)).astype(np.float32)
    vox = voxel_sizes(meta["affine"]) if "affine" in meta else np.ones(3)
    head = brain_outer_mask(torch.as_tensor(img)[None]).squeeze().numpy().astype(bool)
    n_erode = max(1, int(round(erode_mm / float(np.mean(vox)))))
    mask = binary_erosion(head, iterations=n_erode)
    if mask.sum() < 1000:
        mask = head
    f = fwhm_mm(img, mask, vox)
    sm = gaussian_filter(img, sigma=[extra_mm * FWHM_TO_SIGMA / v for v in vox])
    g = fwhm_mm(sm, mask, vox)
    return {"vox_mm": "x".join(f"{v:.2f}" for v in vox),
            "fwhm_x": f[0], "fwhm_y": f[1], "fwhm_z": f[2],
            "fwhm_mm": float(np.exp(np.nanmean(np.log(f)))),
            f"fwhm_mm_plus{extra_mm:g}": float(np.exp(np.nanmean(np.log(g)))),
            "mask_voxels": int(mask.sum())}, img


def middle_slice(img):
    """Axial slice through the centre of mass of the head (last axis)."""
    z = np.arange(img.shape[2])
    w = img.sum(axis=(0, 1))
    k = int(round((z * w).sum() / w.sum())) if w.sum() > 0 else img.shape[2] // 2
    s = img[:, :, k]
    hi = np.percentile(s[s > 0], 99) if (s > 0).any() else 1.0
    return np.rot90(np.clip(s / hi, 0, 1))


def main():
    repo = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input_path", default="", help="Folder that relative --data_types are under (and holding demo.csv)")
    p.add_argument("--data_types", default="tau_raw_backup",
                   help="Comma-separated folders with tau_batch_*.pkl: names under --input_path, or absolute paths")
    p.add_argument("--dataset", default="Gothenburg", help="Used to find each folder's demographics_<dataset>.csv")
    p.add_argument("--max_per_site", type=int, default=20, help="Scans checked per site (0 = all)")
    p.add_argument("--extra_mm", type=float, default=8.0, help="Reference smoothing added to every scan (FWHM mm)")
    p.add_argument("--erode_mm", type=float, default=6.0, help="Head mask shrunk by this much, to leave out the head-air edge")
    p.add_argument("--out_dir", default=str(repo.parent / "results" / "image_sharpness"))
    args = p.parse_args()

    input_path, out_dir = Path(args.input_path), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plus = f"fwhm_mm_plus{args.extra_mm:g}"

    for dt in [d.strip() for d in args.data_types.split(",") if d.strip()]:
        data_dir = Path(dt) if Path(dt).is_absolute() else input_path / dt
        name = data_dir.name
        pkl_files = sorted(data_dir.glob("tau_batch_*.pkl"))
        if not pkl_files:
            print(f"[{name}] no tau_batch_*.pkl files, skipped")
            continue
        demo_csv = data_dir / f"demographics_{args.dataset}.csv"
        if not demo_csv.exists():
            demo_csv = input_path / "demo.csv"
        demo = load_sites(demo_csv)
        site_of = dict(zip(demo["ID"], demo["site"])) if demo is not None else {}
        print(f"[{name}] {len(pkl_files)} pkl files; sites from {demo_csv if demo is not None else '(none)'}")

        rows, n_site, slices, batch_len = [], {}, {}, None
        for pkl_path in pkl_files:
            batch_id = int(pkl_path.stem.split("_")[-1])
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            batch_len = batch_len or len(data)  # IDs index batches the same way src/data.py does
            for idx, sample in enumerate(data):
                sid = batch_id * batch_len + idx
                site = site_of.get(sid, "unknown")
                if args.max_per_site and n_site.get(site, 0) >= args.max_per_site:
                    continue
                n_site[site] = n_site.get(site, 0) + 1
                row, img = scan_row(sample["image"], sample.get("meta") or {}, args.extra_mm, args.erode_mm)
                rows.append({"ID": sid, "site": site, **row})
                slices.setdefault(site, middle_slice(img))
        df = pd.DataFrame(rows)
        df.round(3).to_csv(out_dir / f"sharpness_{name}.csv", index=False)

        by_site = df.groupby("site", dropna=False).agg(
            n=("ID", "size"),
            vox_mm=("vox_mm", lambda v: v.mode().iat[0]),
            fwhm_mm=("fwhm_mm", "median"),
            fwhm_range=("fwhm_mm", lambda v: f"{v.min():.1f}-{v.max():.1f}"),
            fwhm_x=("fwhm_x", "median"), fwhm_y=("fwhm_y", "median"), fwhm_z=("fwhm_z", "median"),
            **{plus: (plus, "median")},
        ).reset_index().sort_values("fwhm_mm")
        by_site.round(2).to_csv(out_dir / f"sharpness_{name}_by_site.csv", index=False)
        print(f"\n===== {name}: {len(df)} scans checked =====")
        with pd.option_context("display.width", 250, "display.max_columns", None):
            print(by_site.round(2).to_string(index=False))

        sites = list(by_site["site"])
        fig, axes = plt.subplots(1, len(sites), figsize=(2.6 * len(sites), 3.2), squeeze=False)
        for ax, site in zip(axes[0], sites):
            ax.imshow(slices[site], cmap="gray", vmin=0, vmax=1)
            fw = by_site.loc[by_site["site"] == site, "fwhm_mm"].iat[0]
            ax.set_title(f"{site}\nFWHM {fw:.1f} mm", fontsize=9)
            ax.axis("off")
        fig.suptitle(f"{name}: one scan per site, middle slice", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / f"sharpness_{name}_slices.png", dpi=120)
        plt.close(fig)

    print(f"\nColumns: fwhm_mm = effective smoothness (geometric mean of x/y/z; higher = blurrier); "
          f"{plus} = the same after an extra {args.extra_mm:g} mm smoothing. A site whose fwhm_mm is close to "
          f"another site's {plus} was probably smoothed before we got it. Compare sites within one data type.")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
