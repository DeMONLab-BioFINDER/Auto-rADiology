#!/usr/bin/env python3
"""What physical size is one voxel of the 128^3 model input?

The model input is made by cropping each scan to its non-zero box (CropForeground) and
stretching that box to --image_shape (Resize), so a voxel's size in mm depends on each scan's
native voxel size and how big the cropped box is. This reads the native images straight from
the tau_batch_*.pkl files (image + meta with the affine), replays the exact pre-resize steps
the pipeline uses (src.data.get_transforms: channel-first, RAS reorientation, CropForeground)
and reports, per scan and per site:
    native shape and voxel size (mm)
    cropped box shape and size (mm)
    mm per voxel after resizing to --image_shape, per axis
    the physical size of a few occlusion mask sizes (in model-input voxels)

Run it on several folders (e.g. tau_raw and tau_T1MNI) to compare them; folders without
tau_batch_*.pkl files are skipped. Reads only, no GPU needed.

Outputs (default <proj_path>/results/image_geometry/, outside the repo):
    image_geometry_<data_type>.csv          - one row per scan
    image_geometry_<data_type>_by_site.csv  - medians and ranges per site
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `src.*` imports

import numpy as np
import pandas as pd
import torch
from monai.data import MetaTensor
from monai.transforms import Compose, Resize

from src.data import get_transforms

AXES = ("x", "y", "z")  # RAS after reorientation: x = left-right, y = back-front, z = bottom-top


def pre_resize_transform():
    """The pipeline's steps up to (not including) Resize, without LoadImage: exactly what
    happens to a scan before it gets stretched to the model input size."""
    steps = get_transforms(crop_foreground=True, ras=True, process=True).transforms[1:]
    cut = next(i for i, t in enumerate(steps) if isinstance(t, Resize))
    return Compose(steps[:cut])


def voxel_sizes(affine):
    """Voxel size per axis in mm = length of each affine column."""
    a = np.asarray(affine, dtype=float)[:3, :3]
    return np.sqrt((a ** 2).sum(axis=0))


def scan_rows(data_dir, tfm, image_shape, mask_sizes):
    pkl_files = sorted(data_dir.glob("tau_batch_*.pkl"))
    if not pkl_files:
        print(f"[{data_dir.name}] no tau_batch_*.pkl files, skipped")
        return []
    rows, batch_len = [], None
    for pkl_path in pkl_files:
        batch_id = int(pkl_path.stem.split("_")[-1])
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        if batch_len is None:
            batch_len = len(data)  # IDs index batches the same way src/data.py does: ID // len(batch 0)
            meta0 = data[0].get("meta") or {}
            print(f"[{data_dir.name}] {len(pkl_files)} pkl files, {batch_len} scans each; meta keys: {sorted(meta0)}")
            if "affine" not in meta0:
                print(f"[{data_dir.name}] WARNING: no affine in meta, mm values below assume 1 mm voxels")
        for idx, sample in enumerate(data):
            img = np.clip(np.nan_to_num(sample["image"], nan=0.0, posinf=0.0, neginf=0.0), 0, None)
            meta = sample.get("meta") or {}
            native_vox = voxel_sizes(meta["affine"]) if "affine" in meta else np.ones(3)
            x = tfm(MetaTensor(torch.as_tensor(img, dtype=torch.float32), meta=meta))
            crop_shape = np.array(x.shape[-3:])
            crop_vox = np.asarray(x.pixdim, dtype=float)[:3] if "affine" in meta else np.ones(3)
            crop_mm = crop_shape * crop_vox
            model_vox = crop_mm / np.array(image_shape)
            row = {"ID": batch_id * batch_len + idx,
                   "file": Path(str(meta.get("filename_or_obj", ""))).name,
                   "native_shape": "x".join(str(s) for s in np.squeeze(img).shape),
                   "cropped_shape": "x".join(str(int(s)) for s in crop_shape)}
            for i, ax in enumerate(AXES):
                row[f"native_vox_mm_{ax}"] = round(float(native_vox[i]), 3)
            for i, ax in enumerate(AXES):
                row[f"cropped_mm_{ax}"] = round(float(crop_mm[i]), 1)
            for i, ax in enumerate(AXES):
                row[f"model_vox_mm_{ax}"] = round(float(model_vox[i]), 3)
            for m in mask_sizes:
                row[f"mask{m}_mm"] = "x".join(f"{m * v:.0f}" for v in model_vox)
            rows.append(row)
    return rows


def site_summary(df, mask_sizes):
    num_cols = [c for c in df.columns if c.startswith(("native_vox_mm_", "cropped_mm_", "model_vox_mm_"))]
    out = []
    for site, g in df.groupby("site", dropna=False):
        r = {"site": site, "n": len(g),
             "most_common_native_shape": g["native_shape"].mode().iat[0],
             "n_native_shapes": g["native_shape"].nunique()}
        for c in num_cols:
            r[f"{c}_median"] = round(float(g[c].median()), 2)
            r[f"{c}_range"] = f"{g[c].min():.2f}-{g[c].max():.2f}"
        for m in mask_sizes:
            r[f"mask{m}_mm_median"] = "x".join(f"{m * g[f'model_vox_mm_{ax}'].median():.0f}" for ax in AXES)
        out.append(r)
    return pd.DataFrame(out)


def main():
    repo = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input_path", required=True, help="Folder holding the data_type folders and demographics.csv")
    p.add_argument("--data_types", default="tau_raw,tau_T1,tau_T1MNI", help="Comma-separated folders under input_path")
    p.add_argument("--image_shape", nargs=3, type=int, default=[128, 128, 128])
    p.add_argument("--mask_sizes", default="4,8,16", help="Occlusion mask sizes (model-input voxels) to convert to mm")
    p.add_argument("--out_dir", default=str(repo.parent / "results" / "image_geometry"))
    args = p.parse_args()

    input_path, out_dir = Path(args.input_path), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_sizes = [int(m) for m in args.mask_sizes.split(",") if m.strip()]
    demo_csv = input_path / "demographics.csv"
    demo = pd.read_csv(demo_csv, index_col=0)[["ID", "site"]] if demo_csv.exists() else None
    if demo is None:
        print(f"WARNING: {demo_csv} not found, per-site summary skipped")
    tfm = pre_resize_transform()

    for dt in [d.strip() for d in args.data_types.split(",") if d.strip()]:
        rows = scan_rows(input_path / dt, tfm, args.image_shape, mask_sizes)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        if demo is not None:
            df = df.merge(demo, on="ID", how="left")
            df.insert(1, "site", df.pop("site"))
        df.to_csv(out_dir / f"image_geometry_{dt}.csv", index=False)

        print(f"\n===== {dt}: {len(df)} scans =====")
        print(f"native shapes: {df['native_shape'].nunique()} different, most common {df['native_shape'].mode().iat[0]}")
        for ax in AXES:
            v = df[f"model_vox_mm_{ax}"]
            print(f"model-input voxel, {ax}: median {v.median():.2f} mm (range {v.min():.2f}-{v.max():.2f})")
        for m in mask_sizes:
            print(f"mask {m} voxels ~ " + " x ".join(f"{m * df[f'model_vox_mm_{ax}'].median():.0f}" for ax in AXES) + " mm (median scan)")
        if demo is not None:
            s = site_summary(df, mask_sizes)
            s.to_csv(out_dir / f"image_geometry_{dt}_by_site.csv", index=False)
            show = ["site", "n", "most_common_native_shape", "n_native_shapes"] + \
                   [f"native_vox_mm_{ax}_median" for ax in AXES] + [f"model_vox_mm_{ax}_median" for ax in AXES] + \
                   [f"mask{m}_mm_median" for m in mask_sizes]
            with pd.option_context("display.width", 250, "display.max_columns", None):
                print(s[show].to_string(index=False))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
