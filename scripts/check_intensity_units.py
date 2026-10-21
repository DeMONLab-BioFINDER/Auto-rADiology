#!/usr/bin/env python3
"""Are the tau-PET images SUVR images, or raw activity values? And what is SUVR 1.0 in the model input?

If the images are already SUVR (divided by a reference region), ordinary brain tissue sits
around 1-2 and the background near 0. Raw activity (e.g. Bq/mL) or scanner counts are in the
hundreds or thousands. This reads the native images from tau_batch_*.pkl and reports, per scan
and per site, the values inside the head (median, 95th and 99th percentile, max).

It also replays the pipeline up to its intensity scaling (src.data.get_transforms: RAS,
CropForeground, Resize, then ScaleIntensityRangePercentiles 1-99 -> 0-1 with clipping, over the
whole volume) and reports which model-input value a native value of 1.0 becomes. That is only
"SUVR 1.0" if the images really are SUVR. Next to it, the value the occlusion 'median_brain'
fill would use, for comparison.

Reads only, no GPU needed. Outputs (default <proj_path>/results/intensity_units/, outside the repo):
    intensity_units_<data_type>.csv          - one row per scan
    intensity_units_<data_type>_by_site.csv  - medians and ranges per site
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
from monai.transforms import Compose, ScaleIntensityRangePercentiles

from src.data import brain_outer_mask, get_transforms
from scripts.check_image_geometry import load_sites


def pre_scale_transform():
    """The pipeline's steps up to (not including) the intensity scaling, without LoadImage, plus
    the scaling step itself (to read its percentile settings)."""
    steps = get_transforms(crop_foreground=True, ras=True, process=True).transforms[1:]
    cut = next(i for i, t in enumerate(steps) if isinstance(t, ScaleIntensityRangePercentiles))
    return Compose(steps[:cut]), steps[cut]


def scan_row(img, meta, pre, scale):
    img = np.clip(np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0), 0, None).astype(np.float32)
    t = torch.as_tensor(img)
    head = brain_outer_mask(t[None] if t.ndim == 3 else t).squeeze().numpy().astype(bool)
    vals = np.squeeze(img)[head] if head.shape == np.squeeze(img).shape else np.squeeze(img)[np.squeeze(img) > 0]

    # replay the pipeline up to the scaling, then the scaling's own percentiles (whole volume, like MONAI)
    x = pre(MetaTensor(t, meta=meta)).numpy()
    lo, hi = np.percentile(x, [scale.lower, scale.upper])
    to_input = lambda v: float(np.clip((v - lo) / (hi - lo) * (scale.b_max - scale.b_min) + scale.b_min,
                                       scale.b_min, scale.b_max)) if hi > lo else float("nan")
    # what the model sees: scaled, then masked; median_brain fill = median of its non-zero voxels
    scaled = np.clip((x - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(x)
    scaled = scaled * brain_outer_mask(torch.as_tensor(scaled)).numpy()
    nz = scaled[scaled != 0]
    return {"head_median": float(np.median(vals)) if vals.size else np.nan,
            "head_p95": float(np.percentile(vals, 95)) if vals.size else np.nan,
            "head_p99": float(np.percentile(vals, 99)) if vals.size else np.nan,
            "max": float(img.max()),
            "scale_p_lo": float(lo), "scale_p_hi": float(hi),
            "share_clipped_to_1": float((x > hi).mean() / max((x > 0).mean(), 1e-9)),
            "value_1.0_in_model_input": to_input(1.0),
            "median_brain_fill": float(np.median(nz)) if nz.size else np.nan}


def guess(head_median):
    if np.isnan(head_median):
        return "?"
    if 0.3 <= head_median <= 5:
        return "looks like SUVR"
    if head_median >= 50:
        return "looks like raw activity/counts"
    return "unclear"


def main():
    repo = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input_path", default="", help="Folder that relative --data_types are under (and holding demo.csv)")
    p.add_argument("--data_types", default="tau_raw_backup",
                   help="Comma-separated folders with tau_batch_*.pkl: names under --input_path, or absolute paths")
    p.add_argument("--dataset", default="Gothenburg", help="Used to find each folder's demographics_<dataset>.csv")
    p.add_argument("--max_per_site", type=int, default=20, help="Scans checked per site (0 = all)")
    p.add_argument("--out_dir", default=str(repo.parent / "results" / "intensity_units"))
    args = p.parse_args()

    input_path, out_dir = Path(args.input_path), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pre, scale = pre_scale_transform()
    print(f"pipeline scaling: percentiles {scale.lower}-{scale.upper} of the whole volume -> "
          f"{scale.b_min}-{scale.b_max}, clip={scale.clip}")

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

        rows, n_site, batch_len = [], {}, None
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
                meta = sample.get("meta") or {}
                row = {"ID": sid, "site": site, "file": Path(str(meta.get("filename_or_obj", ""))).name}
                row.update(scan_row(sample["image"], meta, pre, scale))
                rows.append(row)
        df = pd.DataFrame(rows)
        df["guess"] = df["head_median"].map(guess)
        df.round(4).to_csv(out_dir / f"intensity_units_{name}.csv", index=False)

        by_site = df.groupby("site", dropna=False).agg(
            n=("ID", "size"),
            head_median=("head_median", "median"),
            head_median_range=("head_median", lambda v: f"{v.min():.3g}-{v.max():.3g}"),
            head_p99=("head_p99", "median"),
            max=("max", "median"),
            value_1_in_input=("value_1.0_in_model_input", "median"),
            value_1_in_input_range=("value_1.0_in_model_input", lambda v: f"{v.min():.2f}-{v.max():.2f}"),
            median_brain_fill=("median_brain_fill", "median"),
            clipped=("share_clipped_to_1", "median"),
            guess=("guess", lambda g: g.mode().iat[0]),
        ).reset_index()
        by_site.round(3).to_csv(out_dir / f"intensity_units_{name}_by_site.csv", index=False)
        print(f"\n===== {name}: {len(df)} scans checked =====")
        with pd.option_context("display.width", 250, "display.max_columns", None):
            print(by_site.round(3).to_string(index=False))
        print("example files:", ", ".join(df["file"].dropna().astype(str).unique()[:5]))
    print(f"\nColumns: head_* = native values inside the head; value_1_in_input = model-input value of a native 1.0 "
          f"(= SUVR 1.0 only if the images are SUVR); median_brain_fill = the occlusion fill 'median_brain' would use; "
          f"clipped = share of non-zero voxels above the upper percentile (all become 1).")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
