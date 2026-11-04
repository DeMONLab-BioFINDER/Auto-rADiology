import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `src.*` imports

import numpy as np
import pandas as pd
import torch
from monai.data import MetaTensor
from monai.transforms import Compose, EnsureChannelFirst, LoadImage, Orientation, ScaleIntensityRangePercentiles
from scipy.ndimage import binary_erosion

from src.data import get_transforms


def _iter_pkl_files(pkl_dir: Path):
    return sorted(pkl_dir.glob("tau_batch_*.pkl"))


def _save_batch(tensors, out_dir: Path, batch_id: int):
    batch = torch.stack(tensors, dim=0)
    out_path = out_dir / f"tau_batch_{batch_id:03d}.pt"
    torch.save(batch, out_path)
    return out_path


def _build_transform(args):
    tfm = get_transforms(
        target_shape=tuple(args.image_shape),
        pct_lo=float(args.pct_lo),
        pct_hi=float(args.pct_hi),
        crop_foreground=bool(args.crop_foreground),
        ras=bool(args.ras),
        interp=str(args.interp),
        out_range=(float(args.out_min), float(args.out_max)),
        smooth_sigma_vox=None,
        apply_brain_mask=bool(args.apply_brain_mask),
        process=True,
        head_crop=bool(args.head_crop),
        crop_margin=int(args.crop_margin),
        spacing_mm=args.spacing_mm,
        pct_inside_head=bool(args.pct_inside_head),
    )
    # remove LoadImage (first step) since we already have arrays; split at the intensity scaling so
    # the per-scan record can keep the native values that became 0 and 1
    steps = tfm.transforms[1:]
    cut = next(i for i, t in enumerate(steps) if isinstance(t, ScaleIntensityRangePercentiles))
    return Compose(steps[:cut]), steps[cut], Compose(steps[cut:])


def _load_reference_mask(args):
    """Reference region (e.g. whole cerebellum) as a boolean array on the model grid: loaded and put in
    RAS like the scans, thresholded at 0.5 (probabilistic or 0/1 masks), eroded by --mask_erode voxels
    so warp errors and spill-in at its edge count less. Returns (mask, affine) - the affine is checked
    against every scan, so a mask on another grid stops the conversion instead of silently misaligning."""
    tfm = [LoadImage(image_only=True), EnsureChannelFirst()]
    if args.ras:
        tfm.append(Orientation(axcodes="RAS", labels=None))
    m = Compose(tfm)(args.reference_mask)
    mask = np.asarray(m)[0] > 0.5
    if args.mask_erode:
        mask = binary_erosion(mask, iterations=int(args.mask_erode))
    if not mask.any():
        raise ValueError(f"Reference mask {args.reference_mask} is empty after erosion")
    print(f"Reference mask: {int(mask.sum())} voxels after {args.mask_erode}-voxel erosion, shape {mask.shape}")
    return mask, np.asarray(m.affine, dtype=float)


def _reference_scale(x_pre, mask, mask_affine, args):
    """Divide by the mean of the reference region (-> SUVR) and map to the model input:
    soft (default) = SUVR 0..--suvr_max -> 0..1, clipped, one window for every scan;
    hard (--suvr_threshold t) = 1 where SUVR >= t, else 0 (two-colour reader display, no brightness).
    Only reference voxels with signal (> 0, i.e. inside the head and the scanner field of view)
    count; coverage = their share of the mask. Returns (x, ref_mean, coverage)."""
    if x_pre.shape[1:] != mask.shape or not np.allclose(np.asarray(x_pre.affine, dtype=float), mask_affine, atol=1e-3):
        raise ValueError(f"Reference mask grid {mask.shape} does not match the scan grid {tuple(x_pre.shape[1:])} "
                         "(different shape or affine) - the mask must be on the same MNI grid as the scans")
    vol = np.asarray(x_pre)[0]
    v = vol[mask]
    covered = v > 0
    coverage = float(covered.mean())
    ref = float(v[covered].mean()) if covered.any() else float("nan")
    if not np.isfinite(ref) or ref <= 0:
        return torch.zeros_like(torch.as_tensor(vol))[None], ref, coverage
    suvr = torch.as_tensor(vol / ref, dtype=torch.float32)[None]
    if args.suvr_threshold is not None:
        return (suvr >= float(args.suvr_threshold)).float() * (suvr > 0).float(), ref, coverage
    return (suvr / float(args.suvr_max)).clamp(0.0, 1.0), ref, coverage


def _voxel_mm(affine):
    a = np.asarray(affine, dtype=float)[:3, :3]
    return np.sqrt((a ** 2).sum(axis=0))


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parent.parent).stdout.strip() or None
    except OSError:
        return None


def _save_manifest(out_dir: Path, *, batch_size: int, total: int, image_shape, out_min: float, out_max: float, recipe: dict):
    manifest = {
        "format": "tau_pt_batches_v1",
        "preprocessed": True,
        "batch_size": int(batch_size),
        "total_images": int(total),
        "image_shape": [int(x) for x in image_shape],
        "dtype": "float32",
        "value_range": [float(out_min), float(out_max)],
        "file_pattern": "tau_batch_{batch_id:03d}.pt",
        "recipe": recipe,  # how these batches were made, so two caches can be told apart later
    }
    manifest_path = out_dir / "tau_pt_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main(args):
    pkl_dir = Path(args.pkl_dir)
    if not pkl_dir.exists():
        raise FileNotFoundError(f"Missing pkl_dir: {pkl_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else pkl_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pkl_files = _iter_pkl_files(pkl_dir)
    if not pkl_files:
        raise FileNotFoundError(f"No tau_batch_*.pkl files found in {pkl_dir}")

    batch_size = int(args.batch_size)
    pre, scale, post = _build_transform(args)
    mask, mask_affine = _load_reference_mask(args) if args.reference_mask else (None, None)
    buffer, scan_rows = [], []
    batch_id = 0
    total = 0

    for pkl_path in pkl_files:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        for sample in data:
            img = sample["image"]
            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
            img = np.clip(img, 0, None)

            meta = sample.get("meta")
            load_img = MetaTensor(torch.as_tensor(img, dtype=torch.float32), meta=meta)
            x_pre = pre(load_img)
            if mask is None:
                x = post(x_pre)
            else:
                x, ref_mean, ref_coverage = _reference_scale(x_pre, mask, mask_affine, args)

            # per-scan record: geometry (for true-proportion plots, mm-sized occlusion masks) and the
            # native values the scaling mapped to 0 and 1 (to turn e.g. SUVR 1.0 into a model-input value)
            has_affine = bool(meta) and "affine" in meta
            # x_pre is already resized; its affine (updated by MONAI) gives the mm per output voxel
            out_vox = np.asarray(x_pre.pixdim, dtype=float)[:3] if has_affine else np.ones(3)
            if mask is not None:  # soft: native 0 -> 0, native ref_mean * suvr_max -> 1 (hard: not a linear scale)
                lo, hi = 0.0, ref_mean * float(args.suvr_max)
            else:
                lo, hi = scale.bounds(x_pre) if hasattr(scale, "bounds") else np.percentile(np.asarray(x_pre), [scale.lower, scale.upper])
            # head extent in the output grid; touching the edge with --spacing_mm means the box cut the head
            nz = np.nonzero(np.asarray(x).squeeze() > 0)
            head_vox = np.array([(a.max() - a.min() + 1) if a.size else 0 for a in nz])
            # sides and top only: with --spacing_mm a too-tall box is cut at the bottom (neck) on purpose
            touches_edge = any(a.size and ((a.min() == 0 and i < 2) or a.max() == n - 1)
                               for i, (a, n) in enumerate(zip(nz, args.image_shape)))
            bottom_cut = bool(nz[2].size and nz[2].min() == 0)
            row = {"index": total, "file": Path(str((meta or {}).get("filename_or_obj", ""))).name,
                   "native_shape": "x".join(str(v) for v in np.squeeze(img).shape),
                   "scale_native_lo": float(lo), "scale_native_hi": float(hi)}
            native_vox = _voxel_mm(meta["affine"]) if has_affine else np.ones(3)
            for i, ax in enumerate("xyz"):
                row[f"native_vox_mm_{ax}"] = round(float(native_vox[i]), 3)
            for i, ax in enumerate("xyz"):
                row[f"crop_box_mm_{ax}"] = round(float(out_vox[i] * args.image_shape[i]), 1)
            for i, ax in enumerate("xyz"):
                row[f"out_vox_mm_{ax}"] = round(float(out_vox[i]), 3)
            for i, ax in enumerate("xyz"):
                row[f"head_mm_{ax}"] = round(float(head_vox[i] * out_vox[i]), 1)
            row["head_touches_edge"] = bool(touches_edge)
            row["bottom_cut"] = bottom_cut
            if mask is not None:
                row["reference_mean_native"] = ref_mean
                row["reference_coverage"] = round(ref_coverage, 3)
            scan_rows.append(row)

            if isinstance(x, MetaTensor):
                x = x.as_tensor()
            if x.ndim == 4 and x.shape[0] == 1:
                x = x.squeeze(0)

            buffer.append(x)
            total += 1

            if len(buffer) == batch_size:
                _save_batch(buffer, out_dir, batch_id)
                buffer = []
                batch_id += 1

    if buffer:
        _save_batch(buffer, out_dir, batch_id)
        batch_id += 1

    scans_csv = out_dir / "tau_pt_scans.csv"
    scans = pd.DataFrame(scan_rows)
    scans.to_csv(scans_csv, index=False)
    if mask is not None:
        bad = scans["reference_mean_native"].isna() | (scans["reference_mean_native"] <= 0)
        low = scans["reference_coverage"] < float(args.min_coverage)
        print(f"Reference region: coverage median {scans['reference_coverage'].median():.3f}, "
              f"{int(low.sum())} scans below {args.min_coverage}, {int(bad.sum())} with no signal (saved as all-zero images)")
        if bad.any() or low.any():
            print(f"WARNING: check reference_coverage / reference_mean_native in {scans_csv.name}")
    if args.spacing_mm and scans["head_touches_edge"].any():
        print(f"WARNING: {int(scans['head_touches_edge'].sum())} of {len(scans)} heads touch a side or the top of the "
              f"{args.image_shape} box (possibly cut off): see head_touches_edge in {scans_csv.name}")

    recipe = {k: v for k, v in vars(args).items() if k not in ("out_dir", "batch_size")}
    recipe["git_commit"] = _git_commit()
    manifest_path = _save_manifest(
        out_dir,
        batch_size=batch_size,
        total=total,
        image_shape=args.image_shape,
        out_min=float(args.out_min),
        out_max=float(args.out_max),
        recipe=recipe,
    )

    print(f"Saved {batch_id} batches to {out_dir}")
    print(f"Total images: {total}")
    print(f"Manifest: {manifest_path}")
    print(f"Per-scan geometry/scaling: {scans_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert tau_batch_*.pkl to tau_batch_*.pt batches.")
    parser.add_argument("--pkl_dir", type=str, required=True, help="Directory containing tau_batch_*.pkl files")
    parser.add_argument("--out_dir", type=str, default="", help="Output directory for tau_batch_*.pt (default: pkl_dir)")
    parser.add_argument("--batch_size", type=int, default=50, help="Images per .pt batch file")
    parser.add_argument("--image_shape", nargs=3, type=int, default=[128, 128, 128], help="Output image shape (D H W)")
    parser.add_argument("--pct_lo", type=float, default=1.0, help="Lower percentile for intensity scaling")
    parser.add_argument("--pct_hi", type=float, default=99.0, help="Upper percentile for intensity scaling")
    parser.add_argument("--crop_foreground", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ras", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--interp", type=str, default="trilinear")
    parser.add_argument("--out_min", type=float, default=0.0)
    parser.add_argument("--out_max", type=float, default=1.0)
    parser.add_argument("--apply_brain_mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--head_crop", action=argparse.BooleanOptionalAction, default=False,
                        help="Mask the head on the native scan before cropping, so the crop box is the head, not the field of view")
    parser.add_argument("--crop_margin", type=int, default=0, help="Voxels kept around the crop box")
    parser.add_argument("--spacing_mm", type=float, default=None,
                        help="Resample to cubic voxels of this size and pad/crop to --image_shape instead of stretching")
    parser.add_argument("--pct_inside_head", action=argparse.BooleanOptionalAction, default=False,
                        help="Scaling percentiles over inside-the-head voxels only (use with --head_crop)")

    parser.add_argument("--reference_mask", type=str, default=None,
                        help="NIfTI reference-region mask on the scans' grid (e.g. whole cerebellum in MNI). Replaces the "
                             "percentile scaling: each scan is divided by its mean there (SUVR) and shown on one fixed window")
    parser.add_argument("--mask_erode", type=int, default=1, help="Voxels to erode the reference mask by")
    parser.add_argument("--suvr_max", type=float, default=3.3,
                        help="Soft version: SUVR mapped to 1 (3.3 puts the reader's 1.65 x reference at 0.5)")
    parser.add_argument("--suvr_threshold", type=float, default=None,
                        help="Hard version: 1 where SUVR >= this (reader colour transition, 1.65), else 0")
    parser.add_argument("--min_coverage", type=float, default=0.8,
                        help="Warn about scans whose signal covers less than this share of the reference mask")

    args = parser.parse_args()
    if args.reference_mask and not args.ras:
        parser.error("--reference_mask needs --ras (the mask is put in RAS like the scans)")
    main(args)
