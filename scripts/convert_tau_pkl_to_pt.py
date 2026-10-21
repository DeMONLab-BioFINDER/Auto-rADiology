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
from monai.transforms import Compose, ScaleIntensityRangePercentiles

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
    )
    # remove LoadImage (first step) since we already have arrays; split at the intensity scaling so
    # the per-scan record can keep the native values that became 0 and 1
    steps = tfm.transforms[1:]
    cut = next(i for i, t in enumerate(steps) if isinstance(t, ScaleIntensityRangePercentiles))
    return Compose(steps[:cut]), steps[cut], Compose(steps[cut:])


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
            x = post(x_pre)

            # per-scan record: geometry (for true-proportion plots, mm-sized occlusion masks) and the
            # native values the scaling mapped to 0 and 1 (to turn e.g. SUVR 1.0 into a model-input value)
            has_affine = bool(meta) and "affine" in meta
            # x_pre is already resized; its affine (updated by MONAI) gives the mm per output voxel
            out_vox = np.asarray(x_pre.pixdim, dtype=float)[:3] if has_affine else np.ones(3)
            lo, hi = np.percentile(np.asarray(x_pre), [scale.lower, scale.upper])
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
    pd.DataFrame(scan_rows).to_csv(scans_csv, index=False)

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
    parser.add_argument("--crop_margin", type=int, default=0, help="Native voxels kept around the crop box")

    args = parser.parse_args()
    main(args)
