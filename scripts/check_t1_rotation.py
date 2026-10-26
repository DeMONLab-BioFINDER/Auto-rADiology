#!/usr/bin/env python3
"""How much would resampling a scan into its header's frame rotate it?

tau_T1 has the same voxels as tau_raw; only the header (affine) differs: it holds the rigid
coregistration to the person's T1 MRI. Our recipes keep each scan's own voxel axes (MONAI
Spacing with diagonal=False), so that rotation is never applied. This reads the affine of every
scan in the given tau_batch_*.pkl folders and reports the rotation part (after removing voxel
size and axis flips/swaps), i.e. how far the image would turn if it were resampled onto a grid
aligned with the header's world axes:
    total angle (degrees)
    pitch = about left-right axis (nodding), roll = about back-front axis (ear to shoulder),
    yaw = about bottom-top axis (turning)
Run on tau_raw_backup and tau_T1 to see whether the T1 headers add a real rotation.

It also saves pngs for the most-rotated scans per site: middle slices as stored (what the model
gets now) next to the same scan resampled into the header's frame, so you can see whether that
frame is straighter. Titles hold site and angle, not IDs. Reads only, no GPU needed.

Outputs (default <proj_path>/results/t1_rotation/, outside the repo):
    rotation_<folder>.csv          - one row per scan (ID = position in the pkl batches)
    rotation_<folder>_by_site.csv  - per site: median / 90th percentile / max angle
    turn_<folder>_<site>_<rank>.png
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
from monai.data import MetaTensor
from monai.transforms import EnsureChannelFirst, Orientation, Spacing
from scipy.spatial.transform import Rotation


def residual_rotation(affine):
    """Rotation left in the affine after removing voxel sizes and the nearest axis permutation/flip.
    Returns (total angle, pitch, roll, yaw) in degrees."""
    a = np.asarray(affine, dtype=float)[:3, :3]
    m = a / np.linalg.norm(a, axis=0)  # direction cosines, one column per voxel axis
    p = np.zeros((3, 3))
    for col in np.argsort(-np.abs(m).max(axis=0)):  # most clear-cut axes first
        row = max((r for r in range(3) if not p[r].any()), key=lambda r: abs(m[r, col]))
        p[row, col] = np.sign(m[row, col])
    r = m @ p.T  # pure rotation if the affine has no shear
    u, _, vt = np.linalg.svd(r)
    r = u @ vt
    if np.linalg.det(r) < 0:
        return np.nan, np.nan, np.nan, np.nan
    total = float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))
    pitch, roll, yaw = Rotation.from_matrix(r).as_euler("xyz", degrees=True)
    return total, float(pitch), float(roll), float(yaw)


def load_sites(csv):
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    return df[["ID", "site"]] if {"ID", "site"} <= set(df.columns) else None


def middle_slices(v):
    """Axial, coronal, sagittal slice through the head's centre of mass."""
    w = np.clip(v, 0, None)
    c = [int(round(i)) for i in np.array(np.nonzero(w > 0.2 * w.max())).mean(axis=1)] if w.max() > 0 else \
        [s // 2 for s in v.shape]
    return [v[:, :, c[2]].T, v[:, c[1], :].T, v[c[0], :, :].T]


def turn_png(img, meta, path, title):
    """As stored (own voxel axes, like our recipes) vs resampled onto the header's world axes."""
    x = EnsureChannelFirst(channel_dim="no_channel")(MetaTensor(torch.as_tensor(img, dtype=torch.float32), meta=meta))
    stored = Orientation(axcodes="RAS", labels=None)(Spacing(pixdim=(2.0,) * 3, diagonal=False)(x))
    turned = Orientation(axcodes="RAS", labels=None)(Spacing(pixdim=(2.0,) * 3, diagonal=True)(x))
    fig, ax = plt.subplots(2, 3, figsize=(9, 6))
    for r, (vol, name) in enumerate([(stored, "as stored (current recipes)"), (turned, "turned into header frame")]):
        v = vol[0].numpy()
        hi = np.percentile(v[v > 0], 99) if (v > 0).any() else 1
        for c, (s, view) in enumerate(zip(middle_slices(v), ["axial", "coronal", "sagittal"])):
            ax[r, c].imshow(s, cmap="gray", origin="lower", vmin=0, vmax=hi)
            ax[r, c].set_xticks([]), ax[r, c].set_yticks([])
            ax[r, c].set_title(f"{view}" if r == 0 else "", fontsize=9)
        ax[r, 0].set_ylabel(name, fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    repo = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input_path", default="", help="Folder that relative --data_types are under")
    p.add_argument("--data_types", default="tau_raw_backup,tau_T1",
                   help="Comma-separated folders with tau_batch_*.pkl: names under --input_path, or absolute paths")
    p.add_argument("--dataset", default="Gothenburg", help="Used to find each folder's demographics_<dataset>.csv")
    p.add_argument("--n_png", type=int, default=2, help="Pictures of the most-rotated scans per site")
    p.add_argument("--out_dir", default=str(repo.parent / "results" / "t1_rotation"))
    args = p.parse_args()

    input_path, out_dir = Path(args.input_path), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for dt in [d.strip() for d in args.data_types.split(",") if d.strip()]:
        data_dir = Path(dt) if Path(dt).is_absolute() else input_path / dt
        name = data_dir.name
        pkl_files = sorted(data_dir.glob("tau_batch_*.pkl"))
        if not pkl_files:
            print(f"[{name}] no tau_batch_*.pkl files, skipped")
            continue
        rows, batch_len = [], None
        for pkl_path in pkl_files:
            batch_id = int(pkl_path.stem.split("_")[-1])
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            batch_len = batch_len or len(data)  # IDs index batches like src/data.py: ID // len(batch 0)
            for idx, sample in enumerate(data):
                meta = sample.get("meta") or {}
                if "affine" not in meta:
                    continue
                total, pitch, roll, yaw = residual_rotation(meta["affine"])
                rows.append({"ID": batch_id * batch_len + idx, "pkl": pkl_path.name, "index_in_pkl": idx,
                             "angle_deg": round(total, 2), "pitch_deg": round(pitch, 2),
                             "roll_deg": round(roll, 2), "yaw_deg": round(yaw, 2)})
        df = pd.DataFrame(rows)
        if df.empty:
            print(f"[{name}] no affines found, skipped")
            continue
        demo_csv = data_dir / f"demographics_{args.dataset}.csv"
        if not demo_csv.exists():
            demo_csv = input_path / "demo.csv"
        demo = load_sites(demo_csv)
        df["site"] = df.merge(demo, on="ID", how="left")["site"].fillna("not in demo csv").values \
            if demo is not None else "unknown"
        df.to_csv(out_dir / f"rotation_{name}.csv", index=False)

        g = df.groupby("site")
        by_site = pd.DataFrame({
            "n": g.size(),
            "angle_median": g["angle_deg"].median(),
            "angle_p90": g["angle_deg"].quantile(0.9),
            "angle_max": g["angle_deg"].max(),
            "pitch_abs_median": g["pitch_deg"].apply(lambda s: s.abs().median()),
            "roll_abs_median": g["roll_deg"].apply(lambda s: s.abs().median()),
            "yaw_abs_median": g["yaw_deg"].apply(lambda s: s.abs().median()),
            "share_over_5deg": g["angle_deg"].apply(lambda s: (s > 5).mean()),
        }).round(2)
        by_site.to_csv(out_dir / f"rotation_{name}_by_site.csv")
        print(f"\n===== {name}: {len(df)} scans, from {demo_csv.name} for sites =====")
        print(f"rotation angle: median {df['angle_deg'].median():.1f} deg, 90th pct {df['angle_deg'].quantile(0.9):.1f}, "
              f"max {df['angle_deg'].max():.1f}; over 5 deg: {(df['angle_deg'] > 5).mean():.0%}")
        with pd.option_context("display.width", 200, "display.max_columns", None):
            print(by_site.to_string())

        if args.n_png:
            for site, sub in df.groupby("site"):
                for rank, (_, row) in enumerate(sub.nlargest(args.n_png, "angle_deg").iterrows(), start=1):
                    with open(data_dir / row["pkl"], "rb") as f:
                        sample = pickle.load(f)[int(row["index_in_pkl"])]
                    img = np.clip(np.nan_to_num(sample["image"], nan=0.0, posinf=0.0, neginf=0.0), 0, None)
                    title = (f"{name}, {site}, #{rank} most rotated: {row['angle_deg']:.1f} deg "
                             f"(pitch {row['pitch_deg']:.1f}, roll {row['roll_deg']:.1f}, yaw {row['yaw_deg']:.1f})")
                    turn_png(img, sample["meta"], out_dir / f"turn_{name}_{str(site).replace(' ', '_')}_{rank}.png", title)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
