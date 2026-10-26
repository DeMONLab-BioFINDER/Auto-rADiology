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
gets now) next to the same scan resampled into the header's frame and, with --mni_dir, the same
person's MNI scan (standard upright pose, matched by sID). If the turn is right, the turned row
sits in the same pose as the MNI row. The stored and turned rows are sliced through the same
physical point (the brain's centre), so they show the same place, cut at different angles.
Titles hold site and angle, not IDs.

Independent check (with --raw and --t1 folders and sID columns in both demographics csvs): for
the same person, raw and T1 headers describe the same voxels, so T1 header x inverse(raw header)
is the coregistration alone; its angle is reported per scan (coreg_angle_deg). Reads only, no GPU.

Outputs (default <proj_path>/results/t1_rotation/, outside the repo):
    rotation_<folder>.csv          - one row per scan (ID = position in the pkl batches)
    rotation_<folder>_by_site.csv  - per site: median / 90th percentile / max angle
    coreg_raw_vs_t1.csv            - per matched scan: raw angle, T1 angle, coreg angle and shift
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
    total = rotation_angle(r)
    pitch, roll, yaw = Rotation.from_matrix(r).as_euler("xyz", degrees=True)
    return total, float(pitch), float(roll), float(yaw)


def load_demo(csv):
    """ID, site (and sID if present) from a demographics csv, or None."""
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    if not {"ID", "site"} <= set(df.columns):
        return None
    return df[[c for c in ("ID", "site", "sID") if c in df.columns]]


def rotation_angle(r):
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))


def pkl_index(data_dir):
    """ID -> (pkl file, index in it), with IDs counted like src/data.py (ID // len(batch 0))."""
    files = sorted(data_dir.glob("tau_batch_*.pkl"))
    if not files:
        return {}
    with open(files[0], "rb") as f:
        batch_len = len(pickle.load(f))
    return {int(fp.stem.split("_")[-1]) * batch_len + i: (fp, i) for fp in files for i in range(batch_len)}


def load_sample(where):
    fp, i = where
    with open(fp, "rb") as f:
        return pickle.load(f)[i]


def brain_centre_world(vol):
    """World (mm) point at the brain's centre: centroid of the head voxels in the top 120 mm of the
    head (world z), so neck and shoulders don't pull it down. Same physical point in any resampling."""
    v = vol[0].numpy()
    head = v > 0.2 * np.percentile(v[v > 0], 99) if (v > 0).any() else v > 0
    ijk = np.array(np.nonzero(head))
    if ijk.size == 0:
        return None
    aff = np.asarray(vol.affine, dtype=float)
    xyz = aff[:3, :3] @ ijk + aff[:3, 3:4]
    top = xyz[2] >= xyz[2].max() - 120
    return xyz[:, top].mean(axis=1)


def slices_through(vol, world):
    """Axial, coronal, sagittal slice through the voxel nearest to a world point."""
    v = vol[0].numpy()
    aff = np.asarray(vol.affine, dtype=float)
    c = np.linalg.solve(aff[:3, :3], world - aff[:3, 3]) if world is not None else np.array(v.shape) / 2
    c = [int(np.clip(round(float(c[i])), 0, v.shape[i] - 1)) for i in range(3)]
    return [v[:, :, c[2]].T, v[:, c[1], :].T, v[c[0], :, :].T]


def as_tensor(sample):
    img = np.clip(np.nan_to_num(sample["image"], nan=0.0, posinf=0.0, neginf=0.0), 0, None)
    return EnsureChannelFirst(channel_dim="no_channel")(MetaTensor(torch.as_tensor(img, dtype=torch.float32),
                                                                   meta=sample["meta"]))


def turn_png(sample, path, title, mni_sample=None):
    """As stored (own voxel axes, like our recipes) vs resampled onto the header's world axes,
    plus the same person's MNI scan (standard pose) when given."""
    x = as_tensor(sample)
    stored = Orientation(axcodes="RAS", labels=None)(Spacing(pixdim=(2.0,) * 3, diagonal=False, padding_mode="zeros")(x))
    turned = Orientation(axcodes="RAS", labels=None)(Spacing(pixdim=(2.0,) * 3, diagonal=True, padding_mode="zeros")(x))  # zeros: no edge smear where the turned box leaves the scan
    panels = [(stored, "as stored (current recipes)"), (turned, "turned into header frame")]
    if mni_sample is not None:
        panels.append((Orientation(axcodes="RAS", labels=None)(as_tensor(mni_sample)), "same person, MNI (upright)"))
    centre = brain_centre_world(stored)  # one physical point; stored and turned rows both slice through it
    fig, ax = plt.subplots(len(panels), 3, figsize=(9, 3 * len(panels)))
    for r, (vol, name) in enumerate(panels):
        v = vol[0].numpy()
        hi = np.percentile(v[v > 0], 99) if (v > 0).any() else 1
        world = brain_centre_world(vol) if r == 2 else centre  # MNI is another person-space: its own brain centre
        for c, (s, view) in enumerate(zip(slices_through(vol, world), ["axial", "coronal", "sagittal"])):
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
    p.add_argument("--raw", default="tau_raw_backup", help="Which --data_types folder is the raw one (for the coreg check)")
    p.add_argument("--t1", default="tau_T1", help="Which --data_types folder is the T1 one (for the coreg check)")
    p.add_argument("--mni_dir", default="", help="tau_T1MNI pkl folder (same ID order as input_path/demo.csv): adds "
                                                  "the same person's MNI scan as a third picture row")
    p.add_argument("--out_dir", default=str(repo.parent / "results" / "t1_rotation"))
    args = p.parse_args()

    input_path, out_dir = Path(args.input_path), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    demo_main = load_demo(input_path / "demo.csv")  # raw / MNI order
    mni_where = pkl_index(Path(args.mni_dir)) if args.mni_dir else {}
    tables, affines, demos, dirs = {}, {}, {}, {}
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
                affines.setdefault(name, {})[batch_id * batch_len + idx] = np.asarray(meta["affine"], dtype=float)
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
        demo = load_demo(demo_csv)
        df["site"] = df.merge(demo, on="ID", how="left")["site"].fillna("not in demo csv").values \
            if demo is not None else "unknown"
        tables[name], demos[name], dirs[name] = df, demo, data_dir
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


    # independent check: raw and T1 headers of the same person (matched by sID) differ only by the coregistration
    sid_to_raw = {}
    if demo_main is not None and "sID" in demo_main.columns:
        sid_to_raw = dict(zip(demo_main["sID"], demo_main["ID"]))
    raw_n, t1_n = Path(args.raw).name, Path(args.t1).name
    t1_demo = demos.get(t1_n)
    if raw_n in tables and t1_n in tables and sid_to_raw and t1_demo is not None and "sID" in t1_demo.columns:
        raw_df, t1_df = tables[raw_n].set_index("ID"), tables[t1_n].set_index("ID")
        rows = []
        for t1_id, sid in zip(t1_demo["ID"], t1_demo["sID"]):
            raw_id = sid_to_raw.get(sid)
            if raw_id is None or raw_id not in affines[raw_n] or t1_id not in affines[t1_n]:
                continue
            c = affines[t1_n][t1_id] @ np.linalg.inv(affines[raw_n][raw_id])
            u, _, vt = np.linalg.svd(c[:3, :3])
            rows.append({"sID": sid, "raw_ID": raw_id, "t1_ID": t1_id, "site": raw_df.at[raw_id, "site"],
                         "raw_angle_deg": raw_df.at[raw_id, "angle_deg"], "t1_angle_deg": t1_df.at[t1_id, "angle_deg"],
                         "coreg_angle_deg": round(rotation_angle(u @ vt), 2),
                         "coreg_scale_dev": round(float(np.abs(np.linalg.svd(c[:3, :3])[1] - 1).max()), 4),
                         "coreg_shift_mm": round(float(np.linalg.norm(c[:3, 3])), 1)})
        co = pd.DataFrame(rows)
        co.to_csv(out_dir / "coreg_raw_vs_t1.csv", index=False)
        if co.empty:
            print("\ncoreg check: no scans matched by sID")
        else:
            print(f"\n===== coreg check: {len(co)} scans matched by sID (raw vs T1 header of the same person) =====")
            print("coreg_angle = turn between the two headers (the coregistration alone); coreg_scale_dev ~0 means "
                  "rigid (no resize)")
            with pd.option_context("display.width", 200, "display.max_columns", None):
                print(co.groupby("site")[["raw_angle_deg", "t1_angle_deg", "coreg_angle_deg", "coreg_scale_dev",
                                          "coreg_shift_mm"]].median().round(2).to_string())
    else:
        print("\ncoreg check skipped (needs both folders and an sID column in demo.csv and the T1 demographics csv)")

    # pictures of the most-rotated scans; for T1 scans the same person's MNI scan as the third row
    t1_sid = dict(zip(t1_demo["ID"], t1_demo["sID"])) if t1_demo is not None and "sID" in t1_demo.columns else {}
    for name, df in tables.items():
        if not args.n_png:
            break
        for site, sub in df.groupby("site"):
            for rank, (_, row) in enumerate(sub.nlargest(args.n_png, "angle_deg").iterrows(), start=1):
                with open(dirs[name] / row["pkl"], "rb") as f:
                    sample = pickle.load(f)[int(row["index_in_pkl"])]
                mni_id = int(row["ID"]) if name == raw_n else sid_to_raw.get(t1_sid.get(int(row["ID"])))
                mni_sample = load_sample(mni_where[mni_id]) if mni_id in mni_where else None
                title = (f"{name}, {site}, #{rank} most rotated: {row['angle_deg']:.1f} deg "
                         f"(pitch {row['pitch_deg']:.1f}, roll {row['roll_deg']:.1f}, yaw {row['yaw_deg']:.1f})")
                turn_png(sample, out_dir / f"turn_{name}_{str(site).replace(' ', '_')}_{rank}.png", title, mni_sample)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
