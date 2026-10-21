#!/usr/bin/env python3
"""Does the head mask work at every site, or does background get through (e.g. at Geneva)?

brain_outer_mask (the old recipe) keeps voxels above 0.2 x the volume mean. A big field of
view is mostly air, so its mean is low and background noise can pass the threshold; the "head"
then becomes most of the box and the model input is mostly background. head_mask (the iso2
recipe) uses 0.1 x the 99th percentile instead, which does not depend on the field of view.

Per site, for a sample of scans:
  A) the real model inputs in <input_path>/<cache_type> (tau_batch_*.pt, what the models were
     trained on): share of the box that is non-zero, and share of the box's outer shell that
     is non-zero (a working head mask leaves the outer shell almost empty)
  B) replayed from the native pkls in <input_path>/<pkl_type>: the same numbers for the old
     recipe and for head_mask, plus the head size in mm each mask finds
  C) whether the replayed old recipe matches the cached input (correlation), i.e. whether the
     cache was made with this code
  D) for the iso2 recipe (112x128x96 box at 2 mm): whether the head touches a side or the top of
     the box (= the box is too small and cuts the head; should be 0 everywhere), and whether a
     too-tall head was cut at the bottom (neck; expected for some scans, harmless). For a touch:
     which side, and how far below the top of the head (shallow = brain level, deep = shoulders)
  E) share of the head that is pure white (clipped to 1) in the old and the iso2 recipe
  F) whether the head is already cut in the original scan (touches a side of the scanner's
     field of view): no box can bring that back, and the old inputs are cut the same way
and one png per site: middle slices of the old input (cached, else replayed) and the iso2 replay,
plus, for up to 3 scans per site that touch the box edge (edge_*.png) or are cut at the bottom
(neckcut_*.png): front and side view of the whole iso2 head with the box edges marked.

Reads only, no GPU needed. Outputs (default <proj_path>/results/head_mask_check/, outside the repo).
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
from monai.transforms import Compose

from src.data import get_transforms, head_mask
from scripts.check_image_geometry import load_sites


def shell_share(v, width=4):
    """Share of the outer `width`-voxel shell of the box that is non-zero."""
    inner = np.zeros(v.shape, bool)
    inner[width:-width, width:-width, width:-width] = True
    return float((v[~inner] > 0).mean())


def head_mm(v, vox_mm):
    nz = np.nonzero(v > 0)
    return [float((a.max() - a.min() + 1) * s) if a.size else 0.0 for a, s in zip(nz, vox_mm)]


# box faces a head can touch (bottom left out: cutting the neck there is on purpose)
FACES = {"left": lambda m: m[0], "right": lambda m: m[-1], "back": lambda m: m[:, 0], "front": lambda m: m[:, -1]}


def edge_info(v, vox_mm=2.0):
    """Where a head touches its box: the touching faces, how far below the top of the head the
    highest touching point is (mm; small = brain level, large = neck/shoulders), and whether
    the head was cut at the bottom."""
    m = v > 0
    zs = np.nonzero(m.any(axis=(0, 1)))[0]
    out = {"iso2_touches_edge": False, "iso2_touch_sides": "", "iso2_touch_mm_below_top": np.nan,
           "iso2_bottom_cut": bool(zs.size and zs.min() == 0)}
    if not zs.size:
        return out
    top, sides, touch_z = zs.max(), [], []
    for name, face in FACES.items():
        z = np.nonzero(face(m).any(axis=0))[0]  # heights at which the head reaches this face
        if z.size:
            sides.append(name)
            touch_z.append(z.max())
    if m[:, :, -1].any():
        sides.append("top")
        touch_z.append(top)
    if sides:
        out.update(iso2_touches_edge=True, iso2_touch_sides="+".join(sides),
                   iso2_touch_mm_below_top=float((top - max(touch_z)) * vox_mm))
    return out


def white_share(v):
    """Share of the head (non-zero voxels) clipped to pure white (1)."""
    h = v[v > 0]
    return float((h >= 0.999).mean()) if h.size else np.nan


def silhouette_png(v, out_png, title, vox_mm=2.0):
    """Front and side view of a whole iso2 head: average PET along the view, the head outline,
    the box edges, and a rough line where the brain ends (~120 mm below the top of the head)."""
    m = v > 0
    zs = np.nonzero(m.any(axis=(0, 1)))[0]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.6))
    for ax, axis, name, (lft, rgt) in zip(axes, (1, 0), ("front view (left-right)", "side view (back-front)"),
                                          (("left", "right"), ("back", "front"))):
        n = np.maximum(m.sum(axis=axis), 1)
        img = (v.sum(axis=axis) / n).T
        ax.imshow(img, cmap="gray", origin="lower", vmin=0, vmax=np.percentile(img[img > 0], 99.5) if (img > 0).any() else 1)
        ax.contour(m.any(axis=axis).T, levels=[0.5], colors="cyan", linewidths=0.8)
        h, w = img.shape
        ax.add_patch(plt.Rectangle((-0.5, -0.5), w, h, fill=False, edgecolor="orange", linewidth=2))
        sil = m.any(axis=axis)  # (width, z)
        for col, label in ((0, lft), (w - 1, rgt)):
            z = np.nonzero(sil[col])[0]
            if z.size:
                ax.plot([col] * z.size, z, "r.", markersize=3)
        if sil[:, -1].any():
            x = np.nonzero(sil[:, -1])[0]
            ax.plot(x, [h - 1] * x.size, "r.", markersize=3)
        if zs.size:
            ax.axhline(zs.max() - 120 / vox_mm, color="yellow", linestyle="--", linewidth=0.8)
        ax.set_title(name, fontsize=8)
        ax.axis("off")
    fig.suptitle(title + "\norange = box edge, cyan = head outline, red = head touching the box, "
                 "yellow dashed = ~120 mm below the top (roughly where the brain ends)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def mid_slices(v):
    """Side, front and top middle slices of a (x, y, z) RAS volume, upright."""
    return [v[v.shape[0] // 2].T, v[:, v.shape[1] // 2].T, v[:, :, v.shape[2] // 2].T]


def main():
    repo = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input_path", required=True, help="Folder holding demo.csv, the cache and the pkl folders")
    p.add_argument("--cache_type", default="tau_raw", help="Folder with the tau_batch_*.pt the models trained on")
    p.add_argument("--pkl_type", default="tau_raw_backup", help="Folder with the native tau_batch_*.pkl")
    p.add_argument("--max_per_site", type=int, default=15)
    p.add_argument("--out_dir", default=str(repo.parent / "results" / "head_mask_check"))
    args = p.parse_args()

    input_path, out_dir = Path(args.input_path), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    demo = load_sites(input_path / "demo.csv")
    site_of = dict(zip(demo["ID"], demo["site"])) if demo is not None else {}

    cache_dir = input_path / args.cache_type
    pt_files = sorted(cache_dir.glob("tau_batch_*.pt"))
    pt_batch_size = torch.load(pt_files[0], map_location="cpu", weights_only=True).shape[0] if pt_files else None
    print(f"cache: {cache_dir} ({len(pt_files)} .pt files, {pt_batch_size} per batch)")

    old = Compose(get_transforms(target_shape=(128, 128, 128)).transforms[1:])
    to_ras = Compose(get_transforms().transforms[1:3])  # channel-first + RAS only, native voxels
    new = Compose(get_transforms(target_shape=(112, 128, 96), head_crop=True, crop_margin=2, spacing_mm=2,
                                 pct_inside_head=True).transforms[1:])

    rows, examples, n_site, batch_len, cached_batch = [], {}, {}, None, (None, None)
    edge_examples, cut_examples = [], []
    for pkl_path in sorted((input_path / args.pkl_type).glob("tau_batch_*.pkl")):
        batch_id = int(pkl_path.stem.split("_")[-1])
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        batch_len = batch_len or len(data)
        for idx, sample in enumerate(data):
            sid = batch_id * batch_len + idx
            site = site_of.get(sid, "not in demo.csv")
            if args.max_per_site and n_site.get(site, 0) >= args.max_per_site:
                continue
            n_site[site] = n_site.get(site, 0) + 1
            img = np.clip(np.nan_to_num(sample["image"], nan=0.0, posinf=0.0, neginf=0.0), 0, None).astype(np.float32)
            meta = sample.get("meta") or {}
            x_old = old(MetaTensor(torch.as_tensor(img), meta=meta))
            x_new = new(MetaTensor(torch.as_tensor(img), meta=meta))
            old_vox = np.asarray(x_old.pixdim, dtype=float)[:3] if "affine" in meta else np.ones(3)
            v_old, v_new = x_old.squeeze().numpy(), x_new.squeeze().numpy()

            row = {"ID": sid, "site": site,
                   "replay_old_box_share": float((v_old > 0).mean()), "replay_old_shell_share": shell_share(v_old),
                   "replay_iso2_shell_share": shell_share(v_new)}
            row.update(edge_info(v_new))
            v_native = (to_ras(MetaTensor(torch.as_tensor(img), meta=meta)) if "affine" in meta
                        else torch.as_tensor(img).reshape(1, *np.squeeze(img).shape))
            m_native = head_mask(v_native).squeeze().numpy() > 0
            native_sides = [n for n, f in FACES.items() if f(m_native).any()]
            row["native_cut_sides"] = "+".join(native_sides)
            row["native_cut"] = bool(native_sides)
            row["old_white_share"], row["iso2_white_share"] = white_share(v_old), white_share(v_new)
            for ax, a, b in zip("xyz", head_mm(v_old, old_vox), head_mm(v_new, [2, 2, 2])):
                row[f"old_head_mm_{ax}"], row[f"iso2_head_mm_{ax}"] = round(a, 1), round(b, 1)

            v_cache = None
            if pt_batch_size:
                b = sid // pt_batch_size
                if b < len(pt_files):
                    if cached_batch[0] != b:
                        cached_batch = (b, torch.load(pt_files[b], map_location="cpu", weights_only=True))
                    if sid % pt_batch_size < cached_batch[1].shape[0]:
                        v_cache = cached_batch[1][sid % pt_batch_size].squeeze().numpy()
            if v_cache is not None:
                row["cache_box_share"] = float((v_cache > 0).mean())
                row["cache_shell_share"] = shell_share(v_cache)
                row["cache_vs_replay_r"] = (float(np.corrcoef(v_cache.ravel(), v_old.ravel())[0, 1])
                                            if v_cache.shape == v_old.shape else np.nan)
            rows.append(row)
            if site not in examples:
                examples[site] = (sid, v_cache, v_old, v_new)
            if row["iso2_touches_edge"] and sum(e[0] == site for e in edge_examples) < 3:
                edge_examples.append((site, sid, v_new, f"{row['iso2_touch_sides']} touch, highest "
                                      f"{row['iso2_touch_mm_below_top']:.0f} mm below the top of the head"))
            if row["iso2_bottom_cut"] and sum(e[0] == site for e in cut_examples) < 3:
                cut_examples.append((site, sid, v_new, "cut at the bottom of the box"))

    df = pd.DataFrame(rows)
    df.round(4).to_csv(out_dir / "head_mask_per_scan.csv", index=False)
    cols = [c for c in ["cache_box_share", "cache_shell_share", "replay_old_box_share", "replay_old_shell_share",
                        "replay_iso2_shell_share", "cache_vs_replay_r"] if c in df.columns]
    by_site = df.groupby("site")[cols].median()
    by_site.insert(0, "n", df.groupby("site").size())
    for ax in "xyz":
        by_site[f"old_head_mm_{ax}"] = df.groupby("site")[f"old_head_mm_{ax}"].median()
    for ax in "xyz":
        by_site[f"iso2_head_mm_{ax}"] = df.groupby("site")[f"iso2_head_mm_{ax}"].median()
    for ax in "xyz":
        by_site[f"iso2_head_mm_{ax}_max"] = df.groupby("site")[f"iso2_head_mm_{ax}"].max()
    by_site["n_iso2_touch_edge"] = df.groupby("site")["iso2_touches_edge"].sum().astype(int)
    by_site["n_iso2_bottom_cut"] = df.groupby("site")["iso2_bottom_cut"].sum().astype(int)
    by_site["iso2_touch_min_mm_below_top"] = df.groupby("site")["iso2_touch_mm_below_top"].min()
    by_site["iso2_touch_sides"] = df.groupby("site")["iso2_touch_sides"].apply(
        lambda t: ",".join(sorted({x for v in t if v for x in v.split("+")})))
    by_site["n_native_cut"] = df.groupby("site")["native_cut"].sum().astype(int)
    by_site["native_cut_sides"] = df.groupby("site")["native_cut_sides"].apply(
        lambda t: ",".join(sorted({x for v in t if v for x in v.split("+")})))
    by_site["old_white_share"] = df.groupby("site")["old_white_share"].median()
    by_site["iso2_white_share"] = df.groupby("site")["iso2_white_share"].median()
    if "cache_shell_share" in df.columns:
        by_site["n_mask_failed_in_cache"] = df.groupby("site")["cache_shell_share"].apply(lambda s: int((s > 0.2).sum()))
    by_site.round(3).to_csv(out_dir / "head_mask_by_site.csv")
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(by_site.round(3).to_string())

    pngs = [(site, sid, vc, vo, vn, f"example_{str(site).replace(' ', '_')}.png")
            for site, (sid, vc, vo, vn) in examples.items()]
    for prefix, cases in (("edge", edge_examples), ("neckcut", cut_examples)):
        for site, sid, v_new, note in cases:
            silhouette_png(v_new, out_dir / f"{prefix}_{str(site).replace(' ', '_')}_ID{sid}.png",
                           f"{site}, ID {sid}, iso2: {note}")
    for site, sid, v_cache, v_old, v_new, png_name in pngs:
        # cached input == old recipe replayed (cache_vs_replay_r = 1), so show only one of them
        vols = [("old input (trained on)", v_cache) if v_cache is not None else ("old recipe, replayed", v_old),
                ("iso2 recipe", v_new)]
        fig, axes = plt.subplots(len(vols), 3, figsize=(9, 3 * len(vols)), squeeze=False)
        for r, (title, v) in enumerate(vols):
            for c, (sl, view) in enumerate(zip(mid_slices(v), ["side", "front", "top"])):
                axes[r, c].imshow(sl, cmap="gray", origin="lower")
                axes[r, c].set_title(f"{title}: {view}", fontsize=8)
                axes[r, c].axis("off")
        fig.suptitle(f"{site}, ID {sid} (squares = model voxels, so the old recipe is shown stretched)", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / png_name, dpi=110)
        plt.close(fig)

    print("\nshell_share = share of the box's outer 4-voxel shell that is non-zero: ~0 = mask removed the "
          "background; high = background got through. n_mask_failed_in_cache = scans with shell_share > 0.2.\n"
          "cache_vs_replay_r ~1 = the cached input is what this code makes.\n"
          "n_iso2_touch_edge = iso2 heads touching a side or the top of the 112x128x96 box (edge_*.png). "
          "iso2_touch_min_mm_below_top = the highest touch of any scan, in mm below the top of the head: "
          "above ~120 = only neck/shoulders touch (harmless), below ~120 = the skull at brain level touches "
          "(box too small). n_iso2_bottom_cut = heads cut at the bottom (neckcut_*.png). "
          "iso2_head_mm_*_max vs the box (224 x 256 x 192 mm) = how much room is left.\n"
          "*_white_share = share of the head clipped to pure white (1), old vs iso2 recipe.\n"
          "n_native_cut = heads already touching a side of the original scan's field of view (cut by the "
          "scanner, before any of our steps); an iso2 touch on a scan that is also native_cut is not the box's fault.")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
