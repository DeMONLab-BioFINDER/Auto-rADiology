#!/usr/bin/env python3
"""How close is the MRI-free tau-template alignment to each person's MRI-based MNI scan?

tau_template_pts (submit_convert-tau-raw-2mm-rigid.sh template) and tau_T1MNI_pts are on the same
91x109x91 2 mm grid and in the same scan order, so scan i can be compared voxel by voxel. For every
scan: Pearson r between the two model inputs inside the MNI152 brain mask (1 = lined up exactly
like the MRI route; the MRI route also warps, so expect a bit below 1), plus the size factor and turn
from tau_pt_scans.csv. Per site: medians, scans at the size limit, scans with low r.

Outputs (default <proj_path>/results/template_alignment/, outside the repo; no images of single scans):
    template_vs_mni.csv          - one row per scan (ID = position in the batches), site, r, scale, turn
    template_vs_mni_by_site.csv  - per site summary
    template_vs_mni_mean.png     - mean of all template-aligned inputs vs mean of all MNI inputs (3 slices
                                   each + difference); a blurry or shifted mean = systematic misalignment
Reads only, no GPU needed.
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

# the *_pts MNI inputs are stored in RAS order on the SPM 2 mm MNI grid
MNI_2MM_RAS = np.array([[2.0, 0, 0, -90], [0, 2.0, 0, -126], [0, 0, 2.0, -72], [0, 0, 0, 1]])


def load_pts(folder: Path) -> torch.Tensor:
    files = sorted(folder.glob("tau_batch_*.pt"))
    if not files:
        raise SystemExit(f"No tau_batch_*.pt in {folder}")
    return torch.cat([torch.load(f, map_location="cpu") for f in files], dim=0).float()


def mni_brain_mask(shape):
    from nilearn import datasets, image
    target = nib.Nifti1Image(np.zeros(shape, np.int8), MNI_2MM_RAS)
    return image.resample_to_img(datasets.load_mni152_brain_mask(resolution=2), target, interpolation="nearest",
                                 force_resample=True, copy_header=True).get_fdata() > 0.5


def main(args):
    tmpl_dir, mni_dir = Path(args.template_dir), Path(args.mni_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    a, b = load_pts(tmpl_dir), load_pts(mni_dir)
    if a.shape != b.shape:
        raise SystemExit(f"Different sizes: {tmpl_dir.name} {tuple(a.shape)} vs {mni_dir.name} {tuple(b.shape)} "
                         "(need the same scans in the same order on the same grid)")
    mask = torch.as_tensor(mni_brain_mask(tuple(a.shape[1:])))
    av, bv = a[:, mask], b[:, mask]
    av = av - av.mean(1, keepdim=True)
    bv = bv - bv.mean(1, keepdim=True)
    r = ((av * bv).sum(1) / (av.norm(dim=1) * bv.norm(dim=1)).clamp_min(1e-8)).numpy()

    df = pd.DataFrame({"ID": np.arange(len(r)), "r_brain_vs_mni": r.round(4)})
    scans_csv = tmpl_dir / "tau_pt_scans.csv"
    if scans_csv.exists():
        sc = pd.read_csv(scans_csv)
        keep = [c for c in ("rigid_scale", "rigid_scale_at_limit", "rigid_total_deg", "rigid_pitch_deg") if c in sc]
        df = df.join(sc[keep].reset_index(drop=True))
    site = {}
    for f in args.demo_csv:
        if not Path(f).exists():
            print(f"Skipping missing {f} (its scans get site 'unknown')")
            continue
        d = pd.read_csv(f)
        site.update(dict(zip(d["ID"].astype(int), d["site"].astype(str))))
    df.insert(1, "site", df["ID"].map(site).fillna("unknown"))
    df.to_csv(out_dir / "template_vs_mni.csv", index=False)

    agg = {"n": ("ID", "size"), "r_median": ("r_brain_vs_mni", "median"),
           "r_5th_pct": ("r_brain_vs_mni", lambda x: x.quantile(0.05)),
           f"n_r_below_{args.low_r}": ("r_brain_vs_mni", lambda x: int((x < args.low_r).sum()))}
    if "rigid_scale" in df:
        agg.update(scale_median=("rigid_scale", "median"), scale_5th_pct=("rigid_scale", lambda x: x.quantile(0.05)),
                   scale_95th_pct=("rigid_scale", lambda x: x.quantile(0.95)),
                   n_at_size_limit=("rigid_scale_at_limit", lambda x: int(x.astype(bool).sum())),
                   turn_median_deg=("rigid_total_deg", "median"))
    by_site = df.groupby("site").agg(**agg).round(3)
    by_site.loc["ALL"] = df.assign(site="ALL").groupby("site").agg(**agg).round(3).iloc[0]
    by_site.to_csv(out_dir / "template_vs_mni_by_site.csv")
    print(by_site.to_string())
    low = df.loc[df["r_brain_vs_mni"] < args.low_r, "ID"].tolist()
    if low:
        print(f"\n{len(low)} scans with r < {args.low_r} (look at these first): {low[:40]}{' ...' if len(low) > 40 else ''}")

    ma, mb = a.mean(0).numpy(), b.mean(0).numpy()
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    cx, cy, cz = (s // 2 for s in ma.shape)
    cut = lambda v: (v[cx], v[:, cy], v[:, :, cz])
    vmax = max(ma.max(), mb.max())
    for row, (vol, name, kw) in enumerate([(ma, f"Tau template ({tmpl_dir.name})", dict(cmap="gray", vmin=0, vmax=vmax)),
                                           (mb, f"MNI via MRI ({mni_dir.name})", dict(cmap="gray", vmin=0, vmax=vmax)),
                                           (ma - mb, "Difference", dict(cmap="RdBu_r", vmin=-0.3 * vmax, vmax=0.3 * vmax))]):
        for col, sl in enumerate(cut(vol)):
            axes[row, col].imshow(np.rot90(sl), **kw)
            axes[row, col].axis("off")
        axes[row, 0].set_title(name, loc="left", fontsize=9)
    fig.suptitle(f"Mean model input over {len(r)} scans")
    fig.tight_layout()
    fig.savefig(out_dir / "template_vs_mni_mean.png", dpi=120)
    (out_dir / "inputs.json").write_text(json.dumps(vars(args), indent=1))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    repo = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--template_dir", required=True, help="tau_template_pts folder")
    p.add_argument("--mni_dir", required=True, help="tau_T1MNI_pts folder (same scans, same order)")
    p.add_argument("--demo_csv", nargs="*", default=[], help="CSV(s) with ID and site (demo.csv, demo_AVID_unseen.csv)")
    p.add_argument("--low_r", type=float, default=0.8, help="Flag scans whose brain r is below this")
    p.add_argument("--out_dir", default=str(repo.parent / "results" / "template_alignment"))
    main(p.parse_args())
