#!/usr/bin/env python3
"""How close is the MRI-free tau-template alignment to each person's MRI-based MNI scan?

tau_template_pts (submit_convert-tau-raw-2mm-rigid.sh template) and tau_T1MNI_pts are on the same
91x109x91 2 mm grid. Scans are matched by ID (index column of the template folder's tau_pt_scans.csv,
so a pilot subset works too). For every scan: Pearson r between the two model inputs inside the
MNI152 brain mask (1 = lined up exactly like the MRI route; the MRI route also warps, so expect a bit
below 1), plus the size factor and turn from tau_pt_scans.csv. Per site: medians, scans at the size
limit, scans with low r. Reads one batch at a time, so the full ~2,500 scans fit in memory.

Outputs (default <proj_path>/results/template_alignment/, outside the repo):
    template_vs_mni.csv          - one row per scan: ID, site, r, scale, turn
    template_vs_mni_by_site.csv  - per site summary
    template_vs_mni_mean.png     - mean of all template-aligned inputs vs mean of all MNI inputs (3 slices
                                   each + difference); a blurry or shifted mean = systematic misalignment
    with --sheets: sheet_<site>.png - each scan (rows, up to --sheet_max per site) as 3 slices, template-
                                   aligned next to its own MNI input. Single scans: view on Berzelius
                                   (ThinLinc), do not download.
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


class Batches:
    """Scan by position from a tau_batch_*.pt folder, one batch in memory at a time."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.files = sorted(folder.glob("tau_batch_*.pt"))
        if not self.files:
            raise SystemExit(f"No tau_batch_*.pt in {folder}")
        man = folder / "tau_pt_manifest.json"
        bs = json.loads(man.read_text()).get("batch_size") if man.exists() else None
        self.cache_i, self.cache = None, None
        self.batch_size = int(bs) if bs else int(self._batch(0).shape[0])

    def _batch(self, i):
        if i != self.cache_i:
            self.cache_i, self.cache = i, torch.load(self.files[i], map_location="cpu", weights_only=True).float()
        return self.cache

    def __getitem__(self, pos):
        i = pos // self.batch_size
        if i >= len(self.files):
            raise IndexError(f"Scan position {pos} is beyond {self.folder}")
        return self._batch(i)[pos % self.batch_size]


def mni_brain_mask(shape):
    from nilearn import datasets, image
    target = nib.Nifti1Image(np.zeros(shape, np.int8), MNI_2MM_RAS)
    return image.resample_to_img(datasets.load_mni152_brain_mask(resolution=2), target, interpolation="nearest",
                                 force_resample=True, copy_header=True).get_fdata() > 0.5


def three_slices(vol):
    cx, cy, cz = (s // 2 for s in vol.shape)
    return [np.rot90(vol[cx]), np.rot90(vol[:, cy]), np.rot90(vol[:, :, cz])]


def main(args):
    tmpl_dir, mni_dir = Path(args.template_dir), Path(args.mni_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmpl, mni = Batches(tmpl_dir), Batches(mni_dir)

    scans_csv = tmpl_dir / "tau_pt_scans.csv"
    sc = pd.read_csv(scans_csv) if scans_csv.exists() else None
    n_tmpl = sum(len(torch.load(f, map_location="cpu", weights_only=True)) for f in tmpl.files)
    ids = sc["index"].astype(int).tolist() if sc is not None and "index" in sc else list(range(n_tmpl))
    if len(ids) != n_tmpl:
        raise SystemExit(f"{scans_csv} lists {len(ids)} scans but {tmpl_dir.name} holds {n_tmpl}")
    shape = tuple(tmpl[0].shape)
    if tuple(mni[0].shape) != shape:
        raise SystemExit(f"Different grids: {tmpl_dir.name} {shape} vs {mni_dir.name} {tuple(mni[0].shape)}")
    mask = torch.as_tensor(mni_brain_mask(shape))

    site = {}
    for f in args.demo_csv:
        if not Path(f).exists():
            print(f"Skipping missing {f} (its scans get site 'unknown')")
            continue
        d = pd.read_csv(f)
        site.update(dict(zip(d["ID"].astype(int), d["site"].astype(str))))

    r, sum_a, sum_b = [], torch.zeros(shape), torch.zeros(shape)
    sheets = {}  # site -> list of (ID, template vol, MNI vol)
    for pos, sid in enumerate(ids):
        a, b = tmpl[pos], mni[sid]
        av, bv = a[mask] - a[mask].mean(), b[mask] - b[mask].mean()
        r.append(float((av * bv).sum() / (av.norm() * bv.norm()).clamp_min(1e-8)))
        sum_a += a
        sum_b += b
        if args.sheets:
            lst = sheets.setdefault(site.get(sid, "unknown"), [])
            if len(lst) < args.sheet_max:
                lst.append((sid, a.numpy().copy(), b.numpy().copy()))

    df = pd.DataFrame({"ID": ids, "r_brain_vs_mni": np.round(r, 4)})
    if sc is not None:
        keep = [c for c in ("rigid_r_template", "rigid_scale", "rigid_scale_at_limit", "rigid_total_deg", "rigid_pitch_deg") if c in sc]
        df = df.join(sc[keep].reset_index(drop=True))
    df.insert(1, "site", df["ID"].map(site).fillna("unknown"))
    df.to_csv(out_dir / "template_vs_mni.csv", index=False)

    agg = {"n": ("ID", "size"), "r_median": ("r_brain_vs_mni", "median"),
           "r_5th_pct": ("r_brain_vs_mni", lambda x: x.quantile(0.05)),
           f"n_r_below_{args.low_r}": ("r_brain_vs_mni", lambda x: int((x < args.low_r).sum()))}
    if "rigid_r_template" in df:
        agg.update(r_template_median=("rigid_r_template", "median"))
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

    ma, mb = (sum_a / len(ids)).numpy(), (sum_b / len(ids)).numpy()
    vmax = max(ma.max(), mb.max())
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for row, (vol, name, kw) in enumerate([(ma, f"Tau template ({tmpl_dir.name})", dict(cmap="gray", vmin=0, vmax=vmax)),
                                           (mb, f"MNI via MRI ({mni_dir.name})", dict(cmap="gray", vmin=0, vmax=vmax)),
                                           (ma - mb, "Difference", dict(cmap="RdBu_r", vmin=-0.3 * vmax, vmax=0.3 * vmax))]):
        for col, sl in enumerate(three_slices(vol)):
            axes[row, col].imshow(sl, **kw)
            axes[row, col].axis("off")
        axes[row, 0].set_title(name, loc="left", fontsize=9)
    fig.suptitle(f"Mean model input over {len(ids)} scans")
    fig.tight_layout()
    fig.savefig(out_dir / "template_vs_mni_mean.png", dpi=120)
    plt.close(fig)

    rs = dict(zip(df["ID"], df["r_brain_vs_mni"]))
    scale = dict(zip(df["ID"], df["rigid_scale"])) if "rigid_scale" in df else {}
    for st, rows in sheets.items():
        fig, axes = plt.subplots(len(rows), 6, figsize=(12, 2.1 * len(rows)), squeeze=False)
        for i, (sid, a, b) in enumerate(rows):
            for j, sl in enumerate(three_slices(a) + three_slices(b)):
                axes[i, j].imshow(sl, cmap="gray", vmin=0, vmax=1)
                axes[i, j].axis("off")
            axes[i, 0].set_title(f"ID {sid}  r {rs[sid]:.2f}" + (f"  size {scale[sid]:.2f}" if sid in scale else ""),
                                 loc="left", fontsize=8)
            axes[i, 3].set_title("its MNI scan (MRI route)", loc="left", fontsize=8)
        fig.suptitle(f"{st}: tau-template aligned (left 3) vs MNI (right 3)")
        fig.tight_layout()
        fig.savefig(out_dir / f"sheet_{st}.png", dpi=110)
        plt.close(fig)
    (out_dir / "inputs.json").write_text(json.dumps(vars(args), indent=1))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    repo = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--template_dir", required=True, help="tau_template_pts (or pilot) folder")
    p.add_argument("--mni_dir", required=True, help="tau_T1MNI_pts folder (all scans, in ID order)")
    p.add_argument("--demo_csv", nargs="*", default=[], help="CSV(s) with ID and site (demo.csv, demo_AVID_unseen.csv)")
    p.add_argument("--low_r", type=float, default=0.8, help="Flag scans whose brain r is below this")
    p.add_argument("--sheets", action="store_true", help="Per-site pictures of single scans (pilot; view on Berzelius)")
    p.add_argument("--sheet_max", type=int, default=10, help="Scans per site picture")
    p.add_argument("--out_dir", default=str(repo.parent / "results" / "template_alignment"))
    main(p.parse_args())
