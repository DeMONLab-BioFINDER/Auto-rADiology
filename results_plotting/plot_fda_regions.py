#!/usr/bin/env python3
"""Where readers look: the FDA flortaucipir (TAUVID) visual-read regions, drawn in MNI space.

The TAUVID label (2020, sections 2.5-2.6) defines the read in words, plus one crosshair rule for the temporal lobes:
  positive = increased neocortical uptake in posterolateral temporal (PLT), occipital or parietal/precuneus,
             either hemisphere, with or without frontal;
  negative = no increased neocortical uptake, or uptake only in mesial temporal, anterolateral temporal and/or frontal;
  temporal quadrants = horizontal crosshair "immediately posterior to the brainstem nuclei", vertical crosshair
             "through the widest portion of the temporal pole" -> ALT / AMT / PLT / PMT.
The label gives no masks or coordinates, so this is an approximation for figures only (no analysis uses it).

How it is built:
  atlas      Desikan-Killiany (FreeSurfer aparc) + subcortical labels on MNI152NLin2009cAsym, 1 mm, from
             Lotter LD (2025) FreeSurfer parcellations in MNI space, G-Node, doi:10.12751/g-node.2mnxpm
             (FreeSurfer 8 run on the template itself; desc-smoothed file). The brainstem comes from that dataset's
             full FreeSurfer aseg. Both files are downloaded on first use into --atlas_dir.
  crosshairs measured on the atlas (in MNI mm): y = posterior edge of the brainstem at the midbrain/upper pons
             (z = -20 to -10 mm, median, rounded); x = +/- centre of the temporal pole at its widest axial slice
             (mean of both sides, rounded). Override with --y_cross / --x_cross.
  temporal lobe = superior/middle/inferior temporal, banks STS, transverse temporal, temporal pole, fusiform,
             parahippocampal, entorhinal + hippocampus, amygdala; PLT = y < y_cross and |x| >= x_cross,
             ALT = y >= y_cross and |x| >= x_cross, mesial temporal (AMT + PMT) = |x| < x_cross.
  other regions = whole DK regions in FreeSurfer's lobe grouping (postcentral in parietal; precentral and
             paracentral in frontal); insula and cingulate are left out (the label does not name them).

Outputs (default <proj_path>/results/fda_regions, outside the repo):
  fda_regions_MNI152NLin2009cAsym.nii.gz + fda_regions_labels.json   region map (1 mm, values 1-6)
  fda_regions_axial.png    nine axial slices z = -36..60 mm (the same as voxelwise_visual_read.py's axial.png),
                           crosshairs dashed on every slice that contains temporal lobe
  fda_regions_3d.html      interactive 3D view (drag to rotate; click a name to hide it); needs plotly, skipped
                           with a message if it is missing

Usage (Berzelius login node: light job, about a minute, and it downloads the atlas on first use):
    python results_plotting/plot_fda_regions.py
"""
import argparse
import json
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from nilearn import datasets, image, plotting

GIN = "https://gin.g-node.org/llotter/mni_freesurfer/raw/master/parcellations/MNI152NLin2009cAsym/"
DK_FILE = "seg-aparc.aseg_space-MNI152NLin2009cAsym_desc-smoothed.nii.gz"
DK_TSV = "seg-aparc.aseg_space-MNI152NLin2009cAsym.tsv"
FS_ASEG = "aseg.nii.gz"  # full FreeSurfer aseg (FreeSurfer label numbers; 16 = brainstem)
AXIAL_CUTS = [-36, -24, -12, 0, 12, 24, 36, 48, 60]  # mm, bottom -> top, as in voxelwise_visual_read.py

TEMPORAL = ["superiortemporal", "middletemporal", "inferiortemporal", "bankssts", "transversetemporal",
            "temporalpole", "fusiform", "parahippocampal", "entorhinal"]
OCCIPITAL = ["lateraloccipital", "lingual", "cuneus", "pericalcarine"]
PARIETAL = ["superiorparietal", "inferiorparietal", "supramarginal", "postcentral", "precuneus"]
FRONTAL = ["superiorfrontal", "rostralmiddlefrontal", "caudalmiddlefrontal", "parsopercularis", "parstriangularis",
           "parsorbitalis", "lateralorbitofrontal", "medialorbitofrontal", "frontalpole", "precentral", "paracentral"]
# name, colour, counts towards a positive read
GROUPS = [("Posterolateral temporal (PLT)", "#D62728", True), ("Occipital", "#FF7F0E", True),
          ("Parietal / precuneus", "#E7BA52", True), ("Frontal", "#6B9BD1", False),
          ("Anterolateral temporal (ALT)", "#8C6BB1", False), ("Mesial temporal (AMT + PMT)", "#41AB5D", False)]


def parse_args():
    repo = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--atlas_dir", default=str(repo.parent / "data" / "atlases" / "lotter2025_mni_freesurfer"),
                   help="Where the DK atlas files are (downloaded here if missing).")
    p.add_argument("--y_cross", type=float, default=None, help="Front/back crosshair in MNI mm (default: measured).")
    p.add_argument("--x_cross", type=float, default=None, help="Inner/outer crosshair, |x| in MNI mm (default: measured).")
    p.add_argument("--out_dir", default=str(repo.parent / "results" / "fda_regions"))
    return p.parse_args()


def fetch(atlas_dir: Path) -> None:
    atlas_dir.mkdir(parents=True, exist_ok=True)
    for f in (DK_FILE, DK_TSV, FS_ASEG):
        if not (atlas_dir / f).exists():
            print(f"[download] {GIN}{f}")
            urllib.request.urlretrieve(GIN + f, atlas_dir / f)


def mm_of(mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """MNI mm (x, y, z) of every voxel in a True/False mask."""
    return nib.affines.apply_affine(affine, np.argwhere(mask))


def measure_crosshairs(K, brainstem, temporal_pole_ids, affine):
    bs = mm_of(brainstem, affine)
    edges = [bs[np.abs(bs[:, 2] - z) < 0.5, 1].min() for z in range(-20, -9) if (np.abs(bs[:, 2] - z) < 0.5).any()]
    y_cross = float(np.round(np.median(edges)))
    centres = []
    for lab in temporal_pole_ids:  # left and right temporal pole
        tp = mm_of(K == lab, affine)
        widths = {z: np.ptp(tp[tp[:, 2] == z, 0]) for z in np.unique(tp[:, 2])}
        z_wide = max(widths, key=widths.get)
        s = tp[tp[:, 2] == z_wide, 0]
        centres.append(abs((s.max() + s.min()) / 2))
        print(f"[crosshair] temporal pole {lab}: widest at z = {z_wide:.0f} mm, centre |x| = {centres[-1]:.1f} mm")
    x_cross = float(np.round(np.mean(centres)))
    print(f"[crosshair] brainstem posterior edge at z = -20..-10 mm: y = {min(edges):.0f} to {max(edges):.0f} mm")
    return y_cross, x_cross


def build_regions(atlas_dir: Path, y_cross, x_cross):
    dk = nib.load(atlas_dir / DK_FILE)
    K = np.asarray(dk.dataobj).astype(int)
    names = pd.read_csv(atlas_dir / DK_TSV, sep="\t", header=None, index_col=0)[1]
    ids = lambda regions: [i for i, n in names.items() if n in {f"ctx-{h}-{r}" for h in ("lh", "rh") for r in regions}]
    sub = lambda wanted: [i for i, n in names.items() if n in wanted]
    fs = image.resample_to_img(nib.load(atlas_dir / FS_ASEG), dk, interpolation="nearest",
                               force_resample=True, copy_header=True)
    brainstem = np.asarray(fs.dataobj).astype(int) == 16
    if y_cross is None or x_cross is None:
        y_m, x_m = measure_crosshairs(K, brainstem, ids(["temporalpole"]), dk.affine)
        y_cross = y_m if y_cross is None else y_cross
        x_cross = x_m if x_cross is None else x_cross
    print(f"[crosshair] using y = {y_cross:g} mm (front/back) and |x| = {x_cross:g} mm (inner/outer)")

    xyz = nib.affines.apply_affine(dk.affine, np.indices(K.shape).reshape(3, -1).T).reshape(*K.shape, 3)
    temporal = np.isin(K, ids(TEMPORAL) + sub({"Left-Hippocampus", "Left-Amygdala", "Right-Hippocampus", "Right-Amygdala"}))
    posterior, lateral = xyz[..., 1] < y_cross, np.abs(xyz[..., 0]) >= x_cross
    masks = [temporal & posterior & lateral, np.isin(K, ids(OCCIPITAL)), np.isin(K, ids(PARIETAL)),
             np.isin(K, ids(FRONTAL)), temporal & ~posterior & lateral, temporal & ~lateral]
    lab = np.zeros(K.shape, np.int16)
    for i, m in enumerate(masks, 1):
        lab[m] = i
    return nib.Nifti1Image(lab, dk.affine), temporal, y_cross, x_cross


def plot_axial(img, temporal, y_cross, x_cross, out: Path):
    fig = plt.figure(figsize=(20, 4.9), facecolor="black")
    ax = fig.add_axes([0, 0.17, 1, 0.70])
    d = plotting.plot_roi(img, bg_img=datasets.load_mni152_template(resolution=1), display_mode="z",
                          cut_coords=AXIAL_CUTS, axes=ax, cmap=ListedColormap([g[1] for g in GROUPS]), vmin=1,
                          vmax=len(GROUPS), alpha=0.8, black_bg=True, annotate=False, colorbar=False,
                          interpolation="nearest")
    z0 = img.affine[2, 3]
    for z, a in d.axes.items():
        a.ax.set_title(f"z = {z:g}", color="white", fontsize=10)
        if temporal[..., int(round(z - z0))].any():  # crosshairs only split the temporal lobe
            kw = dict(color="white", lw=0.8, ls="--", alpha=0.9)
            a.ax.axhline(y_cross, **kw)
            a.ax.axvline(-x_cross, **kw)
            a.ax.axvline(x_cross, **kw)
    fig.text(0.5, 0.97, "FDA visual-read regions (TAUVID label) on the Desikan-Killiany atlas, temporal lobe split by "
             "the label's crosshairs", color="white", ha="center", va="top", fontsize=13)
    pos = [Patch(color=c, label=n) for n, c, p in GROUPS if p]
    neg = [Patch(color=c, label=n) for n, c, p in GROUPS if not p]
    for handles, anchor, loc, title in [(pos, (0.1, 0.02), "lower left", "Uptake here makes the scan positive"),
                                        (neg, (0.92, 0.02), "lower right", "Uptake only here = still negative")]:
        leg = fig.legend(handles=handles, loc=loc, bbox_to_anchor=anchor, ncol=3, frameon=False, labelcolor="white",
                         fontsize=10.5, title=title, title_fontsize=10.5)
        leg.get_title().set_color("white")
    fig.savefig(out, dpi=300, facecolor="black")
    plt.close(fig)
    print(f"[save] {out}")


def plot_3d(img, out: Path):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[skip] interactive 3D .html needs plotly (pip install plotly in ai-pet)")
        return
    from scipy.ndimage import gaussian_filter
    grid = nib.Nifti1Image(np.zeros((91, 109, 91), np.int8), np.array(
        [[2.0, 0, 0, -90], [0, 2.0, 0, -126], [0, 0, 2.0, -72], [0, 0, 0, 1]]))  # MNI 2 mm, keeps the file small
    lab = np.rint(image.resample_to_img(img, grid, interpolation="nearest", force_resample=True,
                                        copy_header=True).get_fdata()).astype(int)
    brain = image.resample_to_img(datasets.load_mni152_brain_mask(resolution=2), grid, interpolation="nearest",
                                  force_resample=True, copy_header=True).get_fdata() > 0

    def surface(mask, name, colour, opacity, step=1, sigma=0.8):
        m = gaussian_filter(mask.astype(np.float32), sigma)  # smooth edges, then cut at 0.5
        idx = np.argwhere(m > 0.05)
        lo, hi = np.maximum(idx.min(0) - 2, 0), np.minimum(idx.max(0) + 3, m.shape)
        sub = m[lo[0]:hi[0]:step, lo[1]:hi[1]:step, lo[2]:hi[2]:step]
        ijk = np.stack(np.meshgrid(*[np.arange(a, b, step) for a, b in zip(lo, hi)], indexing="ij"), -1)
        xyz = nib.affines.apply_affine(grid.affine, ijk)
        return go.Isosurface(x=xyz[..., 0].ravel(), y=xyz[..., 1].ravel(), z=xyz[..., 2].ravel(), value=sub.ravel(),
                             isomin=0.5, isomax=1.0, surface_count=1, opacity=opacity, name=name, showlegend=True,
                             colorscale=[[0, colour], [1, colour]], showscale=False, hoverinfo="name",
                             caps=dict(x_show=False, y_show=False, z_show=False),
                             lighting=dict(ambient=0.5, diffuse=0.8, specular=0.2))

    traces = [surface(brain, "Brain (outline)", "#BBBBBB", 0.08, step=2, sigma=1.2)]
    traces += [surface(lab == i, n, c, 0.6) for i, (n, c, _) in enumerate(GROUPS, 1)]
    axis = dict(showbackground=False, color="grey")
    fig = go.Figure(traces)
    fig.update_layout(
        title=dict(text="FDA visual-read regions on the Desikan-Killiany atlas — drag to rotate, scroll to zoom, "
                        "click a name to hide it", x=0.5),
        paper_bgcolor="black", font_color="white", legend=dict(font=dict(size=13), bgcolor="rgba(0,0,0,0)"),
        scene=dict(xaxis_title="left ← x → right (mm)", yaxis_title="back ← y → front (mm)",
                   zaxis_title="bottom ← z → top (mm)", aspectmode="data", bgcolor="black",
                   xaxis=axis, yaxis=axis, zaxis=axis, camera=dict(eye=dict(x=-1.6, y=-0.6, z=0.5))),
        margin=dict(l=0, r=0, t=60, b=0))
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"[save] {out} (open in a browser; needs internet once to load plotly)")


def main():
    args = parse_args()
    atlas_dir, out_dir = Path(args.atlas_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fetch(atlas_dir)
    img, temporal, y_cross, x_cross = build_regions(atlas_dir, args.y_cross, args.x_cross)
    nib.save(img, out_dir / "fda_regions_MNI152NLin2009cAsym.nii.gz")
    labels = {str(i): n for i, (n, _, _) in enumerate(GROUPS, 1)}
    (out_dir / "fda_regions_labels.json").write_text(json.dumps(
        {"labels": labels, "y_cross_mm": y_cross, "x_cross_mm": x_cross,
         "atlas": "Lotter LD (2025) FreeSurfer parcellations in MNI space, doi:10.12751/g-node.2mnxpm"}, indent=1))
    lab = np.asarray(img.dataobj)
    for i, (n, _, _) in enumerate(GROUPS, 1):
        print(f"[region] {n:32s} {int((lab == i).sum()):7d} voxels (1 mm)")
    plot_axial(img, temporal, y_cross, x_cross, out_dir / "fda_regions_axial.png")
    plot_3d(img, out_dir / "fda_regions_3d.html")


if __name__ == "__main__":
    main()
