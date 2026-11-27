#!/usr/bin/env python3
"""One .html per scan: what the reader saw next to what the model saw, two linked NiiVue brains.

  left   the scan as SUVR (divided by its own whole-cerebellum mean) in the readers' colour scale: grey
         (black = low; the readers' display was inverted, white = low) up to 1.65 x cerebellum, then purple / dark blue / cyan / teal / light green / yellow /
         orange / red bands, scale top 5.89 x cerebellum (1.65 at 28% of the bar, as measured on the Moscoso
         supplement colour bars). Readers had a hand-traced cerebellum on reoriented native images; this is the
         Centiloid whole-cerebellum mask in MNI space instead, so it is close to their display, not identical.
  right  the model's input in grey (as under the occlusion PNGs) with the occlusion map drawn exactly like the PNGs
         (src/vis.py): 'pos vs neg' red -> yellow = supports positive, blue -> light cyan = supports negative;
         'class' dark red -> yellow; opacity follows strength (0 = see-through, solid 0.9 at the top and above);
         colour top = the map's 99th percentile or --color_max; vertical colour bar faded the same way on grey.
  click  the values under the cursor, and (with --regions) which FDA visual-read region it is in.
  third  (with --regions) the FDA read regions on the MNI template, in slices in the same plane as the other two
         (default) or as a 3D brain you can turn. Each region has its own checkbox: the three that make a read
         positive (posterolateral temporal, occipital, parietal / precuneus) are ticked on opening; frontal (counts
         only together with those), anterolateral and mesial temporal (do not count) can be ticked.

Page: header with the read, the model call (P >= 0.5) and whether they agree; axial / coronal / sagittal / all three
(all brains together, axial first like the readers); a slice slider with a small template picture showing where the
slice is (also <- -> buttons, scrolling still works); occlusion map choice (+ / -, class, off); "Whole head" shows the
image without the brain mask (the readers saw the whole head: off-target signal in meninges / skull / sinuses then
shows as such); "Go to" buttons jump to the strongest occlusion evidence each way; right-drag pans (NiiVue's default
right-drag changes the contrast, which would change the colour scales), Reset view undoes zoom / pan. The layout
follows the window size (three brains side by side, fewer per row in a narrow window). Voxels are drawn as they are
(nearest neighbour), so the page is as sharp as the occlusion PNGs. NiiVue is loaded from the web, the images are
inside the page.

MNI model only: the FDA regions are in MNI space, and only MNI-model inputs (91 x 109 x 91, 2 mm) sit on that grid.
run_vis.py saves the maps without a position (identity affine); the MNI grid is put back here.

Inputs:
  --vis_dir   an occlusion run folder from run_vis.py / submit_vis-occlusion-cases.sh: <ID>_input.nii.gz,
              <ID>_occlusion_pos_vs_neg.nii.gz, <ID>_occlusion_class.nii.gz, summary.csv
  --pkl_dir + --cbl_mask   original MNI images (tau_batch_*.pkl, same IDs as the run) and the whole-cerebellum mask
              (Centiloid voi_WhlCbl_2mm.nii) for the left brain. Without them the left brain shows the model input on a
              plain heat scale instead (no reader scale: the input is not SUVR).

Usage (Berzelius login node, from the repo; then copy the .html files to the Mac):
    python results_plotting/make_case_viewer.py \\
        --vis_dir <model>/validation/visualization/occlusion_cases_logit_gaussian_mask4_ov0.5 \\
        --pkl_dir /proj/berzelius-2024-156/dataset/tau-PET/tau_T1MNI \\
        --cbl_mask /proj/berzelius-2024-156/users/x_nadpi/data/masks/voi_WhlCbl_2mm.nii \\
        --regions <proj_path>/results/fda_regions/fda_regions_MNI152NLin2009cAsym.nii.gz
Output: <vis_dir>/viewers/<ID>_viewer.html (or --out_dir).
"""
import argparse
import base64
import gzip
import io
import json
import pickle
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image
from matplotlib import colors as mcolors
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt

from make_niivue_viewer import NIIVUE, REGION_COLOURS, REGION_NAMES

SHOWN_FIRST = (1, 2, 3)  # FDA regions ticked on opening: the ones that make a read positive; the others can be ticked
REGION_STATUS = {1: "counts towards +", 2: "counts towards +", 3: "counts towards +",
                 4: "counts only with PLT / occipital / parietal", 5: "does not count", 6: "does not count"}
REGION_OPACITY = 0.75  # FDA regions over the template: the anatomy still shows through

# "About this scan" section: CenTauR regional SUVRs and the SUVR equal to CTRz = 2 for flortaucipir (Villemagne 2023
# equations; the CTRz > 2 positivity cut-off used by Karagianni 2026)
CENTAUR = [("Universal", "Universal", 1.31), ("MetaTemporal", "Meta-temporal", 1.34),
           ("MesialTemporal", "Mesial temporal", 1.35), ("TemporoParietal", "Temporo-parietal", 1.30),
           ("Frontal", "Frontal", 1.29)]
DX_NAMES = {"cu": "Cognitively unimpaired (CU)", "mci": "MCI", "alzcs dem": "AD dementia"}
GRADE = {0: "negative", 1: "positive, moderate", 2: "positive, frequent"}  # read_1..3 / visual_read_3_cat (Stam)
READ_COLS = ["read_1", "read_2", "read_3"]

# MNI model input grid (tau_T1MNI_pts, RAS), the same as voxelwise_visual_read.py
MNI_2MM_RAS = np.array([[2.0, 0, 0, -90], [0, 2.0, 0, -126], [0, 0, 2.0, -72], [0, 0, 0, 1]])
MNI_SHAPE = (91, 109, 91)

# readers' colour scale (memory: visual-read method notes): bar 0..SUVR_TOP, 1.65 x cerebellum at 28% of the bar
SUVR_TRANSITION, TRANSITION_AT = 1.65, 0.28
SUVR_TOP = SUVR_TRANSITION / TRANSITION_AT  # 5.89
BAND_EDGES = [0.28, 0.33, 0.42, 0.52, 0.625, 0.715, 0.81, 0.90, 1.0]  # share of the bar
BAND_COLOURS = ["#7f3f98", "#1f2a9c", "#00bfe8", "#00897b", "#8fdc5a", "#ffe600", "#ff9100", "#e3000f"]
GREY_LOW, GREY_HIGH = "#000000", "#c8c8c8"  # grey below the transition: black = low (the readers used inverted grey,
# white = low; flipped here so the brain sits on black like the right brain - the 1.65 transition and bands are unchanged)

# occlusion colours, copied from src/vis.py (WARM, WARM_COOL) so the viewer matches the PNGs
WARM = ["#c00000", "#ff3000", "#ff9900", "#ffff66"]
POS = ["#c00000", "#ff9900", "#ffff66"]   # WARM_COOL above 0: near zero -> top
NEG = ["#1040ff", "#2aa0ff", "#a0ffff"]   # WARM_COOL below 0: near zero -> most negative
# the same colour maps as src/vis.py, used to colour the map voxel by voxel exactly like the PNGs
WARM_CMAP = mcolors.LinearSegmentedColormap.from_list("warm", WARM)
WARM_COOL = mcolors.LinearSegmentedColormap.from_list(
    "cool_warm_bright", [(0.0, "#a0ffff"), (0.25, "#2aa0ff"), (0.499, "#1040ff"),
                         (0.501, "#c00000"), (0.75, "#ff9900"), (1.0, "#ffff66")])
RGBA = np.dtype([("R", "u1"), ("G", "u1"), ("B", "u1"), ("A", "u1")])
PET_DIM = 1.3  # src/vis.py: grey top = 1.3 x the 99.5th percentile, so yellow still stands out
ZONE_MM = 6  # each FDA region (a thin cortical ribbon in the atlas) is grown this far, so a click near it finds it


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vis_dir", required=True, help="Occlusion run folder (run_vis.py output).")
    p.add_argument("--ids", default=None, help="Comma-separated IDs (default: every <ID>_input.nii.gz in --vis_dir).")
    p.add_argument("--pkl_dir", default=None, help="Original MNI images (tau_batch_*.pkl) for the SUVR / reader view.")
    p.add_argument("--cbl_mask", default=None, help="Whole-cerebellum mask (Centiloid voi_WhlCbl_2mm.nii).")
    p.add_argument("--mask_erode", type=int, default=1, help="Voxels to shrink the cerebellum mask by (as before: 1).")
    p.add_argument("--regions", default=None, help="FDA region map (plot_fda_regions.py): click readout names the region; third brain shows the regions in 3D.")
    p.add_argument("--color_max", type=float, default=None,
                   help="Top of the occlusion colour scale (default: each map's 99th percentile, like the PNGs).")
    p.add_argument("--demo_csv", default=None,
                   help="Demographics csv (ID, site, age, gender, dx, CDR, CDR_SB, apoe, amyloid_status, CL, CenTauR "
                        "SUVRs) for the 'About this scan' section. Has clinical data: the pages stay on Berzelius.")
    p.add_argument("--votes", default=None, help="Per-reader votes csv (ID, read_1..3, visual_read_3_cat).")
    p.add_argument("--female_code", type=float, default=2, help="Value of `gender` meaning female (as describe_dataset.py).")
    p.add_argument("--out_dir", default=None, help="Default <vis_dir>/viewers.")
    args = p.parse_args()
    if bool(args.pkl_dir) != bool(args.cbl_mask):
        p.error("--pkl_dir and --cbl_mask go together (reader view)")
    return args


def nifti_b64(data: np.ndarray, affine: np.ndarray) -> str:
    img = nib.Nifti1Image(np.asarray(data), affine)
    buf = io.BytesIO()
    buf.write(gzip.compress(img.to_bytes(), compresslevel=6))
    return base64.b64encode(buf.getvalue()).decode()


def on_mni_grid(path: Path) -> np.ndarray:
    x = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    if x.shape != MNI_SHAPE:
        raise SystemExit(f"{path.name} is {x.shape}, not the MNI grid {MNI_SHAPE}: this viewer is for the MNI model "
                         "only (2 mm / raw inputs are in each scan's own space, so the FDA regions would not line up).")
    return x


def robust_top(m: np.ndarray, signed: bool, pct: float = 99.0) -> float:
    """src/vis.py robust_top: the pct percentile of the non-zero (absolute) values."""
    v = np.abs(m) if signed else np.clip(m, 0, None)
    v = v[v > 0]
    return float(np.percentile(v, pct)) if v.size else 1.0


def colormap(stops: list, overlay: bool = False) -> dict:
    """NiiVue colormap from [(index 0-255, '#rrggbb'), ...]; NiiVue blends between the given indices.
    overlay=True: index 0 see-through, because NiiVue paints an overlay's values below cal_min in the colour
    at index 0 (opaque there = the whole volume covered)."""
    cm = {"R": [], "G": [], "B": [], "A": [], "I": []}
    if overlay:
        stops = [(0, stops[0][1])] + [(max(i, 1), c) for i, c in stops]
    for n, (idx, hexcol) in enumerate(stops):
        for k, i in zip("RGB", (1, 3, 5)):
            cm[k].append(int(hexcol[i:i + 2], 16))
        cm["A"].append(0 if overlay and n == 0 else 255)
        cm["I"].append(int(idx))
    return cm


def occlusion_rgba(m: np.ndarray, signed: bool, top: float) -> np.ndarray:
    """src/vis.py _overlay_rgba as a colour volume: colour from the map, opacity 0.9 x strength / top (capped).
    Coloured here rather than in NiiVue, which draws every coloured voxel solid in its slice view."""
    cmap, norm = (WARM_COOL, mcolors.Normalize(-top, top)) if signed else (WARM_CMAP, mcolors.Normalize(0, top))
    rgba = cmap(norm(m))
    strength = np.abs(m) if signed else np.clip(m, 0, None)
    rgba[..., 3] = 0.9 * np.clip(strength / top, 0, 1)
    out = np.zeros(m.shape, RGBA)
    for k, c in enumerate("RGBA"):
        out[c] = np.round(rgba[..., k] * 255).astype(np.uint8)
    return out


def faded_colormap(colours: list) -> dict:
    """256-step NiiVue colormap: colours spread evenly, opacity rising from 0 (see-through) to 0.9 at the top,
    like src/vis.py _overlay_rgba (alpha = 0.9 x strength / top)."""
    rgb = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in colours], float)
    f = np.arange(256) / 255
    cm = {k: np.interp(f, np.linspace(0, 1, len(colours)), rgb[:, n]).round().astype(int).tolist()
          for n, k in enumerate("RGB")}
    cm["A"] = (0.9 * 255 * f).round().astype(int).tolist()
    cm["I"] = list(range(256))
    return cm


def faded_css(colours: list, lo: float, hi: float) -> list:
    """CSS gradient stops (bottom lo% -> top hi%) for the same colours and fading, for the colour bar."""
    cm = faded_colormap(colours)
    out = []
    for k in range(0, 256, 32) if hi > lo else range(255, -1, -32):
        pos = lo + (hi - lo) * k / 255
        out.append(f"rgba({cm['R'][k]},{cm['G'][k]},{cm['B'][k]},{cm['A'][k] / 255:.2f}) {pos:.1f}%")
    k = 255
    out.append(f"rgba({cm['R'][k]},{cm['G'][k]},{cm['B'][k]},0.90) {hi:.1f}%") if hi > lo else \
        out.insert(0, f"rgba({cm['R'][k]},{cm['G'][k]},{cm['B'][k]},0.90) {hi:.1f}%")
    return out


def vbar(stops: list, ticks: list, label: str, vt: str) -> str:
    """Vertical colour bar beside the brain, on the PNGs' grey (0.55), ticks on the right."""
    t = "".join(f"<span style='bottom:{b}%'>{txt}</span>" for b, txt in ticks)
    return (f"<div class='vbar occ' data-for='{vt}'><div class='vgrad' style='background:linear-gradient(to top, "
            f"{', '.join(stops)}), #8c8c8c'></div><div class='vticks'>{t}</div><div class='vlabel'>{label}</div></div>")


def even(colours: list) -> list:
    return [(round(k * 255 / (len(colours) - 1)), c) for k, c in enumerate(colours)]


def reader_colormap() -> dict:
    """Inverted grey ramp up to the transition, then flat colour bands (two stops per band, so no blending)."""
    t = round(TRANSITION_AT * 255)
    stops = [(0, GREY_LOW), (t - 1, GREY_HIGH)]
    for lo, hi, c in zip(BAND_EDGES[:-1], BAND_EDGES[1:], BAND_COLOURS):
        stops += [(round(lo * 255), c), (round(hi * 255) - (0 if hi == 1.0 else 1), c)]
    # index 0 (outside the brain) see-through: the page background shows there
    return colormap(stops, overlay=True)


def reader_gradient_css() -> str:
    parts = [f"{GREY_LOW} 0%", f"{GREY_HIGH} {TRANSITION_AT * 100:.1f}%"]
    for lo, hi, c in zip(BAND_EDGES[:-1], BAND_EDGES[1:], BAND_COLOURS):
        parts += [f"{c} {lo * 100:.1f}%", f"{c} {hi * 100:.1f}%"]
    return "linear-gradient(to top, " + ", ".join(parts) + ")"


def load_cbl_mask(path: Path, shape, affine, erode: int) -> np.ndarray:
    from nilearn import image  # only needed for the reader view
    target = nib.Nifti1Image(np.zeros(shape, np.int8), affine)
    m = image.resample_to_img(nib.load(path), target, interpolation="nearest", force_resample=True,
                              copy_header=True).get_fdata() > 0.5
    if erode:
        m = binary_erosion(m, iterations=erode)
    if not m.any():
        raise SystemExit(f"{path.name} is empty after {erode}-voxel erosion")
    return m


def suvr_scans(pkl_dir: Path, ids: set, mask_path: Path, erode: int) -> dict:
    """ID -> (SUVR volume, affine, cerebellum coverage) from the original pkls; IDs index batches like src/data.py."""
    out, mask, batch_len = {}, None, None
    for f in sorted(pkl_dir.glob("tau_batch_*.pkl")):
        b = int(f.stem.split("_")[-1])
        if batch_len is not None and not any(b * batch_len <= i < (b + 1) * batch_len for i in ids):
            continue
        with open(f, "rb") as fh:
            data = pickle.load(fh)
        batch_len = batch_len or len(data)
        for idx, sample in enumerate(data):
            scan_id = b * batch_len + idx
            if scan_id not in ids:
                continue
            x = np.nan_to_num(np.squeeze(np.asarray(sample["image"], dtype=np.float64)), nan=0.0, posinf=0.0, neginf=0.0)
            affine = np.asarray(sample["meta"]["affine"], dtype=float)
            if mask is None:
                mask = load_cbl_mask(mask_path, x.shape, affine, erode)
            v = x[mask]
            coverage = float((v > 0).mean())
            ref = float(v[v > 0].mean()) if (v > 0).any() else float("nan")
            out[scan_id] = (x / ref if np.isfinite(ref) and ref > 0 else np.zeros_like(x), affine, coverage)
        if ids <= set(out):
            break
    return out


def brain_mask(shape, affine, dilate_vox: int) -> np.ndarray:
    """nilearn's MNI152 brain mask on a given grid, grown by a few voxels (keeps the cortex edge, drops scalp / noise)."""
    from nilearn import datasets, image
    target = nib.Nifti1Image(np.zeros(shape, np.int8), affine)
    m = image.resample_to_img(datasets.load_mni152_brain_mask(resolution=1), target, interpolation="nearest",
                              force_resample=True, copy_header=True).get_fdata() > 0
    return binary_dilation(m, iterations=dilate_vox) if dilate_vox else m


def region_zones(path: Path):
    """The FDA regions as zones: inside the brain, every voxel within ZONE_MM of a region's cortex gets that region
    (nearest one). Only used to name the region under the cursor. Returns (zones 0-6, affine)."""
    img = nib.load(path)
    lab = np.asarray(img.dataobj).astype(np.int8)
    vox_mm = np.sqrt((img.affine[:3, :3] ** 2).sum(0))
    dist, idx = distance_transform_edt(lab == 0, sampling=vox_mm, return_indices=True)
    zones = lab[tuple(idx)]
    zones[(dist > ZONE_MM) | ~brain_mask(lab.shape, img.affine, 0)] = 0
    return zones, img.affine


def template_on_grid() -> np.ndarray:
    """nilearn's MNI152 template on the model's 2 mm MNI grid (for the slice locator pictures)."""
    from nilearn import datasets
    t = image.resample_to_img(datasets.load_mni152_template(resolution=2),
                              nib.Nifti1Image(np.zeros(MNI_SHAPE, np.int8), MNI_2MM_RAS),
                              interpolation="continuous", force_resample=True, copy_header=True)
    return np.asarray(t.dataobj, dtype=np.float32)


def png_b64(a: np.ndarray) -> str:
    from matplotlib import image as mpimg
    buf = io.BytesIO()
    mpimg.imsave(buf, a, cmap="gray", vmin=0, vmax=float(np.percentile(a[a > 0], 99.5)), format="png")
    return base64.b64encode(buf.getvalue()).decode()


def locator(in_brain: np.ndarray):
    """Brain extent in mm (slider ranges) and two small template pictures cropped to it: a sagittal one (back on the
    left, like NiiVue) for the axial / coronal slider and an axial one (front at the top, left on the left) for the
    sagittal slider."""
    idx = np.argwhere(in_brain)
    lo, hi = idx.min(0), idx.max(0)
    ext = {a: [float(MNI_2MM_RAS[k, 3] + 2 * lo[k]), float(MNI_2MM_RAS[k, 3] + 2 * hi[k])] for k, a in enumerate("xyz")}
    t = template_on_grid()
    t = t[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1]
    sag = t[t.shape[0] // 2].T[::-1]                             # rows top -> bottom, columns back -> front
    ax = t[:, :, int(round((10 - ext["z"][0]) / 2))].T[::-1]      # z = +10 mm; rows front -> back, columns left -> right
    return ext, {"sag": png_b64(sag), "ax": png_b64(ax)}


def jump_targets(maps: dict, in_brain: np.ndarray) -> list:
    """'Go to' buttons: the strongest occlusion evidence each way, in MNI mm."""
    mm = lambda flat: [float(v) for v in nib.affines.apply_affine(MNI_2MM_RAS, np.unravel_index(flat, MNI_SHAPE))]
    out = []
    if "pos_vs_neg" in maps:
        m = np.where(in_brain, maps["pos_vs_neg"], 0)
        if m.max() > 0:
            out.append({"label": "Strongest pull to +", "mm": mm(np.argmax(m)), "cls": "pos"})
        if m.min() < 0:
            out.append({"label": "Strongest pull to −", "mm": mm(np.argmin(m)), "cls": "neg"})
    elif "class" in maps:
        m = np.where(in_brain, maps["class"], 0)
        if m.max() > 0:
            out.append({"label": "Strongest evidence", "mm": mm(np.argmax(m)), "cls": "pos"})
    return out


def by_id(path):
    if not path:
        return None
    d = pd.read_csv(path)
    d = d.loc[:, ~d.columns.astype(str).str.startswith("Unnamed")]
    d = d[pd.to_numeric(d["ID"], errors="coerce").notna()]
    return d.assign(ID=d["ID"].astype(int)).drop_duplicates("ID").set_index("ID")


def row_of(table, scan_id):
    return table.loc[scan_id] if table is not None and scan_id in table.index else None


def num(row, col):
    v = pd.to_numeric(row.get(col, np.nan), errors="coerce") if row is not None else np.nan
    return None if pd.isna(v) else float(v)


def vote_summary(v) -> tuple[list, str]:
    """[(grade 0/1/2), ...] for each reader who read the scan, and e.g. '2 of 3 readers positive'."""
    reads = [int(num(v, c)) for c in READ_COLS if v is not None and num(v, c) is not None]
    if not reads:
        return [], ""
    pos = sum(r > 0 for r in reads)
    text = ("single reader" if len(reads) == 1 else
            f"{pos} of {len(reads)} readers said positive" + (" · unanimous" if pos in (0, len(reads)) else " · split vote"))
    return reads, text


def info_section(demo, votes, summ, female_code: float) -> str:
    """'About this scan': person, amyloid, reads + model, CenTauR regional SUVRs. Empty if nothing is known."""
    esc = lambda t: str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    dl = lambda items: "<dl>" + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in items if v is not None) + "</dl>"
    cards = []

    if demo is not None:
        age, sex, cdr, cdrsb, apoe = (num(demo, c) for c in ("age", "gender", "CDR", "CDR_SB", "apoe"))
        dx = demo.get("dx", np.nan)
        cards.append(("Person", dl([
            ("Age", f"{age:.0f}" if age is not None else None),
            ("Sex", None if sex is None else ("Female" if sex == female_code else "Male")),
            ("Diagnosis", None if pd.isna(dx) else esc(DX_NAMES.get(str(dx).strip().lower(), dx))),
            ("CDR", f"{cdr:g}" + (f" <span class='muted'>(sum of boxes {cdrsb:g})</span>" if cdrsb is not None else "")
             if cdr is not None else (f"sum of boxes {cdrsb:g}" if cdrsb is not None else None)),
            ("Cohort", None if pd.isna(demo.get("site", np.nan)) else esc(demo["site"]))])))
        amy, cl = num(demo, "amyloid_status"), num(demo, "CL")
        cards.append(("Amyloid and APOE", dl([
            ("Status", None if amy is None else ("<b class='hi'>Aβ+</b>" if amy == 1 else "Aβ−")),
            ("Centiloid", None if cl is None else f"{cl:.0f}" + (" <span class='muted'>(Aβ+ cut-off 24.4)</span>")),
            ("APOE ε4", None if apoe is None else ("carrier" if apoe == 1 else "non-carrier")),
        ])))

    reads, vtext = vote_summary(votes)
    vr = num(summ, "visual_read") if summ is not None else num(demo, "visual_read")
    p = num(summ, "P_positive")
    grade = num(votes, "visual_read_3_cat")
    if reads or vr is not None or p is not None:
        pills = "".join(f"<span class='pill {'pos' if r else 'neg'}' title='Reader {i}: {GRADE[r]}'>"
                        f"Reader {i} <b>{'+' if r else '−'}</b>{' ' + GRADE[r].split(', ')[1] if r else ''}</span>"
                        for i, r in enumerate(reads, 1))
        items = [("Visual read", None if vr is None else ("<b class='hi'>positive</b>" if vr == 1 else "negative")
                  + (f" <span class='muted'>({vtext})</span>" if vtext else "")),
                 ("Tau grade", None if not grade else GRADE.get(int(grade), f"{grade:g}").split(", ")[-1]),
                 ("Model", None if p is None else f"{'positive' if p >= 0.5 else 'negative'} "
                                                  f"<span class='muted'>(P(+) = {p:.2f}, cut-off 0.5)</span>")]
        cards.append(("Visual read and model", (f"<div class='pills'>{pills}</div>" if pills else "") + dl(items)))

    if demo is not None:
        vals = [(lab, num(demo, col), cut) for col, lab, cut in CENTAUR if num(demo, col) is not None]
        if vals:
            lo, hi = 0.8, max(2.2, max(v for _, v, _ in vals) * 1.05)
            pct = lambda v: 100 * (min(max(v, lo), hi) - lo) / (hi - lo)
            bars = "".join(
                f"<div class='sbar'><span class='slab'>{lab}</span><span class='strack'>"
                f"<span class='sfill {'above' if v >= cut else ''}' style='width:{pct(v):.1f}%'></span>"
                f"<span class='scut' style='left:{pct(cut):.1f}%' title='CTRz 2 = SUVR {cut:.2f}'></span></span>"
                f"<span class='sval'>{v:.2f}</span></div>" for lab, v, cut in vals)
            cards.append(("Regional tau SUVR (CenTauR)", bars + "<div class='muted small'>White tick = CTRz 2 "
                          "cut-off for that region (SUVR 1.29–1.35); orange bar = above it. Whole-scan region averages, "
                          "so a focal spot can be missed.</div>"))
    if not cards:
        return ""
    return ("<section id='info'><h2 class='sech'>About this scan</h2><div class='cards'>"
            + "".join(f"<div class='card ic'><h3>{t}</h3>{body}</div>" for t, body in cards) + "</div></section>")


def header_chips(row, set_name: str, vtext: str = "") -> str:
    chips = [f"<span class='chip'>{set_name}</span>"] if set_name else []
    if row is None:
        return "".join(chips)
    vr, p = row.get("visual_read", np.nan), row.get("P_positive", np.nan)
    read = None if pd.isna(vr) else int(vr == 1)
    model = None if pd.isna(p) else int(p >= 0.5)
    if read is not None:
        chips.append(f"<span class='chip'>Visual read {'+' if read else '−'}"
                     + (f" <span class='dim'>{vtext.split(' ·')[0].replace(' readers said positive', '')}</span>" if vtext else "")
                     + "</span>")
    if model is not None:
        chips.append(f"<span class='chip'>Model {'+' if model else '−'} <span class='dim'>P(+) = {p:.2f}</span></span>")
    if read is not None and model is not None:
        chips.append("<span class='chip agree'>Model and read agree</span>" if read == model else
                     f"<span class='chip disagree'>Model {'+' if model else '−'} / Read {'+' if read else '−'}</span>")
    return "".join(chips)


def region_toggles() -> str:
    groups = [("Count towards a positive read", (1, 2, 3)), ("Count only together with those", (4,)),
              ("Do not count", (5, 6))]
    out = ""
    for title, ks in groups:
        items = "".join(f"<label><input type='checkbox' data-region='{k}' {'checked' if k in SHOWN_FIRST else ''}>"
                        f"<i style='background:{REGION_COLOURS[k - 1]}'></i>{REGION_NAMES[k - 1]}</label>" for k in ks)
        out += f"<div class='rgroup'><div class='rtitle'>{title}</div>{items}</div>"
    return out


def main():
    args = parse_args()
    vis_dir = Path(args.vis_dir)
    ids = [int(t) for t in args.ids.split(",")] if args.ids else \
        sorted(int(f.name.split("_")[0]) for f in vis_dir.glob("*_input.nii.gz"))
    if not ids:
        raise SystemExit(f"No <ID>_input.nii.gz in {vis_dir} (runs from before the input was saved need a rerun)")
    out_dir = Path(args.out_dir) if args.out_dir else vis_dir / "viewers"
    out_dir.mkdir(parents=True, exist_ok=True)
    logit = "logit" in vis_dir.name
    set_name = "AVID (external)" if "AVID" in f"{vis_dir.name}/{out_dir.name}" else ""
    summary = pd.read_csv(vis_dir / "summary.csv").set_index("ID") if (vis_dir / "summary.csv").exists() else None
    demo_t, votes_t = by_id(args.demo_csv), by_id(args.votes)
    suvr = suvr_scans(Path(args.pkl_dir), set(ids), Path(args.cbl_mask), args.mask_erode) if args.pkl_dir else {}
    zones_layer = []  # invisible, only for the click readout
    if args.regions:
        zones, z_aff = region_zones(Path(args.regions))
        # onto the 2 mm scan grid: 8x fewer voxels than the 1 mm atlas, loads much faster (e.g. over ThinLinc)
        zones = np.asarray(image.resample_to_img(nib.Nifti1Image(zones.astype(np.int16), z_aff),
                                                 nib.Nifti1Image(np.zeros(MNI_SHAPE, np.int8), MNI_2MM_RAS),
                                                 interpolation="nearest", force_resample=True, copy_header=True).dataobj
                           ).astype(np.int8)
        zones_layer = [{"b64": nifti_b64(zones, MNI_2MM_RAS), "name": "zones.nii.gz", "colormap": "gray",
                        "cal_min": 0.0, "cal_max": 6.0, "opacity": 0.0}]
    in_brain = brain_mask(MNI_SHAPE, MNI_2MM_RAS, 2)
    ext, thumbs = locator(in_brain)
    fda_layers = []
    if args.regions:
        from nilearn import datasets
        tpl = datasets.load_mni152_template(resolution=2)  # 2 mm: 8x fewer voxels, fast 3D even in software rendering
        t = np.asarray(tpl.dataobj, dtype=np.float32)
        t8 = np.clip(255 * t / np.percentile(t[t > 0], 99.5), 0, 255).astype(np.uint8)  # uint8: keeps the page small
        reg = nib.load(args.regions)
        reg = np.asarray(image.resample_to_img(nib.Nifti1Image(np.asarray(reg.dataobj).astype(np.int16), reg.affine),
                                               tpl, interpolation="nearest", force_resample=True,
                                               copy_header=True).dataobj).astype(np.uint8)
        fda_layers = [{"b64": nifti_b64(t8, tpl.affine), "name": "template.nii.gz", "colormap": "gray",
                       "cal_min": 0.0, "cal_max": 255.0, "opacity": 1.0}]
        # one layer per region, so each can be switched on and off
        fda_layers += [{"b64": nifti_b64((reg == k).astype(np.uint8), tpl.affine), "name": f"fda{k}.nii.gz",
                        "colormap": f"fda{k}", "cal_min": 0.0, "cal_max": 1.0, "opacity": REGION_OPACITY,
                        "region": k, "start_hidden": k not in SHOWN_FIRST} for k in range(1, 7)]
    cmaps = {"reader": reader_colormap(),
             **{f"fda{k}": colormap([(0, c), (255, c)], overlay=True) for k, c in enumerate(REGION_COLOURS, 1)}}

    for scan_id in ids:
        x_full = on_mni_grid(vis_dir / f"{scan_id}_input.nii.gz")  # whole scan, for the "whole scan" checkbox
        x = x_full * in_brain  # default view: inside the brain only
        maps = {vt: on_mni_grid(vis_dir / f"{scan_id}_occlusion_{vt}.nii.gz")
                for vt in ("pos_vs_neg", "class") if (vis_dir / f"{scan_id}_occlusion_{vt}.nii.gz").exists()}
        head = x[x > 0]
        grey_top = PET_DIM * float(np.percentile(head, 99.5)) if head.size else 1.0

        # left brain
        left, left_note, left_bar = [], "", ""
        if scan_id in suvr:
            s, s_aff, cov = suvr[scan_id]
            s_full = s
            s = s * brain_mask(s.shape, s_aff, 2)
            # same scan? the SUVR (in RAS) should follow the model input closely
            s_ras = np.asarray(nib.as_closest_canonical(nib.Nifti1Image(s, s_aff)).dataobj)
            both = (s_ras > 0) & (x > 0)
            r = float(np.corrcoef(s_ras[both], x[both])[0, 1]) if s_ras.shape == x.shape and both.sum() > 100 else float("nan")
            print(f"[{scan_id}] cerebellum coverage {cov:.2f}, SUVR vs model input r = {r:.2f}"
                  + ("  <-- WARNING: low, check the IDs / pkl_dir" if not r > 0.8 else "")
                  + ("  <-- WARNING: cerebellum mostly outside the scan" if cov < 0.5 else ""))
            left.append({"b64": nifti_b64(s.astype(np.float32), s_aff), "name": "suvr.nii.gz", "colormap": "reader",
                         "cal_min": 0.0, "cal_max": SUVR_TOP, "opacity": 1.0})
            left.append({"b64": nifti_b64(s_full.astype(np.float32), s_aff), "name": "suvr_full.nii.gz", "colormap": "reader",
                         "cal_min": 0.0, "cal_max": SUVR_TOP, "opacity": 1.0, "toggle": "whole", "start_hidden": True})
            left_title, left_sub = "Reader view", "SUVR (÷ whole cerebellum) in the readers' colour scale"
            ticks = [(0, "0")] + [(round(e * 100, 1), f"{e * SUVR_TOP:.2f}") for e in BAND_EDGES]
            left_bar = ("<div class='vbar'><div class='vgrad' style='background:" + reader_gradient_css() + "'></div>"
                        "<div class='vticks'>" + "".join(f"<span style='bottom:{b}%'>{t}</span>" for b, t in ticks)
                        + "</div><div class='vlabel'>SUVR</div></div>")
            left_note = ("Grey = below 1.65 × cerebellum, the readers' threshold for raised uptake. "
                         "Colours = above it, rising from purple to red.")
        else:
            left.append({"b64": nifti_b64(x, MNI_2MM_RAS), "name": "input_heat.nii.gz", "colormap": "inferno",
                         "cal_min": 0.0, "cal_max": grey_top / PET_DIM, "opacity": 1.0})
            left.append({"b64": nifti_b64(x_full, MNI_2MM_RAS), "name": "input_heat_full.nii.gz", "colormap": "inferno",
                         "cal_min": 0.0, "cal_max": grey_top / PET_DIM, "opacity": 1.0, "toggle": "whole", "start_hidden": True})
            left_title, left_sub = "Tau", "Model input (scaled 0–1 per scan, not SUVR)"
            left_note = "No readers' colour scale: run with --pkl_dir and --cbl_mask for the SUVR view."
        left += zones_layer

        # right brain
        right = [{"b64": nifti_b64(x, MNI_2MM_RAS), "name": "input.nii.gz", "colormap": "gray",
                  "cal_min": 0.0, "cal_max": grey_top, "opacity": 1.0},
                 {"b64": nifti_b64(x_full, MNI_2MM_RAS), "name": "input_full.nii.gz", "colormap": "gray",  # whole scan
                  "cal_min": 0.0, "cal_max": grey_top, "opacity": 1.0, "toggle": "whole", "start_hidden": True}]
        legend_right, bars = "", ""
        for vt, m in maps.items():
            signed = vt == "pos_vs_neg"
            top = args.color_max or robust_top(m, signed)
            hidden = vt == "class" and "pos_vs_neg" in maps
            lay = {"b64": nifti_b64(occlusion_rgba(m, signed, top), MNI_2MM_RAS), "name": f"{vt}_shown.nii.gz",
                   "opacity": 1.0, "toggle": vt, "start_hidden": hidden}
            values = {"b64": nifti_b64(m, MNI_2MM_RAS), "name": f"{vt}.nii.gz", "colormap": "gray",  # never shown,
                      "cal_min": 0.0, "cal_max": 1.0, "opacity": 0.0}                              # click readout only
            if signed:
                unit = "log-odds" if logit else "probability"
                bars += vbar(faded_css(NEG, 50, 0) + faded_css(POS, 50, 100),
                             [(100, f"+{top:.3g}"), (50, "0"), (0, f"−{top:.3g}")], f"pull to + / − ({unit})", vt)
                legend_right += (f"<div class='occ' data-for='{vt}'>Occlusion: hide a small block, see how the model "
                                 "changes. <b class='pos'>Red→yellow</b>: this area pushes it towards positive. "
                                 "<b class='neg'>Blue→cyan</b>: towards negative. Fainter = weaker.</div>")
            else:
                unit = "log-odds" if logit else "probability"
                bars += vbar(faded_css(WARM, 0, 100), [(100, f"{top:.3g}"), (0, "0")],
                             f"support for its call ({unit})", vt)
                legend_right += (f"<div class='occ' data-for='{vt}'>Class map: <b class='pos'>red→yellow</b> = this "
                                 "area supports the model's own call (+ or −). Fainter = weaker.</div>")
            right += [lay, values]

        row = summary.loc[scan_id] if summary is not None and scan_id in summary.index else None
        vtext = vote_summary(row_of(votes_t, scan_id))[1]
        info = info_section(row_of(demo_t, scan_id), row_of(votes_t, scan_id), row, args.female_code)
        first = "pos_vs_neg" if "pos_vs_neg" in maps else "class"
        toggles = ("<div class='group'><span class='glabel'>Occlusion</span><span class='seg' id='maps'>"
                   + "".join(f"<button data-map='{vt}' class='{'on' if vt == first else ''}'>"
                             f"{'+ / −' if vt == 'pos_vs_neg' else 'Class map'}</button>" for vt in maps)
                   + "<button data-map='off'>Off</button></span></div>") if maps else ""
        toggles += ("<div class='group'><label title='The readers saw the whole head, so off-target signal outside the "
                    "brain shows as such'><input type='checkbox' data-toggle='whole'> Whole head</label></div>")
        data = {"cmaps": cmaps, "L": left, "R": right, "F": fda_layers, "ext": ext, "thumbs": thumbs,
                "jumps": jump_targets(maps, in_brain),
                "regionNames": REGION_NAMES, "regionStatus": REGION_STATUS}
        page = TEMPLATE
        for k, v in {"__TITLE__": f"Scan {scan_id}", "__CHIPS__": header_chips(row, set_name, vtext), "__INFO__": info,
                     "__TOGGLES__": toggles, "__LEFT_TITLE__": left_title, "__LEFT_SUB__": left_sub,
                     "__LEFT_BARS__": left_bar, "__LEFT_LEGEND__": left_note, "__RIGHT_BARS__": bars,
                     "__RIGHT_LEGEND__": legend_right or "No occlusion map for this scan.",
                     "__FDA_DISPLAY__": "" if fda_layers else "hidden", "__REGION_TOGGLES__": region_toggles(),
                     "__NIIVUE__": NIIVUE, "__DATA__": json.dumps(data)}.items():
            page = page.replace(k, v)
        out = out_dir / f"{scan_id}_viewer.html"
        out.write_text(page, encoding="utf-8")
        print(f"[save] {out} ({out.stat().st_size / 1e6:.1f} MB)")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ · case viewer</title>
<style>
  :root { --bg: #0d1117; --panel: #161b22; --raised: #1f2631; --line: #2d3644; --text: #e6edf3; --muted: #9aa5b1;
          --accent: #4c8dff; --cross: #46e05a;
          /* brains: as tall as 1.2 x their width, but the page should fit the window (never below 260 px) */
          --brain-max: max(260px, calc(100vh - 390px)); }
  * { box-sizing: border-box; }
  html, body { margin: 0; background: var(--bg); color: var(--text);
               font: 14px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
  #app { padding: 12px 16px 20px; }
  header { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 10px; margin-bottom: 10px; }
  h1 { font-size: 19px; font-weight: 650; margin: 0 6px 0 0; }
  .chip { padding: 2px 10px; border-radius: 999px; font-size: 13px; background: var(--raised);
          border: 1px solid var(--line); white-space: nowrap; }
  .chip .dim { color: var(--muted); margin-left: 4px; }
  .chip.agree { border-color: #2c6e3d; color: #86e3a4; }
  .chip.disagree { border-color: #8a5a12; color: #ffc56e; }
  #status { color: var(--muted); font-size: 13px; margin-left: auto; }
  #status.err { color: #ff8a80; }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }
  #toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 22px; padding: 8px 12px; }
  .group { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 10px; }
  .glabel { color: var(--muted); font-size: 11.5px; text-transform: uppercase; letter-spacing: .05em; }
  .seg { display: inline-flex; border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }
  .seg button { background: transparent; color: var(--text); border: 0; border-right: 1px solid var(--line);
                padding: 4px 11px; font: inherit; font-size: 13px; cursor: pointer; }
  .seg button:last-child { border-right: 0; }
  .seg button:hover { background: var(--raised); }
  .seg button.on { background: var(--accent); color: #fff; }
  .btn { background: var(--raised); color: var(--text); border: 1px solid var(--line); border-radius: 6px;
         padding: 4px 10px; font: inherit; font-size: 13px; cursor: pointer; white-space: nowrap; }
  .btn:hover { border-color: #46546a; }
  .btn.pos { border-color: #7a2b18; } .btn.neg { border-color: #1d3f7a; }
  label { cursor: pointer; white-space: nowrap; font-size: 13px; }
  input[type=checkbox] { accent-color: var(--accent); vertical-align: -2px; }
  #slicegroup { flex: 1 1 380px; flex-wrap: nowrap; }
  #slicegroup .end { color: var(--muted); font-size: 12px; width: 38px; text-align: center; flex: none; }
  input[type=range] { flex: 1; min-width: 90px; accent-color: var(--accent); }
  #slicetext { font-size: 13px; font-variant-numeric: tabular-nums; min-width: 86px; flex: none; }
  #locator { position: relative; height: 54px; flex: none; }
  #locator img { height: 54px; display: block; border-radius: 3px; image-rendering: auto; }
  #locator .line { position: absolute; background: var(--cross); box-shadow: 0 0 2px #000; }
  #readout { display: flex; flex-wrap: wrap; gap: 4px 26px; padding: 7px 12px; margin: 8px 0; min-height: 36px;
             font-size: 13px; font-variant-numeric: tabular-nums; }
  #readout .k { color: var(--muted); margin-right: 6px; }
  #readout .hint { color: var(--muted); }
  #panes { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 330px), 1fr)); gap: 10px;
           align-items: start; }
  .pane { overflow: hidden; display: flex; flex-direction: column; }
  .pane[hidden] { display: none; }
  .phead { display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; padding: 8px 12px;
           border-bottom: 1px solid var(--line); min-height: 58px; }
  .phead h2 { font-size: 14.5px; font-weight: 600; margin: 0; }
  .phead .sub { color: var(--muted); font-size: 12.5px; }
  .body { display: flex; aspect-ratio: 5 / 6; max-height: var(--brain-max); min-height: 260px; background: #000; }
  .cv { flex: 1; min-width: 0; position: relative; }
  .cv canvas { display: block; outline: none; }
  .side { width: 86px; flex: none; position: relative; }
  .vbar { position: absolute; top: 26px; bottom: 26px; left: 8px; right: 0; }
  .vgrad { position: absolute; left: 0; top: 0; bottom: 0; width: 13px; border-radius: 2px; }
  .vticks span { position: absolute; left: 18px; transform: translateY(50%); font-size: 11px; color: #c3cad3;
                 font-variant-numeric: tabular-nums; }
  .vlabel { position: absolute; left: 62px; top: 50%; transform: translate(-50%, -50%) rotate(-90deg);
            white-space: nowrap; font-size: 11px; color: var(--muted); }
  .legend { padding: 8px 12px 10px; border-top: 1px solid var(--line); font-size: 12.5px; color: var(--muted); }
  .legend b.pos { color: #ffb347; font-weight: 600; } .legend b.neg { color: #6fb6ff; font-weight: 600; }
  .regions { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 6px 14px; }
  .rtitle { font-size: 11.5px; text-transform: uppercase; letter-spacing: .04em; margin-bottom: 2px; }
  .regions label { display: block; color: var(--text); white-space: normal; padding: 1px 0; }
  .regions i { display: inline-block; width: 11px; height: 11px; border-radius: 2px; margin: 0 6px 0 4px;
               vertical-align: -1px; }
  #info { margin-top: 14px; }
  .sech { font-size: 15px; font-weight: 600; margin: 0 0 8px; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 260px), 1fr)); gap: 10px; }
  .ic { padding: 10px 14px 12px; }
  .ic h3 { font-size: 12px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .05em;
           margin: 0 0 8px; }
  .ic dl { display: grid; grid-template-columns: max-content 1fr; gap: 4px 14px; margin: 0; font-size: 13.5px; }
  .ic dt { color: var(--muted); } .ic dd { margin: 0; }
  .muted { color: var(--muted); } .small { font-size: 12px; margin-top: 8px; }
  b.hi { color: #ffc56e; font-weight: 600; }
  .pills { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 9px; }
  .pill { font-size: 12.5px; padding: 2px 9px; border-radius: 999px; border: 1px solid var(--line); white-space: nowrap; }
  .pill.pos { border-color: #8a5a12; background: #2a2112; } .pill.neg { background: var(--raised); }
  .sbar { display: grid; grid-template-columns: 118px 1fr 40px; align-items: center; gap: 8px; margin: 5px 0;
          font-size: 13px; }
  .slab { color: var(--muted); } .sval { text-align: right; font-variant-numeric: tabular-nums; }
  .strack { position: relative; height: 10px; background: var(--raised); border-radius: 5px; }
  .sfill { position: absolute; left: 0; top: 0; bottom: 0; background: #5f7896; border-radius: 5px; }
  .sfill.above { background: #e8963a; }
  .scut { position: absolute; top: -3px; bottom: -3px; width: 2px; background: #fff; }
  details { margin-top: 10px; padding: 8px 12px; font-size: 13px; color: var(--muted); }
  details summary { cursor: pointer; color: var(--text); font-weight: 600; }
  details ul { margin: 6px 0 2px; padding-left: 18px; } details li { margin: 3px 0; }
  details + details { margin-top: 6px; }
</style></head>
<body><div id="app">
  <header><h1>__TITLE__</h1>__CHIPS__<span id="status">Loading…</span></header>

  <div id="toolbar" class="card">
    <div class="group"><span class="glabel">Plane</span>
      <span class="seg" id="views"><button data-view="0" class="on">Axial</button><button data-view="1">Coronal</button><button data-view="2">Sagittal</button><button data-view="3">All three</button></span></div>
    <div class="group" id="slicegroup"><span class="glabel">Slice</span>
      <div id="locator" title="Where the slice is"><img alt=""><div class="line"></div></div>
      <button class="btn" id="prev" title="One slice back (2 mm)">‹</button>
      <span class="end" id="end0"></span><input type="range" id="slider" step="2" aria-label="Slice position"><span class="end" id="end1"></span>
      <button class="btn" id="next" title="One slice on (2 mm)">›</button>
      <span id="slicetext"></span></div>
    __TOGGLES__
    <div class="group" id="jumps"><span class="glabel">Go to</span></div>
    <div class="group"><button class="btn" id="reset" title="Undo zoom and panning">Reset view</button></div>
  </div>

  <div id="readout" class="card"><span class="hint">Click a spot on any brain to see its values here.</span></div>

  <div id="panes">
    <section class="pane card"><div class="phead"><div><h2>__LEFT_TITLE__</h2><div class="sub">__LEFT_SUB__</div></div></div>
      <div class="body"><div class="cv"><canvas id="glL" tabindex="0"></canvas></div><div class="side">__LEFT_BARS__</div></div>
      <div class="legend">__LEFT_LEGEND__</div></section>
    <section class="pane card"><div class="phead"><div><h2>Model view</h2><div class="sub">Model input with its occlusion map</div></div></div>
      <div class="body"><div class="cv"><canvas id="glR" tabindex="0"></canvas></div><div class="side">__RIGHT_BARS__</div></div>
      <div class="legend">__RIGHT_LEGEND__</div></section>
    <section class="pane card" id="fdapane" __FDA_DISPLAY__><div class="phead"><div><h2>FDA read regions</h2><div class="sub">On the MNI template · tick to show</div></div>
      <span class="seg" id="fdamode"><button data-fda="slices" class="on">Slices</button><button data-fda="3d">3D</button></span></div>
      <div class="body"><div class="cv"><canvas id="glF" tabindex="0"></canvas></div><div class="side"></div></div>
      <div class="legend regions">__REGION_TOGGLES__</div></section>
  </div>

  __INFO__
  <details class="card"><summary>How to use</summary><ul>
    <li><b>Move through the brain</b> with the slider, the ‹ › buttons, or by scrolling on a brain. The little picture next to the slider shows where the slice is.</li>
    <li><b>Click</b> a spot to see its values above the brains. All brains follow the same spot (green cross).</li>
    <li><b>Zoom</b> with ctrl + scroll (pinch on a trackpad), <b>pan</b> by right-dragging; <i>Reset view</i> undoes both.</li>
    <li><b>Go to</b> jumps to the spot where hiding the scan changed the model's output most.</li>
    <li>In the <b>3D</b> FDA view, drag to turn the brain.</li></ul></details>
  <details class="card"><summary>About these views</summary><ul>
    <li><b>Reader view</b> imitates the readers' display: SUVR divided by the scan's whole-cerebellum mean, grey below 1.65 × cerebellum and coloured bands above, scale top 5.89. The readers used a hand-traced cerebellum on their own reoriented images (and white for low values); here it is the Centiloid cerebellum mask in MNI space, so it is close to what they saw, not identical.</li>
    <li><b>Occlusion</b>: each small block of the scan is hidden in turn and the change in the model's output is recorded there. Colours are scaled to this scan's 99th percentile.</li>
    <li><b>FDA read regions</b>: an atlas version (Desikan-Killiany, temporal lobe split with the label's crosshairs) of the regions named in the TAUVID label. The label itself gives no exact borders. A read is positive with raised uptake in posterolateral temporal, occipital or parietal / precuneus cortex. Frontal counts only with one of those; mesial and anterolateral temporal do not count.</li>
    <li>Only uptake in the cortex counts for the read, not white matter or signal outside the brain.</li></ul></details>
</div>
<script type="module">
const D = __DATA__;
const statusEl = document.getElementById("status");
let Niivue;
try { ({ Niivue } = await import("__NIIVUE__")); }
catch (e) { statusEl.textContent = "Could not load the viewer library (this page needs internet once)."; statusEl.className = "err"; throw e; }

const toUrl = (b64) => URL.createObjectURL(new Blob([Uint8Array.from(atob(b64), c => c.charCodeAt(0))]));
const makeViewer = async (id, layers, opts) => {
  // isNearestInterpolation: each 2 mm voxel one flat block, like the occlusion PNGs (no blur between voxels)
  const nv = new Niivue({ backColor: [0, 0, 0, 1], isColorbar: false, isOrientCube: false, crosshairColor: [0.27, 0.88, 0.35, 1],
                          dragAndDropEnabled: false, isNearestInterpolation: true, multiplanarShowRender: 0, ...opts });
  await nv.attachTo(id);
  nv.opts.dragMode = nv.dragModes.pan;  // right-drag pans (NiiVue's default changes the contrast = the colour scale)
  for (const [name, cm] of Object.entries(D.cmaps)) nv.addColormap(name, cm);
  await nv.loadVolumes(layers.map(l => {
    const v = { url: toUrl(l.b64), name: l.name, opacity: l.start_hidden ? 0 : l.opacity };
    if (l.colormap) v.colormap = l.colormap;  // colour volumes (occlusion) bring their own colours
    return v;
  }));
  layers.forEach((l, i) => { if (l.cal_min !== undefined) { nv.volumes[i].cal_min = l.cal_min; nv.volumes[i].cal_max = l.cal_max; } });
  nv.updateGLVolume();
  nv.setSliceType(0);  // axial, like the readers
  return nv;
};
const L = await makeViewer("glL", D.L), R = await makeViewer("glR", D.R);
L.syncWith(R, { "2d": true }); R.syncWith(L, { "2d": true });  // zoom and pan together
const F = D.F.length ? await makeViewer("glF", D.F, { isNearestInterpolation: false, multiplanarShowRender: 2, show3Dcrosshair: true }) : null;
const all = [L, R, F].filter(Boolean);
statusEl.textContent = "";

// ---- position: one spot (in MNI mm) shared by all brains ----
let view = 0, fdaMode = "slices", busy = false;
const mmOf = (nv) => Array.from(nv.frac2mm(nv.scene.crosshairPos)).slice(0, 3);
const goTo = (mm, from = null) => {
  if (busy) return; busy = true;
  for (const nv of all) if (nv !== from) { nv.scene.crosshairPos = nv.mm2frac(mm); nv.drawScene(); }
  busy = false; update();
};
for (const nv of all) nv.onLocationChange = () => { if (!busy) goTo(mmOf(nv), nv); };

// ---- slice slider ----
const AX = { 0: 2, 1: 1, 2: 0, 3: 2 };  // view -> axis it moves through (All three: the axial slider)
const AXINFO = [{ name: "x", ends: ["Left", "Right"], thumb: "ax" }, { name: "y", ends: ["Back", "Front"], thumb: "sag" },
                { name: "z", ends: ["Bottom", "Top"], thumb: "sag" }];
const slider = document.getElementById("slider"), line = document.querySelector("#locator .line"),
      thumb = document.querySelector("#locator img");
const sgn = (v) => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(0);
const setupSlider = () => {
  const a = AX[view], info = AXINFO[a], [lo, hi] = D.ext[info.name];
  slider.min = lo; slider.max = hi;
  document.getElementById("end0").textContent = info.ends[0];
  document.getElementById("end1").textContent = info.ends[1];
  thumb.src = "data:image/png;base64," + D.thumbs[info.thumb];
  update();
};
const placeLine = (a, v) => {
  const [lo, hi] = D.ext[AXINFO[a].name], f = Math.min(Math.max((v - lo) / (hi - lo), 0), 1) * 100;
  // sagittal picture: back on the left, top at the top (like NiiVue); axial picture: left on the left
  const s = line.style;
  if (a === 2) Object.assign(s, { top: `calc(${100 - f}% - 1px)`, left: 0, width: "100%", height: "2px" });
  else Object.assign(s, { left: `calc(${f}% - 1px)`, top: 0, width: "2px", height: "100%" });
};
slider.oninput = () => { const mm = mmOf(L); mm[AX[view]] = +slider.value; goTo(mm); };
const step = (d) => { slider.value = +slider.value + 2 * d; slider.oninput(); };
document.getElementById("prev").onclick = () => step(-1);
document.getElementById("next").onclick = () => step(1);

// ---- values at the cursor ----
const fmt = (v) => (Math.abs(v) >= 0.1 ? v.toFixed(2) : v.toPrecision(2));
const val = (nv, name) => { const v = nv.volumes.find(v => v.name === name); if (!v) return null;
  return v.getValue(...v.mm2vox(mmOf(nv)), v.frame4D); };
const item = (k, v) => `<span><span class="k">${k}</span>${v}</span>`;
let clicked = false;
for (const id of ["glL", "glR", "glF"]) document.getElementById(id).addEventListener("pointerdown", () => { clicked = true; });
const update = () => {
  const mm = mmOf(L), a = AX[view];
  slider.value = mm[a]; placeLine(a, mm[a]);
  document.getElementById("slicetext").textContent = `${AXINFO[a].name} = ${sgn(mm[a])} mm`;
  if (!clicked) return;
  const side = mm[0] < -1 ? "left" : mm[0] > 1 ? "right" : "midline";
  const parts = [item("Position", `${side} · x ${sgn(mm[0])}, y ${sgn(mm[1])}, z ${sgn(mm[2])} mm`)];
  const s = val(L, "suvr_full.nii.gz");
  if (s !== null) parts.push(item("SUVR", `${s.toFixed(2)} × cerebellum` + (s >= 1.65 ? " (above 1.65)" : "")));
  const pn = val(R, "pos_vs_neg.nii.gz");
  if (pn !== null) parts.push(item("Occlusion", `${pn > 0 ? "+" : ""}${fmt(pn)}` + (pn > 0 ? " → towards +" : pn < 0 ? " → towards −" : "")));
  const c = val(R, "class.nii.gz"); if (c !== null) parts.push(item("Class map", fmt(c)));
  const z = val(L, "zones.nii.gz");
  if (z !== null) { const k = Math.round(z);
    parts.push(item("FDA region", k > 0 ? `${D.regionNames[k - 1]} <span class="k">(${D.regionStatus[k]})</span>` : "none here")); }
  document.getElementById("readout").innerHTML = parts.join("");
};

// ---- buttons ----
const views = document.querySelectorAll("[data-view]");
views.forEach(b => b.onclick = () => {
  view = +b.dataset.view; L.setSliceType(view); R.setSliceType(view);
  if (F && fdaMode === "slices") F.setSliceType(view);
  views.forEach(x => x.classList.toggle("on", x === b)); setupSlider();
});
const fdaButtons = document.querySelectorAll("[data-fda]");
fdaButtons.forEach(b => b.onclick = () => {
  fdaMode = b.dataset.fda;
  if (fdaMode === "3d") { F.setSliceType(4); F.setRenderAzimuthElevation(120, 15); } else F.setSliceType(view);
  fdaButtons.forEach(x => x.classList.toggle("on", x === b));
});
document.querySelectorAll("[data-region]").forEach(c => c.onchange = () => {
  const i = D.F.findIndex(l => l.region === +c.dataset.region);
  F.setOpacity(i, c.checked ? D.F[i].opacity : 0);
});
document.getElementById("reset").onclick = () => { for (const nv of all) nv.setPan2Dxyzmm([0, 0, 0, 1]); };
const jumps = document.getElementById("jumps");
if (!D.jumps.length) jumps.remove();
for (const j of D.jumps) { const b = document.createElement("button"); b.className = `btn ${j.cls}`; b.textContent = j.label;
  b.onclick = () => { clicked = true; goTo(j.mm); }; jumps.appendChild(b); }

let mapOn = document.querySelector("#maps .on")?.dataset.map ?? "off";
const shown = (t) => { if (t !== "whole") return t === mapOn;
  const c = document.querySelector(`[data-toggle="${t}"]`); return !c || c.checked; };
const mapButtons = document.querySelectorAll("[data-map]");
mapButtons.forEach(b => b.onclick = () => { mapOn = b.dataset.map;
  mapButtons.forEach(x => x.classList.toggle("on", x === b)); applyLayers(); });
const applyLayers = () => {  // a layer shows if its checkbox is ticked; a hidden map's colour bar is hidden too
  for (const [nv, layers] of [[L, D.L], [R, D.R]])
    layers.forEach((l, i) => { if (l.toggle) nv.setOpacity(i, shown(l.toggle) ? l.opacity : 0); });
  document.querySelectorAll(".occ").forEach(d => d.style.visibility = shown(d.dataset.for) ? "" : "hidden");
  document.querySelectorAll(".legend .occ").forEach(d => d.style.display = shown(d.dataset.for) ? "" : "none");
};
document.querySelectorAll("[data-toggle]").forEach(c => c.onchange = applyLayers);
applyLayers();
goTo(mmOf(L));
setupSlider();

// for testing: open at a given spot / view from the address, e.g. viewer.html#z10, #coronal, #fda3d
const zh = location.hash.match(/z(-?\d+)/);
if (zh) { clicked = true; goTo([-50, -55, +zh[1]]); }
if (location.hash.includes("coronal")) document.querySelector('[data-view="1"]').click();
if (location.hash.includes("sagittal")) document.querySelector('[data-view="2"]').click();
if (location.hash.includes("all3")) document.querySelector('[data-view="3"]').click();
if (location.hash.includes("fda3d") && F) document.querySelector('[data-fda="3d"]').click();
</script></body></html>
"""

if __name__ == "__main__":
    main()
