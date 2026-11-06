#!/usr/bin/env python3
"""One .html per scan: what the reader saw next to what the model saw, two linked NiiVue brains.

  left   the scan as SUVR (divided by its own whole-cerebellum mean) in the readers' colour scale: inverted grey
         (white = low) up to 1.65 x cerebellum, then purple / dark blue / cyan / teal / light green / yellow /
         orange / red bands, scale top 5.89 x cerebellum (1.65 at 28% of the bar, as measured on the Moscoso
         supplement colour bars). Readers had a hand-traced cerebellum on reoriented native images; this is the
         Centiloid whole-cerebellum mask in MNI space instead, so it is close to their display, not identical.
  right  the model's input in grey (as under the occlusion PNGs) with the occlusion map in the PNGs' colours:
         'pos vs neg' red -> yellow = supports positive, blue -> light cyan = supports negative; 'class' dark red
         -> yellow. Colour top = the map's 99th percentile (same rule as the PNGs) or --color_max.
  both   FDA visual-read regions (plot_fda_regions.py) as outlines in one colour, on/off with a checkbox.

The two brains move together: scroll through axial slices (the readers' view, default), switch to three planes or
a rotating 3D brain. Open the .html by double-clicking; NiiVue is loaded from the web, the images are inside the page.

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
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt

from make_niivue_viewer import NIIVUE, REGION_NAMES

# MNI model input grid (tau_T1MNI_pts, RAS), the same as voxelwise_visual_read.py
MNI_2MM_RAS = np.array([[2.0, 0, 0, -90], [0, 2.0, 0, -126], [0, 0, 2.0, -72], [0, 0, 0, 1]])
MNI_SHAPE = (91, 109, 91)

# readers' colour scale (memory: visual-read method notes): bar 0..SUVR_TOP, 1.65 x cerebellum at 28% of the bar
SUVR_TRANSITION, TRANSITION_AT = 1.65, 0.28
SUVR_TOP = SUVR_TRANSITION / TRANSITION_AT  # 5.89
BAND_EDGES = [0.28, 0.33, 0.42, 0.52, 0.625, 0.715, 0.81, 0.90, 1.0]  # share of the bar
BAND_COLOURS = ["#7f3f98", "#1f2a9c", "#00bfe8", "#00897b", "#8fdc5a", "#ffe600", "#ff9100", "#e3000f"]
GREY_LOW, GREY_HIGH = "#ffffff", "#3a3a3a"  # inverted grey below the transition: white = low

# occlusion colours, copied from src/vis.py (WARM, WARM_COOL) so the viewer matches the PNGs
WARM = ["#c00000", "#ff3000", "#ff9900", "#ffff66"]
POS = ["#c00000", "#ff9900", "#ffff66"]   # WARM_COOL above 0: near zero -> top
NEG = ["#1040ff", "#2aa0ff", "#a0ffff"]   # WARM_COOL below 0: near zero -> most negative
PET_DIM = 1.3  # src/vis.py: grey top = 1.3 x the 99.5th percentile, so yellow still stands out
REGION_OUTLINE = "#ff2bd6"  # one colour for every FDA region (pink: not in the reader scale or the occlusion map)
ZONE_MM = 6  # each FDA region (a thin cortical ribbon in the atlas) is grown this far, so borders become clean lines


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vis_dir", required=True, help="Occlusion run folder (run_vis.py output).")
    p.add_argument("--ids", default=None, help="Comma-separated IDs (default: every <ID>_input.nii.gz in --vis_dir).")
    p.add_argument("--pkl_dir", default=None, help="Original MNI images (tau_batch_*.pkl) for the SUVR / reader view.")
    p.add_argument("--cbl_mask", default=None, help="Whole-cerebellum mask (Centiloid voi_WhlCbl_2mm.nii).")
    p.add_argument("--mask_erode", type=int, default=1, help="Voxels to shrink the cerebellum mask by (as before: 1).")
    p.add_argument("--regions", default=None, help="FDA region map (plot_fda_regions.py), drawn as outlines.")
    p.add_argument("--color_max", type=float, default=None,
                   help="Top of the occlusion colour scale (default: each map's 99th percentile, like the PNGs).")
    p.add_argument("--hide_below", type=float, default=0.1,
                   help="Occlusion values below this share of the top are not coloured (default 0.1).")
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


def even(colours: list) -> list:
    return [(round(k * 255 / (len(colours) - 1)), c) for k, c in enumerate(colours)]


def reader_colormap() -> dict:
    """Inverted grey ramp up to the transition, then flat colour bands (two stops per band, so no blending)."""
    t = round(TRANSITION_AT * 255)
    stops = [(0, GREY_LOW), (t - 1, GREY_HIGH)]
    for lo, hi, c in zip(BAND_EDGES[:-1], BAND_EDGES[1:], BAND_COLOURS):
        stops += [(round(lo * 255), c), (round(hi * 255) - (0 if hi == 1.0 else 1), c)]
    # index 0 (outside the brain) see-through: the page background shows there, and the 3D brain is not a white box
    return colormap(stops, overlay=True)


def reader_gradient_css() -> str:
    parts = [f"{GREY_LOW} 0%", f"{GREY_HIGH} {TRANSITION_AT * 100:.1f}%"]
    for lo, hi, c in zip(BAND_EDGES[:-1], BAND_EDGES[1:], BAND_COLOURS):
        parts += [f"{c} {lo * 100:.1f}%", f"{c} {hi * 100:.1f}%"]
    return "linear-gradient(to right, " + ", ".join(parts) + ")"


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
    (nearest one). Returns (zones 0-6, border lines between two different zones, affine). Lines between zones are
    what is drawn; the zones themselves are only used to name the region under the cursor. For 3D, only the part of
    the lines near the brain's outside is kept (lines on the surface instead of sheets through the brain)."""
    img = nib.load(path)
    lab = np.asarray(img.dataobj).astype(np.int8)
    vox_mm = np.sqrt((img.affine[:3, :3] ** 2).sum(0))
    dist, idx = distance_transform_edt(lab == 0, sampling=vox_mm, return_indices=True)
    zones = lab[tuple(idx)]
    brain = brain_mask(lab.shape, img.affine, 0)
    zones[(dist > ZONE_MM) | ~brain] = 0
    border = np.zeros(lab.shape, bool)
    for ax in range(3):  # one side only, so the line is one voxel thick
        nb = np.roll(zones, 1, axis=ax)
        border |= (zones > 0) & (nb > 0) & (nb != zones)
    surface = brain & ~binary_erosion(brain, iterations=int(round(4 / vox_mm.min())))  # outer 4 mm
    return zones, border.astype(np.uint8), (border & surface).astype(np.uint8), img.affine


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
    summary = pd.read_csv(vis_dir / "summary.csv").set_index("ID") if (vis_dir / "summary.csv").exists() else None
    suvr = suvr_scans(Path(args.pkl_dir), set(ids), Path(args.cbl_mask), args.mask_erode) if args.pkl_dir else {}
    outline = []
    if args.regions:
        zones, border, border3d, z_aff = region_zones(Path(args.regions))
        outline = [{"b64": nifti_b64(border, z_aff), "name": "regions.nii.gz", "colormap": "outline",
                    "cal_min": 0.5, "cal_max": 1.0, "opacity": 1.0, "toggle": "regions", "only": "slices"},
                   {"b64": nifti_b64(border3d, z_aff), "name": "regions3d.nii.gz", "colormap": "outline",
                    "cal_min": 0.5, "cal_max": 1.0, "opacity": 1.0, "toggle": "regions", "only": "render"},
                   {"b64": nifti_b64(zones, z_aff), "name": "zones.nii.gz", "colormap": "gray",  # never shown
                    "cal_min": 0.0, "cal_max": 6.0, "opacity": 0.0}]
    in_brain = brain_mask(MNI_SHAPE, MNI_2MM_RAS, 2)

    for scan_id in ids:
        x = on_mni_grid(vis_dir / f"{scan_id}_input.nii.gz") * in_brain  # shown inside the brain only
        maps = {vt: on_mni_grid(vis_dir / f"{scan_id}_occlusion_{vt}.nii.gz")
                for vt in ("pos_vs_neg", "class") if (vis_dir / f"{scan_id}_occlusion_{vt}.nii.gz").exists()}
        head = x[x > 0]
        grey_top = PET_DIM * float(np.percentile(head, 99.5)) if head.size else 1.0

        # left brain
        left, left_note = [], ""
        if scan_id in suvr:
            s, s_aff, cov = suvr[scan_id]
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
            left_title = "What the reader saw: SUVR (÷ whole cerebellum), readers' colour scale"
            ticks = "".join(f"<span class='{'low' if k % 2 else ''}' style='left:{e * 100:.1f}%'>{e * SUVR_TOP:.2f}</span>"
                            for k, e in enumerate(BAND_EDGES[:-1]))  # every other one lower, so 1.65 / 1.94 do not overlap
            left_note = (f"<div class='cbar'><div class='grad' style='background:{reader_gradient_css()}'></div>"
                         f"<div class='ticks'>{ticks}</div></div>"
                         "<div class='note'>× cerebellum · white→grey = below 1.65 (the readers' transition) · "
                         f"top {SUVR_TOP:.2f}</div>")
        else:
            left.append({"b64": nifti_b64(x, MNI_2MM_RAS), "name": "input_heat.nii.gz", "colormap": "inferno",
                         "cal_min": 0.0, "cal_max": grey_top / PET_DIM, "opacity": 1.0})
            left_title = "Tau (model input, 0–1 per scan; not SUVR — no reader scale without --pkl_dir)"
        left += outline

        # right brain
        right = [{"b64": nifti_b64(x, MNI_2MM_RAS), "name": "input.nii.gz", "colormap": "gray",
                  "cal_min": 0.0, "cal_max": grey_top, "opacity": 1.0}]
        legend_right = ""
        for vt, m in maps.items():
            signed = vt == "pos_vs_neg"
            top = args.color_max or robust_top(m, signed)
            lay = {"b64": nifti_b64(m, MNI_2MM_RAS), "name": f"{vt}.nii.gz", "colormap": "occ_pos" if signed else "occ_warm",
                   "cal_min": args.hide_below * top, "cal_max": top, "opacity": 1.0, "toggle": vt,
                   "start_hidden": vt == "class" and "pos_vs_neg" in maps}
            if signed:
                lay.update(colormapNegative="occ_neg", cal_minNeg=-top, cal_maxNeg=-args.hide_below * top)
                what = ("drop in log-odds of positive when hidden" if logit
                        else "drop in P(positive) minus drop in P(negative)")
                grad = f"linear-gradient(to right, {', '.join(NEG[::-1])}, {', '.join(POS)})"
                legend_right += (f"<div class='occ' data-for='{vt}'><div class='cbar'><div class='grad' style='background:{grad}'></div>"
                                 f"<div class='ticks'><span style='left:0%'>−{top:.3g}</span><span style='left:50%'>0</span>"
                                 f"<span style='left:100%'>+{top:.3g}</span></div></div>"
                                 f"<div class='note'>pos vs neg: {what} · red→yellow supports positive · blue→light cyan "
                                 f"supports negative · below {args.hide_below:g}× top not coloured</div></div>")
            else:
                what = "drop in log-odds of the predicted class" if logit else "drop in P(predicted class)"
                legend_right += (f"<div class='occ' data-for='{vt}'><div class='cbar'><div class='grad' "
                                 f"style='background:linear-gradient(to right, {', '.join(WARM)})'></div>"
                                 f"<div class='ticks'><span style='left:0%'>0</span><span style='left:100%'>{top:.3g}</span>"
                                 f"</div></div><div class='note'>class: {what} when hidden</div></div>")
            right.append(lay)
        right += outline

        title = f"Scan {scan_id}"
        if summary is not None and scan_id in summary.index:
            row = summary.loc[scan_id]
            vr = row.get("visual_read", np.nan)
            title += f" · visual read: {'n/a' if pd.isna(vr) else ('positive' if vr == 1 else 'negative')}"
            if "P_positive" in row and not pd.isna(row["P_positive"]):
                title += f" · model P(positive) = {row['P_positive']:.2f}"
        toggles = "".join(
            f"<label><input type='checkbox' data-toggle='{vt}' {'' if (vt == 'class' and 'pos_vs_neg' in maps) else 'checked'}>"
            f" Occlusion: {'pos vs neg' if vt == 'pos_vs_neg' else 'class'}</label>" for vt in maps)
        if outline:
            toggles += (f"<label><input type='checkbox' data-toggle='regions' checked> "
                        f"<i class='sw' style='background:{REGION_OUTLINE}'></i> FDA region borders</label>")

        page = (TEMPLATE.replace("__NIIVUE__", NIIVUE).replace("__TITLE__", title)
                .replace("__LEFT_TITLE__", left_title).replace("__LEFT_LEGEND__", left_note)
                .replace("__RIGHT_LEGEND__", legend_right).replace("__TOGGLES__", toggles)
                .replace("__REGION_NAMES__", json.dumps(REGION_NAMES)).replace("__LEFT__", json.dumps(left)).replace("__RIGHT__", json.dumps(right))
                .replace("__CMAPS__", json.dumps({
                    "reader": reader_colormap(), "occ_pos": colormap(even(POS), True), "occ_neg": colormap(even(NEG), True),
                    "occ_warm": colormap(even(WARM), True),
                    "outline": colormap([(0, REGION_OUTLINE), (255, REGION_OUTLINE)], True)})))
        out = out_dir / f"{scan_id}_viewer.html"
        out.write_text(page, encoding="utf-8")
        print(f"[save] {out} ({out.stat().st_size / 1e6:.1f} MB)")


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  html, body { margin: 0; height: 100%; background: #000; color: #ddd; font: 14px system-ui, sans-serif; }
  #wrap { display: flex; flex-direction: column; height: 100%; }
  #bar { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; padding: 8px 12px; background: #111; }
  #bar button { background: #222; color: #ddd; border: 1px solid #444; border-radius: 4px; padding: 4px 10px; cursor: pointer; }
  #bar button.on { border-color: #bbb; color: #fff; }
  #bar label { white-space: nowrap; }
  .sw { display: inline-block; width: 11px; height: 11px; vertical-align: -1px; }
  #panes { flex: 1; min-height: 0; display: flex; gap: 4px; }
  .pane { flex: 1; min-width: 0; display: flex; flex-direction: column; }
  .head { padding: 5px 10px; background: #181818; font-weight: 600; }
  .box { flex: 1; min-height: 0; }
  canvas { width: 100%; height: 100%; display: block; outline: none; }
  .legend { padding: 6px 12px 10px; background: #111; min-height: 54px; }
  .cbar { position: relative; width: min(420px, 90%); margin-bottom: 16px; }
  .grad { height: 12px; }
  .ticks span { position: absolute; top: 14px; transform: translateX(-50%); font-size: 11px; color: #bbb; }
  .ticks span.low { top: 27px; }
  .cbar:has(.low) { margin-bottom: 30px; }
  .note { color: #999; font-size: 12px; }
  #status { color: #999; }
  #here { padding: 4px 12px; background: #0b0b0b; color: #ccc; font-size: 13px; min-height: 18px; }
  @media (max-width: 800px) { #panes { flex-direction: column; } }
</style></head>
<body><div id="wrap">
  <div id="bar">
    <b>__TITLE__</b>
    <button data-view="axial" class="on">Axial slices</button>
    <button data-view="planes">Three planes</button>
    <button data-view="render">3D</button>
    <button id="clip">Cut-away (3D)</button>
    __TOGGLES__
    <button id="full">Full screen</button>
    <span id="status">loading…</span>
  </div>
  <div id="here">click a spot to see its values</div>
  <div id="panes">
    <div class="pane"><div class="head">__LEFT_TITLE__</div><div class="box"><canvas id="glL" tabindex="0"></canvas></div>
      <div class="legend">__LEFT_LEGEND__</div></div>
    <div class="pane"><div class="head">What the model saw + occlusion map</div>
      <div class="box"><canvas id="glR" tabindex="0"></canvas></div><div class="legend">__RIGHT_LEGEND__</div></div>
  </div>
  <div class="note" style="padding:4px 12px;background:#111">scroll = move through slices · click = jump there · drag in 3D = rotate · the two brains move together</div>
</div>
<script type="module">
import { Niivue } from "__NIIVUE__";
const cmaps = __CMAPS__;
const sides = { L: __LEFT__, R: __RIGHT__ };
const toUrl = (b64) => URL.createObjectURL(new Blob([Uint8Array.from(atob(b64), c => c.charCodeAt(0))]));
const nvs = {};
for (const [k, layers] of Object.entries(sides)) {
  const nv = new Niivue({ backColor: k === "L" ? [1, 1, 1, 1] : [0, 0, 0, 1], show3Dcrosshair: true, isColorbar: false,
                          isOrientCube: false, crosshairColor: [0.2, 0.9, 0.2, 1], dragAndDropEnabled: false,
                          clipPlaneColor: [0, 0, 0, 0], multiplanarShowRender: 0 });
  await nv.attachTo("gl" + k);
  for (const [name, cm] of Object.entries(cmaps)) nv.addColormap(name, cm);
  await nv.loadVolumes(layers.map(l => {
    const v = { url: toUrl(l.b64), name: l.name, colormap: l.colormap, opacity: l.start_hidden ? 0 : l.opacity };
    if (l.colormapNegative) v.colormapNegative = l.colormapNegative;
    return v;
  }));
  layers.forEach((l, i) => {  // keep the colour ranges exactly as given
    nv.volumes[i].cal_min = l.cal_min; nv.volumes[i].cal_max = l.cal_max;
    if (l.cal_minNeg !== undefined) { nv.volumes[i].cal_minNeg = l.cal_minNeg; nv.volumes[i].cal_maxNeg = l.cal_maxNeg; }
  });
  nv.updateGLVolume();
  nv.setSliceType(0);  // axial, like the readers
  nvs[k] = nv;
}
nvs.L.syncWith(nvs.R, { "2d": true, "3d": true }); nvs.R.syncWith(nvs.L, { "2d": true, "3d": true });
document.getElementById("status").textContent = "";
const both = (fn) => { fn(nvs.L); fn(nvs.R); };
const regionNames = __REGION_NAMES__;
const fmt = (v) => (Math.abs(v) >= 0.1 ? v.toFixed(2) : v.toPrecision(2));
const readout = () => {  // values at the crosshair, from both brains
  const parts = [];
  const val = (nv, name) => { const i = nv.volumes.findIndex(v => v.name === name); if (i < 0) return null;
    const v = nv.volumes[i], vox = v.mm2vox(nv.frac2mm(nv.scene.crosshairPos)); return v.getValue(...vox, v.frame4D); };
  const s = val(nvs.L, "suvr.nii.gz"); if (s !== null) parts.push(`SUVR ${s.toFixed(2)} × cerebellum`);
  const pn = val(nvs.R, "pos_vs_neg.nii.gz"); if (pn !== null) parts.push(`occlusion pos vs neg ${fmt(pn)}`);
  const c = val(nvs.R, "class.nii.gz"); if (c !== null) parts.push(`occlusion class ${fmt(c)}`);
  const z = val(nvs.L, "zones.nii.gz");
  if (z !== null) parts.push(z > 0.5 ? `FDA region: ${regionNames[Math.round(z) - 1]}` : "outside the FDA regions");
  document.getElementById("here").textContent = parts.join("  ·  ");
};
nvs.L.onLocationChange = readout; nvs.R.onLocationChange = readout;

let mode = "slices";
const checked = (t) => { const c = document.querySelector(`[data-toggle="${t}"]`); return !c || c.checked; };
const applyLayers = () => {  // a layer shows if its checkbox is ticked and it belongs to the current view
  for (const k of ["L", "R"]) sides[k].forEach((l, i) => {
    if (!l.toggle) return;
    nvs[k].setOpacity(i, checked(l.toggle) && (!l.only || l.only === mode) ? l.opacity : 0);
  });
  document.querySelectorAll(".occ").forEach(d => d.style.display = checked(d.dataset.for) ? "" : "none");
};
document.querySelectorAll("[data-view]").forEach(b => b.onclick = () => {
  document.querySelectorAll("[data-view]").forEach(x => x.classList.toggle("on", x === b));
  mode = b.dataset.view === "render" ? "render" : "slices";
  // the reader scale is white for low tau: on a white background a 3D brain would vanish, so 3D gets black
  nvs.L.opts.backColor = mode === "render" ? [0, 0, 0, 1] : [1, 1, 1, 1];
  both(nv => nv.setSliceType({ axial: 0, planes: 3, render: 4 }[b.dataset.view]));
  applyLayers();
});
let clipped = false;
document.getElementById("clip").onclick = (e) => {
  clipped = !clipped; e.target.classList.toggle("on", clipped);
  both(nv => { nv.setClipPlane(clipped ? [0, 270, 0] : [2, 0, 0]); if (clipped) nv.setRenderAzimuthElevation(45, 15); });
};
document.querySelectorAll("[data-toggle]").forEach(c => c.onchange = applyLayers);
applyLayers();
document.getElementById("full").onclick = () => document.getElementById("wrap").requestFullscreen();
// open in a given state from the address (for testing), e.g. viewer.html#render+clip, #planes+class
const want = location.hash.slice(1).split("+");
for (const v of ["planes", "render"]) if (want.includes(v)) document.querySelector(`[data-view="${v}"]`).click();
if (want.includes("clip")) document.getElementById("clip").click();
if (want.includes("class")) { document.querySelectorAll('[data-toggle]').forEach(c => {
  if (c.dataset.toggle === "class") c.checked = true; if (c.dataset.toggle === "pos_vs_neg") c.checked = false; }); applyLayers(); }
const z = want.find(w => w.startsWith("z")); if (z) { both(nv => { nv.scene.crosshairPos = nv.mm2frac([-50, -55, +z.slice(1)]); nv.drawScene(); }); readout(); }
</script></body></html>
"""

if __name__ == "__main__":
    main()
