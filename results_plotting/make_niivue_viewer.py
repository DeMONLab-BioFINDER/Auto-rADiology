#!/usr/bin/env python3
"""One self-contained .html viewer (NiiVue) for voxelwise maps: slices + 3D render, full screen, cut-away.

Open the .html by double-clicking it: NiiVue (a browser-based brain viewer) is loaded from the web, and the images
are built into the page, so nothing is installed and nothing is uploaded. Needs internet once to load NiiVue.

Layers (all in MNI space; NiiVue lines them up by their own affines):
  background   the MNI152 template nilearn uses for the PNG figures (default) or --background
  difference   Read + minus Read - map (voxelwise_visual_read.py: difference_read_pos_minus_neg.nii.gz), blue-white-red
               like the PNGs, symmetric -vmax..+vmax. vmax: --vmax, else colour_scales.json next to the file (so it
               matches that run's PNGs), else the 99th percentile of |difference| (the same rule the script uses)
  regions      FDA visual-read regions (plot_fda_regions.py: fda_regions_MNI152NLin2009cAsym.nii.gz), optional,
               a layer that can be switched on and off
Any layer can be left out. Differences smaller than --hide_below x vmax (default 0.1) are see-through, so the brain
shows underneath; set --hide_below 0 to colour every voxel.

Usage:
    python results_plotting/make_niivue_viewer.py \\
        --difference <run folder>/difference_read_pos_minus_neg.nii.gz \\
        --regions <proj_path>/results/fda_regions/fda_regions_MNI152NLin2009cAsym.nii.gz \\
        --out <run folder>/niivue_difference.html
"""
import argparse
import base64
import json
from pathlib import Path

import nibabel as nib
import numpy as np

NIIVUE = "https://cdn.jsdelivr.net/npm/@niivue/niivue@0.69.0/+esm"  # pinned version
# blue-white-red as in matplotlib's RdBu_r; alpha rises from 0 at zero so small values stay see-through
RDBU_POS = {"R": [247, 253, 244, 214, 103], "G": [247, 219, 165, 96, 0], "B": [247, 199, 130, 77, 31],
            "A": [0, 25, 130, 215, 255], "I": [0, 64, 128, 192, 255]}
RDBU_NEG = {"R": [247, 209, 146, 67, 5], "G": [247, 229, 197, 147, 48], "B": [247, 240, 222, 195, 97],
            "A": [0, 25, 130, 215, 255], "I": [0, 64, 128, 192, 255]}
# FDA region colours, the same as plot_fda_regions.py (values 1-6)
REGION_COLOURS = ["#D62728", "#FF7F0E", "#E7BA52", "#6B9BD1", "#8C6BB1", "#41AB5D"]
REGION_NAMES = ["Posterolateral temporal (PLT)", "Occipital", "Parietal / precuneus", "Frontal",
                "Anterolateral temporal (ALT)", "Mesial temporal (AMT + PMT)"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--background", default=None, help="Background NIfTI (default: nilearn's MNI152 template, 1 mm).")
    p.add_argument("--difference", default=None, help="Read + minus Read - NIfTI.")
    p.add_argument("--vmax", type=float, default=None, help="Colour range of the difference: -vmax..+vmax.")
    p.add_argument("--hide_below", type=float, default=0.1,
                   help="Make |difference| below this share of vmax see-through (default 0.1; 0 = show all).")
    p.add_argument("--regions", default=None, help="FDA region map NIfTI (values 1-6).")
    p.add_argument("--title", default="Mean Read + − mean Read −")
    p.add_argument("--out", required=True)
    return p.parse_args()


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def difference_vmax(path: Path, vmax):
    if vmax is not None:
        return float(vmax), "--vmax"
    scales = path.parent / "colour_scales.json"
    if scales.exists():
        return float(json.loads(scales.read_text())["difference_vmax"]), scales.name
    d = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    return float(np.percentile(np.abs(d[d != 0]), 99)), "99th percentile of |difference|"


def region_colormap() -> dict:
    # value k (1-6) sits at index round(k / 6 * 255); flat steps around each so labels keep one colour
    cm = {"R": [0], "G": [0], "B": [0], "A": [0], "I": [0]}
    for k, hexcol in enumerate(REGION_COLOURS, 1):
        r, g, b = (int(hexcol[i:i + 2], 16) for i in (1, 3, 5))
        centre = round(k / 6 * 255)
        for idx in (max(centre - 21, 1), min(centre + 21, 255)) if k < 6 else (centre - 21, 255):
            cm["R"].append(r); cm["G"].append(g); cm["B"].append(b); cm["A"].append(255); cm["I"].append(idx)
    return cm


def main():
    args = parse_args()
    background = Path(args.background) if args.background else None
    if background is None:
        from nilearn import datasets
        tmp = Path(args.out).with_name("_mni152_template_1mm.nii.gz")
        nib.save(datasets.load_mni152_template(resolution=1), tmp)
        background = tmp
    layers = [{"key": "background", "label": "Brain (MNI152 template)", "b64": b64(background),
               "name": "background.nii.gz", "colormap": "gray", "opacity": 1.0}]
    legend = ""
    if args.difference:
        vmax, source = difference_vmax(Path(args.difference), args.vmax)
        print(f"[difference] colour range -{vmax:.3f} to +{vmax:.3f} ({source})")
        layers.append({"key": "difference", "label": args.title, "b64": b64(Path(args.difference)),
                       "name": "difference.nii.gz", "colormap": "rdbu_pos", "colormapNegative": "rdbu_neg",
                       "cal_min": args.hide_below * vmax, "cal_max": vmax,
                       "cal_minNeg": -vmax, "cal_maxNeg": -args.hide_below * vmax, "opacity": 1.0})
        legend += (f"<div class='bar'><span>−{vmax:.2f}</span><div class='grad'></div><span>+{vmax:.2f}</span></div>"
                   "<div class='note'>blue = Read + lower · red = Read + higher"
                   + (f" · |difference| below {args.hide_below * vmax:.2f} not coloured" if args.hide_below else "")
                   + "</div>")
    if args.regions:
        layers.append({"key": "regions", "label": "FDA visual-read regions", "b64": b64(Path(args.regions)),
                       "name": "regions.nii.gz", "colormap": "fda_regions", "cal_min": 0.0, "cal_max": 6.0,
                       "opacity": 0.45, "start_hidden": bool(args.difference)})
        legend += "<div class='regions'>" + "".join(
            f"<span><i style='background:{c}'></i>{n}</span>" for c, n in zip(REGION_COLOURS, REGION_NAMES)) + "</div>"
    if background.name == "_mni152_template_1mm.nii.gz":
        background.unlink()

    toggles = "".join(
        f"<label><input type='checkbox' data-layer='{i}' {'' if l.get('start_hidden') else 'checked'}> {l['label']}</label>"
        for i, l in enumerate(layers) if i > 0)
    page = TEMPLATE.replace("__NIIVUE__", NIIVUE).replace("__TITLE__", args.title).replace("__TOGGLES__", toggles)
    page = page.replace("__LEGEND__", legend).replace("__LAYERS__", json.dumps(layers))
    page = page.replace("__RDBU_POS__", json.dumps(RDBU_POS)).replace("__RDBU_NEG__", json.dumps(RDBU_NEG))
    page = page.replace("__REGIONS_CMAP__", json.dumps(region_colormap()))
    Path(args.out).write_text(page, encoding="utf-8")
    print(f"[save] {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB) — double-click to open in a browser")


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  html, body { margin: 0; height: 100%; background: #000; color: #ddd; font: 14px system-ui, sans-serif; }
  #wrap { display: flex; flex-direction: column; height: 100%; }
  #bar { display: flex; flex-wrap: wrap; gap: 14px; align-items: center; padding: 8px 12px; background: #111; }
  #bar button { background: #222; color: #ddd; border: 1px solid #444; border-radius: 4px; padding: 4px 10px; cursor: pointer; }
  #bar button.on { border-color: #bbb; color: #fff; }
  #bar label { white-space: nowrap; }
  #canvasbox { flex: 1; min-height: 0; }
  canvas { width: 100%; height: 100%; display: block; }
  #legend { padding: 6px 12px 10px; background: #111; display: flex; flex-wrap: wrap; gap: 18px; align-items: center; }
  .bar { display: flex; align-items: center; gap: 6px; }
  .grad { width: 220px; height: 12px; background: linear-gradient(to right, #053061, #4393c3, #f7f7f7, #d6604d, #67001f); }
  .note { color: #999; }
  .regions span { margin-right: 12px; white-space: nowrap; }
  .regions i { display: inline-block; width: 11px; height: 11px; margin-right: 4px; vertical-align: -1px; }
  #status { color: #999; }
</style></head>
<body><div id="wrap">
  <div id="bar">
    <b>__TITLE__</b>
    <button data-view="both" class="on">Slices + 3D</button>
    <button data-view="slices">Slices</button>
    <button data-view="render">3D only</button>
    <button id="clip">Cut-away (3D)</button>
    __TOGGLES__
    <button id="full">Full screen</button>
    <span id="status">loading…</span>
  </div>
  <div id="canvasbox"><canvas id="gl"></canvas></div>
  <div id="legend">__LEGEND__<span class="note">click to move · drag in 3D to rotate · scroll to zoom</span></div>
</div>
<script type="module">
import { Niivue } from "__NIIVUE__";
const layers = __LAYERS__;
const nv = new Niivue({ backColor: [0, 0, 0, 1], show3Dcrosshair: true, isColorbar: false, isOrientCube: true,
                        multiplanarShowRender: 1, dragAndDropEnabled: true,
                        clipPlaneColor: [0, 0, 0, 0] });
await nv.attachTo("gl");
nv.addColormap("rdbu_pos", __RDBU_POS__);
nv.addColormap("rdbu_neg", __RDBU_NEG__);
nv.addColormap("fda_regions", __REGIONS_CMAP__);
const toUrl = (b64) => URL.createObjectURL(new Blob([Uint8Array.from(atob(b64), c => c.charCodeAt(0))]));
await nv.loadVolumes(layers.map(l => {
  const v = { url: toUrl(l.b64), name: l.name, colormap: l.colormap, opacity: l.start_hidden ? 0 : l.opacity };
  if (l.colormapNegative) v.colormapNegative = l.colormapNegative;
  if (l.cal_min !== undefined) { v.cal_min = l.cal_min; v.cal_max = l.cal_max; }
  return v;
}));
// keep the colour range exactly as given (NiiVue would otherwise re-estimate it from the data)
layers.forEach((l, i) => {
  if (l.cal_min !== undefined) { nv.volumes[i].cal_min = l.cal_min; nv.volumes[i].cal_max = l.cal_max; }
  if (l.cal_minNeg !== undefined) { nv.volumes[i].cal_minNeg = l.cal_minNeg; nv.volumes[i].cal_maxNeg = l.cal_maxNeg; }
});
nv.updateGLVolume();
nv.setSliceType(3);  // multiplanar: three slices + 3D
document.getElementById("status").textContent = "";

document.querySelectorAll("[data-view]").forEach(b => b.onclick = () => {
  document.querySelectorAll("[data-view]").forEach(x => x.classList.toggle("on", x === b));
  if (b.dataset.view === "render") nv.setSliceType(4);
  else { nv.opts.multiplanarShowRender = b.dataset.view === "both" ? 1 : 0; nv.setSliceType(3); }
});
let clipped = false;
document.getElementById("clip").onclick = (e) => {
  clipped = !clipped; e.target.classList.toggle("on", clipped);
  nv.setClipPlane(clipped ? [0, 270, 0] : [2, 0, 0]);  // cut through the middle; drag to rotate
  if (clipped) nv.setRenderAzimuthElevation(45, 15);  // turn so the cut face is in view
};
document.querySelectorAll("[data-layer]").forEach(c => c.onchange = () => {
  const i = +c.dataset.layer; nv.setOpacity(i, c.checked ? layers[i].opacity : 0);
});
document.getElementById("full").onclick = () => document.getElementById("wrap").requestFullscreen();
// open in a given state from the address, e.g. viewer.html#render+clip, #slices+regions, #render+az200
const want = location.hash.slice(1).split("+");
if (want.includes("render")) document.querySelector('[data-view="render"]').click();
if (want.includes("slices")) document.querySelector('[data-view="slices"]').click();
if (want.includes("clip")) document.getElementById("clip").click();
const az = want.find(w => w.startsWith("az")); if (az) nv.setRenderAzimuthElevation(+az.slice(2), 15);
if (want.includes("regions")) { const c = document.querySelector('[data-layer="2"]'); if (c && !c.checked) c.click(); }
</script></body></html>
"""

if __name__ == "__main__":
    main()
