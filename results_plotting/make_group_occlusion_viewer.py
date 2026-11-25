#!/usr/bin/env python3
"""Group-average occlusion maps in a NiiVue viewer: slices + 3D, one group at a time, FDA regions optional.

Uses the group_mean_<group>.nii.gz files occlusion_group_maps.py writes (average 'pos vs neg' occlusion map of each
model / read group; red = hiding that spot lowers the log-odds of positive, i.e. it supports positive; blue = supports
negative). Same colour range as the group montage (default -0.06 to +0.06), the same for every group, so colours compare.
Background: nilearn's MNI152 template (1 mm), as in the montage.

Usage (local is fine, no images of scans needed):
    python results_plotting/make_group_occlusion_viewer.py <occlusion folder with group_mean_*.nii.gz> \\
        --label "held-out test" [--regions fda_regions_MNI152NLin2009cAsym.nii.gz] --out viewer.html
Output: one self-contained .html (double-click to open; NiiVue loads from the web).
"""
import argparse
import json
from pathlib import Path

import nibabel as nib

from make_niivue_viewer import (NIIVUE, RDBU_NEG, RDBU_POS, REGION_COLOURS, REGION_NAMES, TEMPLATE, b64,
                                region_colormap)

GROUPS = [("TP", "Model + / Read +"), ("TN", "Model − / Read −"), ("FN", "Model − / Read +"), ("FP", "Model + / Read −")]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", help="Folder with group_mean_TP/TN/FN(/FP).nii.gz")
    p.add_argument("--label", default="", help="Data set name for the title, e.g. 'held-out test'.")
    p.add_argument("--vmax", type=float, default=0.06, help="Colour range -vmax..+vmax (default: as the montage).")
    p.add_argument("--hide_below", type=float, default=0.1, help="Leave |value| below this share of vmax uncoloured.")
    p.add_argument("--regions", default=None, help="FDA region map (values 1-6), shown on request.")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    folder = Path(args.folder)
    from nilearn import datasets
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(args.out).with_name("_mni152_template_1mm.nii.gz")
    nib.save(datasets.load_mni152_template(resolution=1), tmp)
    layers = [{"key": "background", "label": "Brain", "b64": b64(tmp), "name": "background.nii.gz",
               "colormap": "gray", "opacity": 1.0}]
    tmp.unlink()
    lo, hi = args.hide_below * args.vmax, args.vmax
    first = True
    for code, name in GROUPS:
        f = folder / f"group_mean_{code}.nii.gz"
        if not f.exists():
            continue
        layers.append({"key": code, "label": name, "b64": b64(f), "name": f"{code}.nii.gz", "colormap": "rdbu_pos",
                       "colormapNegative": "rdbu_neg", "cal_min": lo, "cal_max": hi, "cal_minNeg": -hi, "cal_maxNeg": -lo,
                       "opacity": 1.0, "start_hidden": not first, "group": True})
        first = False
    if len(layers) == 1:
        raise SystemExit(f"No group_mean_*.nii.gz in {folder}")
    if args.regions:
        layers.append({"key": "regions", "label": "FDA visual-read regions", "b64": b64(Path(args.regions)),
                       "name": "regions.nii.gz", "colormap": "fda_regions", "cal_min": 0.0, "cal_max": 6.0,
                       "opacity": 0.45, "start_hidden": True})

    toggles = "<span>Group:</span>" + "".join(
        f"<label><input type='radio' name='grp' data-layer='{i}' {'' if l.get('start_hidden') else 'checked'}> "
        f"{l['label']}</label>" for i, l in enumerate(layers) if l.get("group"))
    toggles += "".join(f"<label><input type='checkbox' data-layer='{i}'> {l['label']}</label>"
                       for i, l in enumerate(layers) if l["key"] == "regions")
    legend = (f"<div class='bar'><span>−{args.vmax:g}</span><div class='grad'></div><span>+{args.vmax:g}</span></div>"
              "<div class='note'>change in log-odds of positive when the spot is hidden: red = supports positive, "
              "blue = supports negative; group average</div>")
    if args.regions:
        legend += "<div class='regions'>" + "".join(
            f"<span><i style='background:{c}'></i>{n}</span>" for c, n in zip(REGION_COLOURS, REGION_NAMES)) + "</div>"
    title = "Group-average occlusion" + (f", {args.label}" if args.label else "")
    radios_js = """
// one group at a time: picking a group hides the others
const radios = [...document.querySelectorAll("input[name=grp]")];
radios.forEach(r => r.addEventListener("change", () =>
  radios.forEach(q => nv.setOpacity(+q.dataset.layer, q.checked ? layers[+q.dataset.layer].opacity : 0))));
</script>"""
    page = (TEMPLATE.replace("__NIIVUE__", NIIVUE).replace("__TITLE__", title).replace("__TOGGLES__", toggles)
            .replace("__LEGEND__", legend).replace("__LAYERS__", json.dumps(layers))
            .replace("__RDBU_POS__", json.dumps(RDBU_POS)).replace("__RDBU_NEG__", json.dumps(RDBU_NEG))
            .replace("__REGIONS_CMAP__", json.dumps(region_colormap())))
    assert page.count("</script>") == 1
    page = page.replace("</script>", radios_js)
    Path(args.out).write_text(page, encoding="utf-8")
    print(f"[save] {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB) - double-click to open")


if __name__ == "__main__":
    main()
