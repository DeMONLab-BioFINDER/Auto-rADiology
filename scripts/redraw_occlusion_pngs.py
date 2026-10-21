#!/usr/bin/env python3
"""Redraw the occlusion pngs of a finished run from its saved maps, without recomputing anything.

New layout (src/vis.py save_mip_png): PET alone, map on PET as projections, and slices through
the map's strongest point; opacity follows strength (0 invisible); colour scale top = 99th
percentile instead of the single strongest voxel; faded colour bar.

  default    each scan on its own scale -> {ID}_occlusion_{type}_mip.png (replaces the old ones):
             for reading where the model looks within each scan
  --shared   one scale for all scans in the folder -> {ID}_occlusion_{type}_mip_shared.png:
             for comparing how much the model cares between scans
  --both     both of the above

The model input under the colours comes from {ID}_input.nii.gz (runs from now on save it), or
for older runs from the cached tau_batch_*.pt in --data_dir (same IDs as the run).

Example:
  python scripts/redraw_occlusion_pngs.py --vis_dir <model>/validation/visualization/occlusion \
      --data_dir /proj/berzelius-2024-156/users/x_nadpi/data/tau_raw --both
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `src.*` imports

import torch

from src.vis import redraw_map_pngs


def cache_loader(data_dir):
    """ID -> (D,H,W) model input from tau_batch_*.pt, indexed like src/data.py (ID // batch size)."""
    data_dir = Path(data_dir)
    manifest = data_dir / "tau_pt_manifest.json"
    batch_size = json.loads(manifest.read_text())["batch_size"] if manifest.exists() else \
        torch.load(data_dir / "tau_batch_000.pt", map_location="cpu", weights_only=True).shape[0]
    cache = {}

    def load(pid):
        b = pid // batch_size
        if b not in cache:
            cache.clear()
            cache[b] = torch.load(data_dir / f"tau_batch_{b:03d}.pt", map_location="cpu", weights_only=True)
        return cache[b][pid % batch_size].squeeze().numpy()
    return load


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--vis_dir", required=True, nargs="+", help="Folder(s) with {ID}_occlusion_*.nii.gz")
    p.add_argument("--data_dir", default=None, help="Cache folder with tau_batch_*.pt (only needed for runs without {ID}_input.nii.gz)")
    p.add_argument("--vis_name", default="occlusion")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--shared", action="store_true", help="Only the shared-scale pngs")
    g.add_argument("--both", action="store_true", help="Own-scale and shared-scale pngs")
    p.add_argument("--no_slices", action="store_true", help="Leave out the row of slices through the strongest point")
    args = p.parse_args()

    load = cache_loader(args.data_dir) if args.data_dir else None
    for d in args.vis_dir:
        print(f"===== {d} =====")
        for shared in ([False, True] if args.both else [args.shared]):
            redraw_map_pngs(d, args.vis_name, shared_scale=shared, load_input=load, peak_slices=not args.no_slices)


if __name__ == "__main__":
    main()
