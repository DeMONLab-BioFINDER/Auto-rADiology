#!/usr/bin/env python3
"""Combine the chunk folders of submit_vis-occlusion-all.sh into one folder.

Moves every scan's files from <base>_chunk<k>/ into <base>/ (a scan already in <base> is replaced by the newer
chunk copy), joins the chunks' summary.csv into <base>/summary.csv (one row per scan), removes the chunk
folders once they are empty, and reports how many scans <base> now holds. The chunks' group_average_* files
are dropped (each covered one chunk only); group maps by outcome are made by the analysis scripts.

Usage:
    python results_plotting/merge_occlusion_chunks.py \\
        <best_model_folder>/validation/visualization/occlusion_all_logit_median_brain_mask4_ov0.5 [--expected 456]
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("base", help="Combined folder (the chunk folders are <base>_chunk<k> next to it).")
    p.add_argument("--expected", type=int, default=None, help="Number of scans there should be (warns if fewer).")
    args = p.parse_args()
    base = Path(args.base)
    chunks = sorted(base.parent.glob(base.name + "_chunk*"))
    if not chunks and not base.exists():
        raise SystemExit(f"No {base} and no {base.name}_chunk* folders next to it")
    base.mkdir(exist_ok=True)

    summaries = [pd.read_csv(base / "summary.csv")] if (base / "summary.csv").exists() else []
    for c in chunks:
        for f in c.iterdir():
            if f.name == "summary.csv":
                summaries.append(pd.read_csv(f))
                f.unlink()
            elif f.name.startswith("group_average_"):
                f.unlink()
            elif f.is_file():
                shutil.move(str(f), base / f.name)
        if not any(c.iterdir()):
            c.rmdir()
        else:
            print(f"[keep] {c} is not empty (subfolders?), left in place")
    if summaries:
        s = pd.concat(summaries, ignore_index=True).drop_duplicates("ID", keep="last").sort_values("ID")
        s.to_csv(base / "summary.csv", index=False)

    n_maps = len(list(base.glob("*_occlusion_class.nii.gz")))
    print(f"[merge] {len(chunks)} chunk folders -> {base}: {n_maps} scans with maps"
          + (f", {len(s)} rows in summary.csv" if summaries else ""))
    if args.expected and n_maps < args.expected:
        print(f"[WARNING] {args.expected - n_maps} scans missing - resubmit the same sbatch command "
              "(finished scans are skipped), then merge again")


if __name__ == "__main__":
    main()
