#!/usr/bin/env python3
"""Per scan: signal and model evidence inside the clusters of one voxelwise map.

Made for the precuneus clusters of 'Model - scans: confident Read + vs Read -' (held-out test): do most of the 12
confident Model - / Read + scans carry that signal, or do one or two drive it? And how much of the model's evidence
falls there?
  - On the set the clusters came from, this is DESCRIPTIVE only (the group difference is there by construction); the
    question is how many scans carry it.
  - On another set (AVID) the same clusters are an independent check.

Clusters: voxels with TFCE FWE p < .05 and t > 0 in <analysis>_logp_tfce.nii.gz / _t.nii.gz (voxelwise_stats_visual_read.py).
Per scan (SUVR on the readers' scale, whole cerebellum = 1, like the focality analysis):
    cluster_suvr    mean SUVR inside the clusters
    cortex_suvr     mean cortical SUVR (DK cortex outline), for comparison
    above_neg90     cluster_suvr above the 90th percentile of the set's Model - / Read - scans (True/False)
    occ_share       share of the model's positive occlusion evidence (cortex + clusters) that falls in the clusters;
                    chance = the clusters' share of those voxels (printed)
Outputs (--out_dir): cluster_by_group.csv (no IDs), cluster_per_scan.csv (HAS IDs - keep local), cluster_by_scan.png

Usage (Berzelius, CPU; see sbatch_scripts/submit_cluster-by-scan.sh):
    python results_plotting/cluster_by_scan.py <run_dir> --set test --clusters <folder>/confident_neg_pos_vs_neg \\
        --pkl_dir <tau_T1MNI> --ref_mask <masks>/voi_WhlCbl_2mm.nii --demo_csv <demo.csv> [--occ_dir <occlusion_all_...>]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

from atlas_utils import DKCortex
from focality_by_group import COLOURS, GROUPS, group_of, on_grid
from voxelwise_stats_visual_read import load_table
from voxelwise_visual_read import suvr_images


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--set", default="test", choices=["test", "avid"])
    p.add_argument("--clusters", required=True, help="Path prefix of the voxelwise map, e.g. <folder>/confident_neg_pos_vs_neg")
    p.add_argument("--pkl_dir", required=True)
    p.add_argument("--ref_mask", required=True)
    p.add_argument("--atlas_dir", default=None)
    p.add_argument("--occ_dir", default=None)
    p.add_argument("--dataset", default="Gothenburg")
    p.add_argument("--ext_dataset", default="AVID_unseen")
    p.add_argument("--demo_csv", default=None)
    p.add_argument("--ext_demo_csv", default=None)
    p.add_argument("--female_code", type=float, default=2)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table = load_table(Path(args.run_dir), args).set_index("ID")
    table["prob"] = 1 / (1 + np.exp(-table["logodds"]))
    logp, tmap = nib.load(f"{args.clusters}_logp_tfce.nii.gz"), nib.load(f"{args.clusters}_t.nii.gz")
    occ_dir = Path(args.occ_dir) if args.occ_dir else None

    clus = cortex = None
    rows, skipped, refs = [], [], {}
    for scan_id, x, aff in suvr_images(Path(args.pkl_dir), table.index, Path(args.ref_mask), skipped, refs):
        img = nib.as_closest_canonical(nib.Nifti1Image(x.astype(np.float32), aff))  # RAS, like the occlusion maps
        x, aff = np.asarray(img.dataobj, dtype=np.float32), img.affine
        if clus is None:
            clus = (on_grid(logp, x.shape, aff) > -np.log10(0.05)) & (on_grid(tmap, x.shape, aff) > 0)
            cortex = DKCortex(x.shape, aff, args.atlas_dir).mask
            vox_ml = float(abs(np.linalg.det(aff[:3, :3]))) / 1000
            area = cortex | clus
            chance = clus.sum() / area.sum()
            print(f"[clusters] {int(clus.sum())} voxels ({clus.sum() * vox_ml:.1f} ml); chance share of evidence "
                  f"{chance:.2%}")
            if clus.sum() == 0:
                raise SystemExit("No significant voxels in that map")
        ok = clus & (x != 0)
        if ok.sum() < 0.5 * clus.sum():
            skipped.append(scan_id)
            continue
        r = table.loc[scan_id]
        row = {"ID": scan_id, "group": group_of(r["read"], r["call"], r["prob"]), "prob": float(r["prob"]),
               "cluster_suvr": float(x[ok].mean()), "cortex_suvr": float(x[cortex & (x != 0)].mean())}
        if occ_dir is not None and (occ_dir / f"{scan_id}_occlusion_pos_vs_neg.nii.gz").exists():
            occ = np.clip(np.asarray(nib.load(occ_dir / f"{scan_id}_occlusion_pos_vs_neg.nii.gz").dataobj), 0, None)
            tot = occ[area].sum()
            row["occ_share"] = float(occ[clus].sum() / tot) if tot > 0 else np.nan
        rows.append(row)
    if skipped:
        print(f"[data] {len(skipped)} scans left out (reference or clusters not covered)")
    d = pd.DataFrame(rows)
    neg90 = d.loc[d["group"] == GROUPS[1], "cluster_suvr"].quantile(0.9)
    d["above_neg90"] = d["cluster_suvr"] > neg90
    d.round(4).to_csv(out / "cluster_per_scan.csv", index=False)

    cols = ["cluster_suvr", "cortex_suvr"] + (["occ_share"] if "occ_share" in d else [])
    g = d.groupby("group")[cols].median().reindex(GROUPS)
    g.insert(0, "n", d["group"].value_counts().reindex(GROUPS).fillna(0).astype(int))
    g["n_above_neg90"] = d.groupby("group")["above_neg90"].sum().reindex(GROUPS).fillna(0).astype(int)
    g.round(4).to_csv(out / "cluster_by_group.csv")

    set_label = "held-out test" if args.set == "test" else args.ext_dataset.removesuffix("_unseen")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    groups = [k for k in GROUPS[:4] if (d["group"] == k).any()]
    rng = np.random.default_rng(0)
    for i, k in enumerate(groups):
        v = d.loc[d["group"] == k, "cluster_suvr"]
        big = k in GROUPS[2:]
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=40 if big else 10, alpha=0.9 if big else 0.4,
                   color=COLOURS[k], edgecolor="white" if big else "none", linewidth=0.5)
        ax.hlines(v.median(), i - 0.28, i + 0.28, color="black", lw=2)
    ax.axhline(neg90, color="grey", ls="--", lw=1)
    ax.text(len(groups) - 0.5, neg90, "90th percentile of\nModel − / Read −", fontsize=9, color="grey", va="bottom", ha="right")
    ax.set_xticks(range(len(groups)), [k.replace(" (", "\n(") for k in groups], fontsize=10)
    ax.set_ylabel("SUVR in the precuneus clusters (× whole cerebellum)", fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"Signal in the voxelwise clusters, {set_label}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "cluster_by_scan.png", dpi=200, facecolor="white")
    plt.close(fig)

    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(f"\nCluster SUVR, 90th percentile of Model − / Read −: {neg90:.3f}\n" + g.round(3).to_string())
    print(f"\nSaved to {out} (cluster_per_scan.csv has IDs - keep it local)")


if __name__ == "__main__":
    main()
