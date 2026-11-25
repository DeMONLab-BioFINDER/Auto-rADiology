#!/usr/bin/env python3
"""What kind of tau does the model miss? Focality of cortical tau per scan, and whether the model's evidence sits on it.

Step 1 - focality. Per scan, from the original MNI images as SUVR on the readers' scale (divided by the whole-cerebellum
mean, like the case viewer), inside the cortex (DK cortex outline):
    peak          mean SUVR of the hottest 2 ml of cortex (how hot the hottest spot is)
    extent_ml     cortex above 1.65 x cerebellum (what a reader sees in colour; the readers' threshold), in ml - a
                  spatial-extent measure like tau-SPEX
    meta          CenTauR meta-temporal SUVR on the same images (the amount held equal)
    (also saved, not tested: mean cortical SUVR, largest blob above 1.65, number of blobs)
Group map (no statistics): per voxel, the share of each group's scans above 1.65 (where a reader would see colour).

Step 2 - does the model's evidence sit on the tau? (needs --occ_dir: run_vis.py occlusion maps of every scan,
<ID>_occlusion_pos_vs_neg.nii.gz on the MNI model grid). Per scan, inside the cortex:
    hot_share     share of the model's positive evidence that falls on the scan's own hottest 2 ml of cortex
                  (about 0.2% = no preference, 2 of about 1,060 ml; higher = the model's evidence sits on the hot spot). A share, so it does not
                  depend on how strong the model's output was.

Tests, within Read + scans (Model - vs Model +, the question of the discordant analysis): each measure alone and at the
same meta-temporal SUVR (logistic regression, likelihood-ratio test), FDR across the measures; and Mann-Whitney of each
measure for Model - / Read + vs Model + / Read + and vs Model - / Read -.
If the model ignores small focal tau: Model - / Read + scans have a high peak but little extent at the same
meta-temporal SUVR, and a low hot_share. If it misses weak, spread-out tau: a low peak.

Usage (Berzelius, CPU; see sbatch_scripts/submit_focality.sh):
    python results_plotting/focality_by_group.py <run_dir> --set test --pkl_dir <tau_T1MNI> \\
        --ref_mask <masks>/voi_WhlCbl_2mm.nii --demo_csv <demo.csv> [--occ_dir <occlusion_all_... folder>]
Outputs (<run>/figures/focality_<set>/): focality_by_group.csv, focality_tests.csv, focality_scatter.png,
    focality_maps.png (no IDs); focality_per_scan.csv (HAS IDs - keep local)
"""
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from scipy.ndimage import label

from atlas_utils import DKCortex, centaur_masks
from plot_utils import CUSTOM_PALETTE, fdr_bh
from voxelwise_stats_visual_read import load_table
from voxelwise_visual_read import suvr_images

THRESH = 1.65  # readers' grey -> colour transition (x cerebellum)
HOT_ML = 2.0  # the hot spot = the hottest 2 ml of cortex (about a 1.5 cm blob): small enough for focal tau
MIN_BLOB = 3
CONFIDENT = 0.1
GROUPS = ["Model + / Read +", "Model − / Read −", "Model − / Read + (confident)", "Model − / Read + (other)",
          "Model + / Read −"]
COLOURS = {GROUPS[0]: CUSTOM_PALETTE[3], GROUPS[1]: CUSTOM_PALETTE[0], GROUPS[2]: CUSTOM_PALETTE[1],
           GROUPS[3]: "#E9A9C2", GROUPS[4]: CUSTOM_PALETTE[4]}
MEASURES = ["peak", "extent_ml", "hot_share"]  # kept simple on purpose: three numbers per scan
LABELS = {"peak": "Peak (hottest 2 ml of cortex)", "extent_ml": "Cortex above 1.65 (ml)",
          "largest_ml": "Largest blob above 1.65 (ml)", "n_blobs": "Number of blobs above 1.65",
          "mean": "Mean cortical SUVR", "hot_share": "Model evidence on the hot spot (share)",
          "rho": "Model evidence vs tau (Spearman)"}
CUTS = [-24, -12, 0, 12, 24, 36, 48]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--set", default="test", choices=["test", "avid"])
    p.add_argument("--pkl_dir", required=True)
    p.add_argument("--ref_mask", required=True, help="Whole-cerebellum mask (voi_WhlCbl_2mm.nii): the readers' scale.")
    p.add_argument("--centaur_dir", default=None, help="CenTauR masks (default: the --ref_mask folder).")
    p.add_argument("--atlas_dir", default=None)
    p.add_argument("--occ_dir", default=None, help="Occlusion maps of every scan (step 2).")
    p.add_argument("--dataset", default="Gothenburg")
    p.add_argument("--ext_dataset", default="AVID_unseen")
    p.add_argument("--demo_csv", default=None)
    p.add_argument("--ext_demo_csv", default=None)
    p.add_argument("--female_code", type=float, default=2)
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


def group_of(read, call, prob) -> str:
    if read == 1 and call == 1:
        return GROUPS[0]
    if read == 0 and call == 0:
        return GROUPS[1]
    if read == 1:
        return GROUPS[2] if prob <= CONFIDENT else GROUPS[3]
    return GROUPS[4]


def per_scan(x: np.ndarray, cortex: np.ndarray, meta: np.ndarray, vox_ml: float, occ=None):
    """Focality measures (and evidence measures if occ is given) of one SUVR volume; also its hot / above masks."""
    cov = cortex & (x != 0)
    v = x[cov]
    if v.size < 0.5 * cortex.sum():
        return None, None, None
    n_hot = max(1, int(round(HOT_ML / vox_ml)))
    hot_thr = np.sort(v)[-n_hot]
    hot = cov & (x >= hot_thr)
    above = cov & (x >= THRESH)
    lab, n = label(above, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(lab.ravel())[1:] if n else np.array([])
    m = {"mean": float(v.mean()), "peak": float(x[hot].mean()), "extent_ml": float(above.sum() * vox_ml),
         "largest_ml": float(sizes.max() * vox_ml) if sizes.size else 0.0,
         "n_blobs": int((sizes >= MIN_BLOB).sum()),
         "meta": float(x[meta & (x != 0)].mean()) if (meta & (x != 0)).any() else np.nan}
    if occ is not None:
        e = np.clip(occ[cov], 0, None)  # positive evidence only
        m["hot_share"] = float(np.clip(occ[hot], 0, None).sum() / e.sum()) if e.sum() > 0 else np.nan
        m["rho"] = float(stats.spearmanr(v, occ[cov]).correlation)
    return m, hot, above


def logit_tests(d: pd.DataFrame, measures: list) -> pd.DataFrame:
    """Within Read + scans: Model - ~ measure, and ~ measure + meta-temporal SUVR (LR test); FDR per column."""
    rp = d[d["read"] == 1].copy()
    rp["wrong"] = (rp["call"] == 0).astype(int)
    rows = []
    for m in measures:
        dd = rp.dropna(subset=[m, "meta"]).copy()
        if dd[m].std() == 0 or dd["wrong"].sum() < 3:
            continue
        dd["z"] = (dd[m] - dd[m].mean()) / dd[m].std()
        row = {"measure": LABELS[m], "n": len(dd), "n_model_neg": int(dd["wrong"].sum())}
        for prefix, adj in (("plain_", []), ("", ["meta"])):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    full = smf.logit("wrong ~ " + " + ".join(["z", *adj]), data=dd).fit(disp=0)
                    red = smf.logit("wrong ~ " + (" + ".join(adj) or "1"), data=dd).fit(disp=0)
                lo, hi = full.conf_int().loc["z"]
                row.update({f"{prefix}or_per_sd": np.exp(full.params["z"]), f"{prefix}or_lo": np.exp(lo),
                            f"{prefix}or_hi": np.exp(hi), f"{prefix}p": stats.chi2.sf(2 * (full.llf - red.llf), 1)})
            except Exception as err:
                row[f"{prefix}note"] = f"could not fit ({err})"
        rows.append(row)
    t = pd.DataFrame(rows)
    for prefix in ("plain_", ""):
        if f"{prefix}p" in t:
            ok = t[f"{prefix}p"].notna()
            t.loc[ok, f"{prefix}p_fdr"] = fdr_bh(t.loc[ok, f"{prefix}p"])
    return t


def group_tests(d: pd.DataFrame, measures: list) -> pd.DataFrame:
    neg_pos = d["group"].isin(GROUPS[2:4])
    comps = [("Model − / Read + vs Model + / Read +", neg_pos, d["group"] == GROUPS[0]),
             ("Model − / Read + vs Model − / Read −", neg_pos, d["group"] == GROUPS[1]),
             ("Confident Model − / Read + vs Model − / Read −", d["group"] == GROUPS[2], d["group"] == GROUPS[1])]
    rows = []
    for name, a, b in comps:
        part = []
        for m in measures:
            x, y = d.loc[a, m].dropna(), d.loc[b, m].dropna()
            p = stats.mannwhitneyu(x, y).pvalue if len(x) >= 3 and len(y) >= 3 else np.nan
            part.append({"comparison": name, "measure": LABELS[m], "n_a": len(x), "n_b": len(y),
                         "median_a": x.median(), "median_b": y.median(), "p": p})
        part = pd.DataFrame(part)
        ok = part["p"].notna()
        part.loc[ok, "p_fdr"] = fdr_bh(part.loc[ok, "p"])
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def scatter(d: pd.DataFrame, measures: list, set_label: str, out: Path):
    """Each measure against meta-temporal SUVR (the amount), one dot per scan."""
    fig, axes = plt.subplots(1, len(measures), figsize=(4.4 * len(measures), 4.6), squeeze=False)
    for ax, m in zip(axes[0], measures):
        for g in GROUPS:
            s = d[d["group"] == g]
            big = g in GROUPS[2:]
            ax.scatter(s["meta"], s[m], s=44 if big else 13, alpha=0.9 if big else 0.35, color=COLOURS[g],
                       edgecolor="white" if big else "none", linewidth=0.5, zorder=3 if big else 2,
                       label=f"{g} (n = {len(s)})")
        ax.set_xlabel("Meta-temporal SUVR (× whole cerebellum)", fontsize=11)
        ax.set_title(LABELS[m], fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Focality of cortical tau, {set_label}", fontsize=15)
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def maps_figure(freq: dict, affine, set_label: str, out: Path):
    """Rows = groups: per voxel, the share of the group's scans above 1.65 x cerebellum."""
    from nilearn import datasets, plotting
    bg = datasets.load_mni152_template(resolution=1)
    groups = [g for g in GROUPS[:4] if g in freq]
    fig, axes = plt.subplots(len(groups), 1, figsize=(13, 2.3 * len(groups)), facecolor="white", squeeze=False)
    for i, g in enumerate(groups):
        for j, (key, title, vmax) in enumerate((("above", "Above 1.65 × cerebellum", 1.0),)):
            img = nib.Nifti1Image(freq[g][key].astype(np.float32), affine)
            plotting.plot_stat_map(img, bg_img=bg, display_mode="z", cut_coords=CUTS, axes=axes[i][j], cmap="hot_r",
                                   vmax=vmax, threshold=0.02, colorbar=(i == 0), black_bg=False, draw_cross=False,
                                   annotate=False, dim=-0.3, figure=fig)
            if i == 0:
                axes[i][j].set_title(f"{title}: share of each group's scans (where a reader sees colour)", fontsize=13)
        axes[i][0].text(-0.01, 0.5, f"{g}\n(n = {freq[g]['n']})", transform=axes[i][0].transAxes, ha="right",
                        va="center", fontsize=12)
    fig.suptitle(f"Where the tau is, per group, {set_label}", fontsize=15, x=0.08, ha="left")
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    run = Path(args.run_dir)
    out = Path(args.out_dir) if args.out_dir else run / "figures" / f"focality_{args.set}"
    out.mkdir(parents=True, exist_ok=True)
    table = load_table(run, args).set_index("ID")
    table["prob"] = 1 / (1 + np.exp(-table["logodds"]))
    occ_dir = Path(args.occ_dir) if args.occ_dir else None

    cortex = meta = None
    rows, skipped, refs, freq, no_occ = [], [], {}, {}, 0
    for scan_id, x, aff in suvr_images(Path(args.pkl_dir), table.index, Path(args.ref_mask), skipped, refs):
        img = nib.as_closest_canonical(nib.Nifti1Image(x.astype(np.float32), aff))  # RAS, as the occlusion maps
        x, aff = np.asarray(img.dataobj, dtype=np.float32), img.affine
        if cortex is None:
            grid_aff = aff
            cortex = DKCortex(x.shape, aff, args.atlas_dir).mask
            meta = centaur_masks(x.shape, aff, Path(args.centaur_dir) if args.centaur_dir
                                 else Path(args.ref_mask).parent)["Meta-temporal"]
            vox_ml = float(abs(np.linalg.det(aff[:3, :3]))) / 1000
            print(f"[grid] {x.shape}, cortex {int(cortex.sum())} voxels ({cortex.sum() * vox_ml:.0f} ml)")
        occ = None
        if occ_dir is not None:
            f = occ_dir / f"{scan_id}_occlusion_pos_vs_neg.nii.gz"
            if f.exists():
                occ = np.asarray(nib.load(f).dataobj, dtype=np.float32)
                if occ.shape != x.shape:
                    raise SystemExit(f"{f.name} is {occ.shape}, the image {x.shape}: occlusion maps must be from the MNI model")
            else:
                no_occ += 1
        r = table.loc[scan_id]
        m, hot, above = per_scan(x, cortex, meta, vox_ml, occ)
        if m is None:
            skipped.append(scan_id)
            continue
        g = group_of(r["read"], r["call"], r["prob"])
        rows.append({"ID": scan_id, "read": int(r["read"]), "call": int(r["call"]), "prob": float(r["prob"]),
                     "group": g, **m})
        f_ = freq.setdefault(g, {"hot": np.zeros(x.shape), "above": np.zeros(x.shape), "n": 0})
        f_["hot"] += hot; f_["above"] += above; f_["n"] += 1
    if skipped:
        print(f"[data] {len(skipped)} scans left out (reference region or cortex not covered)")
    if occ_dir is not None:
        print(f"[occlusion] maps found for {len(rows) - no_occ} of {len(rows)} scans")
    d = pd.DataFrame(rows)
    for g in freq:
        freq[g]["hot"] /= freq[g]["n"]; freq[g]["above"] /= freq[g]["n"]
    measures = [m for m in MEASURES if m in d and d[m].notna().sum() > 0]

    d.round(4).to_csv(out / "focality_per_scan.csv", index=False)
    summ = d.groupby("group")[["meta", *measures, "mean", "largest_ml", "n_blobs"]].median().reindex(GROUPS).round(3)
    summ.insert(0, "n", d["group"].value_counts().reindex(GROUPS).fillna(0).astype(int))
    summ.to_csv(out / "focality_by_group.csv")
    lt, gt = logit_tests(d, measures), group_tests(d, measures)
    pd.concat([lt.assign(test="Model − within Read + (logistic, plain and at the same meta-temporal SUVR)"),
               gt.assign(test="Mann-Whitney")], ignore_index=True).round(4).to_csv(out / "focality_tests.csv", index=False)
    set_label = "held-out test" if args.set == "test" else args.ext_dataset.removesuffix("_unseen")
    scatter(d, measures, set_label,
            out / "focality_scatter.png")
    maps_figure(freq, grid_aff, set_label, out / "focality_maps.png")
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print("\nMedian per group:\n" + summ.to_string())
        print("\nWithin Read + scans, odds of Model − per SD (plain / same meta-temporal SUVR):\n" + lt.round(3).to_string(index=False))
        print("\nGroup comparisons (Mann-Whitney, FDR across measures):\n" + gt.round(3).to_string(index=False))
    print(f"\nSaved to {out} (focality_per_scan.csv has IDs - keep it local)")


if __name__ == "__main__":
    main()
