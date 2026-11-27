#!/usr/bin/env python3
"""Follow-up on the voxelwise result (Model - / Read + vs Model - / Read -, held-out test): which misses carry the
extra tau, who they are, and did the model react to it?

The "area" = the significant voxels of the voxelwise map (model_neg_read_pos_vs_neg, TFCE FWE p < .05, misses higher;
voxelwise_stats_visual_read.py). Per scan, two numbers inside the area:
  area SUVR      mean SUVR (same images, reference and 6 mm smoothing as the voxelwise test)
  area evidence  mean of the scan's occlusion map ("pos vs neg", log-odds) over the area: how much hiding a patch
                 there lowers the model's log-odds, on average. Occlusion patches overlap, so this is a per-patch
                 average, NOT the total push of the area: it cannot be added up and compared with the distance to
                 the threshold directly.

Questions (one set at a time; the area always comes from the held-out test map):
  1. Which misses carry the extra tau? Area SUVR within the misses by confidence band (confident P <= .1,
     moderately confident .1-.3, uncertain .3-.7), reader agreement (unanimous / split 2-1 / single reader) and
     reader grade (moderate / frequent). Descriptive, with Kruskal-Wallis / Mann-Whitney p-values (small groups).
  2. Who are they? Demographics of the misses with area SUVR above vs below the misses' median (descriptive).
  3. Did the model see it? Across all scans: area evidence against area SUVR. OLS (HC3)
        area evidence ~ area SUVR + miss            (miss = Model - / Read + vs every other scan)
     The 'miss' term says whether the misses get less evidence than their tau predicts. A miss term near 0 = the
     model reacts to the area's tau like in every other scan (it sees it), there is just not enough tau.
  4. How far from positive? Per miss: its log-odds (the threshold 0.5 is log-odds 0), so the distance to positive,
     next to the area evidence; plus the median of both for Model + / Read + scans to read them against.
Caveat: the area was found by comparing misses with negatives on the test set, so "misses have more tau there than
negatives" is circular on test (not on AVID). Comparisons within the misses and question 3 are not.

Usage (Berzelius; reads the images and occlusion maps, CPU):
    python results_plotting/elevated_area_followup.py <run_dir> --set test \\
        --area_dir <run>/figures/voxelwise_discordant_test \\
        --occlusion_dir <run>/validation/visualization/occlusion_all_logit_median_brain_mask4_ov0.5 \\
        --pkl_dir /proj/berzelius-2024-156/dataset/tau-PET/tau_T1MNI --ref_mask <masks>/voi_CerebGry_tau_2mm.nii \\
        --votes <data>/IDs_reads.csv --demo_csv <data>/demo.csv
Outputs (<run>/figures/elevated_area_<set>/; no IDs except area_per_scan.csv):
    area_by_group.csv, misses_by_subgroup.csv, misses_demographics.csv, evidence_vs_tau.csv, misses_gap.csv
    (sorted, no IDs), elevated_area.png; area_per_scan.csv (has IDs - stays on Berzelius)
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
from nilearn import image
from scipy import stats

from analyze_reader_votes import load_votes
from make_case_viewer import MNI_2MM_RAS, MNI_SHAPE, on_mni_grid
from plot_utils import CONFUSION_CATEGORY_PALETTE, add_subgroup_columns
from voxelwise_visual_read import suvr_images

ANALYSIS = "model_neg_read_pos_vs_neg"
GROUPS = {(1, 1): "Model + / Read +", (1, 0): "Model − / Read +", (0, 0): "Model − / Read −", (0, 1): "Model + / Read −"}
COLOUR = {"Model + / Read +": CONFUSION_CATEGORY_PALETTE["True Positive"],
          "Model − / Read −": CONFUSION_CATEGORY_PALETTE["True Negative"],
          "Model − / Read +": CONFUSION_CATEGORY_PALETTE["False Negative"],
          "Model + / Read −": CONFUSION_CATEGORY_PALETTE["False Positive"]}
MISS = "Model − / Read +"
CLIP = 1e-4


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--set", default="test", choices=["test", "avid"])
    p.add_argument("--area_dir", required=True, help=f"Voxelwise output folder with {ANALYSIS}_logp_tfce/_t.nii.gz (test).")
    p.add_argument("--occlusion_dir", required=True, help="Merged occlusion folder of this set (maps + summary.csv).")
    p.add_argument("--pkl_dir", required=True)
    p.add_argument("--ref_mask", required=True)
    p.add_argument("--votes", default=None, help="Per-reader vote csv (IDs_reads.csv / AVIDIDs_reads.csv).")
    p.add_argument("--demo_csv", required=True)
    p.add_argument("--female_code", type=float, default=2)
    p.add_argument("--fwhm", type=float, default=6.0, help="Smoothing in mm, as in the voxelwise test.")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


def area_masks(area_dir: Path, alpha: float):
    """The significant 'misses higher' voxels, as a NIfTI on the voxelwise grid and as a mask on the occlusion grid."""
    lp = nib.load(area_dir / f"{ANALYSIS}_logp_tfce.nii.gz")
    t = nib.load(area_dir / f"{ANALYSIS}_t.nii.gz").get_fdata()
    m = (lp.get_fdata() > -np.log10(alpha)) & (t > 0)
    if not m.any():
        raise SystemExit(f"No significant voxels in {area_dir}")
    img = nib.Nifti1Image(m.astype(np.int8), lp.affine)
    occ = image.resample_to_img(img, nib.Nifti1Image(np.zeros(MNI_SHAPE, np.int8), MNI_2MM_RAS), interpolation="nearest",
                                force_resample=True, copy_header=True).get_fdata() > 0
    print(f"[area] {int(m.sum())} voxels ({int(occ.sum())} on the occlusion grid)")
    return img, occ


def band(prob: float) -> str:
    c = min(prob, 1 - prob)
    return "Confident" if c <= 0.1 else ("Uncertain" if c >= 0.3 else "Moderately confident")


def load_table(args) -> pd.DataFrame:
    s = pd.read_csv(Path(args.occlusion_dir) / "summary.csv").dropna(subset=["visual_read", "pred"])
    d = pd.DataFrame({"ID": s["ID"].astype(int), "read": s["visual_read"].astype(int), "call": s["pred"].astype(int),
                      "prob": s["P_positive"].astype(float)})
    p = d["prob"].clip(CLIP, 1 - CLIP)
    d["logodds"] = np.log(p / (1 - p))
    d["group"] = [GROUPS[(r, c)] for r, c in zip(d["read"], d["call"])]
    d["band"] = d["prob"].map(band)
    demo = pd.read_csv(args.demo_csv)
    demo = demo.loc[:, ~demo.columns.str.startswith("Unnamed")].drop_duplicates("ID")
    add_subgroup_columns(demo, args.female_code)
    keep = ["ID", "MetaTemporal", "age", "_g_dx", "_g_sex", "_g_apoe", "_g_amy", "_g_site"]
    d = d.merge(demo[[c for c in keep if c in demo.columns]], on="ID", how="left")
    if args.votes:
        v = load_votes(Path(args.votes))
        v["grade"] = v["visual_read_3_cat"].map({1: "moderate", 2: "frequent"})
        d = d.merge(v[["ID", "agreement", "grade"]], on="ID", how="left")
    return d


def measure(d: pd.DataFrame, args, area_img, area_occ) -> pd.DataFrame:
    rows, skipped, refs, mask = [], [], {}, None
    occ_dir = Path(args.occlusion_dir)
    for sid, x, aff in suvr_images(Path(args.pkl_dir), pd.Index(d["ID"]), Path(args.ref_mask), skipped, refs):
        img = nib.Nifti1Image(x.astype(np.float32), aff)
        if mask is None:
            mask = image.resample_to_img(area_img, img, interpolation="nearest", force_resample=True,
                                         copy_header=True).get_fdata() > 0
        if args.fwhm:
            x = image.smooth_img(img, args.fwhm).get_fdata()
        inside = mask & (x != 0)
        f = occ_dir / f"{sid}_occlusion_pos_vs_neg.nii.gz"
        ev = float(on_mni_grid(f)[area_occ].mean()) if f.exists() else np.nan
        rows.append({"ID": sid, "area_SUVR": x[inside].mean() if inside.sum() >= 0.9 * mask.sum() else np.nan,
                     "area_evidence": ev})
    if skipped:
        print(f"[data] {len(skipped)} scans left out: reference region not covered")
    return d.merge(pd.DataFrame(rows), on="ID", how="inner")


def med(s):
    s = s.dropna()
    return round(float(s.median()), 3) if len(s) else np.nan


def by_subgroup(m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, order in (("band", ["Confident", "Moderately confident", "Uncertain"]),
                       ("agreement", ["unanimous", "split 2-1", "single reader"]), ("grade", ["frequent", "moderate"])):
        if col not in m:
            continue
        groups = [(lvl, m[m[col] == lvl]) for lvl in order if (m[col] == lvl).any()]
        vals = [g["area_SUVR"].dropna() for _, g in groups if g["area_SUVR"].notna().sum() >= 2]
        p = np.nan
        if len(vals) == 2:
            p = stats.mannwhitneyu(*vals).pvalue
        elif len(vals) > 2:
            p = stats.kruskal(*vals).pvalue
        for lvl, g in groups:
            rows.append({"split by": col, "level": lvl, "n": len(g), "median area SUVR": med(g["area_SUVR"]),
                         "median area evidence": med(g["area_evidence"]), "median log-odds": med(g["logodds"]),
                         "median meta-temporal SUVR": med(g["MetaTemporal"]), "p (area SUVR across levels)": p})
    return pd.DataFrame(rows)


def demographics(m: pd.DataFrame) -> pd.DataFrame:
    cut = m["area_SUVR"].median()
    rows = []
    for half, g in (("higher area SUVR", m[m["area_SUVR"] > cut]), ("lower area SUVR", m[m["area_SUVR"] <= cut])):
        r = {"misses": half, "n": len(g), "median area SUVR": med(g["area_SUVR"]), "median age": med(g["age"])}
        for col, lvl, name in (("_g_amy", "Aβ positive", "amyloid +"), ("_g_dx", "CU", "cognitively unimpaired"),
                               ("_g_apoe", "Carrier", "APOE ε4 carrier"), ("_g_sex", "Female", "female")):
            if col in g:
                k, n = int((g[col] == lvl).sum()), int(g[col].notna().sum())
                r[name] = f"{k}/{n}"
        if "_g_site" in g:
            r["sites"] = ", ".join(f"{k} {v}" for k, v in g["_g_site"].value_counts().items())
        rows.append(r)
    return pd.DataFrame(rows)


def evidence_vs_tau(d: pd.DataFrame) -> pd.DataFrame:
    e = d.dropna(subset=["area_SUVR", "area_evidence"]).copy()
    e["miss"] = (e["group"] == MISS).astype(int)
    rows = []
    for f, term in (("area_evidence ~ area_SUVR", "area_SUVR"), ("area_evidence ~ area_SUVR + miss", "area_SUVR"),
                    ("area_evidence ~ area_SUVR + miss", "miss")):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = smf.ols(f, e).fit(cov_type="HC3")
        lo, hi = r.conf_int().loc[term]
        rows.append({"model": f, "term": term, "estimate": r.params[term], "lo": lo, "hi": hi, "p": r.pvalues[term],
                     "n": int(r.nobs), "r2": r.rsquared})
    rho, p = stats.spearmanr(e["area_SUVR"], e["area_evidence"])
    rows.append({"model": "Spearman, all scans", "term": "area_SUVR vs area_evidence", "estimate": rho, "p": p, "n": len(e)})
    return pd.DataFrame(rows)


def figure(d: pd.DataFrame, gap: pd.DataFrame, title: str, out: Path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
    for g in ["Model − / Read −", "Model + / Read +", "Model + / Read −", MISS]:
        s = d[d["group"] == g]
        a.scatter(s["area_SUVR"], s["area_evidence"], s=40 if g == MISS else 14, alpha=0.9 if g == MISS else 0.45,
                  color=COLOUR[g], edgecolor="white", linewidth=0.4, label=f"{g} (n = {len(s)})", zorder=3 if g == MISS else 2)
    a.axhline(0, color="0.6", lw=0.8)
    a.set_xlabel("Tau in the area (mean SUVR)", fontsize=12)
    a.set_ylabel("Model evidence in the area\n(mean occlusion drop, log-odds)", fontsize=12)
    a.set_title("Does the model react to the tau in the area?", fontsize=13)
    a.legend(fontsize=9, frameon=False)
    y = np.arange(len(gap))
    shade = {"Confident": "#8c5a00", "Moderately confident": COLOUR[MISS], "Uncertain": "#f3d39a"}
    for lvl, c in shade.items():
        s = (gap["band"] == lvl).to_numpy()
        if s.any():
            b.barh(y[s], gap.loc[s, "distance to positive (log-odds)"], color=c, label=lvl)
    b.set_yticks([])
    b.set_ylabel("Misses (one bar each)", fontsize=12)
    b.set_xlabel("Log-odds needed to reach positive", fontsize=12)
    b.set_title("Each miss: how far from a positive call", fontsize=13)
    b.legend(fontsize=9, frameon=False, loc="lower right")
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    run = Path(args.run_dir)
    out = Path(args.out_dir) if args.out_dir else run / "figures" / f"elevated_area_{args.set}"
    out.mkdir(parents=True, exist_ok=True)
    area_img, area_occ = area_masks(Path(args.area_dir), args.alpha)
    d = measure(load_table(args), args, area_img, area_occ)
    d.to_csv(out / "area_per_scan.csv", index=False)
    m = d[d["group"] == MISS]
    print(f"[data] {len(d)} scans, {len(m)} misses")

    groups = pd.DataFrame([{"group": g, "n": int((d["group"] == g).sum()),
                            "median area SUVR": med(d.loc[d["group"] == g, "area_SUVR"]),
                            "median area evidence": med(d.loc[d["group"] == g, "area_evidence"]),
                            "median log-odds": med(d.loc[d["group"] == g, "logodds"])} for g in COLOUR])
    sub, dem, fit = by_subgroup(m), demographics(m), evidence_vs_tau(d)
    gap = m.assign(**{"distance to positive (log-odds)": -m["logodds"]}).sort_values("distance to positive (log-odds)")
    gap = gap[["band", "agreement", "grade", "area_SUVR", "area_evidence", "distance to positive (log-odds)"]
              if "agreement" in gap else ["band", "area_SUVR", "area_evidence", "distance to positive (log-odds)"]]
    for name, t in (("area_by_group", groups), ("misses_by_subgroup", sub), ("misses_demographics", dem),
                    ("evidence_vs_tau", fit), ("misses_gap", gap.round(3))):
        t.to_csv(out / f"{name}.csv", index=False)
    label = "held-out test" if args.set == "test" else "AVID (area from the held-out test)"
    figure(d, gap.reset_index(drop=True), f"The elevated-tau area of the voxelwise map, {label}", out / "elevated_area.png")
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.float_format", "{:.3g}".format):
        for name, t in (("By group", groups), ("Misses by subgroup", sub), ("Misses: demographics", dem),
                        ("Evidence vs tau (OLS, HC3)", fit), ("Misses: distance to positive", gap)):
            print(f"\n{name}:\n{t.to_string(index=False)}")
    print(f"\nSaved to {out} (area_per_scan.csv has IDs - keep it on Berzelius)")


if __name__ == "__main__":
    main()
