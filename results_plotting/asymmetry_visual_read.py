#!/usr/bin/env python3
"""Left-right asymmetry of tau, and whether it explains when the visual read, regional SUVR and the model
disagree. Karagianni et al. (2026) saw that most read-positive scans that CenTauR quantification calls negative
have focal, asymmetric uptake; a bilateral regional average dilutes one-sided tau. Does the read (and the model)
call asymmetric scans positive more often than their bilateral SUVR would predict?

Per scan (MNI images, SUVR = image / its own mean in the reference region, as in voxelwise_visual_read.py):
left and right mean SUVR in each CenTauR region (masks split at the midline, x = 0 excluded), and the
laterality index LI (%) = 100 x (right - left) / (right + left), as in Anijärv et al. 2025 (Nat Commun 16:8232).
|LI| = how one-sided, regardless of side. (Columns keep the old names <region>_AI / <region>_absAI; until
26 Nov 2026 they held (left - right) / mean(left, right) = -LI / 50, i.e. the same ranking and per-SD results.) The meta-temporal region is the main measure
(decided before looking: it is the region the rest of the analysis adjusts for); temporoparietal is secondary.
Added 26 Nov 2026: the Universal CenTauR region (the overlap of the tracer AD masks, containing the four regional
masks: one general 'is the scan one-sided overall?' check) and the remaining CenTauR regions (mesial temporal,
frontal), each analysed the same way. Expect frontal |LI| to be noisy (little tau at this stage) and mesial
temporal |LI| to pick up off-target signal. A region whose mask is not in --centaur_dir is skipped.

Analyses (one set at a time, test and AVID never pooled):
  1. Sanity: the bilateral SUVR computed here vs the MetaTemporal SUVR in the demographics (Pearson r).
  2. Who calls asymmetric scans positive, at the same tau on the HOTTER side?
       read:   logistic  read ~ |LI| (z) + cr(<region>_hot, df=3) + age + sex [+ site]
     <region>_hot = the higher of the left and right SUVR. Holding the hotter side equal is the fair test: at the
     same bilateral average a one-sided scan has a hotter side by construction, so 'called positive more often' would
     be partly built in. (Until 26 Nov 2026 the bilateral MetaTemporal SUVR was held equal; those rows are kept in
     asymmetry_tests.csv as 'check: same bilateral average', outside the FDR.)
       model:  logistic  model call ~ the same
       output: OLS (HC3) model log-odds ~ read + the same
     -> odds ratios per SD of |LI| for read and model side by side (as in regress_model_output_visual_read.py).
  3. Read + scans only: |LI| in detected (Model + / Read +) vs missed (Model - / Read +) scans (Mann-Whitney),
     and logistic  model call ~ |LI| + spline of the hotter side + age + sex [+ site].
  4. Does the model's advantage over meta-temporal SUVR grow with asymmetry? AUC of model and of SUVR in scans
     below / above the median |LI| (paired bootstrap of the AUC difference in each half).
FDR (Benjamini-Hochberg) over the |LI| tests of analyses 2-3, per region (p_fdr, unchanged), and over all regions
together (p_fdr_all_regions). Exploratory.

Usage:
    python results_plotting/asymmetry_visual_read.py <final_run_dir> --set test \\
        --pkl_dir /proj/berzelius-2024-156/dataset/tau-PET/tau_T1MNI \\
        --ref_mask <masks>/voi_CerebGry_tau_2mm.nii --centaur_dir <folder with Meta_CenTauR.nii.gz, TP_CenTauR.nii.gz>
Outputs (<run_dir>/figures/asymmetry_<set>/): asymmetry_tests.csv, auc_by_asymmetry.csv, asymmetry.png (no IDs);
asymmetry_per_scan.csv (has IDs).
--from_per_scan <asymmetry_per_scan.csv>: rerun the statistics from an earlier run's per-scan table (no images needed).
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
from nilearn import image
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

from plot_utils import CONFUSION_CATEGORY_PALETTE, add_subgroup_columns, external_name, find_demo_csv, resolve_path
from regress_model_output_visual_read import CLIP, find_ext_demo
from voxelwise_visual_read import suvr_images

REGIONS = {"MetaTemporal": "Meta_CenTauR.nii.gz", "TemporoParietal": "TP_CenTauR.nii.gz",
           "Universal": "CenTauR.nii.gz", "MesialTemporal": "Mesial_CenTauR.nii.gz", "Frontal": "Frontal_CenTauR.nii.gz"}
TITLES = {"MetaTemporal": "Meta-temporal", "TemporoParietal": "Temporoparietal", "Universal": "Universal",
          "MesialTemporal": "Mesial temporal", "Frontal": "Frontal"}
SPLINE = "cr(MetaTemporal, df=3)"  # check only: the bilateral average (spreadsheet value)
GROUPS = {(1, 1): "Model + / Read +", (1, 0): "Model − / Read +", (0, 0): "Model − / Read −", (0, 1): "Model + / Read −"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--set", default="test", choices=["test", "avid"])
    p.add_argument("--pkl_dir", default=None)
    p.add_argument("--ref_mask", default=None)
    p.add_argument("--from_per_scan", default=None, help="Rerun the statistics from this asymmetry_per_scan.csv.")
    p.add_argument("--centaur_dir", default=None, help="Folder with the CenTauR masks (Meta_CenTauR.nii.gz, TP_CenTauR.nii.gz, CenTauR.nii.gz, "
                   "Mesial_CenTauR.nii.gz, Frontal_CenTauR.nii.gz; missing ones are skipped).")
    p.add_argument("--dataset", default="Gothenburg")
    p.add_argument("--ext_dataset", default="AVID_unseen")
    p.add_argument("--demo_csv", default=None)
    p.add_argument("--ext_demo_csv", default=None)
    p.add_argument("--female_code", type=float, default=2)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


def load_table(run: Path, args) -> pd.DataFrame:
    if args.set == "test":
        f = run / "evaluation" / args.dataset / f"Eval_{args.dataset}_results.csv"
        demo_path = resolve_path(args.demo_csv, find_demo_csv)
    else:
        hits = sorted((run / "validation").glob(f"External_validation_{args.ext_dataset}__*_zeroshot_results.csv"))
        f = hits[0] if hits else run / "validation" / "missing"
        demo_path = resolve_path(args.ext_demo_csv, lambda: find_ext_demo(args.ext_dataset))
    if demo_path is None or not f.exists():
        raise SystemExit(f"Missing predictions ({f}) or demographics for the {args.set} set")
    d = pd.read_csv(f)[["ID_ind", "y", "pred", "prob"]]
    demo = pd.read_csv(demo_path)
    demo = demo.loc[:, ~demo.columns.astype(str).str.startswith("Unnamed")]
    d = d.merge(demo, left_on="ID_ind", right_on="ID", how="left")
    add_subgroup_columns(d, args.female_code)
    p = d["prob"].clip(CLIP, 1 - CLIP)
    return pd.DataFrame({
        "ID": d["ID_ind"].astype(int), "prob": d["prob"], "logodds": np.log(p / (1 - p)),
        "read": d["y"].astype(int), "call": d["pred"].astype(int),
        "MetaTemporal": pd.to_numeric(d["MetaTemporal"], errors="coerce"),
        "age10": pd.to_numeric(d["age"], errors="coerce") / 10, "sex": d["_g_sex"], "site": d["_g_site"],
    })


def hemisphere_masks(centaur_dir: Path, shape, affine) -> dict:
    """{region: (left mask, right mask)} on the scans' grid; left = MNI x < 0."""
    target = nib.Nifti1Image(np.zeros(shape, np.int8), affine)
    ijk = np.indices(shape).reshape(3, -1).T
    x = nib.affines.apply_affine(affine, ijk)[:, 0].reshape(shape)
    out = {}
    for name, fname in REGIONS.items():
        if not (centaur_dir / fname).exists():
            print(f"[mask] {name}: {fname} not in {centaur_dir} - skipped")
            continue
        img = nib.load(centaur_dir / fname)
        img = nib.Nifti1Image(np.squeeze(np.asarray(img.dataobj)), img.affine)
        m = image.resample_to_img(img, target, interpolation="nearest", force_resample=True,
                                  copy_header=True).get_fdata() > 0
        out[name] = (m & (x < 0), m & (x > 0))
        print(f"[mask] {name}: {int(out[name][0].sum())} left, {int(out[name][1].sum())} right voxels")
    return out


def asymmetry(table: pd.DataFrame, args) -> pd.DataFrame:
    rows, masks, skipped, refs = [], None, [], {}
    for scan_id, x, aff in suvr_images(Path(args.pkl_dir), pd.Index(table["ID"]), Path(args.ref_mask), skipped, refs):
        if masks is None:
            masks = hemisphere_masks(Path(args.centaur_dir), x.shape, aff)
        row = {"ID": scan_id}
        for name, (lm, rm) in masks.items():
            left, right = x[lm & (x != 0)], x[rm & (x != 0)]
            # a hemisphere mostly outside the field of view would give a biased mean: left out
            if left.size < 0.9 * lm.sum() or right.size < 0.9 * rm.sum():
                row.update({f"{name}_L": np.nan, f"{name}_R": np.nan})
                continue
            L, R = left.mean(), right.mean()
            row.update({f"{name}_L": L, f"{name}_R": R, f"{name}_bilateral": x[(lm | rm) & (x != 0)].mean(),
                        f"{name}_AI": 100 * (R - L) / (R + L)})  # laterality index LI (%), Anijärv et al. 2025
        rows.append(row)
    if skipped:
        print(f"[data] {len(skipped)} scans left out: reference region not covered")
    a = pd.DataFrame(rows)
    for name in REGIONS:
        if f"{name}_AI" in a:
            a[f"{name}_absAI"] = a[f"{name}_AI"].abs()
    return table.merge(a, on="ID", how="inner")


def covariates(d: pd.DataFrame, region: str = None) -> str:
    """Held equal: the hotter side of the region (default) or, with region=None, the bilateral average (check)."""
    terms = [f"cr({region}_hot, df=3)" if region else SPLINE, "age10", "C(sex)"]
    if d["site"].nunique() > 1:
        terms.append("C(site)")
    return " + ".join(terms)


def fit(kind: str, f: str, d: pd.DataFrame, term: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if kind == "ols":
                r = smf.ols(f, d).fit(cov_type="HC3")
                est, ci = r.params[term], r.conf_int().loc[term].to_numpy()
            else:
                r = smf.logit(f, d).fit(disp=0, maxiter=500)
                if not r.mle_retvals.get("converged", True) or abs(r.params[term]) > 10:
                    return None
                est, ci = np.exp(r.params[term]), np.exp(r.conf_int().loc[term].to_numpy())
            return {"estimate": est, "lo": ci[0], "hi": ci[1], "p": r.pvalues[term], "n": int(r.nobs)}
        except Exception:
            return None


def paired_auc_diff(y, a, b, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    diffs, n = [], len(y)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        diffs.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return np.percentile(diffs, [2.5, 97.5]) if diffs else (np.nan, np.nan)


def main():
    args = parse_args()
    run = Path(args.run_dir)
    out = Path(args.out_dir) if args.out_dir else run / "figures" / f"asymmetry_{args.set}"
    out.mkdir(parents=True, exist_ok=True)
    if args.from_per_scan:
        d = pd.read_csv(args.from_per_scan)
    else:
        if not (args.pkl_dir and args.ref_mask and args.centaur_dir):
            raise SystemExit("--pkl_dir, --ref_mask and --centaur_dir are needed (or --from_per_scan)")
        d = asymmetry(load_table(run, args), args)
    regions = [n for n in REGIONS if f"{n}_L" in d and f"{n}_absAI" in d]
    print(f"[data] regions: {', '.join(regions)}")
    for name in regions:
        d[f"{name}_hot"] = d[[f"{name}_L", f"{name}_R"]].max(axis=1)
    d = d.dropna(subset=["MetaTemporal", "age10", "sex"])
    d["group"] = [GROUPS[(r, c)] for r, c in zip(d["read"], d["call"])]
    d.to_csv(out / "asymmetry_per_scan.csv", index=False)
    print(f"[data] {len(d)} scans ({int(d['read'].sum())} read +)")

    ok = d[["MetaTemporal", "MetaTemporal_bilateral"]].dropna()
    r = stats.pearsonr(ok["MetaTemporal"], ok["MetaTemporal_bilateral"])[0]
    print(f"[check] meta-temporal SUVR computed here vs demographics: r = {r:.3f} (n = {len(ok)})")

    tests, aucs = [], []
    for name in regions:
        ai = f"{name}_absAI"
        e = d.dropna(subset=[ai]).copy()
        e["z_ai"] = (e[ai] - e[ai].mean()) / e[ai].std()
        rp = e[e["read"] == 1]
        rows, checks = [], []
        for held, target, region in (("same hotter side", rows, name), ("check: same bilateral average", checks, None)):
            cov = covariates(e, region)
            specs = [("read", "logit", f"read ~ z_ai + {cov}", e),
                     ("model call", "logit", f"call ~ z_ai + {cov}", e),
                     ("model log-odds (same read)", "ols", f"logodds ~ read + z_ai + {cov}", e),
                     ("model call, read + only", "logit", f"call ~ z_ai + {covariates(rp, region)}", rp)]
            for label, kind, f, data in specs:
                res = fit(kind, f, data, "z_ai")
                target.append({"region": name, "outcome": label, "held equal": held,
                               "effect": "odds ratio per SD |LI|" if kind == "logit" else "log-odds per SD |LI|",
                               **(res or {"note": "could not be estimated (too few scans / separation)"})})
        det, mis = rp.loc[rp["call"] == 1, ai], rp.loc[rp["call"] == 0, ai]
        if len(det) >= 5 and len(mis) >= 5:
            u = stats.mannwhitneyu(det, mis)
            rows.append({"region": name, "outcome": "|LI| detected vs missed (read +)",
                         "effect": "median |LI| detected − missed", "estimate": det.median() - mis.median(),
                         "lo": np.nan, "hi": np.nan, "p": u.pvalue, "n": len(det) + len(mis)})
        ok = [x for x in rows if pd.notna(x.get("p"))]
        if ok:
            q = multipletests([x["p"] for x in ok], method="fdr_bh")[1]
            for x, qq in zip(ok, q):
                x["p_fdr"] = qq
        tests += rows + checks

        med = e[ai].median()
        for half, s in [("symmetric (below median |LI|)", e[e[ai] <= med]), ("asymmetric (above median)", e[e[ai] > med])]:
            y = s["read"].to_numpy()
            if y.min() == y.max():
                continue
            am, asu = roc_auc_score(y, s["prob"]), roc_auc_score(y, s["MetaTemporal"])
            lo, hi = paired_auc_diff(y, s["prob"].to_numpy(), s["MetaTemporal"].to_numpy(), args.n_boot)
            aucs.append({"region": name, "half": half, "n": len(s), "n_read_pos": int(y.sum()),
                         "median_absAI": s[ai].median(), "auc_model": am, "auc_suvr": asu,
                         "diff": am - asu, "diff_lo": lo, "diff_hi": hi})

    tests, aucs = pd.DataFrame(tests), pd.DataFrame(aucs)
    main = tests["p"].notna() & (tests["held equal"] != "check: same bilateral average")
    if main.any():
        tests.loc[main, "p_fdr_all_regions"] = multipletests(tests.loc[main, "p"], method="fdr_bh")[1]
    tests.to_csv(out / "asymmetry_tests.csv", index=False)
    aucs.to_csv(out / "auc_by_asymmetry.csv", index=False)

    # figure: |LI| by outcome group, per region (same group order and colours as the other discordance figures)
    colour = {"Model + / Read +": CONFUSION_CATEGORY_PALETTE["True Positive"],
              "Model − / Read −": CONFUSION_CATEGORY_PALETTE["True Negative"],
              "Model − / Read +": CONFUSION_CATEGORY_PALETTE["False Negative"],
              "Model + / Read −": CONFUSION_CATEGORY_PALETTE["False Positive"]}
    order = [g for g in colour if (d["group"] == g).sum() > 0]
    set_name = "held-out test" if args.set == "test" else external_name(args.ext_dataset)
    fig, axes = plt.subplots(1, len(regions), figsize=(5.6 * len(regions), 5.2), sharey=True)
    for ax, name in zip(np.atleast_1d(axes), regions):
        vals = [d.loc[d["group"] == g, f"{name}_absAI"].dropna() for g in order]
        bp = ax.boxplot(vals, widths=0.55, showfliers=False, patch_artist=True,
                        medianprops={"color": "0.15", "linewidth": 2}, whiskerprops={"color": "0.45"},
                        capprops={"color": "0.45"}, boxprops={"linewidth": 0})
        for patch, g in zip(bp["boxes"], order):
            patch.set_facecolor(colour[g]); patch.set_alpha(0.35)
        for i, (v, g) in enumerate(zip(vals, order), 1):
            ax.scatter(np.random.default_rng(i).normal(i, 0.07, len(v)), v, s=14, alpha=0.7, color=colour[g],
                       edgecolor="white", linewidth=0.3, zorder=3)
        ax.set_xticks(range(1, len(order) + 1), [f"{g.replace(' / ', ' /' + chr(10))}\n(n = {len(v)})"
                                                 for g, v in zip(order, vals)], fontsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_title(TITLES.get(name, name), fontsize=14)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.92")
    np.atleast_1d(axes)[0].set_ylabel("Left–right asymmetry |LI| (%)", fontsize=14)
    fig.suptitle(f"Left–right asymmetry, {set_name}", fontsize=16)
    fig.tight_layout()
    fig.savefig(out / "asymmetry.png", dpi=200)
    plt.close(fig)

    with pd.option_context("display.width", 200, "display.float_format", "{:.3g}".format):
        print("\nMedian |LI| by group:")
        print(d.groupby("group")[[f"{n}_absAI" for n in regions]].median().reindex(order).to_string())
        print("\nAsymmetry tests (FDR within region):")
        print(tests.to_string(index=False))
        print("\nAUC model vs meta-temporal SUVR, by asymmetry:")
        print(aucs.to_string(index=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
