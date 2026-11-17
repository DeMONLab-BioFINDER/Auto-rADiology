#!/usr/bin/env python3
"""Who pushes the model towards "positive", at the same tau level? Multivariable regressions on the
model's output, with the visual read as the comparison.

Only a handful of scans are Model + / Read - (2 on test, 1 on AVID), so right/wrong can't be modelled in
that direction. Instead the outcome is the model's log-odds of "positive" (continuous, all scans), which
keeps all the information in the probabilities.

Models (each set separately - test and AVID are never pooled):
  1. all scans:      log-odds ~ read + MetaTemporal SUVR (spline) + age + sex + diagnosis [+ site]
                     -> how much a variable shifts the model's output when the read and the tau level are the same
  2. read + scans:   log-odds ~ MetaTemporal SUVR (spline) + age + sex + diagnosis [+ site]
  3. read - scans:   the same                                    (2 and 3: does the effect differ by read?)
  4. read vs model:  logistic regression of the read, and of the model's call (at 0.5), on the same variables,
                     all scans -> odds ratios side by side: where do readers and model react differently?
Then the same four with one extra variable at a time - APOE e4 carrier, Centiloid (per 25 CL), CDR-SB - each on
the scans that have it (they are missing for whole cohorts: CDR-SB for all of Geneva, APOE / Centiloid for
40% of NACC), so the main model keeps everyone.

Log-odds: probabilities clipped to [1e-4, 1 - 1e-4] (log-odds -9.2..9.2), because a saturated softmax gives
0 or 1. OLS with robust (HC3) standard errors, since those log-odds are far from normal. The spline (natural
cubic, 3 df) lets the SUVR effect bend, so the other variables are compared at truly the same tau level.
Site is a fixed effect when the set has more than one. FDR (Benjamini-Hochberg) within each model over the
reported variables (age, sex, diagnosis, extra variable). Exploratory.

Usage:
    python results_plotting/regress_model_output_visual_read.py <final_run_dir> [--sets test avid] \\
        [--demo_csv ../data/demo.csv] [--ext_demo_csv ../data/demo_AVID_unseen.csv]
Outputs (<run_dir>/figures/regressions/, no IDs): regressions_<set>.csv (every coefficient), regressions_<set>.png
"""
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from plot_utils import add_subgroup_columns, find_demo_csv, resolve_path

CLIP = 1e-4
SPLINE = "cr(MetaTemporal, df=3)"
EXTRAS = {"apoe_e4": "APOE ε4 carrier", "cl25": "Centiloid (per 25 CL)", "CDR_SB": "CDR-SB (per point)"}
LABELS = {"age10": "Age (per 10 years)", "C(sex)[T.Male]": "Male sex",
          "C(dx, Treatment('CU'))[T.MCI]": "MCI (vs CU)", "C(dx, Treatment('CU'))[T.AD dementia]": "AD dementia (vs CU)",
          "read": "Visual read +", **EXTRAS}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--sets", nargs="+", default=["test", "avid"], choices=["test", "avid"])
    p.add_argument("--dataset", default="Gothenburg")
    p.add_argument("--ext_dataset", default="AVID_unseen")
    p.add_argument("--demo_csv", default=None)
    p.add_argument("--ext_demo_csv", default=None)
    p.add_argument("--female_code", type=float, default=2)
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


def find_ext_demo(name):
    here = Path(__file__).resolve()
    return next((c for c in [here.parents[1] / "data" / f"demo_{name}.csv", here.parents[2] / "data" / f"demo_{name}.csv"]
                 if c.exists()), None)


def load(run: Path, which: str, args) -> pd.DataFrame:
    if which == "test":
        f = run / "evaluation" / args.dataset / f"Eval_{args.dataset}_results.csv"
        demo_path = resolve_path(args.demo_csv, find_demo_csv)
    else:
        hits = sorted((run / "validation").glob(f"External_validation_{args.ext_dataset}__*_zeroshot_results.csv"))
        if not hits:
            raise SystemExit(f"No {args.ext_dataset} result in {run}/validation")
        f = hits[0]
        demo_path = resolve_path(args.ext_demo_csv, lambda: find_ext_demo(args.ext_dataset))
    if demo_path is None or not f.exists():
        raise SystemExit(f"Missing predictions or demographics for the {which} set")
    d = pd.read_csv(f)[["ID_ind", "y", "pred", "prob"]]
    demo = pd.read_csv(demo_path)
    demo = demo.loc[:, ~demo.columns.astype(str).str.startswith("Unnamed")]
    d = d.merge(demo, left_on="ID_ind", right_on="ID", how="left")
    add_subgroup_columns(d, args.female_code)  # same diagnosis / sex / APOE / site coding as the subgroup analysis
    p = d["prob"].clip(CLIP, 1 - CLIP)
    out = pd.DataFrame({
        "logodds": np.log(p / (1 - p)), "read": d["y"].astype(int), "call": d["pred"].astype(int),
        "MetaTemporal": pd.to_numeric(d["MetaTemporal"], errors="coerce"),
        "age10": pd.to_numeric(d["age"], errors="coerce") / 10,
        "sex": d.get("_g_sex"), "dx": d.get("_g_dx"), "site": d.get("_g_site"),
        "apoe_e4": pd.to_numeric(d.get("apoe"), errors="coerce"),
        "cl25": pd.to_numeric(d.get("CL"), errors="coerce") / 25,
        "CDR_SB": pd.to_numeric(d.get("CDR_SB"), errors="coerce"),
    })
    # "Other" diagnoses (FTLD etc., ~1%) are too few to estimate and would not mean one thing; left out here
    return out[out["dx"].isin(["CU", "MCI", "AD dementia"])]


def formula(lhs: str, d: pd.DataFrame, with_read: bool, extra: str | None) -> str:
    terms = (["read"] if with_read else []) + [SPLINE, "age10", "C(sex)", "C(dx, Treatment('CU'))"]
    if d["site"].nunique() > 1:
        terms.append("C(site)")
    if extra:
        terms.append(extra)
    return f"{lhs} ~ " + " + ".join(terms)


def fit(kind: str, d: pd.DataFrame, f: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if kind == "ols":
                return smf.ols(f, d).fit(cov_type="HC3")
            res = smf.logit(f, d).fit(disp=0, maxiter=500)
            # perfect separation shows up as huge coefficients on the reported variables (an odds ratio beyond
            # e^10); the spline's own coefficients can be large without anything being wrong, so they are not checked
            reported = [t for t in res.params.index if t in LABELS]
            ok = res.mle_retvals.get("converged", True) and (res.params[reported].abs() < 10).all()
            return res if ok else None
        except Exception:
            return None


def tidy(res, model: str, extra: str | None, n: int, odds: bool) -> list[dict]:
    keep = [t for t in res.params.index if t in LABELS and t != "read"]
    ci = res.conf_int()
    rows = [{"model": model, "extra": extra or "none", "term": LABELS[t], "n": n,
             "estimate": res.params[t], "lo": ci.loc[t, 0], "hi": ci.loc[t, 1], "p": res.pvalues[t]} for t in keep]
    if rows:
        fdr = multipletests([r["p"] for r in rows], method="fdr_bh")[1]
        for r, q in zip(rows, fdr):
            r["p_fdr"] = q
            if odds:
                r["estimate"], r["lo"], r["hi"] = np.exp(r["estimate"]), np.exp(r["lo"]), np.exp(r["hi"])
    return rows


def run_set(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for extra in [None, *EXTRAS]:
        cols = ["logodds", "read", "call", "MetaTemporal", "age10", "sex", "dx", "site"] + ([extra] if extra else [])
        dd = d.dropna(subset=[c for c in cols if c != "site" or d["site"].notna().any()])
        specs = [("1 shift in model log-odds (all scans, same read and tau)", "ols", "logodds", dd, True),
                 ("2 model log-odds, read + scans", "ols", "logodds", dd[dd["read"] == 1], False),
                 ("3 model log-odds, read - scans", "ols", "logodds", dd[dd["read"] == 0], False),
                 ("4a odds ratio of read +", "logit", "read", dd, False),
                 ("4b odds ratio of model call +", "logit", "call", dd, False)]
        for name, kind, lhs, data, with_read in specs:
            res = fit(kind, data, formula(lhs, data, with_read, extra))
            if res is None:
                rows.append({"model": name, "extra": extra or "none", "term": "(not estimable)", "n": len(data)})
                continue
            rows += tidy(res, name, extra, int(res.nobs), odds=(kind == "logit"))
    return pd.DataFrame(rows)


def figure(t: pd.DataFrame, title: str, out: Path):
    """Left: shift in the model's log-odds at the same read and tau (model 1). Right: odds ratios, read vs model."""
    m1 = t[t["model"].str.startswith("1") & t["estimate"].notna()]
    m1 = m1[(m1["extra"] == "none") | (m1["term"] == m1["extra"].map(EXTRAS))]
    m4 = t[t["model"].str.startswith("4") & t["estimate"].notna()]
    m4 = m4[(m4["extra"] == "none") | (m4["term"] == m4["extra"].map(EXTRAS))]
    terms = list(dict.fromkeys(m1["term"].tolist() + m4["term"].tolist()))
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.5 * len(terms) + 1.6), dpi=170, sharey=True)
    y = {term: i for i, term in enumerate(terms[::-1])}
    for _, r in m1.iterrows():
        axes[0].errorbar(r["estimate"], y[r["term"]], xerr=[[r["estimate"] - r["lo"]], [r["hi"] - r["estimate"]]],
                         fmt="o", color="#1f4e79", capsize=2.5, ms=5)
    axes[0].axvline(0, color="#999", lw=0.8)
    axes[0].set_xlabel("Shift in model log-odds (same read and tau)")
    for (model, colour, dy) in [("4a", "#7f7f7f", 0.15), ("4b", "#c0504d", -0.15)]:
        for _, r in m4[m4["model"].str.startswith(model)].iterrows():
            axes[1].errorbar(r["estimate"], y[r["term"]] + dy, xerr=[[r["estimate"] - r["lo"]], [r["hi"] - r["estimate"]]],
                             fmt="o", color=colour, capsize=2.5, ms=5,
                             label="Visual read" if model == "4a" else "Model call")
    axes[1].axvline(1, color="#999", lw=0.8); axes[1].set_xscale("log")
    axes[1].set_xlabel("Odds ratio of being called + (same tau)")
    h, l = axes[1].get_legend_handles_labels(); keep = dict(zip(l, h))
    axes[1].legend(keep.values(), keep.keys(), fontsize=8, loc="lower right")
    axes[0].set_yticks(list(y.values())); axes[0].set_yticklabels(list(y.keys()), fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=11); fig.tight_layout(); fig.savefig(out); plt.close(fig)


def main():
    args = parse_args()
    run = Path(args.run_dir).resolve()
    out = Path(args.out_dir) if args.out_dir else run / "figures" / "regressions"
    out.mkdir(parents=True, exist_ok=True)
    names = {"test": "Held-out test", "avid": "External (AVID)"}
    for which in args.sets:
        d = load(run, which, args)
        t = run_set(d)
        t.to_csv(out / f"regressions_{which}.csv", index=False)
        figure(t, f"{names[which]} (n = {len(d)})", out / f"regressions_{which}.png")
        print(f"\n===== {names[which]}: {len(d)} scans (CU / MCI / AD dementia) =====")
        for (model, extra), g in t.groupby(["model", "extra"], sort=False):
            g = g[g["term"] != "(not estimable)"]
            if g.empty:
                print(f"  {model} [+{extra}]: not estimable"); continue
            odds = model.startswith("4")
            print(f"  {model}" + (f"  [+ {EXTRAS[extra]}]" if extra != "none" else "") + f"  (n = {int(g['n'].iloc[0])})")
            for _, r in g.iterrows():
                if extra != "none" and r["term"] != EXTRAS[extra]:
                    continue  # with an extra variable, only show that variable (the rest is in the csv)
                star = " *" if r["p_fdr"] < 0.05 else ""
                print(f"      {r['term']:<24} {'OR' if odds else 'β'} {r['estimate']:6.2f} [{r['lo']:.2f}, {r['hi']:.2f}]"
                      f"  p = {r['p']:.3g}  FDR {r['p_fdr']:.3g}{star}")
    print(f"\n* = FDR < 0.05 within that model.  Saved to {out}")


if __name__ == "__main__":
    main()
