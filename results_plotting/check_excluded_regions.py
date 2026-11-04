#!/usr/bin/env python3
"""Does the model use regions the FDA rule ignores (mesial temporal, frontal)?

The read counts only posterior neocortex (posterolateral temporal, occipital, parietal/precuneus);
uptake isolated to mesial temporal, anterolateral temporal and/or frontal is negative, with or without
it a posterior positive stays positive. So among scans whose posterior regions are NOT elevated, a
model following the rule should not call more of them positive when mesial or frontal uptake is high.

"Elevated" = CenTauRz > 2 for the region (FTP SUVR cut-offs from Villemagne 2023, plot_utils):
posterior = MetaTemporal or TemporoParietal; excluded = MesialTemporal, Frontal. Bilateral ROI means,
so a focal posterior spot can hide in a "not elevated" scan - which is why the readers' own calls are
shown next to the model's: the readers are the reference for how often that happens. A model that uses
the excluded regions shows a rise in % Model + from the "none" row to the mesial / frontal rows that is
clearly larger than the rise in % read +.

Table: among posterior-not-elevated scans, per group (none / mesial only / frontal only / both):
n, median Meta SUVR, % read +, and per run % Model + (call at 0.5) and median probability.
Adjusted: in the same scans, logistic regressions of the read and of each run's call on Meta and TP SUVR
(continuous) + mesial elevated + frontal elevated; odds ratios with 95% CI for the two excluded regions
(larger for a model than for the read = the model leans on that region more than the readers do).
Descriptive: groups can be small; no formal model-vs-read test.

Predictions: <run>/oof_predictions.csv (5-fold CV, ~1800 scans) if present, else the run's test
predictions (evaluation/<dataset>/Eval_<dataset>_results.csv). All runs must have the same scans.

Usage:
    python results_plotting/check_excluded_regions.py <run> [<run> ...] --labels avg attention
Outputs: <proj_path>/results/excluded_regions/<labels>.csv and _odds_ratios.csv (outside the repo; no IDs).
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from plot_utils import CENTAUR_FTP_CTRZ2_SUVR, find_demo_csv, resolve_path

POSTERIOR = ["MetaTemporal", "TemporoParietal"]
EXCLUDED = ["MesialTemporal", "Frontal"]


def parse_args():
    repo = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+")
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--dataset", default="Gothenburg")
    p.add_argument("--demo_csv", default=None, help="Demographics csv with the regional SUVRs (default data/demo.csv).")
    p.add_argument("--out_dir", default=str(repo.parent / "results" / "excluded_regions"))
    args = p.parse_args()
    args.labels = args.labels or [f"run{i + 1}" for i in range(len(args.runs))]
    if len(args.labels) != len(args.runs):
        p.error("--labels needs one name per run")
    return args


def load_preds(run: str, dataset: str) -> tuple[pd.DataFrame, str]:
    oof = Path(run) / "oof_predictions.csv"
    test = Path(run) / "evaluation" / dataset / f"Eval_{dataset}_results.csv"
    f = oof if oof.exists() else test
    if not f.exists():
        raise SystemExit(f"No oof_predictions.csv or {test.relative_to(run)} in {run}")
    return pd.read_csv(f)[["ID_ind", "y", "pred", "prob"]], ("out-of-fold" if f == oof else "test")


def odds_ratios(y: pd.Series, d: pd.DataFrame) -> dict:
    """OR (95% CI) for mesial / frontal elevated, adjusted for Meta and TP SUVR (z-scored)."""
    X = pd.DataFrame({r: (d[r] - d[r].mean()) / d[r].std() for r in POSTERIOR})
    X["mesial elevated"] = d["mesial_elev"].astype(float)
    X["frontal elevated"] = d["frontal_elev"].astype(float)
    keys = ("mesial elevated", "frontal elevated")
    n_pos = int(y.sum())
    if n_pos < 5 or len(y) - n_pos < 5:
        return {k: f"too few + calls ({n_pos})" for k in keys}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.Logit(y.astype(int), sm.add_constant(X)).fit(disp=0, maxiter=500)
    except Exception:  # e.g. perfect separation in a small subset
        return {k: "not estimable" for k in keys}
    if not res.mle_retvals.get("converged", True) or (res.params.abs() > 15).any():
        return {k: "not estimable (separation)" for k in keys}
    ci = res.conf_int()
    return {k: f"{np.exp(res.params[k]):.2f} [{np.exp(ci.loc[k, 0]):.2f}-{np.exp(ci.loc[k, 1]):.2f}]"
            for k in keys}


def main():
    args = parse_args()
    m, kinds = None, set()
    for lab, run in zip(args.labels, args.runs):
        d, kind = load_preds(run, args.dataset)
        kinds.add(kind)
        d = d.rename(columns={"pred": f"pred_{lab}", "prob": f"prob_{lab}"})
        if m is None:
            m = d
        else:
            n0 = len(m)
            m = m.merge(d.drop(columns="y"), on="ID_ind", how="inner")
            if len(m) != n0 or len(m) != len(d):
                raise SystemExit("The runs do not have the same scans")
    if len(kinds) > 1:
        raise SystemExit("Mix of out-of-fold and test predictions - give runs of one kind")

    demo_path = resolve_path(args.demo_csv, find_demo_csv)
    if demo_path is None:
        raise FileNotFoundError("No demographics csv found - pass --demo_csv.")
    demo = pd.read_csv(demo_path)[["ID"] + POSTERIOR + EXCLUDED]
    m = m.merge(demo, left_on="ID_ind", right_on="ID", how="left").dropna(subset=POSTERIOR + EXCLUDED)

    cut = CENTAUR_FTP_CTRZ2_SUVR
    m["posterior_elev"] = (m["MetaTemporal"] > cut["MetaTemporal"]) | (m["TemporoParietal"] > cut["TemporoParietal"])
    m["mesial_elev"] = m["MesialTemporal"] > cut["MesialTemporal"]
    m["frontal_elev"] = m["Frontal"] > cut["Frontal"]
    sub = m[~m["posterior_elev"]].copy()
    sub["group"] = np.select([sub["mesial_elev"] & sub["frontal_elev"], sub["mesial_elev"], sub["frontal_elev"]],
                             ["mesial + frontal", "mesial only", "frontal only"], "none")

    rows = []
    for grp in ["none", "mesial only", "frontal only", "mesial + frontal"]:
        g = sub[sub["group"] == grp]
        row = {"group": grp, "n": len(g), "Meta median": round(g["MetaTemporal"].median(), 2) if len(g) else np.nan,
               "% read +": round(100 * g["y"].mean(), 1) if len(g) else np.nan}
        for lab in args.labels:
            row[f"% {lab} Model +"] = round(100 * g[f"pred_{lab}"].mean(), 1) if len(g) else np.nan
            row[f"{lab} prob median"] = round(g[f"prob_{lab}"].median(), 2) if len(g) else np.nan
        rows.append(row)
    table = pd.DataFrame(rows).set_index("group")

    ors = pd.DataFrame({"read": odds_ratios(sub["y"], sub),
                        **{f"{lab} call": odds_ratios(sub[f"pred_{lab}"], sub) for lab in args.labels}})

    print(f"{len(m)} {kinds.pop()} scans; {len(sub)} with posterior regions NOT elevated "
          f"(Meta <= {cut['MetaTemporal']:.2f} and TP <= {cut['TemporoParietal']:.2f}); "
          f"elevated = mesial > {cut['MesialTemporal']:.2f}, frontal > {cut['Frontal']:.2f} (CTRz > 2)\n")
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(table.to_string())
        print("\nOdds ratio for being called + (read, or each model's call), adjusted for Meta and TP SUVR:")
        print(ors.to_string())

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = "_".join(args.labels)
    table.to_csv(out / f"{name}.csv")
    ors.to_csv(out / f"{name}_odds_ratios.csv")
    print(f"\nSaved {out / name}.csv and {name}_odds_ratios.csv")


if __name__ == "__main__":
    main()
