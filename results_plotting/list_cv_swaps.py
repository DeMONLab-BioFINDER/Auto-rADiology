#!/usr/bin/env python3
"""Which scans does model B get right that model A gets wrong, and the reverse? (5-fold CV, out-of-fold)

compare_cv_runs_oof.py counts these swaps (e.g. 38 only attention agrees with the read, 30 only avg);
this lists them with what is known about each scan, so the two groups can be compared: if B's gains
are one kind of scan (e.g. focal read + with moderate SUVR) and its losses another (e.g. read - with
high mesial temporal SUVR or off-target), the swap is systematic, not seed noise.

Same folds required (checked, as in compare_cv_runs_oof.py). At the call threshold 0.5 (the pred column).

Usage:
    python results_plotting/list_cv_swaps.py <cv_run_A> <cv_run_B> --labels avg attention
Prints a summary per group and the IDs (for results_plotting/inspect_scans.py); saves one row per scan to
<proj_path>/results/cv_oof_comparison/<A>_vs_<B>_swaps.csv (outside the repo; has IDs, not for slides).
"""
import argparse
from pathlib import Path

import pandas as pd

from compare_cv_runs_oof import load_oof
from plot_utils import find_demo_csv, resolve_path

REGIONS = ["MetaTemporal", "TemporoParietal", "MesialTemporal", "Frontal", "Universal"]
INFO = ["site", "dx", "age", "amyloid_status", "CL", "CDR_SB"]


def parse_args():
    repo = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.add_argument("--labels", nargs=2, default=["A", "B"])
    p.add_argument("--demo_csv", default=None, help="Demographics csv (default data/demo.csv).")
    p.add_argument("--out_dir", default=str(repo.parent / "results" / "cv_oof_comparison"))
    return p.parse_args()


def summary(g: pd.DataFrame, la: str, lb: str) -> dict:
    out = {"n": len(g)}
    for r in REGIONS:
        if r in g:
            out[f"{r} median"] = round(g[r].median(), 2)
    if "MesialTemporal" in g and "MetaTemporal" in g:  # mesial much higher than meta: isolated mesial uptake (read ignores it)
        out["Mesial - Meta median"] = round((g["MesialTemporal"] - g["MetaTemporal"]).median(), 2)
    if "CL" in g:
        out["CL median"] = round(g["CL"].median(), 1)
    if "amyloid_status" in g:
        out["% amyloid +"] = round(100 * g["amyloid_status"].mean(), 0)
    if "dx" in g:
        out["dx"] = ", ".join(f"{k} {v}" for k, v in g["dx"].value_counts().items())
    if "site" in g:
        out["site"] = ", ".join(f"{k} {v}" for k, v in g["site"].value_counts().items())
    out[f"prob {la} median"] = round(g["prob_a"].median(), 2)
    out[f"prob {lb} median"] = round(g["prob_b"].median(), 2)
    return out


def main():
    args = parse_args()
    la, lb = args.labels
    a, b = load_oof(args.run_a), load_oof(args.run_b)
    m = a.merge(b, on="ID_ind", how="inner", suffixes=("_a", "_b"))
    if len(m) != len(a) or len(m) != len(b) or (m["fold_a"] != m["fold_b"]).any() or (m["y_a"] != m["y_b"]).any():
        raise SystemExit("Not the same scans / folds / reads in the two runs")
    m = m.rename(columns={"y_a": "y", "fold_a": "fold"}).drop(columns=["y_b", "fold_b"])

    demo_path = resolve_path(args.demo_csv, find_demo_csv)
    if demo_path is None:
        raise FileNotFoundError("No demographics csv found - pass --demo_csv.")
    demo = pd.read_csv(demo_path)
    cols = ["ID"] + [c for c in INFO + REGIONS if c in demo.columns]
    m = m.merge(demo[cols], left_on="ID_ind", right_on="ID", how="left").drop(columns=["ID"])

    ok_a, ok_b = m["pred_a"] == m["y"], m["pred_b"] == m["y"]
    m["group"] = None
    m.loc[ok_b & ~ok_a, "group"] = f"only {lb} right"
    m.loc[ok_a & ~ok_b, "group"] = f"only {la} right"
    sw = m[m["group"].notna()].copy()
    sw["read"] = sw["y"].map({1: "read +", 0: "read -"})

    rows = {f"{grp}, {read}": summary(g, la, lb) for (grp, read), g in sw.groupby(["group", "read"])}
    table = pd.DataFrame(rows)
    print(f"{len(sw)} scans where exactly one of {la} / {lb} agrees with the read (out of {len(m)})\n")
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.max_colwidth", 60):
        print(table.to_string())

    print("\nIDs (for inspect_scans.py):")
    for (grp, read), g in sw.groupby(["group", "read"]):
        print(f"  {grp}, {read}: {' '.join(str(int(i)) for i in g.sort_values('MetaTemporal' if 'MetaTemporal' in g else 'ID_ind')['ID_ind'])}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    keep = ["ID_ind", "group", "read", "fold", "prob_a", "prob_b"] + [c for c in INFO + REGIONS if c in sw]
    path = out / f"{la}_vs_{lb}_swaps.csv"
    sw.sort_values(["group", "read", "ID_ind"])[keep].rename(columns={"prob_a": f"prob_{la}", "prob_b": f"prob_{lb}"}).to_csv(path, index=False)
    print(f"\nSaved {path} (has IDs - not for slides)")


if __name__ == "__main__":
    main()
