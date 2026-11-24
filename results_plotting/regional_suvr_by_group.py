#!/usr/bin/env python3
"""How much tau is in each CenTauR region, per model / read group? (Goes next to the occlusion table.)

Per group (Model + / Read +, Model − / Read −, Model − / Read +, Model + / Read −): median SUVR in the CenTauR
meta-temporal, temporoparietal, mesial temporal and frontal regions (the spreadsheet SUVRs). Model − / Read + vs
Model + / Read +: Mann-Whitney per region, FDR across the four regions. Model + / Read − is shown, not tested.

Usage:
    python results_plotting/regional_suvr_by_group.py <run_dir> --demo_csv <demo.csv> [--dataset AVID_unseen]
Output (no IDs): regional_suvr_by_group.csv next to the run's other figures (validation/figures for an external set).
"""
import argparse
from pathlib import Path

import pandas as pd
from scipy import stats

from plot_utils import fdr_bh, find_demo_csv, resolve_path, resolve_predictions_csv

REGIONS = [("MetaTemporal", "Meta-temporal"), ("TemporoParietal", "Temporoparietal"),
           ("MesialTemporal", "Mesial temporal"), ("Frontal", "Frontal")]
GROUPS = [("Model + / Read +", 1, 1), ("Model − / Read −", 0, 0), ("Model − / Read +", 1, 0), ("Model + / Read −", 0, 1)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir"); p.add_argument("--dataset", default="Gothenburg"); p.add_argument("--demo_csv", default=None)
    args = p.parse_args()
    preds_path, out_dir = resolve_predictions_csv(Path(args.run_dir).resolve(), args.dataset)
    preds = pd.read_csv(preds_path)[["ID_ind", "y", "pred"]].apply(pd.to_numeric, errors="coerce").dropna()
    demo_path = resolve_path(args.demo_csv, find_demo_csv)
    if demo_path is None:
        raise FileNotFoundError("No demographics csv found - pass --demo_csv.")
    demo = pd.read_csv(demo_path)
    df = preds.merge(demo, left_on="ID_ind", right_on="ID", how="inner")

    rows = []
    for name, y, pred in GROUPS:
        g = df[(df["y"] == y) & (df["pred"] == pred)]
        rows.append({"group": name, "n": len(g),
                     **{lab: round(pd.to_numeric(g[col], errors="coerce").median(), 2) for col, lab in REGIONS if col in g}})
    table = pd.DataFrame(rows)
    tp, fn = (df[(df["y"] == 1) & (df["pred"] == k)] for k in (1, 0))
    tests = []
    for col, lab in REGIONS:
        a, b = (pd.to_numeric(x[col], errors="coerce").dropna() for x in (fn, tp))
        tests.append({"region": lab, "p": stats.mannwhitneyu(a, b).pvalue if len(a) > 1 and len(b) > 1 else float("nan")})
    tests = pd.DataFrame(tests)
    tests["p_fdr"] = fdr_bh(tests["p"])
    test_row = {"group": "Model − / Read + vs Model + / Read +: FDR p", "n": ""}
    test_row.update({r["region"]: (f"{r['p_fdr']:.3f}" if pd.notna(r["p_fdr"]) else "") for _, r in tests.iterrows()})
    table = pd.concat([table, pd.DataFrame([test_row])], ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "regional_suvr_by_group.csv", index=False)
    print(table.to_string(index=False))
    print(f"\nSaved {out_dir / 'regional_suvr_by_group.csv'}")


if __name__ == "__main__":
    main()
