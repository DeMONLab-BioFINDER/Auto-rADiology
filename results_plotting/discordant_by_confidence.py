#!/usr/bin/env python3
"""Every scan where the model and the visual read disagree, split by how sure the model was.

Confidence bands, the pre-specified cut-offs of analyze_model_confidence_visual_read.py (PRIMARY):
    confident            probability <= 0.1 or >= 0.9
    moderately confident 0.1-0.3 or 0.7-0.9
    uncertain            0.3-0.7
Both directions: Model - / Read + and Model + / Read - (the latter usually a handful of scans: described, never
tested). Question in plain words: are the confident disagreements different kinds of scans from the near misses
(e.g. little tau, symmetric, amyloid negative vs borderline tau, split readers)? Descriptive only: no tests,
groups are small, read the n first.

Joined per scan (each optional except the confidence and demographics files):
    model probability, call, read          visual_read_confidence_groups.csv (analyze_model_confidence_visual_read.py)
    site, diagnosis, amyloid, Centiloid,   demographics csv
    regional SUVRs
    reader votes and tau score             per-reader vote csv (IDs_reads.csv / AVIDIDs_reads.csv)
    left-right asymmetry |LI| (%)          asymmetry_per_scan.csv (asymmetry_visual_read.py)
    where the model found its evidence     centaur_shares_per_scan.csv (occlusion_group_maps.py)

Usage:
    python results_plotting/discordant_by_confidence.py \
        --confidence <run>/figures/visual_read_confidence_groups.csv --demo_csv <demo.csv> \
        [--votes IDs_reads.csv] [--asymmetry <run>/figures/asymmetry/asymmetry_per_scan.csv] \
        [--occlusion <occlusion folder>/group_maps/centaur_shares_per_scan.csv] [--label "Held-out test"] [--out_dir ...]
Outputs (default: next to the confidence csv, discordant_by_confidence/):
    counts.csv        direction x confidence band (no IDs)
    band_profile.csv  per direction x band: n and simple summaries (no IDs)
    cases.csv         one row per disagreeing scan with everything joined (has IDs - keep off slides)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_model_confidence_visual_read import PRIMARY
from analyze_reader_votes import load_votes

BANDS = ["Confident", "Moderately confident", "Uncertain"]
DIRECTIONS = ["Model − / Read +", "Model + / Read −"]
SUVRS = ["MetaTemporal", "TemporoParietal", "MesialTemporal", "Frontal", "Universal"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--confidence", required=True, help="visual_read_confidence_groups.csv of the set.")
    p.add_argument("--demo_csv", required=True, help="Demographics csv of the set (ID, site, dx, amyloid_status, CL, SUVRs).")
    p.add_argument("--votes", default=None, help="Per-reader vote csv (optional).")
    p.add_argument("--asymmetry", default=None, help="asymmetry_per_scan.csv (optional).")
    p.add_argument("--occlusion", default=None, help="centaur_shares_per_scan.csv (optional).")
    p.add_argument("--label", default="", help="Name of the set for the printout, e.g. 'Held-out test'.")
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


def band(prob: pd.Series) -> pd.Series:
    lo_c, hi_c = PRIMARY["confident"]
    lo_u, hi_u = PRIMARY["uncertain"]
    out = pd.Series("Moderately confident", index=prob.index)
    out[(prob <= lo_c) | (prob >= hi_c)] = "Confident"
    out[(prob >= lo_u) & (prob <= hi_u)] = "Uncertain"
    return out


def load_cases(args) -> pd.DataFrame:
    conf = pd.read_csv(args.confidence)
    d = conf[conf["y"] != conf["pred"]].copy()
    d["direction"] = np.where(d["y"] == 1, DIRECTIONS[0], DIRECTIONS[1])
    d["band"] = band(d["prob"])
    d = d.rename(columns={"ID_ind": "ID"})[["ID", "direction", "band", "prob"]]

    demo = pd.read_csv(args.demo_csv)
    cols = [c for c in ["site", "dx", "age", "amyloid_status", "CL", "apoe", "CDR_SB", *SUVRS] if c in demo.columns]
    d = d.merge(demo[["ID", *cols]], on="ID", how="left")
    if args.votes:
        v = load_votes(Path(args.votes).expanduser())
        v["read category"] = v["visual_read_3_cat"].map({0: "-", 1: "moderate", 2: "frequent"})
        d = d.merge(v[["ID", "agreement", "positive_votes", "n_readers", "read category"]], on="ID", how="left")
    if args.asymmetry:
        a = pd.read_csv(args.asymmetry)
        keep = [c for c in ["MetaTemporal_absAI", "TemporoParietal_absAI"] if c in a.columns]
        d = d.merge(a[["ID", *keep]], on="ID", how="left")
    if args.occlusion:
        o = pd.read_csv(args.occlusion)
        keep = {c: f"evidence share {c}" for c in o.columns
                if c in ("Meta-temporal", "Temporoparietal", "Mesial temporal", "Frontal", "Outside CenTauR regions",
                         "Meta-temporal amount")
                or c.startswith("Outside cortex")}
        d = d.merge(o[["ID", *keep]].rename(columns=keep), on="ID", how="left")
    missing = int(d[cols].isna().all(axis=1).sum()) if cols else 0
    if missing:
        print(f"[WARNING] {missing} disagreeing scans are not in {args.demo_csv}")
    return d


def pct(s: pd.Series):
    s = s.dropna()
    return f"{100 * s.mean():.0f}% ({int(s.sum())}/{len(s)})" if len(s) else ""


def med(s: pd.Series, fmt="{:.2f}"):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return fmt.format(s.median()) if len(s) else ""


def profile(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for direction in DIRECTIONS:
        for b in BANDS:
            s = d[(d["direction"] == direction) & (d["band"] == b)]
            if not len(s):
                continue
            row = {"direction": direction, "band": b, "n": len(s), "median probability": med(s["prob"], "{:.3f}"),
                   "median MetaTemporal SUVR": med(s.get("MetaTemporal", pd.Series(dtype=float)))}
            if "amyloid_status" in s:
                row["amyloid +"] = pct(pd.to_numeric(s["amyloid_status"], errors="coerce"))
            if "dx" in s:
                row["cognitively unimpaired"] = pct(s["dx"].where(s["dx"].notna()).eq("CU").where(s["dx"].notna()))
            if "agreement" in s:
                tri = s[s["n_readers"] == 3]
                row["unanimous (of 3-reader scans)"] = pct((tri["agreement"] == "unanimous").astype(float))
                row["single reader"] = pct((s["n_readers"] == 1).astype(float).where(s["n_readers"].notna()))
                if direction == DIRECTIONS[0]:
                    row["graded moderate (vs frequent)"] = pct(
                        s["read category"].map({"moderate": 1.0, "frequent": 0.0}))
            if "MetaTemporal_absAI" in s:
                row["median |LI| (%) meta-temporal"] = med(s["MetaTemporal_absAI"], "{:.1f}")
            if "evidence share Meta-temporal" in s:
                row["median evidence share meta-temporal"] = med(s["evidence share Meta-temporal"])
            if "evidence share Meta-temporal amount" in s:
                row["median evidence amount meta-temporal"] = med(s["evidence share Meta-temporal amount"], "{:.0f}")
            off = [c for c in s.columns if c.startswith("evidence share Outside cortex")]
            if off:
                row["median evidence share outside cortex"] = med(s[off[0]])
            if "site" in s:
                row["sites"] = ", ".join(f"{k} {v}" for k, v in s["site"].value_counts().items())
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    d = load_cases(args)
    out = Path(args.out_dir) if args.out_dir else Path(args.confidence).parent / "discordant_by_confidence"
    out.mkdir(parents=True, exist_ok=True)

    counts = (pd.crosstab(d["direction"], d["band"]).reindex(index=DIRECTIONS, columns=BANDS, fill_value=0))
    counts["all"] = counts.sum(axis=1)
    prof = profile(d)
    counts.to_csv(out / "counts.csv")
    prof.to_csv(out / "band_profile.csv", index=False)
    d.sort_values(["direction", "band", "prob"]).to_csv(out / "cases.csv", index=False)

    lo_c, hi_c = PRIMARY["confident"]
    lo_u, hi_u = PRIMARY["uncertain"]
    print(f"{args.label or Path(args.confidence).parent}: {len(d)} disagreeing scans. Bands: confident <= {lo_c} / "
          f">= {hi_c}, uncertain {lo_u}-{hi_u}, moderately confident in between")
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.max_colwidth", 60):
        print(f"\nCounts:\n{counts.to_string()}")
        print(f"\nProfile per direction x band (descriptive, n first):\n{prof.T.to_string(header=False)}")
    print(f"\nSaved counts.csv, band_profile.csv (no IDs) and cases.csv (has IDs - keep off slides) to {out}")


if __name__ == "__main__":
    main()
