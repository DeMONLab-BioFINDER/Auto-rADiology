#!/usr/bin/env python3
"""One table for the Model - / Read + scans: which characteristics differ from the Model + / Read + scans, and does
the difference stay at the same tau level (meta-temporal SUVR)? Collation only: every number is read from an
existing output, nothing is re-tested, except the plain meta-temporal SUVR comparison itself (Mann-Whitney), which no
other script tests.

Rows (read + scans only; Model + / Read - scans are too few for tests and are described case by case elsewhere):
    Meta-temporal SUVR          median Model - vs Model +; Mann-Whitney (computed here, from the confidence csv)
    Diagnosis, sex, APOE e4,    % Model - per level; Fisher / permutation chi-square, FDR (subgroup analysis:
    amyloid, site, age            visual_read_subgroup_performance.csv / _tests.csv)
    CDR-SB beyond diagnosis     within-diagnosis permutation (subgroup analysis, secondary family)
    -> same tau level           odds ratio of Model - [95% CI], likelihood-ratio p (visual_read_errors_at_same_suvr.csv,
                                "Meta-temporal (main)"; only run for variables significant in the plain test)
    Left-right asymmetry        plain: median |LI| Model + minus Model - (Mann-Whitney); same tau level: odds ratio per
                                SD of |LI| at the same bilateral meta-temporal SUVR (+ age, sex, site), inverted here
                                to the odds of Model - so every row reads the same way (asymmetry_tests.csv)
    Reader tau score            plain: % Model - moderate vs frequent, Fisher (reader votes 1_*); same tau level:
                                6_grade_at_same_suvr.csv
Odds ratios below 1 = less often Model - (e.g. amyloid positive scans are less often Model - / Read +).

Usage:
    python results_plotting/discordant_summary_table.py --figures <run>/figures \
        [--asymmetry_tests <run>/figures/asymmetry/asymmetry_tests.csv] [--votes_dir <reader_votes/<run>_Gothenburg>] \
        [--label "Held-out test"] [--out <file.csv>]
    For AVID: --figures <run>/validation/figures (or figures_AVID_unseen) and the AVID asymmetry / votes outputs.
Output (no IDs): discordant_summary_table.csv in --figures (or --out).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

MAIN = "Meta-temporal (main)"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--figures", required=True, help="Folder with the set's visual_read_*.csv outputs.")
    p.add_argument("--asymmetry_tests", default=None, help="asymmetry_tests.csv of the same set (optional).")
    p.add_argument("--votes_dir", default=None, help="analyze_reader_votes.py output folder of the same set (optional).")
    p.add_argument("--label", default="")
    p.add_argument("--out", default=None)
    return p.parse_args()


def fmt_p(p) -> str:
    if p is None or pd.isna(p):
        return ""
    return "< .001" if p < 0.001 else f"{p:.3f}".lstrip("0")


def fmt_or(o, lo, hi) -> str:
    return "" if pd.isna(o) else f"{o:.2f} [{lo:.2f}, {hi:.2f}]"


def read_csv(path: Path):
    return pd.read_csv(path) if path and path.exists() else None


def suvr_row(fig: Path) -> dict | None:
    conf = read_csv(fig / "visual_read_confidence_groups.csv")
    if conf is None or "MetaTemporal_SUVR" not in conf:
        return None
    rp = conf[conf["y"] == 1]
    neg, pos = rp.loc[rp["pred"] == 0, "MetaTemporal_SUVR"].dropna(), rp.loc[rp["pred"] == 1, "MetaTemporal_SUVR"].dropna()
    p = mannwhitneyu(neg, pos).pvalue if len(neg) and len(pos) else np.nan
    return {"characteristic": "Meta-temporal SUVR (tau amount)",
            "Model − / Read + vs Model + / Read +": f"median {neg.median():.2f} vs {pos.median():.2f} (n {len(neg)} / {len(pos)})",
            "plain p": fmt_p(p), "plain FDR": "", "same tau: OR of Model − [95% CI]": "(this is the tau level)",
            "same tau p": "", "source": "visual_read_confidence_groups.csv (Mann-Whitney computed here)"}


def subgroup_rows(fig: Path) -> list[dict]:
    tests, perf = read_csv(fig / "visual_read_subgroup_tests.csv"), read_csv(fig / "visual_read_subgroup_performance.csv")
    same = read_csv(fig / "visual_read_errors_at_same_suvr.csv")
    if tests is None:
        return []
    rows = []
    for _, t in tests[tests["metric"] == "sensitivity"].iterrows():
        var = t["variable"]
        rates = ""
        if perf is not None and var in set(perf["variable"]):
            lv = perf[(perf["variable"] == var) & (perf["tested"].astype(str) == "True")]
            rates = "; ".join(f"{r.level} {int(r.sensitivity_n - r.sensitivity_k)}/{int(r.sensitivity_n)}"
                              for r in lv.itertuples() if pd.notna(r.sensitivity_n) and r.sensitivity_n > 0)
        elif var == "Age":
            rates = f"median age {t['median_age_model_wrong']:.1f} vs {t['median_age_model_right']:.1f}"
        elif str(var).startswith("CDR-SB"):
            rates = f"median CDR-SB {t['median_cdrsb_model_wrong']} vs {t['median_cdrsb_model_right']}"
        row = {"characteristic": var, "Model − / Read + vs Model + / Read +": rates,
               "plain p": fmt_p(t["p_value"]), "plain FDR": fmt_p(t["p_value_fdr_bh"]),
               "same tau: OR of Model − [95% CI]": ("not run (plain test not significant)" if t["p_value_fdr_bh"] >= 0.05
                                                    else "not run in the same-tau analysis"), "same tau p": "",
               "source": f"visual_read_subgroup_tests.csv ({t['test']}, {t['family']} family)"}
        if same is not None:
            s = same[(same["metric"] == "sensitivity") & (same["variable"] == var) & (same["adjusted_for"] == MAIN)]
            if len(s):
                s = s.iloc[0]
                row["same tau: OR of Model − [95% CI]"] = fmt_or(s["or"], s["or_lo"], s["or_hi"]) or str(s.get("note", ""))
                row["same tau p"] = f"{fmt_p(s['p'])} (FDR {fmt_p(s.get('p_fdr_main'))})"
                row["source"] += " + visual_read_errors_at_same_suvr.csv"
        rows.append(row)
    return rows


def asymmetry_row(path: Path | None) -> dict | None:
    a = read_csv(path) if path else None
    if a is None:
        return None
    a = a[a["region"] == "MetaTemporal"]
    plain = a[a["outcome"].str.contains(r"^\|(?:AI|LI)\| detected vs missed", regex=True)]
    adj = a[a["outcome"] == "model call, read + only"]
    row = {"characteristic": "Left-right asymmetry, meta-temporal |LI| (%)",
           "Model − / Read + vs Model + / Read +": "", "plain p": "", "plain FDR": "",
           "same tau: OR of Model − [95% CI]": "", "same tau p": "",
           "source": "asymmetry_tests.csv (OR inverted from Model + to Model −)"}
    if len(plain):
        r = plain.iloc[0]
        row.update({"Model − / Read + vs Model + / Read +": f"median |LI| lower in Model − by {r['estimate']:.1f} %-points",
                    "plain p": fmt_p(r["p"]), "plain FDR": fmt_p(r["p_fdr"])})
    if len(adj):
        r = adj.iloc[0]
        row.update({"same tau: OR of Model − [95% CI]": fmt_or(1 / r["estimate"], 1 / r["hi"], 1 / r["lo"]) + " per SD",
                    "same tau p": f"{fmt_p(r['p'])} (FDR {fmt_p(r['p_fdr'])})"})
    return row


def grade_row(votes_dir: Path | None) -> dict | None:
    if not votes_dir:
        return None
    by, tests, grade = (read_csv(votes_dir / f) for f in
                        ["1_disagreement_by_group.csv", "1_tests.csv", "6_grade_at_same_suvr.csv"])
    row = {"characteristic": "Reader tau score: moderate vs frequent", "Model − / Read + vs Model + / Read +": "",
           "plain p": "", "plain FDR": "", "same tau: OR of Model − [95% CI]": "", "same tau p": "",
           "source": "reader votes 1_disagreement_by_group / 1_tests / 6_grade_at_same_suvr"}
    if by is not None:
        g = by[(by["read"] == "Read +") & by["group"].isin(["moderate", "frequent"])].set_index("group")
        if len(g) == 2:
            row["Model − / Read + vs Model + / Read +"] = "; ".join(
                f"{k} {int(float(g.loc[k, 'Model - / Read +']))}/{int(g.loc[k, 'scans'])}" for k in ["moderate", "frequent"])
    if tests is not None:
        t = tests[(tests["read"] == "Read +") & (tests["test"] == "moderate vs frequent")]
        if len(t):
            row["plain p"] = fmt_p(t.iloc[0]["Fisher p"])
    if grade is not None and "OR moderate vs frequent" in grade:
        r = grade[grade["adjusted_for"] == "MetaTemporal SUVR"]
        if len(r) and pd.notna(r.iloc[0]["OR moderate vs frequent"]):
            r = r.iloc[0]
            row["same tau: OR of Model − [95% CI]"] = fmt_or(r["OR moderate vs frequent"], r["OR lo"], r["OR hi"])
            row["same tau p"] = fmt_p(r["LR p"])
        elif len(r):
            row["same tau: OR of Model − [95% CI]"] = str(r.iloc[0].get("note", ""))
    return row


def main():
    args = parse_args()
    fig = Path(args.figures)
    rows = [suvr_row(fig), *subgroup_rows(fig), asymmetry_row(Path(args.asymmetry_tests) if args.asymmetry_tests else None),
            grade_row(Path(args.votes_dir) if args.votes_dir else None)]
    table = pd.DataFrame([r for r in rows if r])
    if table.empty:
        raise SystemExit(f"No inputs found in {fig}")
    out = Path(args.out) if args.out else fig / "discordant_summary_table.csv"
    table.to_csv(out, index=False)
    print(f"{args.label or fig}: Model − / Read + vs Model + / Read + (read + scans)")
    with pd.option_context("display.width", 250, "display.max_colwidth", 70):
        print(table.drop(columns="source").to_string(index=False))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
