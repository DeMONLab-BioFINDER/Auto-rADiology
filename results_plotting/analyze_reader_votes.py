#!/usr/bin/env python3
"""What do the per-reader votes say about the model's disagreements with the visual read?

Each scan gets a reader-agreement group from the per-reader votes:
    unanimous     - 3 readers, all the same call (3-0)
    split 2-1     - 3 readers, one disagreed with the majority
    single reader - 1 reader (all NACC, part of ADNI) - agreement unknown, never counted as unanimous
Positive reads are also split by the readers' tau score (visual_read_3_cat: 1 = moderate, 2 = frequent;
both count as positive). Exploratory only: the binary read is the label (advice from the data provider).

Checks (all at the run's own `pred`, i.e. its 0.5 threshold):
  1. Disagreement by group: per read (- and +), scans, disagreements and median probability per
     agreement group and tau score. Fisher's exact tests: unanimous vs split 2-1; 3 readers vs 1 reader
     (all sites, and within ADNI only, since single-reader scans are otherwise all NACC = a site
     difference); moderate vs frequent (read + only, exploratory).
  2. Unanimous-only rescoring: agreement with read + / read - and AUC on all scans vs on scans with
     3 readers who all agreed.
  3. Model vs one human reader: for 3-reader scans, each reader is compared with the other two whenever
     those two agree; the model is compared with the same two-reader consensus on the same scan-reader
     pairs. Descriptive (pairs from one scan are not independent).
  4. Probability vs number of positive votes (3-reader scans): per 0-3 positive votes, median probability,
     % called positive by the model and % in the uncertain band (0.3-0.7, the pre-set confidence cut-off).
     Spearman of probability vs votes within each read; Fisher's exact test of the uncertain-band share in
     split vs unanimous scans.
  5. Case list: every scan the model disagrees with, with its votes, site, diagnosis, amyloid, CDR-SB and
     regional SUVRs from the demographics csv (has IDs - not for slides). Printed: counts per vote group
     with median MetaTemporal SUVR.
Groups are small: read the counts first and the p-values second.

Usage:
    python results_plotting/analyze_reader_votes.py <run_dir> --votes IDs_reads.csv
    python results_plotting/analyze_reader_votes.py <run_dir> --votes AVIDIDs_reads.csv --dataset AVID_unseen
    python results_plotting/analyze_reader_votes.py <cv_run_dir> --votes IDs_reads.csv --oof
The demographics csv defaults to data/demo_<dataset>.csv, then data/demo.csv (in the repo or next to it).
<run_dir> uses the run's held-out test (or zero-shot validation) predictions; with --oof, a CV run's
out-of-fold predictions (every training-pool scan once, predicted by the fold model that did not train on it).
Relative run dirs are looked up under <proj_path>/results.

Output (default <proj_path>/results/reader_votes/<run>[_oof]_<dataset>/, outside the repo):
    1_disagreement_by_group.csv, 1_tests.csv, 2_unanimous_rescoring.csv, 3_model_vs_reader.csv,
    4_prob_by_votes.csv, 4_tests.csv (no IDs)
    5_disagreement_cases.csv (has IDs - keep off slides)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr
from sklearn.metrics import roc_auc_score

from plot_utils import resolve_path, resolve_predictions_csv

GROUP_ORDER = ["unanimous", "split 2-1", "single reader"]
READ_COLS = ["read_1", "read_2", "read_3"]
UNCERTAIN = (0.3, 0.7)
DEMO_COLS = ["site", "dx", "age", "amyloid_status", "CL", "CDR_SB",
             "MetaTemporal", "MesialTemporal", "TemporoParietal", "Frontal", "Universal"]


def parse_args():
    repo = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="Run dir (final / final-val model), or a CV run dir with --oof.")
    p.add_argument("--votes", required=True, help="Per-reader vote csv (ID, cohort, read_1..3, visual_read_binary, visual_read_3_cat).")
    p.add_argument("--dataset", default="Gothenburg", help="Dataset name used in the result filenames.")
    p.add_argument("--oof", action="store_true", help="Use the CV run's oof_predictions.csv.")
    p.add_argument("--demo_csv", default=None, help="Demographics csv for the case list (default: see above).")
    p.add_argument("--results_root", default=str(repo.parent / "results"))
    p.add_argument("--out_dir", default=str(repo.parent / "results" / "reader_votes"))
    return p.parse_args()


def load_votes(path: Path) -> pd.DataFrame:
    v = pd.read_csv(path)
    missing = {"ID", "cohort", "visual_read_binary", "visual_read_3_cat", *READ_COLS} - set(v.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    reads = v[READ_COLS]
    v["n_readers"] = reads.notna().sum(axis=1)
    if not v["n_readers"].isin([1, 3]).all():
        raise ValueError(f"{path}: expected 1 or 3 reads per scan, found {sorted(v['n_readers'].unique())}")
    v["positive_votes"] = (reads > 0).where(reads.notna()).sum(axis=1).astype(int)
    if ((v["positive_votes"] > v["n_readers"] / 2).astype(int) != v["visual_read_binary"]).any():
        raise ValueError(f"{path}: visual_read_binary is not the majority of the reads for some scans")
    unanimous = (v["positive_votes"] == 0) | (v["positive_votes"] == v["n_readers"])
    v["agreement"] = np.select([v["n_readers"] == 1, unanimous], ["single reader", "unanimous"], "split 2-1")
    return v[["ID", "cohort", "visual_read_binary", "visual_read_3_cat", *READ_COLS,
              "n_readers", "positive_votes", "agreement"]]


def find_dataset_demo_csv(dataset: str) -> Path | None:
    repo = Path(__file__).resolve().parent.parent
    candidates = [repo / "data" / f"demo_{dataset}.csv", repo.parent / "data" / f"demo_{dataset}.csv",
                  repo / "data" / "demo.csv", repo.parent / "data" / "demo.csv"]
    return next((p for p in candidates if p.exists()), None)


def load_preds(args) -> pd.DataFrame:
    run_dir = Path(args.run)
    if not (run_dir.is_absolute() or run_dir.exists()):
        run_dir = Path(args.results_root) / run_dir
    path = run_dir / "oof_predictions.csv" if args.oof else resolve_predictions_csv(run_dir, args.dataset)[0]
    d = pd.read_csv(path)[["ID_ind", "y", "prob", "pred"]]
    print(f"{path}: {len(d)} scans")
    return d


def fisher(sub: pd.DataFrame, col: str, a, b, outcome: str = "disagrees") -> tuple[str, float]:
    """Fisher's exact test of the rate of `outcome` between groups a and b of `col`."""
    x, y = sub[sub[col] == a][outcome], sub[sub[col] == b][outcome]
    if len(x) == 0 or len(y) == 0:
        return f"{a} vs {b}", np.nan
    table = [[x.sum(), len(x) - x.sum()], [y.sum(), len(y) - y.sum()]]
    return f"{a} vs {b}", fisher_exact(table)[1]


def pct(s: pd.Series) -> float:
    return round(100 * s.mean(), 1) if len(s) else np.nan


def disagreement_by_group(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for read, sub in df.groupby("read"):
        direction = "Model + / Read -" if read == "Read -" else "Model - / Read +"
        groups = [("agreement", g) for g in GROUP_ORDER]
        if read == "Read +":
            groups += [("read category", c) for c in ["moderate", "frequent"]]
        for col, g in groups:
            s = sub[sub[col] == g]
            if len(s):
                rows.append({"read": read, "group": g, "scans": len(s), direction: int(s["disagrees"].sum()),
                             "% disagree": pct(s["disagrees"]), "median prob": round(s["prob"].median(), 3)})
    table = pd.DataFrame(rows)
    for c in ["Model + / Read -", "Model - / Read +"]:
        if c in table:
            table[c] = table[c].astype("Int64").astype(object).where(table[c].notna(), "")

    tests = []
    for read, sub in df.groupby("read"):
        sub = sub.assign(readers=np.where(sub["n_readers"] == 1, "1 reader", "3 readers"))
        adni = sub[sub["cohort"] == "ADNI"]
        pairs = [("all", *fisher(sub, "agreement", "unanimous", "split 2-1")),
                 ("all", *fisher(sub, "readers", "3 readers", "1 reader")),
                 ("ADNI only", *fisher(adni, "readers", "3 readers", "1 reader"))]
        if read == "Read +":
            pairs.append(("all", *fisher(sub, "read category", "moderate", "frequent")))
        tests += [{"read": read, "scans": who, "test": t, "Fisher p": round(p, 4)} for who, t, p in pairs]
    return table, pd.DataFrame(tests)


def unanimous_rescoring(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, s in [("all scans", df), ("3 readers, unanimous", df[df["agreement"] == "unanimous"])]:
        auc = roc_auc_score(s["y"], s["prob"]) if s["y"].nunique() == 2 else np.nan
        rows.append({"scans": name, "n": len(s), "read +": int((s["y"] == 1).sum()),
                     "% agree, read +": pct(~s.loc[s["y"] == 1, "disagrees"]),
                     "% agree, read -": pct(~s.loc[s["y"] == 0, "disagrees"]),
                     "AUC": round(auc, 3)})
    return pd.DataFrame(rows)


def model_vs_reader(df: pd.DataFrame) -> pd.DataFrame:
    tri = df[df["n_readers"] == 3]
    votes = (tri[READ_COLS].to_numpy() > 0).astype(int)
    pred = tri["pred"].to_numpy()
    pairs = []
    for i in range(3):
        j, k = [x for x in range(3) if x != i]
        both = votes[:, j] == votes[:, k]
        consensus = votes[both, j]
        pairs.append(pd.DataFrame({"consensus": consensus, "reader agrees": votes[both, i] == consensus,
                                   "model agrees": pred[both] == consensus}))
    pairs = pd.concat(pairs)
    rows = []
    for name, s in [("other two say +", pairs[pairs["consensus"] == 1]),
                    ("other two say -", pairs[pairs["consensus"] == 0]), ("all", pairs)]:
        rows.append({"consensus": name, "scan-reader pairs": len(s),
                     "% one reader agrees": pct(s["reader agrees"]), "% model agrees": pct(s["model agrees"])})
    return pd.DataFrame(rows)


def prob_by_votes(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tri = df[df["n_readers"] == 3].assign(uncertain=lambda d: d["prob"].between(*UNCERTAIN))
    table = (tri.groupby("positive_votes")
             .agg(scans=("prob", "size"), median_prob=("prob", "median"),
                  pct_model_positive=("pred", lambda s: pct(s == 1)), pct_uncertain=("uncertain", pct))
             .round(3).reset_index().rename(columns={"positive_votes": "positive votes (of 3)"}))
    tests = []
    for read, sub in tri.groupby("read"):
        rho, p = spearmanr(sub["positive_votes"], sub["prob"]) if sub["positive_votes"].nunique() > 1 and sub["prob"].nunique() > 1 else (np.nan, np.nan)
        tests.append({"read": read, "test": "Spearman prob vs positive votes", "rho": round(rho, 3), "p": round(p, 4)})
    t, p = fisher(tri, "agreement", "split 2-1", "unanimous", outcome="uncertain")
    tests.append({"read": "both", "test": f"uncertain band share, {t} (Fisher)", "rho": np.nan, "p": round(p, 4)})
    return table, pd.DataFrame(tests)


def disagreement_cases(df: pd.DataFrame, demo: pd.DataFrame | None) -> pd.DataFrame:
    cases = df[df["disagrees"]].copy()
    cases["direction"] = np.where(cases["y"] == 0, "Model + / Read -", "Model - / Read +")
    cols = ["ID_ind", "direction", "prob", "cohort", "agreement", "positive_votes", "n_readers",
            "read category", *READ_COLS]
    cases = cases[cols]
    if demo is not None:
        demo_cols = [c for c in DEMO_COLS if c in demo.columns]
        cases = cases.merge(demo[["ID"] + demo_cols], left_on="ID_ind", right_on="ID", how="left").drop(columns="ID")
        n_missing = int(cases[demo_cols].isna().all(axis=1).sum())
        if n_missing:
            print(f"[WARNING] {n_missing} disagreement scans are not in the demographics csv.")
    return cases.sort_values(["direction", "agreement", "read category", "prob"])


def main():
    args = parse_args()
    preds, votes = load_preds(args), load_votes(Path(args.votes).expanduser())
    df = preds.merge(votes, left_on="ID_ind", right_on="ID", how="left")
    n_missing = int(df["agreement"].isna().sum())
    if n_missing:
        print(f"[WARNING] {n_missing} scans have no votes in {args.votes} - left out.")
        df = df.dropna(subset=["agreement"])
    bad = int((df["y"] != df["visual_read_binary"]).sum())
    if bad:
        raise ValueError(f"{bad} scans have a different visual read in the predictions and the vote file - "
                         "the IDs do not refer to the same scans.")
    df["disagrees"] = df["y"] != df["pred"]
    df["read"] = df["y"].map({0: "Read -", 1: "Read +"})
    df["read category"] = df["visual_read_3_cat"].map({0: "-", 1: "moderate", 2: "frequent"})

    out = {}
    out["1_disagreement_by_group"], out["1_tests"] = disagreement_by_group(df)
    out["2_unanimous_rescoring"] = unanimous_rescoring(df)
    out["3_model_vs_reader"] = model_vs_reader(df)
    out["4_prob_by_votes"], out["4_tests"] = prob_by_votes(df)
    demo_path = resolve_path(args.demo_csv, lambda: find_dataset_demo_csv(args.dataset))
    if demo_path is None:
        print("[WARNING] no demographics csv found - case list without site / SUVR (pass --demo_csv).")
    else:
        print(f"Demographics: {demo_path}")
    cases = disagreement_cases(df, pd.read_csv(demo_path) if demo_path else None)

    titles = {
        "1_disagreement_by_group": "1. Disagreement with the read, by reader agreement and tau score",
        "1_tests": "1. Disagreement rate, Fisher's exact test",
        "2_unanimous_rescoring": "2. All scans vs scans with 3 readers who all agreed",
        "3_model_vs_reader": "3. Agreement with the other two readers' consensus: one reader vs the model",
        "4_prob_by_votes": f"4. Model probability by number of positive votes (3-reader scans; uncertain = {UNCERTAIN[0]}-{UNCERTAIN[1]})",
        "4_tests": "4. Tests",
    }
    print(f"\n{args.dataset}{' (out-of-fold)' if args.oof else ''}: {len(df)} scans")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        for key, table in out.items():
            print(f"\n{titles[key]}:\n{table.to_string(index=False)}")
        print("\n5. Votes of the scans the model disagrees with:")
        g = cases.groupby(["direction", "agreement", "read category"])
        summary = g.size().rename("scans").to_frame()
        summary["median prob"] = g["prob"].median().round(3)
        if "MetaTemporal" in cases:
            summary["median MetaTemporal"] = g["MetaTemporal"].median().round(2)
        if "site" in cases:
            summary["sites"] = g["site"].agg(lambda s: ", ".join(f"{k} {v}" for k, v in s.value_counts().items()))
        print(summary.to_string())

    out_dir = Path(args.out_dir) / f"{Path(args.run).name}{'_oof' if args.oof else ''}_{args.dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, table in out.items():
        table.to_csv(out_dir / f"{key}.csv", index=False)
    cases.to_csv(out_dir / "5_disagreement_cases.csv", index=False)
    print(f"\nSaved to {out_dir} (5_disagreement_cases.csv has IDs - keep it off slides).")


if __name__ == "__main__":
    main()
