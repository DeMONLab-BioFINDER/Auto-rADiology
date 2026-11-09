#!/usr/bin/env python3
"""Do the model's disagreements with the visual read sit in scans the readers themselves were unsure of?

Each scan gets a reader-agreement group from the per-reader votes:
    unanimous     - 3 readers, all the same call (3-0)
    split 2-1     - 3 readers, one disagreed with the majority
    single reader - 1 reader (e.g. NACC, part of ADNI)
Positive reads are also split by the readers' tau score (visual_read_3_cat: 1 = moderate, 2 = frequent;
both count as positive). Exploratory only: the binary read is the label (advice from the data provider).

Per read (- and +) and group: number of scans, how many the model disagrees with, and the model's median
probability. Tests (pre-specified): within each read, Fisher's exact test of disagreement in unanimous vs
split 2-1 scans, and in 3-reader vs single-reader scans; for read + scans, moderate vs frequent (exploratory). Groups are small,
so read the counts first and the p-values second.

Usage:
    python results_plotting/analyze_reader_votes.py <run_dir> --votes IDs_reads.csv
    python results_plotting/analyze_reader_votes.py <run_dir> --votes AVIDIDs_reads.csv --dataset AVID_unseen
    python results_plotting/analyze_reader_votes.py <cv_run_dir> --votes IDs_reads.csv --oof
<run_dir> uses the run's held-out test (or zero-shot validation) predictions; with --oof, a CV run's
out-of-fold predictions (every training-pool scan once, predicted by the fold model that did not train on it).
Relative run dirs are looked up under <proj_path>/results.

Output (default <proj_path>/results/reader_votes/, outside the repo; no IDs):
    reader_votes_<run>_<dataset>.csv - the table that is printed
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from plot_utils import resolve_predictions_csv

GROUP_ORDER = ["unanimous", "split 2-1", "single reader"]
READ_COLS = ["read_1", "read_2", "read_3"]


def parse_args():
    repo = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="Run dir (final / final-val model), or a CV run dir with --oof.")
    p.add_argument("--votes", required=True, help="Per-reader vote csv (ID, read_1..3, visual_read_binary, visual_read_3_cat).")
    p.add_argument("--dataset", default="Gothenburg", help="Dataset name used in the result filenames.")
    p.add_argument("--oof", action="store_true", help="Use the CV run's oof_predictions.csv.")
    p.add_argument("--results_root", default=str(repo.parent / "results"))
    p.add_argument("--out_dir", default=str(repo.parent / "results" / "reader_votes"))
    return p.parse_args()


def load_votes(path: Path) -> pd.DataFrame:
    v = pd.read_csv(path)
    missing = {"ID", "visual_read_binary", "visual_read_3_cat", *READ_COLS} - set(v.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    reads = v[READ_COLS]
    n_reads = reads.notna().sum(axis=1)
    if not n_reads.isin([1, 3]).all():
        raise ValueError(f"{path}: expected 1 or 3 reads per scan, found {sorted(n_reads.unique())}")
    n_pos = (reads > 0).where(reads.notna()).sum(axis=1)
    if ((n_pos > n_reads / 2).astype(int) != v["visual_read_binary"]).any():
        raise ValueError(f"{path}: visual_read_binary is not the majority of the reads for some scans")
    unanimous = (n_pos == 0) | (n_pos == n_reads)
    v["agreement"] = np.select([n_reads == 1, unanimous], ["single reader", "unanimous"], "split 2-1")
    return v[["ID", "visual_read_binary", "visual_read_3_cat", "agreement"]]


def load_preds(args) -> pd.DataFrame:
    run_dir = Path(args.run)
    if not (run_dir.is_absolute() or run_dir.exists()):
        run_dir = Path(args.results_root) / run_dir
    path = run_dir / "oof_predictions.csv" if args.oof else resolve_predictions_csv(run_dir, args.dataset)[0]
    d = pd.read_csv(path)[["ID_ind", "y", "prob", "pred"]]
    print(f"{path}: {len(d)} scans")
    return d


def fisher(sub: pd.DataFrame, col: str, a, b) -> tuple[str, float]:
    """Fisher's exact test of disagreement rate between groups a and b of `col`."""
    x, y = sub[sub[col] == a]["disagrees"], sub[sub[col] == b]["disagrees"]
    if len(x) == 0 or len(y) == 0:
        return f"{a} vs {b}", np.nan
    table = [[x.sum(), len(x) - x.sum()], [y.sum(), len(y) - y.sum()]]
    return f"{a} vs {b}", fisher_exact(table)[1]


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
                             "% disagree": round(100 * s["disagrees"].mean(), 1),
                             "median prob": round(s["prob"].median(), 3)})
    table = pd.DataFrame(rows)
    for c in ["Model + / Read -", "Model - / Read +"]:
        if c in table:
            table[c] = table[c].astype("Int64").astype(object).where(table[c].notna(), "")

    tests = []
    for read, sub in df.groupby("read"):
        sub = sub.assign(n_readers=np.where(sub["agreement"] == "single reader", "1 reader", "3 readers"))
        pairs = [fisher(sub, "agreement", "unanimous", "split 2-1"), fisher(sub, "n_readers", "3 readers", "1 reader")]
        if read == "Read +":
            pairs.append(fisher(sub, "read category", "moderate", "frequent"))
        tests += [{"read": read, "test": t, "Fisher p": round(p, 4)} for t, p in pairs]
    tests = pd.DataFrame(tests)

    print(f"\n{args.dataset}{' (out-of-fold)' if args.oof else ''}: {len(df)} scans")
    print(table.to_string(index=False))
    print("\nDisagreement rate, Fisher's exact test:")
    print(tests.to_string(index=False))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"reader_votes_{Path(args.run).name}{'_oof' if args.oof else ''}_{args.dataset}.csv"
    pd.concat([table, tests]).to_csv(out_dir / name, index=False)
    print(f"\nSaved {out_dir / name}")


if __name__ == "__main__":
    main()
