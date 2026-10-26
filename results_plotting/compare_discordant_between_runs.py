#!/usr/bin/env python3
"""Compare the discordant cases (model vs visual read) of two or more visual_read runs scan by scan,
e.g. raw vs 2 mm vs MNI, on the same held-out test scans (or AVID).

Every scan that at least one run disagrees on is kept, in its direction (Model + / Visual read -
for read-negative scans, Model - / Visual read + for read-positive scans). Two views:

  Which runs disagree on it (the "pattern"), e.g. "all", "only raw", "raw + 2 mm". Scans every run
  disagrees on are probably borderline or unusual scans; scans only one run disagrees on point to
  something about that run's input.

  Each pair of runs A -> B (in the order given):
      now agrees        - A disagrees with the read, B agrees
      still disagrees   - both disagree
      newly disagrees   - A agrees, B disagrees

Each run's own `pred` column is used as is (i.e. the decision threshold that run was evaluated at).
Descriptive only - the groups are too small for tests.

Usage:
    python results_plotting/compare_discordant_between_runs.py <raw_final> <2mm_final> <MNI_final> \\
        --labels raw "2 mm" MNI
    python results_plotting/compare_discordant_between_runs.py <raw_final> <2mm_final> --labels raw "2 mm" \\
        --dataset AVID_unseen --demo_csv data/demo_AVID_unseen.csv
Relative run dirs are looked up under <proj_path>/results.

Outputs (default <proj_path>/results/discordant_comparison/<labels>, outside the repo):
    discordant_comparison_<dataset>.csv          - one row per scan any run disagrees on (has IDs: not for slides)
    discordant_patterns_<dataset>.csv            - per direction: how many scans each pattern has, with medians
    discordant_pairs_<dataset>.csv               - per direction and pair: now agrees / still / newly disagrees
    discordant_patterns_by_site_<dataset>.csv    - patterns per site and direction
The tables are also printed.
"""
import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd

from plot_utils import CONFUSION_CATEGORY_LABELS, confusion_display_label, find_demo_csv, resolve_path, resolve_predictions_csv

DEMO_COLS = ["site", "age", "dx", "amyloid_status", "apoe", "MetaTemporal"]
DIRECTIONS = ["False Positive", "False Negative"]
PAIR_ORDER = ["now agrees", "still disagrees", "newly disagrees"]


def parse_args():
    repo = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="Two or more run dirs, reference first (e.g. raw, then 2 mm, then MNI).")
    p.add_argument("--labels", nargs="+", default=None, help="One short name per run (default A, B, C, ...).")
    p.add_argument("--dataset", default="Gothenburg", help="Dataset name used in the result filenames.")
    p.add_argument("--demo_csv", default=None, help="Demographics csv (default: data/demo_<dataset>.csv, then data/demo.csv).")
    p.add_argument("--results_root", default=str(repo.parent / "results"))
    p.add_argument("--out_dir", default=None,
                   help="Output folder (default <proj_path>/results/discordant_comparison/<labels>).")
    args = p.parse_args()
    if len(args.runs) < 2:
        p.error("give at least two runs")
    args.labels = args.labels or [chr(ord("A") + i) for i in range(len(args.runs))]
    if len(args.labels) != len(args.runs) or len(set(args.labels)) != len(args.labels):
        p.error("--labels needs one distinct name per run")
    if args.out_dir is None:
        name = "_vs_".join(l.replace(" ", "") for l in args.labels)
        args.out_dir = str(repo.parent / "results" / "discordant_comparison" / name)
    return args


def find_dataset_demo_csv(dataset: str) -> Path | None:
    repo = Path(__file__).resolve().parent.parent
    candidates = [repo / "data" / f"demo_{dataset}.csv", repo.parent / "data" / f"demo_{dataset}.csv"]
    return next((p for p in candidates if p.exists()), None)


def load_preds(run_dir: Path, dataset: str) -> pd.DataFrame:
    preds_path, _ = resolve_predictions_csv(run_dir, dataset)
    df = pd.read_csv(preds_path)
    missing = {"ID_ind", "y", "prob", "pred"} - set(df.columns)
    if missing:
        raise ValueError(f"{preds_path} is missing columns {sorted(missing)} - not a visual_read results csv.")
    df = df[["ID_ind", "y", "prob", "pred"]].apply(pd.to_numeric, errors="coerce").dropna()
    df[["ID_ind", "y", "pred"]] = df[["ID_ind", "y", "pred"]].astype(int)
    print(f"{preds_path}: {len(df)} scans")
    return df


def pattern_name(disagree: list[bool], labels: list[str]) -> str:
    who = [l for l, d in zip(labels, disagree) if d]
    if len(who) == len(labels):
        return "all"
    return ("only " if len(who) == 1 else "") + " + ".join(who)


def pattern_order(labels: list[str]) -> list[str]:
    """'all' first, then fewer and fewer runs, each in the order given."""
    n = len(labels)
    out = ["all"]
    for k in range(n - 1, 0, -1):
        for idx in combinations(range(n), k):
            out.append(pattern_name([i in idx for i in range(n)], labels))
    return out


def main():
    args = parse_args()
    labels = args.labels
    results_root, out_dir = Path(args.results_root), Path(args.out_dir)

    def run_dir(d):
        d = Path(d)
        return d if d.is_absolute() or d.exists() else results_root / d

    runs = [load_preds(run_dir(r), args.dataset) for r in args.runs]
    shared = set.intersection(*[set(r["ID_ind"]) for r in runs])
    for label, r in zip(labels, runs):
        if len(r) != len(shared):
            print(f"[WARNING] {label} has {len(r) - len(shared)} scans not in every run - comparing the "
                  f"{len(shared)} shared scans only.")

    df = runs[0][runs[0]["ID_ind"].isin(shared)][["ID_ind", "y"]].copy()
    for label, r in zip(labels, runs):
        r = r.rename(columns={"y": f"y_{label}", "prob": f"prob_{label}", "pred": f"pred_{label}"})
        df = df.merge(r, on="ID_ind")
        bad = int((df["y"] != df[f"y_{label}"]).sum())
        if bad:
            raise ValueError(f"{bad} scans have a different visual read in {labels[0]} and {label} - "
                             "the IDs do not refer to the same scans.")
        df = df.drop(columns=f"y_{label}")
        df[f"disagrees_{label}"] = df["y"] != df[f"pred_{label}"]

    demo_path = resolve_path(args.demo_csv, lambda: find_dataset_demo_csv(args.dataset) or find_demo_csv())
    if demo_path is None:
        raise FileNotFoundError("Could not find a demographics csv - pass --demo_csv explicitly.")
    df_demo = pd.read_csv(demo_path)
    demo_cols = [c for c in DEMO_COLS if c in df_demo.columns]
    df = df.merge(df_demo[["ID"] + demo_cols], left_on="ID_ind", right_on="ID", how="left").drop(columns="ID")

    # Direction comes from the visual read: read - scans can only be Model + / Read -, read + only Model - / Read +.
    df["direction"] = df["y"].map({0: "False Positive", 1: "False Negative"})
    dis_cols = [f"disagrees_{l}" for l in labels]
    any_dis = df[df[dis_cols].any(axis=1)].copy()
    any_dis["direction_label"] = any_dis["direction"].map(confusion_display_label)
    any_dis["disagreeing runs"] = [pattern_name(list(row), labels) for row in any_dis[dis_cols].to_numpy()]
    order = pattern_order(labels)

    print(f"\n{args.dataset}: {len(df)} shared scans, runs: {', '.join(labels)}")
    n_read = df["direction"].value_counts()
    for direction in DIRECTIONS:
        lab = confusion_display_label(direction)
        totals = ", ".join(f"{l} {int(any_dis.loc[any_dis['direction'] == direction, f'disagrees_{l}'].sum())}"
                           for l in labels)
        print(f"  {lab} (out of {int(n_read.get(direction, 0))} read {'-' if direction == 'False Positive' else '+'} "
              f"scans): {totals}")

    # Patterns: which runs disagree, with medians
    med_cols = [f"prob_{l}" for l in labels] + [c for c in ["MetaTemporal", "age"] if c in any_dis]
    pat = (any_dis.groupby(["direction_label", "disagreeing runs"])[med_cols].median().round(2))
    pat.insert(0, "n", any_dis.groupby(["direction_label", "disagreeing runs"]).size())
    pat = pat.reindex([(confusion_display_label(d), o) for d in DIRECTIONS for o in order]).dropna(subset=["n"])
    pat["n"] = pat["n"].astype(int)
    print("\nWhich runs disagree with the visual read (medians; prob = each run's probability of positive):")
    print(pat.to_string())

    # Pairs A -> B
    pair_rows = []
    for (la, lb) in combinations(labels, 2):
        da, db = any_dis[f"disagrees_{la}"], any_dis[f"disagrees_{lb}"]
        status = pd.Series(None, index=any_dis.index, dtype=object)
        status[da & ~db], status[da & db], status[~da & db] = PAIR_ORDER
        for direction in DIRECTIONS:
            s = status[any_dis["direction"] == direction].value_counts()
            pair_rows.append({"direction": confusion_display_label(direction), "pair": f"{la} -> {lb}",
                              f"disagree in A": int(s.get("now agrees", 0) + s.get("still disagrees", 0)),
                              **{k: int(s.get(k, 0)) for k in PAIR_ORDER},
                              f"disagree in B": int(s.get("still disagrees", 0) + s.get("newly disagrees", 0))})
    pairs = pd.DataFrame(pair_rows)
    print("\nPairs A -> B (now agrees = A disagreed, B agrees with the read):")
    print(pairs.to_string(index=False))

    out_dir.mkdir(parents=True, exist_ok=True)
    pat.to_csv(out_dir / f"discordant_patterns_{args.dataset}.csv")
    pairs.to_csv(out_dir / f"discordant_pairs_{args.dataset}.csv", index=False)
    if "site" in any_dis.columns:
        cols = [f"{confusion_display_label(d)}: {o}" for d in DIRECTIONS for o in order]
        by_site = (any_dis.assign(col=any_dis["direction_label"] + ": " + any_dis["disagreeing runs"])
                   .groupby(["site", "col"]).size().unstack(fill_value=0))
        by_site = by_site.reindex(columns=[c for c in cols if c in by_site.columns])
        by_site.columns.name = None
        by_site.insert(0, "test scans", df.groupby("site").size().reindex(by_site.index))
        print("\nPer site (scans in each pattern):")
        print(by_site.to_string())
        by_site.to_csv(out_dir / f"discordant_patterns_by_site_{args.dataset}.csv")

    any_dis["disagreeing runs"] = pd.Categorical(any_dis["disagreeing runs"], categories=order, ordered=True)
    any_dis = (any_dis.drop(columns=["direction"]).sort_values(["direction_label", "disagreeing runs"])
               .rename(columns={"y": "visual_read"}))
    first = ["ID_ind", "direction_label", "disagreeing runs", "visual_read"]
    any_dis = any_dis[first + [c for c in any_dis.columns if c not in first]]
    any_dis.to_csv(out_dir / f"discordant_comparison_{args.dataset}.csv", index=False)
    print(f"\nSaved to {out_dir} (the per-scan csv has IDs - keep them off slides).")


if __name__ == "__main__":
    main()
