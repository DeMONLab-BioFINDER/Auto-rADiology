#!/usr/bin/env python3
"""Compare the discordant cases (model vs visual read) of two visual_read runs scan by scan, e.g.
the old raw input vs the 2 mm input, on the same held-out test scans (or AVID).

For each direction (Model + / Read - and Model - / Read +), every scan that either run gets
wrong is labelled:
    fixed        - run A wrong, run B right
    still wrong  - both wrong
    new          - run A right, run B wrong
Each run's own `pred` column is used as is (i.e. the decision threshold that run was evaluated at).
Descriptive only - the groups are too small for tests.

Usage:
    python results_plotting/compare_discordant_between_runs.py <raw_final_run> <2mm_final_run> --labels raw "2 mm"
    python results_plotting/compare_discordant_between_runs.py <raw_final_run> <2mm_final_run> --labels raw "2 mm" \\
        --dataset AVID_unseen --demo_csv data/demo_AVID_unseen.csv
Relative run dirs are looked up under <proj_path>/results.

Outputs (default <proj_path>/results/discordant_comparison, outside the repo):
    discordant_comparison_<dataset>.csv          - one row per scan either run gets wrong (has IDs: not for slides)
    discordant_comparison_by_site_<dataset>.csv  - fixed / still wrong / new per site and direction
The summary table is also printed.
"""
import argparse
from pathlib import Path

import pandas as pd

from plot_utils import CONFUSION_CATEGORY_LABELS, confusion_display_label, find_demo_csv, resolve_path, resolve_predictions_csv

DEMO_COLS = ["site", "age", "dx", "amyloid_status", "apoe", "MetaTemporal"]
DIRECTIONS = ["False Positive", "False Negative"]
STATUS_ORDER = ["fixed", "still wrong", "new"]


def parse_args():
    repo = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_a", help="Reference run (e.g. the old raw input).")
    p.add_argument("run_b", help="Run to compare against it (e.g. the 2 mm input).")
    p.add_argument("--labels", nargs=2, default=["A", "B"], metavar=("LABEL_A", "LABEL_B"))
    p.add_argument("--dataset", default="Gothenburg", help="Dataset name used in the result filenames.")
    p.add_argument("--demo_csv", default=None, help="Demographics csv (default: data/demo_<dataset>.csv, then data/demo.csv).")
    p.add_argument("--results_root", default=str(repo.parent / "results"))
    p.add_argument("--out_dir", default=str(repo.parent / "results" / "discordant_comparison"))
    return p.parse_args()


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
    df["category"] = [CONFUSION_CATEGORY_LABELS[(t, p)] for t, p in zip(df["y"], df["pred"])]
    print(f"{preds_path}: {len(df)} scans")
    return df


def main():
    args = parse_args()
    label_a, label_b = args.labels
    results_root, out_dir = Path(args.results_root), Path(args.out_dir)

    def run_dir(d):
        d = Path(d)
        return d if d.is_absolute() or d.exists() else results_root / d

    df_a, df_b = load_preds(run_dir(args.run_a), args.dataset), load_preds(run_dir(args.run_b), args.dataset)
    only_a = set(df_a["ID_ind"]) - set(df_b["ID_ind"])
    only_b = set(df_b["ID_ind"]) - set(df_a["ID_ind"])
    if only_a or only_b:
        print(f"[WARNING] Test sets differ: {len(only_a)} scans only in {label_a}, {len(only_b)} only in "
              f"{label_b}. Comparing the {len(df_a) - len(only_a)} shared scans only.")

    df = df_a.merge(df_b, on="ID_ind", suffixes=("_a", "_b"))
    if (df["y_a"] != df["y_b"]).any():
        raise ValueError(f"{int((df['y_a'] != df['y_b']).sum())} scans have a different visual read in the "
                         "two runs - the IDs do not refer to the same scans.")
    df = df.rename(columns={"y_a": "y"}).drop(columns="y_b")

    demo_path = resolve_path(args.demo_csv, lambda: find_dataset_demo_csv(args.dataset) or find_demo_csv())
    if demo_path is None:
        raise FileNotFoundError("Could not find a demographics csv - pass --demo_csv explicitly.")
    df_demo = pd.read_csv(demo_path)
    demo_cols = [c for c in DEMO_COLS if c in df_demo.columns]
    df = df.merge(df_demo[["ID"] + demo_cols], left_on="ID_ind", right_on="ID", how="left").drop(columns="ID")

    # Direction comes from the visual read: read - scans can only be Model + / Read -, read + only Model - / Read +.
    df["direction"] = df["y"].map({0: "False Positive", 1: "False Negative"})
    wrong_a, wrong_b = df["y"] != df["pred_a"], df["y"] != df["pred_b"]
    df["status"] = None
    df.loc[wrong_a & ~wrong_b, "status"] = "fixed"
    df.loc[wrong_a & wrong_b, "status"] = "still wrong"
    df.loc[~wrong_a & wrong_b, "status"] = "new"
    changed = df[df["status"].notna()].copy()
    changed["direction_label"] = changed["direction"].map(confusion_display_label)

    print(f"\n{args.dataset}: {len(df)} shared scans, {label_a} vs {label_b}\n")
    summary = []
    for direction in DIRECTIONS:
        sub = changed[changed["direction"] == direction]
        counts = sub["status"].value_counts()
        summary.append({
            "": confusion_display_label(direction),
            label_a: int(counts.get("fixed", 0) + counts.get("still wrong", 0)),
            f"fixed by {label_b}": int(counts.get("fixed", 0)),
            "still wrong": int(counts.get("still wrong", 0)),
            f"new in {label_b}": int(counts.get("new", 0)),
            label_b: int(counts.get("still wrong", 0) + counts.get("new", 0)),
        })
    print(pd.DataFrame(summary).to_string(index=False))

    medians = (changed.groupby(["direction_label", "status"])[[c for c in ["prob_a", "prob_b", "MetaTemporal", "age"] if c in changed]]
               .median().round(2).reindex(STATUS_ORDER, level="status"))
    medians.insert(0, "n", changed.groupby(["direction_label", "status"]).size())
    print("\nMedians per group (prob_a/prob_b = each run's probability):")
    print(medians.rename(columns={"prob_a": f"prob {label_a}", "prob_b": f"prob {label_b}"}).to_string())

    out_dir.mkdir(parents=True, exist_ok=True)
    if "site" in changed.columns:
        cols = [f"{confusion_display_label(d)}: {s}" for d in DIRECTIONS for s in STATUS_ORDER]
        by_site = (changed.assign(col=changed["direction_label"] + ": " + changed["status"])
                   .groupby(["site", "col"]).size().unstack(fill_value=0)
                   .reindex(columns=cols, fill_value=0))
        by_site.columns.name = None
        by_site.insert(0, "test scans", df.groupby("site").size().reindex(by_site.index))
        print("\nPer site:")
        print(by_site.to_string())
        by_site.to_csv(out_dir / f"discordant_comparison_by_site_{args.dataset}.csv")

    changed = changed.rename(columns={"prob_a": f"prob_{label_a}", "prob_b": f"prob_{label_b}",
                                      "pred_a": f"pred_{label_a}", "pred_b": f"pred_{label_b}"})
    changed = changed.drop(columns=["category_a", "category_b", "direction"]).sort_values(["direction_label", "status"])
    changed.to_csv(out_dir / f"discordant_comparison_{args.dataset}.csv", index=False)
    print(f"\nSaved to {out_dir} (the per-scan csv has IDs - keep them off slides).")


if __name__ == "__main__":
    main()
