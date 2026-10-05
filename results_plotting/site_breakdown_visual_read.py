#!/usr/bin/env python3
"""Per-site (or other subgroup) breakdown of a visual-read run's classification
performance. An aggregate PPV/specificity can look strong while hiding poor
behavior on one or two specific sites - this reveals whether that's happening,
using the same stats already computed for the pooled run (see
compute_classification_stats_from_preds in plot_utils.py)."""

import argparse
from pathlib import Path

import pandas as pd

from plot_utils import (
    compute_classification_stats_from_preds,
    find_demo_csv,
    resolve_existing_path,
    resolve_path,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-site classification stats for a visual_read run (same run_dir/--dataset "
                     "conventions as plot_visual_read_results.py)."
    )
    parser.add_argument("run_dir", help="Path to a results run directory trained with --targets visual_read.")
    parser.add_argument("--dataset", default="Gothenburg", help="Dataset name used in the result filenames.")
    parser.add_argument("--group_col", default="site", help="Demographic column to break results down by.")
    parser.add_argument(
        "--demo_csv",
        default=None,
        help="Path to the demographics csv (defaults to data/demo.csv if found). Set this "
             "explicitly for a dataset with its own demo file, e.g. demo_AVID_unseen.csv.",
    )
    parser.add_argument(
        "--out_csv",
        default=None,
        help="Where to save the per-group stats csv (defaults next to the run's other figures).",
    )
    return parser.parse_args()


def find_zeroshot_results_csv(run_dir: Path, dataset: str) -> Path | None:
    """Results from a run_val.py --few_shot 0 external validation (e.g. AVID)."""
    matches = sorted((run_dir / "validation").glob(f"External_validation_{dataset}__*_zeroshot_results.csv"))
    return matches[0] if matches else None


def find_dataset_demo_csv(dataset: str) -> Path | None:
    script_path = Path(__file__).resolve()
    demo_candidates = [
        script_path.parents[1] / "data" / f"demo_{dataset}.csv",
        script_path.parents[2] / "data" / f"demo_{dataset}.csv",
    ]
    return next((p for p in demo_candidates if p.exists()), None)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()

    zeroshot_path = find_zeroshot_results_csv(run_dir, args.dataset)
    if zeroshot_path is not None:
        preds_path = zeroshot_path
        out_dir = run_dir / "validation" / "figures"
    else:
        evaluation_dir = run_dir / "evaluation" / args.dataset
        preds_path = resolve_existing_path(
            [
                run_dir / "validation" / args.dataset / f"Test_{args.dataset}_results.csv",
                evaluation_dir / f"Eval_{args.dataset}_results.csv",
            ],
            "predictions csv",
        )
        out_dir = run_dir / "figures"

    df_preds = pd.read_csv(preds_path)
    required_cols = {"ID_ind", "y", "pred"}
    missing = required_cols - set(df_preds.columns)
    if missing:
        raise ValueError(
            f"{preds_path} is missing columns {sorted(missing)} — "
            "this run may not have been trained with a visual_read classification target."
        )

    demo_path = resolve_path(args.demo_csv, lambda: find_dataset_demo_csv(args.dataset) or find_demo_csv())
    if demo_path is None:
        raise FileNotFoundError("Could not find a demographics csv — pass --demo_csv explicitly.")
    df_demo = pd.read_csv(demo_path)
    if args.group_col not in df_demo.columns:
        raise ValueError(f"'{args.group_col}' not found in {demo_path}")

    df = pd.merge(
        df_preds[["ID_ind", "y", "pred"]],
        df_demo[["ID", args.group_col]],
        left_on="ID_ind", right_on="ID", how="left",
    )
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["pred"] = pd.to_numeric(df["pred"], errors="coerce")
    df = df.dropna(subset=["y", "pred", args.group_col])
    if df.empty:
        raise ValueError("No rows left after merging predictions with demographics — check ID columns match.")

    rows = []
    for group_val, g in df.groupby(args.group_col):
        stats = compute_classification_stats_from_preds(
            g["y"].astype(int).to_numpy(), g["pred"].astype(int).to_numpy()
        )
        rows.append({args.group_col: group_val, **stats})

    overall = compute_classification_stats_from_preds(
        df["y"].astype(int).to_numpy(), df["pred"].astype(int).to_numpy()
    )
    rows.append({args.group_col: "ALL", **overall})

    df_out = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_csv) if args.out_csv else out_dir / f"visual_read_performance_stats_by_{args.group_col}.csv"
    df_out.to_csv(out_csv, index=False)

    print(df_out.to_string(index=False))
    print(f"[done] saved to {out_csv}")


if __name__ == "__main__":
    main()
