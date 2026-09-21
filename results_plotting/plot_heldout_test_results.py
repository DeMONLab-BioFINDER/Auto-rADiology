#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns

from plot_utils import (
    CUSTOM_PALETTE,
    combine_dx_groups,
    find_demo_csv,
    format_suvr_label,
    infer_targets,
    make_bland_altman_panel,
    make_subgroup_mae_panel,
    make_true_vs_predicted_panel,
    resolve_existing_path,
    resolve_path,
)

DEFAULT_RESULTS_ROOT = Path("../../results")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create simple summary plots for MetaTemporal regression results."
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        default=None,
        help="Path to one results run directory. If omitted, use the newest MetaTemporal run.",
    )
    parser.add_argument(
        "--dataset",
        default="Gothenburg",
        help="Dataset name used in the validation filenames.",
    )
    parser.add_argument(
        "--target-name",
        default="MetaTemporal",
        help="Display name for axis labels and titles.",
    )
    parser.add_argument(
        "--demo_csv",
        default=None,
        help="Path to the demographics csv (defaults to data/demo.csv if found). Set this "
             "explicitly for a dataset with its own demo file, e.g. demo_test_subset_tau_raw.csv.",
    )
    return parser.parse_args()


def infer_latest_run_dir(results_root: Path, target_name: str) -> Path:
    if not results_root.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_root}")

    candidates = sorted(
        [
            path
            for path in results_root.iterdir()
            if path.is_dir() and target_name.lower() in path.name.lower()
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"Could not find any run folders in {results_root} matching {target_name!r}"
        )
    return candidates[0]


def load_results(run_dir: Path, dataset: str):
    evaluation_dir = run_dir / "evaluation" / dataset
    preds_path = resolve_existing_path(
        [
            run_dir / "validation" / dataset / f"Test_{dataset}_results.csv",
            evaluation_dir / f"Eval_{dataset}_results.csv",
        ],
        "predictions csv",
    )
    metrics_path = resolve_existing_path(
        [
            run_dir / "validation" / dataset / f"Test_{dataset}_metrics.csv",
            run_dir / "validation" / f"{dataset}Test_{dataset}_metrics.csv",
            evaluation_dir / f"Eval_{dataset}_metrics.csv",
        ],
        "metrics csv",
    )

    df_preds = pd.read_csv(preds_path)
    df_metrics = pd.read_csv(metrics_path)

    required_cols = {"y", "pred"}
    has_long_cols = required_cols.issubset(df_preds.columns)
    has_wide_cols = bool(infer_targets(df_preds))
    if not (has_long_cols or has_wide_cols):
        raise ValueError(
            f"Predictions csv has neither 'y'/'pred' columns nor '<target>_y'/'<target>_pred' columns: {preds_path}"
        )

    return df_metrics, df_preds


def wide_to_long_predictions(df_wide: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    """Reshape `<target>_y`/`<target>_pred` columns into one row per subject-target pair
    with generic `y`/`pred` columns, keeping all other (e.g. demographic) columns as-is."""
    id_vars = [c for c in df_wide.columns if not (c.endswith("_y") or c.endswith("_pred"))]
    frames = []
    for target in targets:
        y_col, pred_col = f"{target}_y", f"{target}_pred"
        if y_col not in df_wide.columns or pred_col not in df_wide.columns:
            continue
        sub = df_wide[id_vars + [y_col, pred_col]].rename(columns={y_col: "y", pred_col: "pred"})
        sub["target"] = target
        frames.append(sub)
    return pd.concat(frames, ignore_index=True) if frames else df_wide


def main():
    sns.set_theme(style="whitegrid", palette=CUSTOM_PALETTE)

    args = parse_args()
    if args.run_dir is None:
        run_dir = infer_latest_run_dir(DEFAULT_RESULTS_ROOT.resolve(), args.target_name)
    else:
        run_dir = Path(args.run_dir).resolve()
    out_dir = run_dir / "figures"
    out_dir.mkdir(exist_ok=True)

    _, df_preds = load_results(run_dir, args.dataset)
    demo_path = resolve_path(args.demo_csv, find_demo_csv)

    if demo_path is None:
        print("[WARNING] Could not find demographics csv for MAE panel.")
        return

    df_demo = pd.read_csv(demo_path)
    if "ID_ind" in df_preds.columns and "ID" in df_demo.columns:
        df_merged = pd.merge(df_preds, df_demo, left_on="ID_ind", right_on="ID", how="left")
    else:
        df_merged = pd.concat([df_preds.reset_index(drop=True), df_demo.reset_index(drop=True)], axis=1)

    if "dx" in df_merged.columns:
        df_merged["dx_grouped"] = combine_dx_groups(df_merged["dx"])

    targets = infer_targets(df_merged)
    if targets:
        # Multi-target regional-SUVR results: <target>_y/<target>_pred columns.
        make_true_vs_predicted_panel(
            df_merged,
            out_dir,
            targets,
            title=f"Test Set: Reference vs Predicted {format_suvr_label()}",
            filename="suvr_true_vs_predicted_test.png",
            stats_csv_path=out_dir / "suvr_true_vs_predicted_test_stats.csv",
        )
        make_bland_altman_panel(df_merged, out_dir, targets, filename="suvr_bland_altman_test.png")
        df_long = wide_to_long_predictions(df_merged, targets)
    else:
        # Legacy single-target runs already have plain y/pred columns.
        df_long = df_merged

    make_subgroup_mae_panel(df_long, out_dir, filename="suvr_mae_subgroup_test.png")
    print(f"[done] figures saved to {out_dir}")


if __name__ == "__main__":
    main()
