#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd

import plot_utils as mod


def parse_args():
    parser = argparse.ArgumentParser(description="Create true-vs-predicted scatter plots for AVID unseen validation runs.")
    parser.add_argument("--final_run_dir", required=True, help="Run directory for the final model.")
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Directory for the combined comparison figure. Defaults to a folder under the final run directory.",
    )
    return parser.parse_args()


def find_results_csv(run_dir: Path) -> Path:
    matches = sorted(run_dir.glob("External_validation_AVID_unseen__*_zeroshot_results.csv"))
    if not matches:
        raise FileNotFoundError(f"Could not find validation results CSV in {run_dir}")
    return matches[0]


def load_prediction_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed")]
    if "ID_ind" in df.columns:
        df["ID_ind"] = pd.to_numeric(df["ID_ind"], errors="coerce")
    return df


def main():
    args = parse_args()
    mod.sns.set_theme(style="whitegrid", palette=mod.CUSTOM_PALETTE)

    run_dir = Path(args.final_run_dir).resolve()
    csv_path = find_results_csv(run_dir)
    df = load_prediction_table(csv_path)
    targets = mod.infer_targets(df)
    if not targets:
        raise ValueError(f"Could not infer target columns from {csv_path}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir / "figures_unseen_scatter_panel"
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_path = out_dir / "true_vs_predicted_final_unseen.png"
    if combined_path.exists():
        print(f"[skip] {combined_path} exists")
        return

    mod.make_true_vs_predicted_panel(
        df,
        out_dir,
        targets,
        title="AVID Unseen Validation: Reference vs Predicted SUVR",
        filename=combined_path.name,
        stats_csv_path=out_dir / "true_vs_predicted_final_unseen_stats.csv",
    )
    mod.make_bland_altman_panel(df, out_dir, targets, filename="bland_altman_panel_unseen.png")
    print(f"Saved combined comparison figure to: {combined_path}")


if __name__ == "__main__":
    main()
