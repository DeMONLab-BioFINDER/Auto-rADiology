#!/usr/bin/env python3

import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_plot_module():
    script_path = Path(__file__).resolve().parent / "plot_metatemporal_results.py"
    spec = importlib.util.spec_from_file_location("plot_metatemporal_results", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(description="Create true-vs-predicted scatter plots for AVID unseen validation runs.")
    parser.add_argument("--final_run_dir", required=True, help="Run directory for the final model.")
    parser.add_argument("--four_x_run_dir", required=True, help="Run directory for the 4x model.")
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


def infer_targets(df: pd.DataFrame) -> list[str]:
    targets = []
    for col in df.columns:
        if col.endswith("_y") and f"{col[:-2]}_pred" in df.columns:
            targets.append(col[:-2])
    return targets


def load_prediction_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed")]
    if "ID_ind" in df.columns:
        df["ID_ind"] = pd.to_numeric(df["ID_ind"], errors="coerce")
    return df


def plot_scatter(ax, df: pd.DataFrame, target: str, mod, model_label: str):
    y_true = pd.to_numeric(df[f"{target}_y"], errors="coerce").to_numpy()
    y_pred = pd.to_numeric(df[f"{target}_pred"], errors="coerce").to_numpy()
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[finite]
    y_pred = y_pred[finite]

    ax.scatter(
        y_true,
        y_pred,
        alpha=0.85,
        s=38,
        edgecolor="white",
        linewidth=0.45,
        color=mod.SEABORN_COLORS[0],
    )

    lo = float(np.nanmin([y_true.min(), y_pred.min()]))
    hi = float(np.nanmax([y_true.max(), y_pred.max()]))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1.2, alpha=0.5)

    if len(y_true) > 1:
        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)
        ax.plot([lo, hi], p([lo, hi]), linestyle="-", color="black", linewidth=1.8)
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        corr = np.nan

    display_target = mod.pretty_region_name(target)
    mod.style_axes(
        ax,
        f"{display_target} SUVR",
        f"Reference {display_target} SUVR",
        f"Predicted {display_target} SUVR",
    )
    ax.text(
        0.96,
        0.96,
        f"r = {corr:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )


def save_individual_plots(df: pd.DataFrame, targets: list[str], run_dir: Path, model_label: str, mod):
    out_root = run_dir / "figures_unseen_scatter"
    out_root.mkdir(parents=True, exist_ok=True)

    for target in targets:
        out_dir = out_root / target
        out_dir.mkdir(parents=True, exist_ok=True)

        y_true = pd.to_numeric(df[f"{target}_y"], errors="coerce").to_numpy()
        y_pred = pd.to_numeric(df[f"{target}_pred"], errors="coerce").to_numpy()
        finite = np.isfinite(y_true) & np.isfinite(y_pred)
        target_df = pd.DataFrame({"y": y_true[finite], "pred": y_pred[finite]})

        fig, ax = plt.subplots(figsize=(6.4, 6.4))
        ax.scatter(
            target_df["y"],
            target_df["pred"],
            alpha=0.85,
            s=38,
            edgecolor="white",
            linewidth=0.45,
            color=mod.SEABORN_COLORS[0],
        )
        lo = np.nanmin([target_df["y"].min(), target_df["pred"].min()])
        hi = np.nanmax([target_df["y"].max(), target_df["pred"].max()])
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1.2, alpha=0.5, label="Perfect prediction")
        z = np.polyfit(target_df["y"], target_df["pred"], 1)
        p = np.poly1d(z)
        ax.plot([lo, hi], p([lo, hi]), linestyle="-", color="black", linewidth=1.8, label="Fitted line")
        mod.style_legend(ax, loc="upper left")
        display_target = mod.pretty_region_name(target)
        mod.style_axes(
            ax,
            f"{model_label}: Reference vs Predicted\n{display_target} SUVR",
            f"Reference {display_target} SUVR",
            f"Predicted {display_target} SUVR",
        )
        corr = float(np.corrcoef(target_df["y"], target_df["pred"])[0, 1]) if len(target_df) > 1 else np.nan
        ax.text(
            0.96,
            0.96,
            f"r = {corr:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        mod.finalize_figure(fig)
        fig.savefig(out_dir / "true_vs_predicted.png", dpi=300)
        plt.close(fig)


def main():
    args = parse_args()
    mod = load_plot_module()
    mod.sns.set_theme(style="whitegrid", palette=mod.CUSTOM_PALETTE)

    runs = [
        (Path(args.final_run_dir).resolve(), "Final"),
        (Path(args.four_x_run_dir).resolve(), "4x"),
    ]

    tables = []
    target_sets = []
    for run_dir, model_label in runs:
        csv_path = find_results_csv(run_dir)
        df = load_prediction_table(csv_path)
        targets = infer_targets(df)
        if not targets:
            raise ValueError(f"Could not infer target columns from {csv_path}")
        tables.append((run_dir, model_label, df, csv_path, targets))
        target_sets.append(targets)

    common_targets = target_sets[0]
    for targets in target_sets[1:]:
        if targets != common_targets:
            raise ValueError(f"Target columns do not match across runs: {target_sets}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else runs[0][0] / "figures_unseen_scatter_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    for run_dir, model_label, df, _, targets in tables:
        save_individual_plots(df, targets, run_dir, model_label, mod)

    n_rows = len(tables)
    n_cols = len(common_targets)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 5.2 * n_rows), squeeze=False)

    for row_idx, (_, model_label, df, _, _) in enumerate(tables):
        for col_idx, target in enumerate(common_targets):
            ax = axes[row_idx][col_idx]
            plot_scatter(ax, df, target, mod, model_label)
            if row_idx == 0:
                ax.set_title(mod.wrap_plot_title(mod.pretty_region_name(target) + " SUVR"), fontsize=mod.TITLE_SIZE, pad=10)

        axes[row_idx][0].text(
            -0.18,
            0.5,
            model_label,
            transform=axes[row_idx][0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=mod.TITLE_SIZE,
            fontweight="bold",
        )

    fig.suptitle("AVID unseen validation: reference vs predicted", y=0.995, fontsize=mod.TITLE_SIZE)
    mod.finalize_figure(fig, rect=(0.0, 0.0, 0.98, 0.97))
    combined_path = out_dir / "true_vs_predicted_final_vs_4x.png"
    fig.savefig(combined_path, dpi=300)
    plt.close(fig)

    print(f"Saved combined comparison figure to: {combined_path}")
    for run_dir, model_label, _, csv_path, _ in tables:
        print(f"Saved per-target figures for {model_label} using {csv_path} under {run_dir / 'figures_unseen_scatter'}")


if __name__ == "__main__":
    main()