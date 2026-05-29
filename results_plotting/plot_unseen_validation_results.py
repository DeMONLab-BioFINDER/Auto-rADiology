#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_heldout_test_results as mod


PREFERRED_TARGET_ORDER = ["MetaTemporal", "MesialTemporal", "TemporoParietal", "Frontal"]


def save_figure(fig, out_path: Path, **kwargs) -> bool:
    if out_path.exists():
        print(f"[skip] {out_path} exists")
        return False
    fig.savefig(out_path, **kwargs)
    return True


def order_targets(targets: list[str]) -> list[str]:
    target_map = {t.lower(): t for t in targets}
    ordered = []
    for pref in PREFERRED_TARGET_ORDER:
        key = pref.lower()
        if key in target_map:
            ordered.append(target_map[key])
    remaining = [t for t in targets if t not in ordered]
    return ordered + remaining


def parse_args():
    parser = argparse.ArgumentParser(description="Create true-vs-predicted scatter plots for AVID unseen validation runs.")
    parser.add_argument("--final_run_dir", required=True, help="Run directory for the final model.")
    parser.add_argument("--four_x_run_dir", default=None, help="Run directory for the 4x model (optional).")
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


def get_valid_target_values(df: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray]:
    y_true = pd.to_numeric(df[f"{target}_y"], errors="coerce").to_numpy()
    y_pred = pd.to_numeric(df[f"{target}_pred"], errors="coerce").to_numpy()
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[valid], y_pred[valid]


def get_plot_limits(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    lo = float(np.nanmin([y_true.min(), y_pred.min()]))
    hi = float(np.nanmax([y_true.max(), y_pred.max()]))
    return lo, hi


def hide_unused_axes(axes, start_idx: int, total_slots: int, n_cols: int):
    for idx in range(start_idx, total_slots):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        axes[row_idx][col_idx].set_visible(False)


def add_panel_labels(axes, labels: list[str]):
    flat_axes = [ax for row in axes for ax in row]
    for ax, label in zip(flat_axes, labels):
        ax.annotate(
            label,
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(-14, 8),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            clip_on=False,
        )


def plot_scatter(ax, df: pd.DataFrame, target: str, mod):
    y_true, y_pred = get_valid_target_values(df, target)
    ax.scatter(
        y_true,
        y_pred,
        alpha=0.85,
        s=38,
        edgecolor="white",
        linewidth=0.45,
        color=mod.SEABORN_COLORS[0],
    )
    lo, hi = get_plot_limits(y_true, y_pred)
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1.2, alpha=0.5)
    if len(y_true) > 1:
        fit_coeffs = np.polyfit(y_true, y_pred, 1)
        fit_fn = np.poly1d(fit_coeffs)
        ax.plot([lo, hi], fit_fn([lo, hi]), linestyle="-", color="black", linewidth=1.8)
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        corr = np.nan

    mod.style_axes(ax, mod.pretty_region_name(target), "Reference SUVR", "Predicted SUVR")
    ax.text(
        0.96,
        0.96,
        f"r = {corr:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )


def main():
    args = parse_args()
    mod.sns.set_theme(style="whitegrid", palette=mod.CUSTOM_PALETTE)

    runs = [(Path(args.final_run_dir).resolve(), "Final")]
    if args.four_x_run_dir:
        runs.append((Path(args.four_x_run_dir).resolve(), "4x"))

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

    common_targets = order_targets(common_targets)

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    elif len(runs) > 1:
        out_dir = runs[0][0] / "figures_unseen_scatter_compare"
    else:
        out_dir = runs[0][0] / "figures_unseen_scatter_panel"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_cols = 2 if len(common_targets) > 1 else 1
    n_rows = int(np.ceil(len(common_targets) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 5.2 * n_rows), squeeze=False)

    _, model_label, df, _, _ = tables[0]
    for idx, target in enumerate(common_targets):
        r = idx // n_cols
        c = idx % n_cols
        plot_scatter(axes[r][c], df, target, mod)

    hide_unused_axes(axes, len(common_targets), n_rows * n_cols, n_cols)
    add_panel_labels(axes, [chr(ord("A") + i) for i in range(len(common_targets))])

    mod.finalize_figure(fig, rect=(0.0, 0.0, 0.99, 0.99))
    combined_path = out_dir / "true_vs_predicted_final_unseen.png"
    save_figure(fig, combined_path, dpi=300)
    plt.close(fig)

    print(f"Saved combined comparison figure to: {combined_path}")


if __name__ == "__main__":
    main()