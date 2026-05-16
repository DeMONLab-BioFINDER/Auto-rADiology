#!/usr/bin/env python3

import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PREFERRED_TARGET_ORDER = ["MetaTemporal", "TemporoParietal", "Frontal", "MesialTemporal"]


def save_figure(fig: plt.Figure, out_path: Path, **kwargs) -> bool:
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


def load_plot_module():
    script_path = Path(__file__).resolve().parent / "plot_metatemporal_results.py"
    spec = importlib.util.spec_from_file_location("plot_metatemporal_results", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def draw_scatter(ax, y_true: np.ndarray, y_pred: np.ndarray, mod):
    ax.scatter(
        y_true,
        y_pred,
        alpha=0.85,
        s=38,
        edgecolor="white",
        linewidth=0.45,
        color=mod.SEABORN_COLORS[0],
    )


def draw_reference_and_fit_lines(ax, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    lo, hi = get_plot_limits(y_true, y_pred)
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1.2, alpha=0.5)

    if len(y_true) > 1:
        fit_coeffs = np.polyfit(y_true, y_pred, 1)
        fit_fn = np.poly1d(fit_coeffs)
        ax.plot([lo, hi], fit_fn([lo, hi]), linestyle="-", color="black", linewidth=1.8)
        return float(np.corrcoef(y_true, y_pred)[0, 1])

    return np.nan


def get_target_label(target: str, mod) -> str:
    display_target = mod.pretty_region_name(target)
    return mod.format_ctrz_label(display_target)


def hide_unused_axes(axes, start_idx: int, total_slots: int, n_cols: int):
    for idx in range(start_idx, total_slots):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        axes[row_idx][col_idx].set_visible(False)


def compute_per_region_metrics(df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    rows = []
    for target in targets:
        y_true, y_pred = get_valid_target_values(df, target)
        if len(y_true) == 0:
            rows.append(
                {
                    "target": target,
                    "n": 0,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "r2": np.nan,
                    "pearson_r": np.nan,
                }
            )
            continue

        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan
        corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else np.nan
        rows.append(
            {
                "target": target,
                "n": int(len(y_true)),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "pearson_r": corr,
            }
        )

    return pd.DataFrame(rows)


def plot_scatter(ax, df: pd.DataFrame, target: str, mod):
    y_true, y_pred = get_valid_target_values(df, target)
    draw_scatter(ax, y_true, y_pred, mod)
    corr = draw_reference_and_fit_lines(ax, y_true, y_pred)

    target_label = get_target_label(target, mod)
    mod.style_axes(
        ax,
        target_label,
        f"Reference {target_label}",
        f"Predicted {target_label}",
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

        y_true, y_pred = get_valid_target_values(df, target)

        fig, ax = plt.subplots(figsize=(6.4, 6.4))
        draw_scatter(ax, y_true, y_pred, mod)
        lo, hi = get_plot_limits(y_true, y_pred)
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1.2, alpha=0.5, label="Perfect prediction")
        fit_coeffs = np.polyfit(y_true, y_pred, 1)
        fit_fn = np.poly1d(fit_coeffs)
        ax.plot([lo, hi], fit_fn([lo, hi]), linestyle="-", color="black", linewidth=1.8, label="Fitted line")
        mod.style_legend(ax, loc="upper left")
        target_label = get_target_label(target, mod)
        mod.style_axes(
            ax,
            f"{model_label}: Reference vs Predicted\n{target_label}",
            f"Reference {target_label}",
            f"Predicted {target_label}",
        )
        corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else np.nan
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
        save_figure(fig, out_dir / "true_vs_predicted.png", dpi=300)
        plt.close(fig)


def main():
    args = parse_args()
    mod = load_plot_module()
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

    for run_dir, model_label, df, _, targets in tables:
        save_individual_plots(df, targets, run_dir, model_label, mod)

    if len(tables) == 1:
        n_cols = 2 if len(common_targets) > 1 else 1
        n_rows = int(np.ceil(len(common_targets) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 5.2 * n_rows), squeeze=False)
        _, model_label, df, _, _ = tables[0]

        for idx, target in enumerate(common_targets):
            r = idx // n_cols
            c = idx % n_cols
            ax = axes[r][c]
            plot_scatter(ax, df, target, mod)
            target_label = get_target_label(target, mod)
            ax.set_title(mod.wrap_plot_title(target_label), fontsize=mod.TITLE_SIZE, pad=10)

        hide_unused_axes(axes, len(common_targets), n_rows * n_cols, n_cols)
    else:
        n_rows = len(tables)
        n_cols = len(common_targets)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 5.2 * n_rows), squeeze=False)

        for row_idx, (_, model_label, df, _, _) in enumerate(tables):
            for col_idx, target in enumerate(common_targets):
                ax = axes[row_idx][col_idx]
                plot_scatter(ax, df, target, mod)
                if row_idx == 0:
                    target_label = get_target_label(target, mod)
                    ax.set_title(mod.wrap_plot_title(target_label), fontsize=mod.TITLE_SIZE, pad=10)

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

    fig.suptitle(
        f"AVID Unseen Test Set: Reference vs Predicted {mod.CTRZ_LABEL}",
        y=0.995,
        fontsize=mod.TITLE_SIZE,
    )
    mod.finalize_figure(fig, rect=(0.0, 0.0, 0.98, 0.97))
    combined_name = "true_vs_predicted_final_vs_4x.png" if len(runs) > 1 else "true_vs_predicted_final.png"
    combined_path = out_dir / combined_name
    save_figure(fig, combined_path, dpi=300)
    plt.close(fig)

    metrics_path = out_dir / "per_region_metrics.csv"
    metrics_df = compute_per_region_metrics(tables[0][2], common_targets)
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved per-region metrics to: {metrics_path}")

    print(f"Saved combined comparison figure to: {combined_path}")
    for run_dir, model_label, _, csv_path, _ in tables:
        print(f"Saved per-target figures for {model_label} using {csv_path} under {run_dir / 'figures_unseen_scatter'}")


if __name__ == "__main__":
    main()
