#!/usr/bin/env python3

import argparse
from itertools import combinations
from pathlib import Path
import re
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy import stats


DEFAULT_RESULTS_ROOT = Path("../../results")
CUSTOM_PALETTE = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#9467bd",  # purple
    "#2ca02c",  # green
    "#f1c40f",  # yellow
    "#ff7f0e",  # orange
    "#e377c2",  # pink
]
SEABORN_COLORS = sns.color_palette(CUSTOM_PALETTE)
SEABORN_CMAP = sns.color_palette("viridis", as_cmap=True)

TITLE_SIZE = 17
LABEL_SIZE = 15
TICK_SIZE = 12
LEGEND_SIZE = 11
CTRZ_LABEL = "CTR$_{z}$"


def combine_dx_groups(s: pd.Series) -> pd.Series:
    s = s.astype("string").fillna("NA")
    s_norm = s.str.strip().str.lower()
    mask_other = s_norm.str.startswith(
        ("others", "other neurodegenerative"),
        na=False,
    )
    out = s.copy()
    out.loc[mask_other] = "others"
    return out


def fdr_bh(p_values: pd.Series) -> np.ndarray:
    p = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    n = len(p)
    if n == 0:
        return np.array([])

    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    out = np.empty(n, dtype=float)
    out[order] = adjusted
    return out


def get_display_order(values: pd.Series, col: str):
    present = set(values.astype("string").fillna("NA"))
    preferred_orders = {
        "gender": ["1", "2", "NA"],
        "amyloid_status": ["0.0", "1.0", "NA", "0", "1"],
        "apoe": ["0.0", "1.0", "NA", "0", "1"],
    }

    if col in preferred_orders:
        order = [x for x in preferred_orders[col] if x in present]
        remaining = [x for x in values.value_counts().index if x not in order]
        return order + remaining

    return list(values.value_counts().index)


def get_category_palette(order):
    palette = {}
    for i, value in enumerate(order):
        palette[value] = CUSTOM_PALETTE[i % len(CUSTOM_PALETTE)]
    return palette


def pretty_region_name(name: str) -> str:
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(name))
    return re.sub(r"\s+", " ", name).strip()


def format_ctrz_label(display_target=None) -> str:
    if display_target:
        return f"{display_target} {CTRZ_LABEL}"
    return CTRZ_LABEL


def wrap_plot_title(title: str, width: int = 36) -> str:
    return textwrap.fill(str(title), width=width, break_long_words=False, break_on_hyphens=False)


def finalize_figure(fig, *, rect=(0.0, 0.0, 0.97, 0.97)):
    fig.tight_layout(rect=rect)


def save_figure(fig, out_path: Path, **kwargs) -> bool:
    if out_path.exists():
        print(f"[skip] {out_path} exists")
        return False
    fig.savefig(out_path, **kwargs)
    return True


def style_axes(ax, title: str, xlabel: str, ylabel: str, *, xrotation: int = 0):
    ax.set_title(wrap_plot_title(title), fontsize=TITLE_SIZE, pad=12, loc="center")
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.tick_params(axis="x", rotation=xrotation, labelsize=TICK_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_SIZE)
    ax.grid(alpha=0.28)


def style_legend(ax, **kwargs):
    legend = ax.legend(fontsize=LEGEND_SIZE, frameon=False, **kwargs)
    if legend is not None and legend.get_title() is not None:
        legend.get_title().set_fontsize(LEGEND_SIZE)
    return legend


def make_group_mae_boxstrip_plots(
    df: pd.DataFrame,
    out_dir: Path,
    group_cols: list[str],
    target_name=None,
):
    df_plot = df.copy()
    df_plot["abs_error"] = (df_plot["pred"] - df_plot["y"]).abs()
    display_target = pretty_region_name(target_name) if target_name else None
    target_label = format_ctrz_label(display_target) if display_target else CTRZ_LABEL

    for col in group_cols:
        if col not in df_plot.columns:
            continue

        g = df_plot[[col, "abs_error"]].copy()
        if col == "age":
            age = pd.to_numeric(g[col], errors="coerce")
            age_bins = [50, 60, 70, 80, 90, 110]
            age_labels = ["50-60", "60-70", "70-80", "80-90", "90+"]
            g[col] = pd.cut(
                age,
                bins=age_bins,
                labels=age_labels,
                include_lowest=True,
                right=False,
            )

        g[col] = g[col].astype("string").fillna("NA")
        keep = g[col].value_counts()
        min_n_for_plot = 2 if col == "dx_grouped" else 5
        keep = keep[keep >= min_n_for_plot].index
        g = g[g[col].isin(keep)]
        if g.empty or g[col].nunique() < 2:
            continue

        if col == "age":
            age_order = ["50-60", "60-70", "70-80", "80-90", "90+"]
            present = set(g[col].astype("string"))
            order = [x for x in age_order if x in present]
        else:
            order = get_display_order(g[col], col)
        palette_map = get_category_palette(order)

        fig, ax = plt.subplots(figsize=(6.4, 6.4))
        sns.boxplot(
            data=g,
            x=col,
            y="abs_error",
            hue=col,
            order=order,
            ax=ax,
            palette=palette_map,
            legend=False,
            dodge=False,
            fliersize=0,
            width=0.46,
            linewidth=1.3,
            showmeans=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "black",
                "markeredgecolor": "white",
                "markersize": 6.5,
                "markeredgewidth": 0.7,
                "zorder": 6,
            },
        )
        sns.stripplot(
            data=g,
            x=col,
            y="abs_error",
            hue=col,
            order=order,
            ax=ax,
            palette=palette_map,
            alpha=0.65,
            jitter=0.14,
            size=4.8,
            edgecolor="white",
            linewidth=0.5,
            legend=False,
        )
        col_display = "dx" if col == "dx_grouped" else col
        rotation = 25 if col == "dx_grouped" else 0
        if display_target:
            title = f"{target_label} Absolute Error by {col_display}"
        else:
            title = f"Absolute Error by {col_display}"
        label_map = {
            "apoe": "APOE",
            "amyloid_status": "Amyloid Status",
            "cdr": "CDR",
        }
        if col_display == "dx":
            x_label = "DX"
        else:
            key = col_display.lower()
            x_label = label_map.get(key, col_display.replace("_", " ").title())
        style_axes(
            ax,
            title,
            x_label,
            f"Absolute Error ({CTRZ_LABEL})",
            xrotation=rotation,
        )
        ax.margins(x=0.03)
        ax.grid(axis="y", alpha=0.25)
        finalize_figure(fig)
        save_figure(fig, out_dir / f"mae_boxstrip_by_{col}.png", dpi=300)
        plt.close(fig)


def run_group_significance_tests(df: pd.DataFrame, out_dir: Path, group_cols: list[str]):
    df_stat = df.copy()
    df_stat["abs_error"] = (df_stat["pred"] - df_stat["y"]).abs()
    rows = []

    for col in group_cols:
        if col not in df_stat.columns:
            continue

        g = df_stat[[col, "abs_error"]].copy()
        if col == "age":
            age = pd.to_numeric(g[col], errors="coerce")
            g[col] = pd.cut(
                age,
                bins=[50, 60, 70, 80, 90, 110],
                labels=["50-60", "60-70", "70-80", "80-90", "90+"],
                include_lowest=True,
                right=False,
            )

        g[col] = g[col].astype("string").fillna("NA")
        groups = {k: v["abs_error"].dropna().to_numpy() for k, v in g.groupby(col)}
        groups = {k: v for k, v in groups.items() if len(v) >= 5}
        keys = sorted(groups.keys())
        if len(keys) < 2:
            continue

        for a, b in combinations(keys, 2):
            t, p = stats.ttest_ind(groups[a], groups[b], equal_var=False)
            rows.append(
                {
                    "group": col,
                    "group_a": a,
                    "group_b": b,
                    "n_a": len(groups[a]),
                    "n_b": len(groups[b]),
                    "mae_a": float(np.mean(groups[a])),
                    "mae_b": float(np.mean(groups[b])),
                    "delta_mae": float(np.mean(groups[a]) - np.mean(groups[b])),
                    "t_stat": float(t),
                    "p_value": float(p),
                }
            )

    if rows:
        out = pd.DataFrame(rows).sort_values(["group", "p_value"])
        out["p_value_fdr_bh"] = fdr_bh(out["p_value"])
        out["significant_fdr_0_05"] = out["p_value_fdr_bh"] < 0.05
        out.to_csv(out_dir / "group_significance_ttests_mae.csv", index=False)


def make_site_raw_value_plot(df: pd.DataFrame, out_dir: Path, target_name: str):
    if "site" not in df.columns:
        return
    display_target = pretty_region_name(target_name)
    target_label = format_ctrz_label(display_target)
    tmp = df[["site", "y", "pred"]].copy()
    tmp["site"] = tmp["site"].astype("string").fillna("NA")
    long_df = tmp.melt(id_vars="site", value_vars=["y", "pred"], var_name="value_type", value_name="value")
    long_df["value_type"] = long_df["value_type"].map({"y": "Reference", "pred": "Predicted"})

    site_order = get_display_order(tmp["site"], "site")
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    sns.boxplot(
        data=long_df,
        x="site",
        y="value",
        hue="value_type",
        order=site_order,
        palette=[SEABORN_COLORS[0], SEABORN_COLORS[3]],
        fliersize=0,
        width=0.55,
        ax=ax,
    )
    sns.stripplot(
        data=long_df,
        x="site",
        y="value",
        hue="value_type",
        order=site_order,
        dodge=True,
        alpha=0.65,
        size=5.0,
        palette=[SEABORN_COLORS[0], SEABORN_COLORS[3]],
        edgecolor="white",
        linewidth=0.45,
        ax=ax,
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[:2], labels[:2], title="Value", loc="best", fontsize=LEGEND_SIZE, frameon=False)
    style_axes(
        ax,
        f"Reference and Predicted {target_label} by Site",
        "Site",
        target_label,
        xrotation=20,
    )
    ax.grid(axis="y", alpha=0.25)
    finalize_figure(fig)
    save_figure(fig, out_dir / "site_raw_value_distribution.png", dpi=300)
    plt.close(fig)


def make_site_error_correlation_plot(df: pd.DataFrame, out_dir: Path, target_name: str):
    if "site" not in df.columns:
        return
    display_target = pretty_region_name(target_name)
    target_label = format_ctrz_label(display_target)

    tmp = df[["site", "y", "pred"]].copy()
    tmp["site"] = tmp["site"].astype("string").fillna("NA")
    tmp["abs_error"] = (tmp["pred"] - tmp["y"]).abs()
    site_order = get_display_order(tmp["site"], "site")
    site_palette = get_category_palette(site_order)

    fig, ax = plt.subplots(figsize=(7.8, 6.6))
    rows = []

    for site in site_order:
        site_df = tmp[tmp["site"] == site].copy()
        if site_df.empty:
            continue

        color = site_palette[site]
        ax.scatter(
            site_df["y"],
            site_df["abs_error"],
            s=34,
            alpha=0.45,
            color=color,
            edgecolor="white",
            linewidth=0.25,
        )

        if len(site_df) >= 2:
            z = np.polyfit(site_df["y"], site_df["abs_error"], 1)
            p = np.poly1d(z)
            xline = np.linspace(site_df["y"].min(), site_df["y"].max(), 100)
            ax.plot(xline, p(xline), color=color, linewidth=2.3, label=site)
            corr = np.corrcoef(site_df["y"], site_df["abs_error"])[0, 1]
        else:
            corr = np.nan

        rows.append(
            {
                "site": site,
                "n": len(site_df),
                "pearson_r_true_suvr_vs_abs_error": corr,
                "mae": float(site_df["abs_error"].mean()),
            }
        )

    style_axes(
        ax,
        f"Absolute Error vs Reference\n{target_label} by Site",
        f"Reference {target_label}",
        f"Absolute Error ({CTRZ_LABEL})",
    )
    style_legend(ax, title="site", loc="upper right")
    finalize_figure(fig)
    save_figure(fig, out_dir / "site_error_vs_true_correlation.png", dpi=300)
    plt.close(fig)

    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "site_error_vs_true_correlation.csv", index=False)


def make_training_distribution_plot(run_dir: Path, out_dir: Path, target_name: str):
    train_candidates = [
        run_dir / "Hold-out_training-set.csv",
        run_dir / "splits" / "Hold-out_training-set.csv",
        run_dir / "train-test-split" / "train-test-split_training-set.csv",
        run_dir / "train-test-split" / "preds" / "train-test-split_training-set.csv",
    ]
    train_path = next((path for path in train_candidates if path.exists()), None)
    if train_path is None:
        return
    display_target = pretty_region_name(target_name)
    target_label = format_ctrz_label(display_target)
    dft = pd.read_csv(train_path)
    if "dx" in dft.columns:
        dft["dx_grouped"] = combine_dx_groups(dft["dx"])

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.5))
    axes = axes.flatten()

    if target_name in dft.columns:
        sns.histplot(dft[target_name], bins=20, kde=True, ax=axes[0], color=SEABORN_COLORS[0])
        style_axes(axes[0], f"Training Set\n{target_label}", target_label, "Count")
    else:
        axes[0].set_visible(False)

    cat_cols = ["site", "dx_grouped", "gender", "amyloid_status", "apoe"]
    for i, col in enumerate(cat_cols, start=1):
        if i >= len(axes):
            break
        if col in dft.columns:
            tmp = dft[[col]].copy()
            tmp[col] = tmp[col].astype("string").fillna("NA")
            order = get_display_order(tmp[col], col)
            sns.countplot(data=tmp, x=col, order=order, ax=axes[i], color=SEABORN_COLORS[0])

            label = "dx" if col == "dx_grouped" else col
            if col == "dx_grouped":
                style_axes(axes[i], f"Training\ndistribution: {label}", label, "Count (log scale)", xrotation=35)
                axes[i].set_yscale("log")
                axes[i].tick_params(axis="x", rotation=35, labelsize=8)
                for tick in axes[i].get_xticklabels():
                    tick.set_horizontalalignment("right")
            else:
                style_axes(axes[i], f"Training\ndistribution: {label}", label, "Count", xrotation=0)
                axes[i].tick_params(axis="x", rotation=0, labelsize=9)

            counts = tmp[col].value_counts().reindex(order)
            for patch, count in zip(axes[i].patches, counts):
                axes[i].text(
                    patch.get_x() + patch.get_width() / 2,
                    patch.get_height(),
                    str(int(count)),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=0,
                )
        else:
            axes[i].set_visible(False)

    finalize_figure(fig)
    save_figure(fig, out_dir / "training_distribution_overview.png", dpi=300)
    plt.close(fig)

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
    return parser.parse_args()


def resolve_existing_path(candidates, label: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {label}. Checked:\n{searched}")


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
    curve_path = resolve_existing_path(
        [
            run_dir / "train-test-split" / "trainning_metrics_per_epoch.csv",
            run_dir / "train-test-split" / "metrics" / "trainning_metrics_per_epoch.csv",
        ],
        "training curve csv",
    )

    df_preds = pd.read_csv(preds_path)
    df_metrics = pd.read_csv(metrics_path)
    df_curve = pd.read_csv(curve_path)
    df_curve = df_curve.loc[:, ~df_curve.columns.str.contains(r"^Unnamed")]

    required_cols = {"y", "pred"}
    missing = required_cols - set(df_preds.columns)
    if missing:
        raise ValueError(f"Predictions csv is missing columns: {sorted(missing)}")

    return df_metrics, df_preds, df_curve


def make_training_curve(df_curve: pd.DataFrame, out_dir: Path):
    has_r2 = "r2" in df_curve.columns
    nrows = 3 if has_r2 else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(8.6, 9.8), sharex=True)

    ax_loss = axes[0]
    ax_loss.plot(
        df_curve["epoch"],
        df_curve["train_loss"],
        label="train loss",
        linewidth=2,
        color=SEABORN_COLORS[0],
    )
    style_axes(ax_loss, "Training Loss", "", "Loss")
    style_legend(ax_loss, loc="best")

    ax_err = axes[1]
    if "mae" in df_curve.columns:
        ax_err.plot(
            df_curve["epoch"],
            df_curve["mae"],
            label="MAE",
            linewidth=2,
            color=SEABORN_COLORS[1],
        )
    if "rmse" in df_curve.columns:
        ax_err.plot(
            df_curve["epoch"],
            df_curve["rmse"],
            label="RMSE",
            linewidth=2,
            color=SEABORN_COLORS[3],
        )
    style_axes(ax_err, "Regression Error Metrics", "", "Error")
    style_legend(ax_err, loc="best")

    if has_r2:
        ax_r2 = axes[2]
        ax_r2.plot(
            df_curve["epoch"],
            df_curve["r2"],
            label="R2",
            linewidth=2,
            color=SEABORN_COLORS[2],
        )
        style_axes(ax_r2, "R2 Across Epochs", "Epoch", "R2")
        style_legend(ax_r2, loc="best")
    else:
        ax_err.set_xlabel("Epoch", fontsize=LABEL_SIZE)

    fig.suptitle("Training-Set Regression Metrics Per Epoch", y=0.995, fontsize=TITLE_SIZE)
    finalize_figure(fig)
    save_figure(fig, out_dir / "training_curves_regression.png", dpi=300)
    plt.close(fig)


def make_training_curve_cut_first4(df_curve: pd.DataFrame, out_dir: Path):
    """
    Plots MAE/RMSE and R2, skipping the first 4 epochs (i.e., starting from epoch 5).
    """
    df_cut = df_curve[df_curve["epoch"] > 4].reset_index(drop=True)
    has_r2 = "r2" in df_cut.columns

    # MAE & RMSE plot
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    if "mae" in df_cut.columns:
        ax.plot(
            df_cut["epoch"],
            df_cut["mae"],
            label="MAE",
            linewidth=2,
            color=SEABORN_COLORS[1],
        )
    if "rmse" in df_cut.columns:
        ax.plot(
            df_cut["epoch"],
            df_cut["rmse"],
            label="RMSE",
            linewidth=2,
            color=SEABORN_COLORS[3],
        )
    style_axes(ax, "MAE & RMSE (Epoch > 4)", "Epoch", "Error")
    style_legend(ax, loc="best")
    finalize_figure(fig)
    save_figure(fig, out_dir / "training_mae_rmse_cut_first4.png", dpi=300)
    plt.close(fig)

    # R2 plot
    if has_r2:
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        ax.plot(
            df_cut["epoch"],
            df_cut["r2"],
            label="R2",
            linewidth=2,
            color=SEABORN_COLORS[2],
        )
        style_axes(ax, "R2 (Epoch > 4)", "Epoch", "R2")
        style_legend(ax, loc="best")
        finalize_figure(fig)
        save_figure(fig, out_dir / "training_r2_cut_first4.png", dpi=300)
        plt.close(fig)


def make_training_curve_cut_first4_panel(df_curve: pd.DataFrame, out_dir: Path):
    """
    Combined panel (like training_curves_regression) using only epochs > 4.
    This gives clearer scaling after unstable early-epoch behavior.
    """
    df_cut = df_curve[df_curve["epoch"] > 4].reset_index(drop=True)
    if df_cut.empty:
        return

    has_r2 = "r2" in df_cut.columns
    nrows = 3 if has_r2 else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(8.6, 9.8), sharex=True)

    ax_loss = axes[0]
    ax_loss.plot(
        df_cut["epoch"],
        df_cut["train_loss"],
        label="train loss",
        linewidth=2,
        color=SEABORN_COLORS[0],
    )
    style_axes(ax_loss, "Training Loss (Epoch > 4)", "", "Loss")
    style_legend(ax_loss, loc="best")

    ax_err = axes[1]
    if "mae" in df_cut.columns:
        ax_err.plot(
            df_cut["epoch"],
            df_cut["mae"],
            label="MAE",
            linewidth=2,
            color=SEABORN_COLORS[1],
        )
    if "rmse" in df_cut.columns:
        ax_err.plot(
            df_cut["epoch"],
            df_cut["rmse"],
            label="RMSE",
            linewidth=2,
            color=SEABORN_COLORS[3],
        )
    style_axes(ax_err, "Regression Error Metrics (Epoch > 4)", "", "Error")
    style_legend(ax_err, loc="best")

    if has_r2:
        ax_r2 = axes[2]
        ax_r2.plot(
            df_cut["epoch"],
            df_cut["r2"],
            label="R2",
            linewidth=2,
            color=SEABORN_COLORS[2],
        )
        style_axes(ax_r2, "R2 Across Epochs (Epoch > 4)", "Epoch", "R2")
        style_legend(ax_r2, loc="best")
    else:
        ax_err.set_xlabel("Epoch", fontsize=LABEL_SIZE)

    fig.suptitle("Training-Set Regression Metrics Per Epoch (Epoch > 4)", y=0.995, fontsize=TITLE_SIZE)
    finalize_figure(fig)
    save_figure(fig, out_dir / "training_curves_regression_cut_first4.png", dpi=300)
    plt.close(fig)


def make_scatter_plot(df_preds: pd.DataFrame, out_dir: Path, target_name: str):
    display_target = pretty_region_name(target_name)
    target_label = format_ctrz_label(display_target)
    y_true = df_preds["y"].to_numpy()
    y_pred = df_preds["pred"].to_numpy()

    lo = np.nanmin([y_true.min(), y_pred.min()])
    hi = np.nanmax([y_true.max(), y_pred.max()])
    corr = np.corrcoef(y_true, y_pred)[0, 1] if len(df_preds) > 1 else np.nan

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.scatter(
        y_true,
        y_pred,
        alpha=0.85,
        s=38,
        edgecolor="white",
        linewidth=0.45,
        color=SEABORN_COLORS[0],
    )
    # x=y reference line (faint)
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1.2, alpha=0.5, label="Perfect prediction")
    # Fitted regression line
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    ax.plot([lo, hi], p([lo, hi]), linestyle="-", color="black", linewidth=1.8, label="Fitted line")
    style_legend(ax, loc="upper left")
    style_axes(
        ax,
        f"Reference vs Predicted\n{target_label}",
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
    finalize_figure(fig)
    save_figure(fig, out_dir / "true_vs_predicted.png", dpi=300)
    plt.close(fig)


def make_residual_plot(df_preds: pd.DataFrame, out_dir: Path, target_name: str):
    residuals = df_preds["pred"] - df_preds["y"]

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    ax.scatter(
        df_preds["y"],
        residuals,
        alpha=0.82,
        s=34,
        edgecolor="white",
        linewidth=0.4,
        color=SEABORN_COLORS[0],
    )
    ax.axhline(0, linestyle="--", color="black", linewidth=1.6)
    display_target = pretty_region_name(target_name)
    target_label = format_ctrz_label(display_target)
    style_axes(
        ax,
        f"Residuals vs Reference\n{target_label}",
        f"Reference {target_label}",
        f"Residual (Predicted - Reference {CTRZ_LABEL})",
    )
    finalize_figure(fig)
    save_figure(fig, out_dir / "residuals_vs_true.png", dpi=300)
    plt.close(fig)


def make_residual_histogram(df_preds: pd.DataFrame, out_dir: Path):
    residuals = df_preds["pred"] - df_preds["y"]

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    ax.hist(residuals, bins=20, alpha=0.9, color=SEABORN_COLORS[0], edgecolor="white")
    ax.axvline(0, linestyle="--", color="black", linewidth=1.6)
    style_axes(ax, "Residual Distribution", f"Residual (Predicted - Reference {CTRZ_LABEL})", "Count")
    finalize_figure(fig)
    save_figure(fig, out_dir / "residual_histogram.png", dpi=300)
    plt.close(fig)


def make_true_value_histogram(df_preds: pd.DataFrame, out_dir: Path, target_name: str):
    """
    Plots a histogram of the true target values to visualize their distribution.
    """
    y_true = df_preds["y"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    ax.hist(y_true, bins=20, alpha=0.9, color=SEABORN_COLORS[0], edgecolor="white")
    display_target = pretty_region_name(target_name)
    target_label = format_ctrz_label(display_target)
    style_axes(ax, f"Distribution of Reference\n{target_label}", f"Reference {target_label}", "Count")
    finalize_figure(fig)
    save_figure(fig, out_dir / "true_value_histogram.png", dpi=300)
    plt.close(fig)


def add_error_bins(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    y = pd.to_numeric(df["y"], errors="coerce")
    y = y.dropna()
    if y.empty:
        df["y_bin"] = "all"
        return df

    # 5 equal width bins
    y_min = float(y.min())
    y_max = float(y.max())
    if y_max <= y_min:
        # Fallback for very narrow ranges.
        edges = np.linspace(y_min, y_max, 3)
    else:
        edges = np.linspace(y_min, y_max, 6)

    labels = [f"{edges[i]:.2f}-{edges[i + 1]:.2f}" for i in range(len(edges) - 1)]
    df["y_bin"] = pd.cut(
        df["y"],
        bins=edges,
        labels=labels,
        include_lowest=True,
        duplicates="drop",
    )
    df["y_bin"] = df["y_bin"].astype("string").fillna("out_of_range")
    return df


def summarize_error_by_bin(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["error"] = df["pred"] - df["y"]
    df["abs_error"] = np.abs(df["error"])
    summary = (
        df.groupby("y_bin", observed=False)
        .agg(
            n=("abs_error", "size"),
            mae=("abs_error", "mean"),
            rmse=("error", lambda x: float(np.sqrt(np.mean(x**2)))),
            bias=("error", "mean"),
        )
        .reset_index()
    )
    return summary


def make_mae_by_bin_plot(summary: pd.DataFrame, out_dir: Path, target_name=None):
    display_target = pretty_region_name(target_name) if target_name else None
    target_label = format_ctrz_label(display_target) if display_target else CTRZ_LABEL
    title_prefix = f"{target_label} " if display_target else ""
    fig, ax1 = plt.subplots(figsize=(6.8, 6.8))
    x = np.arange(len(summary))

    ax1.bar(x, summary["mae"], color=SEABORN_COLORS[0], alpha=1.0, zorder=2)
    ax1.set_ylabel("MAE", color=SEABORN_COLORS[0], fontsize=LABEL_SIZE)
    ax1.set_xticks(x)
    ax1.set_xticklabels(summary["y_bin"], rotation=20, ha="right", fontsize=TICK_SIZE)
    ax1.set_xlabel(f"Reference {CTRZ_LABEL} Bin", fontsize=LABEL_SIZE)
    ax1.set_title(f"{title_prefix}MAE by Reference {CTRZ_LABEL} Bin", fontsize=TITLE_SIZE, pad=10)
    ax1.tick_params(axis="y", labelsize=TICK_SIZE)
    ax1.grid(False)

    ax2 = ax1.twinx()
    ax2.set_zorder(ax1.get_zorder() + 1)
    ax2.patch.set_alpha(0)
    ax2.plot(
        x,
        summary["n"],
        color=SEABORN_COLORS[5],
        marker="o",
        linewidth=2.2,
        zorder=5,
    )
    ax2.set_ylabel("Sample count", color=SEABORN_COLORS[5], fontsize=LABEL_SIZE)
    ax2.tick_params(axis="y", labelsize=TICK_SIZE)
    ax2.grid(False)

    finalize_figure(fig)
    save_figure(fig, out_dir / "mae_by_true_bin.png", dpi=300)
    plt.close(fig)


def make_residual_trend_plot(df: pd.DataFrame, out_dir: Path, target_name: str):
    df = df.copy()
    df["error"] = df["pred"] - df["y"]
    df = df.sort_values("y").reset_index(drop=True)

    # Smooth trend without extra dependencies.
    win = max(25, min(80, len(df) // 8))
    df["residual_smooth"] = df["error"].rolling(window=win, center=True, min_periods=10).mean()

    fig, (ax, ax_hist) = plt.subplots(
        2,
        1,
        figsize=(7.8, 6.8),
        sharex=True,
        gridspec_kw={"height_ratios": [4.5, 1.2], "hspace": 0.05},
    )
    ax.scatter(
        df["y"],
        df["error"],
        alpha=0.24,
        s=24,
        color=SEABORN_COLORS[6],
        edgecolor="white",
        linewidth=0.15,
        label="Subject residuals",
    )
    ax.plot(
        df["y"],
        df["residual_smooth"],
        color=SEABORN_COLORS[3],
        linewidth=2.3,
        label="Smoothed average residual",
    )
    ax.axhline(0, linestyle="--", color="black", linewidth=1.2)
    display_target = pretty_region_name(target_name)
    target_label = format_ctrz_label(display_target)
    style_axes(
        ax,
        f"Residuals Across Reference\n{target_label} Values",
        "",
        f"Residual (Predicted - Reference {CTRZ_LABEL})",
    )
    style_legend(ax, loc="upper right")

    ax_hist.hist(df["y"], bins=20, color=SEABORN_COLORS[0], alpha=0.9, edgecolor="white")
    style_axes(ax_hist, "", f"Reference {target_label}", "Number of Test Subjects")
    ax_hist.grid(False)
    finalize_figure(fig)
    save_figure(fig, out_dir / "residual_trend_vs_true.png", dpi=300)
    plt.close(fig)


def save_subgroup_bin_summary(df: pd.DataFrame, out_dir: Path, subgroup_cols: list[str]):
    df = df.copy()
    df["error"] = df["pred"] - df["y"]
    df["abs_error"] = np.abs(df["error"])
    rows = []

    for col in subgroup_cols:
        if col not in df.columns:
            continue
        tmp = df[["y_bin", col, "error", "abs_error"]].copy()
        tmp[col] = tmp[col].astype("string").fillna("NA")
        grouped = (
            tmp.groupby(["y_bin", col], observed=False)
            .agg(
                n=("abs_error", "size"),
                mae=("abs_error", "mean"),
                rmse=("error", lambda x: float(np.sqrt(np.mean(x**2)))),
                bias=("error", "mean"),
            )
            .reset_index()
            .rename(columns={col: "group_value"})
        )
        grouped.insert(0, "group", col)
        rows.append(grouped)

    if rows:
        out = pd.concat(rows, ignore_index=True)
        out.to_csv(out_dir / "subgroup_error_by_true_bin.csv", index=False)


def print_summary(df_metrics: pd.DataFrame, df_preds: pd.DataFrame, target_name: str):
    y_true = df_preds["y"].to_numpy()
    y_pred = df_preds["pred"].to_numpy()
    residuals = y_pred - y_true

    corr = np.corrcoef(y_true, y_pred)[0, 1] if len(df_preds) > 1 else np.nan
    bias = float(np.mean(residuals))
    abs_err = np.abs(residuals)

    print("Test metrics from saved csv:")
    print(df_metrics.to_string(index=False))
    print()
    print(f"Test set size: {len(df_preds)}")
    print(f"Reference {target_name} {CTRZ_LABEL} range: {y_true.min():.3f} to {y_true.max():.3f}")
    print(f"Predicted {target_name} {CTRZ_LABEL} range: {y_pred.min():.3f} to {y_pred.max():.3f}")
    print(f"Pearson r: {corr:.3f}")
    print(f"Mean residual (bias): {bias:.3f}")
    print(f"Median absolute error: {np.median(abs_err):.3f}")
    print(
        f"90th percentile absolute error: {np.percentile(abs_err, 90):.3f}"
    )
    print(
        "\nNote: in this --no-tune run, the per-epoch csv reflects the metric"
        " used during training on the training/test split workflow."
    )


def main():
    sns.set_theme(style="whitegrid", palette=CUSTOM_PALETTE)

    args = parse_args()
    if args.run_dir is None:
        run_dir = infer_latest_run_dir(DEFAULT_RESULTS_ROOT.resolve(), args.target_name)
    else:
        run_dir = Path(args.run_dir).resolve()
    out_dir = run_dir / "figures"
    out_dir.mkdir(exist_ok=True)

    # Create figures_demographics directory inside the run directory
    demo_fig_dir = run_dir / "figures_demographics"
    demo_fig_dir.mkdir(exist_ok=True)

    error_out_dir = run_dir / "figures_error_analysis"
    error_out_dir.mkdir(exist_ok=True)

    df_metrics, df_preds, df_curve = load_results(run_dir, args.dataset)
    df_for_error = df_preds.copy()

    # Merge with demographics and plot true vs predicted colored by each demographic
    script_path = Path(__file__).resolve()
    demo_candidates = [
        script_path.parents[1] / "data" / "demo.csv",  # Auto-rADiology/data/demo.csv
        script_path.parents[2] / "data" / "demo.csv",  # ../data/demo.csv
    ]
    demo_path = next((p for p in demo_candidates if p.exists()), None)

    n_demo_plots = 0
    if demo_path is not None:
        df_demo = pd.read_csv(demo_path)
        # Merge on ID_ind (preds) <-> ID (demo.csv)
        if "ID_ind" in df_preds.columns and "ID" in df_demo.columns:
            df_merged = pd.merge(df_preds, df_demo, left_on="ID_ind", right_on="ID", how="left")
        else:
            # Fallback: try index-based concat
            df_merged = pd.concat([df_preds.reset_index(drop=True), df_demo.reset_index(drop=True)], axis=1)

        if "dx" in df_merged.columns:
            df_merged["dx_grouped"] = combine_dx_groups(df_merged["dx"])

        df_for_error = df_merged.copy()

        # Keep only biologically meaningful demographic columns.
        preferred_demo_cols = [
            "site",
            "age",
            "gender",
            "sex",
            "dx_grouped",
            "apoe",
            "amyloid_status",
            "CDR",
        ]
        demo_cols = [col for col in preferred_demo_cols if col in df_merged.columns]
        if not demo_cols:
            print("[WARNING] No preferred demographic columns found in merged dataframe.")
        else:
            print(f"[INFO] Demographic plots will be generated for: {', '.join(demo_cols)}")

        for col in demo_cols:
            vals = df_merged[col]
            # Skip if all values are nan or only one unique value
            if vals.isnull().all() or vals.nunique() <= 1:
                continue
            fig, ax = plt.subplots(figsize=(6, 6))

            # Keep age continuous (gradient), but treat other low-cardinality numerics as categorical.
            is_categorical = (col != "age") and (
                (not pd.api.types.is_numeric_dtype(vals)) or (vals.nunique(dropna=True) <= 12)
            )
            if is_categorical:
                cat_vals = vals.astype("string").fillna("NA")
                categories = get_display_order(cat_vals, col)
                cat_to_code = {cat: i for i, cat in enumerate(categories)}
                color_codes = cat_vals.map(cat_to_code).astype(float)
                palette_map = get_category_palette(categories)
                palette = [palette_map[cat] for cat in categories]
                cmap = ListedColormap(palette)

                scatter = ax.scatter(
                    df_merged["y"],
                    df_merged["pred"],
                    c=color_codes,
                    cmap=cmap,
                    alpha=0.82,
                    s=36,
                    edgecolor="white",
                    linewidth=0.3,
                    label=None,
                )
            else:
                num_vals = pd.to_numeric(vals, errors="coerce")
                if num_vals.isnull().all():
                    plt.close(fig)
                    continue
                scatter = ax.scatter(
                    df_merged["y"],
                    df_merged["pred"],
                    c=num_vals,
                    cmap=SEABORN_CMAP,
                    alpha=0.84,
                    s=36,
                    edgecolor="white",
                    linewidth=0.3,
                    label=None,
                )
            lo = np.nanmin([df_merged["y"].min(), df_merged["pred"].min()])
            hi = np.nanmax([df_merged["y"].max(), df_merged["pred"].max()])
            # x=y reference line (faint)
            ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1.2, alpha=0.5, label="Perfect prediction")
            # Fitted regression line
            z = np.polyfit(df_merged["y"].dropna(), df_merged["pred"].dropna(), 1)
            p = np.poly1d(z)
            ax.plot([lo, hi], p([lo, hi]), linestyle="-", color="black", linewidth=1.8, label="Fitted line")
            col_display = "dx" if col == "dx_grouped" else col
            target_label = format_ctrz_label(pretty_region_name(args.target_name))
            style_axes(
                ax,
                f"Reference vs Predicted {target_label}\nColored by {col_display}",
                f"Reference {target_label}",
                f"Predicted {target_label}",
            )

            # Add legend for categorical, colorbar for numeric
            if is_categorical:
                handles = [
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        label=str(cat),
                        markerfacecolor=palette[i],
                        markersize=8,
                    )
                    for i, cat in enumerate(categories)
                ]
                ncol = 2 if len(categories) > 4 else 1
                ax.legend(
                    handles=handles,
                    title=None,
                    loc="upper left" if col == "dx_grouped" else "lower right",
                    ncol=ncol,
                    frameon=True,
                    facecolor="white",
                    framealpha=0.88,
                    fontsize=9 if col == "dx_grouped" else LEGEND_SIZE,
                    borderpad=0.4,
                    labelspacing=0.35,
                    handletextpad=0.5,
                    columnspacing=1.0,
                )
            else:
                cbar = fig.colorbar(scatter, ax=ax, label=col)
                cbar.ax.tick_params(labelsize=TICK_SIZE)
                cbar.set_label(col, fontsize=LABEL_SIZE)
            finalize_figure(fig)
            fname = f"true_vs_predicted_by_{col}.png"
            save_figure(fig, demo_fig_dir / fname, dpi=300)
            plt.close(fig)
            n_demo_plots += 1
    else:
        searched = "\n".join(str(p) for p in demo_candidates)
        print(f"[WARNING] Could not find demographics csv. Checked:\n{searched}")

    make_training_curve(df_curve, out_dir)
    make_training_curve_cut_first4(df_curve, out_dir)
    make_training_curve_cut_first4_panel(df_curve, out_dir)
    make_scatter_plot(df_preds, out_dir, args.target_name)
    make_residual_plot(df_preds, out_dir, args.target_name)
    make_residual_histogram(df_preds, out_dir)
    make_true_value_histogram(df_preds, out_dir, args.target_name)

    # Conditioned error analysis: global and subgroup performance by true-value bin.
    df_err = add_error_bins(df_for_error)
    bin_summary = summarize_error_by_bin(df_err)
    bin_summary.to_csv(error_out_dir / "global_error_by_true_bin.csv", index=False)
    make_mae_by_bin_plot(bin_summary, error_out_dir, args.target_name)
    make_residual_trend_plot(df_err, error_out_dir, args.target_name)
    save_subgroup_bin_summary(
        df_err,
        error_out_dir,
        subgroup_cols=["dx_grouped", "amyloid_status", "apoe", "CDR", "site", "gender", "sex"],
    )

    # New group-level diagnostics using MAE (subject-level absolute error).
    group_cols = ["dx_grouped", "amyloid_status", "apoe", "CDR", "site", "gender", "sex", "age"]
    make_group_mae_boxstrip_plots(df_for_error, error_out_dir, group_cols, args.target_name)
    run_group_significance_tests(df_for_error, error_out_dir, group_cols)
    make_site_raw_value_plot(df_for_error, error_out_dir, args.target_name)
    make_site_error_correlation_plot(df_for_error, error_out_dir, args.target_name)
    make_training_distribution_plot(run_dir, error_out_dir, args.target_name)

    print_summary(df_metrics, df_preds, args.target_name)

    print(f"\nSaved figures to: {out_dir}")

    print(f"Saved demographic plots to: {demo_fig_dir} (count: {n_demo_plots})")
    print(f"Saved conditioned error analysis to: {error_out_dir}")


if __name__ == "__main__":
    main()
