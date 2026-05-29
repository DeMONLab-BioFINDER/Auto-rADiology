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


def combine_dx_groups(s: pd.Series) -> pd.Series:
    s = s.astype("string").fillna("NA")
    s_norm = s.str.strip().str.lower()
    dx_map = {
        "cu": "CU",
        "mci": "MCI",
        "alzcs dem": "AlzCS dem",
    }
    out = s_norm.map(dx_map).fillna("Other").astype("string")
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
        "dx_grouped": ["CU", "MCI", "AlzCS dem", "Other"],
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
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(name))

def format_suvr_label(display_target=None) -> str:
    if display_target:
        return f"{display_target} SUVR"
    return "SUVR"


def wrap_plot_title(title: str, width: int = 36) -> str:
    return textwrap.fill(str(title), width=width, break_long_words=False, break_on_hyphens=False)


def finalize_figure(fig, *, rect=(0.0, 0.0, 0.97, 0.97)):
    fig.tight_layout(rect=rect)


def save_figure(fig, out_path: Path, **kwargs) -> bool:
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

def resolve_existing_path(candidates, label: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {label}. Checked:\n{searched}")

PREFERRED_TARGET_ORDER = ["MetaTemporal", "MesialTemporal", "TemporoParietal", "Frontal"]

def order_targets(targets: list[str]) -> list[str]:
    target_map = {t.lower(): t for t in targets}
    ordered = []
    for pref in PREFERRED_TARGET_ORDER:
        key = pref.lower()
        if key in target_map:
            ordered.append(target_map[key])
    remaining = [t for t in targets if t not in ordered]
    return ordered + remaining

def find_demo_csv() -> Path | None:
    script_path = Path(__file__).resolve()
    demo_candidates = [
        script_path.parents[1] / "data" / "demo.csv",
        script_path.parents[2] / "data" / "demo.csv",
    ]
    return next((p for p in demo_candidates if p.exists()), None)

def infer_targets(df_preds: pd.DataFrame) -> list[str]:
    targets = []
    for col in df_preds.columns:
        if col.endswith("_y"):
            target = col[:-2]
            pred_col = f"{target}_pred"
            if pred_col in df_preds.columns:
                targets.append(target)
    return targets

def get_valid_target_values(df: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray]:
    y_true = pd.to_numeric(df[f"{target}_y"], errors="coerce").to_numpy()
    y_pred = pd.to_numeric(df[f"{target}_pred"], errors="coerce").to_numpy()
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[valid], y_pred[valid]

def get_plot_limits(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    lo = float(np.nanmin([y_true.min(), y_pred.min()]))
    hi = float(np.nanmax([y_true.max(), y_pred.max()]))
    return lo, hi

def add_panel_labels(axes, labels: list[str], *, xytext=(-14, 8), fontsize=14):
    flat_axes = [ax for row in axes for ax in row]
    for ax, label in zip(flat_axes, labels):
        ax.annotate(
            label,
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=xytext,
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=fontsize,
            fontweight="bold",
            clip_on=False,
        )

def hide_unused_axes(axes, start_idx: int, total_slots: int, n_cols: int):
    for idx in range(start_idx, total_slots):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        axes[row_idx][col_idx].set_visible(False)

def make_true_vs_predicted_panel(
    df_preds_wide: pd.DataFrame,
    out_dir: Path,
    targets: list[str],
    title: str,
    filename: str,
):
    targets = order_targets(targets)
    if not targets:
        return

    n_cols = 2 if len(targets) > 1 else 1
    n_rows = math.ceil(len(targets) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 5.2 * n_rows), squeeze=False)

    for idx, target in enumerate(targets):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r][c]

        y_true, y_pred = get_valid_target_values(df_preds_wide, target)
        if len(y_true) == 0:
            ax.set_visible(False)
            continue

        ax.scatter(
            y_true,
            y_pred,
            alpha=0.85,
            s=38,
            edgecolor="white",
            linewidth=0.45,
            color=SEABORN_COLORS[0],
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

        display_target = pretty_region_name(target)
        style_axes(
            ax,
            display_target,
            "Reference SUVR",
            "Predicted SUVR",
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
    hide_unused_axes(axes, len(targets), n_rows * n_cols, n_cols)
    add_panel_labels(axes, [chr(ord("A") + i) for i in range(len(targets))])

    finalize_figure(fig, rect=(0.0, 0.0, 0.99, 0.99))
    save_figure(fig, out_dir / filename, dpi=300)
    plt.close(fig)

def make_suvr_distribution_panel(
    demo_path: Path,
    out_dir: Path,
    targets: list[str],
    title: str,
    filename: str,
):
    df_demo = pd.read_csv(demo_path)
    targets = [t for t in order_targets(targets) if t in df_demo.columns]
    if not targets:
        return

    n_cols = 2 if len(targets) > 1 else 1
    n_rows = math.ceil(len(targets) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.6 * n_rows), squeeze=False)

    for idx, target in enumerate(targets):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r][c]

        values = pd.to_numeric(df_demo[target], errors="coerce").dropna()
        if values.empty:
            ax.set_visible(False)
            continue

        sns.histplot(values, bins=22, kde=True, ax=ax, color=SEABORN_COLORS[0])
        target_label = format_suvr_label(pretty_region_name(target))
        style_axes(
            ax,
            target_label.replace(" SUVR", ""),
            "SUVR",
            "Count",
        )

    hide_unused_axes(axes, len(targets), n_rows * n_cols, n_cols)
    add_panel_labels(axes, [chr(ord("A") + i) for i in range(len(targets))], xytext=(-3, 2), fontsize=13)

    finalize_figure(fig, rect=(0.0, 0.0, 0.99, 0.99))
    save_figure(fig, out_dir / filename, dpi=300)
    plt.close(fig)

def run_multi_regression_figures(run_dir: Path, *, dedup: bool = True):
    validation_dir = run_dir / "validation" / "Gothenburg"
    evaluation_dir = run_dir / "evaluation" / "Gothenburg"
    preds_path = resolve_existing_path(
        [
            validation_dir / "Test_Gothenburg_results.csv",
            evaluation_dir / "Eval_Gothenburg_results.csv",
        ],
        "predictions csv",
    )
    test_split_path = resolve_existing_path(
        [
            run_dir / "train-test-split" / "train-test-split_testing-set.csv",
            run_dir / "train-test-split" / "preds" / "train-test-split_testing-set.csv",
            run_dir / "splits" / "Hold-out_testing-set.csv",
        ],
        "test split csv",
    )

    sns.set_theme(style="whitegrid", palette=CUSTOM_PALETTE)

    df_preds_wide = pd.read_csv(preds_path)
    if dedup:
        df_preds_wide = df_preds_wide.drop_duplicates().reset_index(drop=True)

    df_demo = pd.read_csv(test_split_path).copy()
    if "ID" not in df_demo.columns and "Unnamed: 0" in df_demo.columns:
        df_demo = df_demo.rename(columns={"Unnamed: 0": "ID"})
    df_demo = df_demo.loc[:, ~df_demo.columns.str.contains(r"^Unnamed")]
    if dedup:
        df_demo = df_demo.drop_duplicates().reset_index(drop=True)

    targets = infer_targets(df_preds_wide)
    if not targets:
        raise ValueError("Could not infer target names from prediction csv columns.")

    shared_fig_dir = run_dir / "figures_multi" / "_shared"
    shared_fig_dir.mkdir(parents=True, exist_ok=True)

    make_true_vs_predicted_panel(
        df_preds_wide,
        shared_fig_dir,
        targets,
        title=f"Test Set: Reference vs Predicted {format_suvr_label()}",
        filename="true_vs_predicted_panel_test.png",
    )

    demo_path = find_demo_csv()
    if demo_path is not None:
        make_suvr_distribution_panel(
            demo_path,
            shared_fig_dir,
            targets,
            title=f"Reference {format_suvr_label()} Distributions",
            filename="suvr_distribution_panel.png",
        )
    else:
        print("[WARNING] Could not find demo.csv for SUVR distribution panel.")

    print(f"[done] paper-only shared figures saved to {shared_fig_dir}")


def make_group_mae_boxstrip_plots(
    df: pd.DataFrame,
    out_dir: Path,
    group_cols: list[str],
    target_name=None,
):
    df_plot = df.copy()
    df_plot["abs_error"] = (df_plot["pred"] - df_plot["y"]).abs()
    display_target = pretty_region_name(target_name) if target_name else None
    target_label = format_suvr_label(display_target) if display_target else "SUVR"

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
        min_n_for_plot = 2 if col == "dx_grouped" else 5

        if col == "age":
            age_order = ["50-60", "60-70", "70-80", "80-90", "90+"]
            present = set(g[col].astype("string"))
            order = [x for x in age_order if x in present]
        else:
            order = get_display_order(g[col], col)
        order = [x for x in order if (g[col] == x).sum() >= min_n_for_plot]
        if len(order) < 2:
            continue
        palette_map = get_category_palette(order)

        fig, ax = plt.subplots(figsize=(6.4, 6.4))
        # Do not rotate x-tick labels (keep horizontal for readability)
        rotation = 0
        col_display = "dx" if col == "dx_grouped" else col
        # Remove in-figure title for Nature-style minimal plots
        title = ""
        label_map = {
            "dx": "Diagnosis group",
            "site": "Acquisition site",
            "age": "Age group",
            "gender": "Sex",
            "sex": "Sex",
            "apoe": "APOE status",
            "amyloid_status": "Amyloid status",
            "cdr": "CDR",
        }
        x_label = label_map.get(col_display.lower(), col_display.replace("_", " ").title())

        sns.boxplot(
            data=g,
            x=col,
            y="abs_error",
            hue=col,
            hue_order=order,
            order=order,
            ax=ax,
            palette=palette_map,
            dodge=False,
            fliersize=0,
            width=0.55,
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
            medianprops={"color": "black", "linewidth": 1.35},
            whiskerprops={"color": "black", "linewidth": 1.1},
            capprops={"color": "black", "linewidth": 1.1},
        )
        sns.stripplot(
            data=g,
            x=col,
            y="abs_error",
            order=order,
            hue=col,
            hue_order=order,
            ax=ax,
            dodge=False,
            jitter=0.18,
            size=3.2,
            linewidth=0.25,
            edgecolor="white",
            palette=palette_map,
            alpha=0.75,
            legend=False,
            zorder=5,
        )
        if ax.legend_ is not None:
            ax.legend_.remove()
        style_axes(
            ax,
            title,
            x_label,
            f"Absolute Error (SUVR)",
            xrotation=rotation,
        )
        # Add sample size labels under each x-tick (e.g., "n=12").
        try:
            counts = [int((g[col] == val).sum()) for val in order]
            xtick_labels = [f"{str(val)}\n(n={cnt})" for val, cnt in zip(order, counts)]
            ax.set_xticklabels(xtick_labels, rotation=rotation, fontsize=TICK_SIZE)
            # If this is the site plot, n= lines push ticklabels up — lower the x-axis label slightly.
            if col == "site":
                try:
                    ax.xaxis.set_label_coords(0.5, -0.12)
                except Exception:
                    # Fallback: increase labelpad if direct coords adjustment fails
                    try:
                        ax.xaxis.labelpad = (ax.xaxis.labelpad if hasattr(ax.xaxis, "labelpad") else 4) + 6
                    except Exception:
                        pass
        except Exception:
            # Fallback: do nothing if labeling fails for unexpected dtypes
            pass
        whisker_max = np.nan
        data_max = np.nan
        for _, group_df in g.groupby(col):
            values = group_df["abs_error"].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            group_max = float(np.nanmax(values))
            data_max = group_max if not np.isfinite(data_max) else max(data_max, group_max)
            q1, q3 = np.nanpercentile(values, [25, 75])
            iqr = q3 - q1
            group_whisker = q3 + 1.5 * iqr
            if not np.isfinite(group_whisker) or group_whisker <= 0:
                group_whisker = group_max
            whisker_max = group_whisker if not np.isfinite(whisker_max) else max(whisker_max, group_whisker)

        if np.isfinite(data_max) and data_max > 0:
            ax.set_ylim(-0.05 * data_max, data_max * 1.08)
        ax.set_box_aspect(1)
        ax.margins(x=0.03)
        ax.grid(axis="y", alpha=0.25)
        # Use a slightly wider rect and larger pad to avoid right-side cropping
        finalize_figure(fig, rect=(0.0, 0.0, 0.995, 0.985))
        save_figure(fig, out_dir / f"mae_boxstrip_by_{col}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
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
    target_label = format_suvr_label(display_target)
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
    target_label = format_suvr_label(display_target)

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
            f"Absolute Error (SUVR)",
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
    target_label = format_suvr_label(display_target)
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


def make_training_loss_curve(df_curve: pd.DataFrame, out_dir: Path):
    if "train_loss" not in df_curve.columns:
        return

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(
        df_curve["epoch"],
        df_curve["train_loss"],
        label="train loss",
        linewidth=2,
        color=SEABORN_COLORS[0],
    )
    style_axes(ax, "Training Loss", "Epoch", "Loss")
    style_legend(ax, loc="best")
    finalize_figure(fig)
    save_figure(fig, out_dir / "training_loss_curve.png", dpi=300)
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
    target_label = format_suvr_label(display_target)
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
    target_label = format_suvr_label(display_target)
    style_axes(
        ax,
        f"Residuals vs Reference\n{target_label}",
        f"Reference {target_label}",
        f"Residual (Predicted - Reference SUVR)",
    )
    finalize_figure(fig)
    save_figure(fig, out_dir / "residuals_vs_true.png", dpi=300)
    plt.close(fig)


def make_residual_histogram(df_preds: pd.DataFrame, out_dir: Path):
    residuals = df_preds["pred"] - df_preds["y"]

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    ax.hist(residuals, bins=20, alpha=0.9, color=SEABORN_COLORS[0], edgecolor="white")
    ax.axvline(0, linestyle="--", color="black", linewidth=1.6)
    style_axes(ax, "Residual Distribution", f"Residual (Predicted - Reference SUVR)", "Count")
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
    target_label = format_suvr_label(display_target)
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
    target_label = format_suvr_label(display_target) if display_target else "SUVR"
    title_prefix = f"{target_label} " if display_target else ""
    fig, ax1 = plt.subplots(figsize=(6.8, 6.8))
    x = np.arange(len(summary))

    ax1.bar(x, summary["mae"], color=SEABORN_COLORS[0], alpha=1.0, zorder=2)
    ax1.set_ylabel("MAE", color=SEABORN_COLORS[0], fontsize=LABEL_SIZE)
    ax1.set_xticks(x)
    ax1.set_xticklabels(summary["y_bin"], rotation=20, ha="right", fontsize=TICK_SIZE)
    ax1.set_xlabel(f"Reference SUVR Bin", fontsize=LABEL_SIZE)
    ax1.set_title(f"{title_prefix}MAE by Reference SUVR Bin", fontsize=TITLE_SIZE, pad=10)
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


def _draw_sig_bracket(ax, x1, x2, y, text, linewidth=1.0, fontsize=9):
    # Draw a simple significance bracket between two x positions (in data coords: x indices).
    ylim = ax.get_ylim()
    span = ylim[1] - ylim[0] if ylim[1] > ylim[0] else 1.0
    pad = span * 0.02
    y_top = y
    y_mid = y_top - pad
    bracket_y = [y_mid - pad, y_top, y_top, y_mid - pad]
    ax.plot([x1, x1, x2, x2], bracket_y, color="black", linewidth=linewidth, clip_on=False)
    # Draw label slightly above the bracket with a white bbox for readability
    ax.text((x1 + x2) / 2, y_top + pad, text, ha="center", va="bottom", fontsize=fontsize, clip_on=False, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8})


def make_dx_site_two_panel_mae(df_long: pd.DataFrame, out_dir: Path):
    """Create a two-panel figure: (A) MAE by diagnosis group, (B) MAE by site.

    Expects `df_long` to contain columns: `ID_ind` (or `ID`), `y`, `pred`, `dx_grouped`, `site`.
    Computes per-subject MAE as the subject-wise mean absolute error across targets.
    """
    df = df_long.copy()
    id_col = None
    for candidate in ("ID_ind", "ID", "subject_id"):
        if candidate in df.columns:
            id_col = candidate
            break
    if id_col is None:
        raise ValueError("No subject ID column found in dataframe (expected ID_ind or ID)")

    # Compute per-subject MAE
    df["abs_error"] = (pd.to_numeric(df.get("pred"), errors="coerce") - pd.to_numeric(df.get("y"), errors="coerce")).abs()
    subj = (
        df.groupby(id_col, observed=False)
        .agg(mae=("abs_error", "mean"))
        .reset_index()
    )

    # Attach demographics (take first non-null per subject)
    demo_cols = ["dx_grouped", "site"]
    for col in demo_cols:
        if col in df.columns:
            mapping = df.dropna(subset=[col]).drop_duplicates(subset=[id_col]).set_index(id_col)[col]
            subj[col] = subj[id_col].map(mapping).astype("string").fillna("NA")
        else:
            subj[col] = "NA"

    # Panel A: Diagnosis order by severity (exclude the catch-all 'Other')
    dx_order = ["CU", "MCI", "AlzCS dem"]
    present_dx = [d for d in dx_order if d in subj["dx_grouped"].unique()]

    # Panel B: site order by median MAE ascending
    site_medians = subj.groupby("site", observed=False)["mae"].median().sort_values()
    site_order = list(site_medians.index)

    # Create three columns: left plot, spacer, right plot. Spacer guarantees a visible gap.
    # Make spacer even smaller per user request (~0.01)
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 5.2), gridspec_kw={"width_ratios": [1, 0.01, 1]})
    # axes: [0]=dx, [1]=spacer, [2]=site
    spacer_ax = axes[1]
    spacer_ax.set_visible(False)
    fig.subplots_adjust(wspace=0.01, hspace=0.02)

    # Common styling
    palette = get_category_palette(present_dx if present_dx else ["NA"]) if present_dx else {"NA": CUSTOM_PALETTE[0]}

    # A: Diagnosis (left)
    ax = axes[0]
    plot_df = subj[subj["dx_grouped"] .isin(present_dx)].copy()
    if not plot_df.empty:
        sns.boxplot(data=plot_df, x="dx_grouped", y="mae", order=present_dx, ax=ax, palette=get_category_palette(present_dx), width=0.6, fliersize=0)
        # Draw jittered points colored by group to match the box palette, with white edge
        palette_map_dx = get_category_palette(present_dx)
        for i, grp in enumerate(present_dx):
            vals = plot_df.loc[plot_df["dx_grouped"] == grp, "mae"].dropna().to_numpy()
            if vals.size == 0:
                continue
            rng = np.random.default_rng(seed=0)
            jitter = rng.normal(0, 0.08, size=len(vals))
            xs = np.full(len(vals), i) + jitter
            ax.scatter(xs, vals, color=palette_map_dx[grp], edgecolor="white", linewidth=0.4, s=28, alpha=0.75, zorder=5)
        # Add n counts under ticks
        counts = [int((plot_df["dx_grouped"] == val).sum()) for val in present_dx]
        xtick_labels = [f"{val}\n(n={c})" for val, c in zip(present_dx, counts)]
        ax.set_xticklabels(xtick_labels, rotation=0, fontsize=TICK_SIZE)
        ax.set_xlabel("")
        ax.set_ylabel("MAE (SUVR)")
        ax.grid(axis="y", alpha=0.25)

        # Significance annotations removed per user request (no bracket or p-value markers)

    else:
        ax.set_visible(False)

    # B: Site (right)
    ax = axes[2]
    plot_df = subj[subj["site"].isin(site_order)].copy()
    if not plot_df.empty:
        sns.boxplot(data=plot_df, x="site", y="mae", order=site_order, ax=ax, palette=get_category_palette(site_order), width=0.6, fliersize=0)
        # Jittered colored points per site
        palette_map_site = get_category_palette(site_order)
        for i, grp in enumerate(site_order):
            vals = plot_df.loc[plot_df["site"] == grp, "mae"].dropna().to_numpy()
            if vals.size == 0:
                continue
            rng = np.random.default_rng(seed=1)
            jitter = rng.normal(0, 0.08, size=len(vals))
            xs = np.full(len(vals), i) + jitter
            ax.scatter(xs, vals, color=palette_map_site[grp], edgecolor="white", linewidth=0.4, s=28, alpha=0.75, zorder=5)
        counts = [int((plot_df["site"] == val).sum()) for val in site_order]
        xtick_labels = [f"{val}\n(n={c})" for val, c in zip(site_order, counts)]
        ax.set_xticklabels(xtick_labels, rotation=0, fontsize=TICK_SIZE)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(axis="y", alpha=0.25)

        # Significance annotations removed per user request (no bracket or p-value markers)
    else:
        ax.set_visible(False)

    # Panel letters (A, B) only for the visible axes (left and right)
    for ax_, label in zip([axes[0], axes[2]], ["A", "B"]):
        if ax_.get_visible():
            ax_.annotate(label, xy=(0, 1), xycoords="axes fraction", xytext=(-14, 8), textcoords="offset points", ha="right", va="bottom", fontsize=14, fontweight="bold", clip_on=False)
    # Ensure y-axis covers the full observed per-subject MAE range (avoid unexpected autoscaling/clipping)
    try:
        y_max = float(np.nanmax(subj["mae"].to_numpy(dtype=float)))
        if np.isfinite(y_max) and y_max > 0:
            y_lim = (-0.05 * y_max, y_max * 1.08)
            for ax_set in (axes[0], axes[2]):
                if ax_set.get_visible():
                    ax_set.set_ylim(y_lim)
    except Exception:
        # If something goes wrong, fall back to autoscaling
        pass
    # Finalize and save
    finalize_figure(fig, rect=(0.0, 0.0, 0.995, 0.995))
    save_figure(fig, out_dir / "mae_dx_site_panel.png", dpi=300)
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
    target_label = format_suvr_label(display_target)
    style_axes(
        ax,
        f"Residuals Across Reference\n{target_label} Values",
        "",
        f"Residual (Predicted - Reference SUVR)",
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
    print(f"Reference {target_name} SUVR range: {y_true.min():.3f} to {y_true.max():.3f}")
    print(f"Predicted {target_name} SUVR range: {y_pred.min():.3f} to {y_pred.max():.3f}")
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
    demo_path = next((p for p in [
        Path(__file__).resolve().parents[1] / "data" / "demo.csv",
        Path(__file__).resolve().parents[2] / "data" / "demo.csv",
    ] if p.exists()), None)

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

    make_dx_site_two_panel_mae(df_merged, out_dir)
    print(f"[done] paper-only MAE panel saved to {out_dir}")


if __name__ == "__main__":
    main()
