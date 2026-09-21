#!/usr/bin/env python3
"""Shared plotting helpers and figure builders used by the results_plotting scripts."""

import math
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

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

TITLE_SIZE = 17
LABEL_SIZE = 15
TICK_SIZE = 12
LEGEND_SIZE = 11

PREFERRED_TARGET_ORDER = ["MetaTemporal", "MesialTemporal", "TemporoParietal", "Frontal"]


# ------------------------------
# Formatting helpers
# ------------------------------
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


def order_targets(targets: list[str]) -> list[str]:
    target_map = {t.lower(): t for t in targets}
    ordered = []
    for pref in PREFERRED_TARGET_ORDER:
        key = pref.lower()
        if key in target_map:
            ordered.append(target_map[key])
    remaining = [t for t in targets if t not in ordered]
    return ordered + remaining


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


def compute_regression_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    n = len(y_true)
    if n == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "r2": np.nan, "pearson_r": np.nan, "bias": np.nan}

    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    bias = float(np.mean(error))
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1]) if n > 1 else np.nan

    return {"n": n, "mae": mae, "rmse": rmse, "r2": r2, "pearson_r": pearson_r, "bias": bias}


def compute_region_stats_table(df_preds_wide: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    rows = []
    for target in order_targets(targets):
        y_true, y_pred = get_valid_target_values(df_preds_wide, target)
        rows.append({"target": target, **compute_regression_stats(y_true, y_pred)})
    return pd.DataFrame(rows)


def find_demo_csv() -> Path | None:
    script_path = Path(__file__).resolve()
    demo_candidates = [
        script_path.parents[1] / "data" / "demo.csv",
        script_path.parents[2] / "data" / "demo.csv",
    ]
    return next((p for p in demo_candidates if p.exists()), None)


def resolve_existing_path(candidates, label: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {label}. Checked:\n{searched}")


# ------------------------------
# Figure scaffolding helpers
# ------------------------------
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


# ------------------------------
# Figure builders
# ------------------------------
def make_true_vs_predicted_panel(
    df_preds_wide: pd.DataFrame,
    out_dir: Path,
    targets: list[str],
    title: str,
    filename: str,
    stats_csv_path: Path | None = None,
):
    targets = order_targets(targets)
    if not targets:
        return

    n_cols = 2 if len(targets) > 1 else 1
    n_rows = math.ceil(len(targets) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 5.2 * n_rows), squeeze=False)

    stats_rows = []
    for idx, target in enumerate(targets):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r][c]

        y_true, y_pred = get_valid_target_values(df_preds_wide, target)
        if len(y_true) == 0:
            ax.set_visible(False)
            continue

        stats = compute_regression_stats(y_true, y_pred)
        stats_rows.append({"target": target, **stats})

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

        display_target = pretty_region_name(target)
        style_axes(
            ax,
            display_target,
            "Reference SUVR",
            "Predicted SUVR",
        )
        stats_text = (
            f"r = {stats['pearson_r']:.3f}\n"
            f"MAE = {stats['mae']:.3f}\n"
            f"RMSE = {stats['rmse']:.3f}\n"
            f"R² = {stats['r2']:.3f}"
        )
        ax.text(
            0.96,
            0.96,
            stats_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=TICK_SIZE,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    hide_unused_axes(axes, len(targets), n_rows * n_cols, n_cols)
    add_panel_labels(axes, [chr(ord("A") + i) for i in range(len(targets))])

    finalize_figure(fig, rect=(0.0, 0.0, 0.99, 0.99))
    save_figure(fig, out_dir / filename, dpi=300)
    plt.close(fig)

    if stats_csv_path is not None and stats_rows:
        pd.DataFrame(stats_rows).to_csv(stats_csv_path, index=False)


def make_bland_altman_panel(
    df_preds_wide: pd.DataFrame,
    out_dir: Path,
    targets: list[str],
    filename: str,
):
    """Mean-vs-difference agreement plot (Bland-Altman) per region."""
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

        mean_vals = (y_true + y_pred) / 2
        diff_vals = y_pred - y_true
        bias = float(np.mean(diff_vals))
        sd = float(np.std(diff_vals, ddof=1)) if len(diff_vals) > 1 else 0.0
        loa_lo, loa_hi = bias - 1.96 * sd, bias + 1.96 * sd

        ax.scatter(mean_vals, diff_vals, alpha=0.85, s=38, edgecolor="white", linewidth=0.45, color=SEABORN_COLORS[0])
        ax.axhline(0, color="lightgray", linewidth=1.0)
        ax.axhline(bias, color="black", linewidth=1.6)
        ax.axhline(loa_hi, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.axhline(loa_lo, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)

        display_target = pretty_region_name(target)
        style_axes(
            ax,
            display_target,
            "Mean of reference & predicted SUVR",
            "Predicted − Reference SUVR",
        )
        stats_text = f"Bias = {bias:.3f}\n+1.96 SD = {loa_hi:.3f}\n−1.96 SD = {loa_lo:.3f}"
        ax.text(
            0.96,
            0.96,
            stats_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=TICK_SIZE,
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

    single_plot = len(targets) == 1
    for idx, target in enumerate(targets):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r][c]

        values = pd.to_numeric(df_demo[target], errors="coerce").dropna()
        if values.empty:
            ax.set_visible(False)
            continue

        sns.histplot(values, bins=22, kde=True, ax=ax, color=SEABORN_COLORS[0])
        if single_plot:
            plot_title = title
        else:
            target_label = format_suvr_label(pretty_region_name(target))
            plot_title = target_label.replace(" SUVR", "")
        style_axes(
            ax,
            plot_title,
            "SUVR",
            "Count",
        )

    hide_unused_axes(axes, len(targets), n_rows * n_cols, n_cols)
    if not single_plot:
        add_panel_labels(axes, [chr(ord("A") + i) for i in range(len(targets))], xytext=(-3, 2), fontsize=13)

    finalize_figure(fig, rect=(0.0, 0.0, 0.99, 0.99))
    save_figure(fig, out_dir / filename, dpi=300)
    plt.close(fig)


SUBGROUP_COLUMNS = [
    ("dx_grouped", "Diagnosis"),
    ("site", "Site"),
    ("gender", "Sex"),
    ("apoe", "APOE"),
    ("amyloid_status", "Amyloid status"),
    ("age_group", "Age group"),
    ("CDR", "CDR"),
]

_AGE_BIN_EDGES = [0, 60, 70, 80, 90, 200]
_AGE_BIN_LABELS = ["<60", "60-70", "70-80", "80-90", "90+"]


def bin_age_group(age: pd.Series) -> pd.Series:
    return pd.cut(pd.to_numeric(age, errors="coerce"), bins=_AGE_BIN_EDGES, labels=_AGE_BIN_LABELS, right=False)


def _subgroup_order(subj: pd.DataFrame, col: str) -> list[str]:
    if col == "site":
        return list(subj.groupby("site", observed=False)["mae"].median().sort_values().index)
    if col == "dx_grouped":
        present = set(subj["dx_grouped"].unique())
        return [d for d in ["CU", "MCI", "AlzCS dem"] if d in present]
    if col == "age_group":
        present = set(subj["age_group"].astype("string").dropna().unique())
        return [a for a in _AGE_BIN_LABELS if a in present]
    return get_display_order(subj[col], col)


def make_subgroup_mae_panel(df_long: pd.DataFrame, out_dir: Path, filename: str = "mae_subgroup_panel.png", n_cols: int = 3, min_group_n: int = 2):
    """Per-subject MAE boxplots across whichever subgroup columns are available in `df_long`.

    Expects `df_long` to contain `ID_ind`/`ID`, `y`, `pred`, plus whatever of
    SUBGROUP_COLUMNS (or `age`, binned into `age_group`) are present.
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

    if "age" in df.columns and "age_group" not in df.columns:
        df["age_group"] = bin_age_group(df["age"])

    df["abs_error"] = (pd.to_numeric(df.get("pred"), errors="coerce") - pd.to_numeric(df.get("y"), errors="coerce")).abs()
    subj = df.groupby(id_col, observed=False).agg(mae=("abs_error", "mean")).reset_index()

    for col, _ in SUBGROUP_COLUMNS:
        if col in df.columns:
            mapping = df.dropna(subset=[col]).drop_duplicates(subset=[id_col]).set_index(id_col)[col]
            subj[col] = subj[id_col].map(mapping).astype("string").fillna("NA")

    present_cols = [(col, label) for col, label in SUBGROUP_COLUMNS if col in subj.columns and subj[col].nunique() > 1]
    if not present_cols:
        return

    n = len(present_cols)
    n_cols = min(n_cols, n)
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 4.8 * n_rows), squeeze=False)

    for idx, (col, label) in enumerate(present_cols):
        r, c = idx // n_cols, idx % n_cols
        ax = axes[r][c]

        order = [o for o in _subgroup_order(subj, col) if (subj[col] == o).sum() >= min_group_n]
        if len(order) < 2:
            ax.set_visible(False)
            continue

        plot_df = subj[subj[col].isin(order)]
        palette = get_category_palette(order)
        sns.boxplot(data=plot_df, x=col, y="mae", order=order, ax=ax, palette=palette, width=0.6, fliersize=0)
        for i, grp in enumerate(order):
            vals = plot_df.loc[plot_df[col] == grp, "mae"].dropna().to_numpy()
            if vals.size == 0:
                continue
            rng = np.random.default_rng(seed=idx)
            xs = np.full(len(vals), i) + rng.normal(0, 0.08, size=len(vals))
            ax.scatter(xs, vals, color=palette[grp], edgecolor="white", linewidth=0.4, s=24, alpha=0.75, zorder=5)

        counts = [int((plot_df[col] == val).sum()) for val in order]
        ax.set_xticklabels([f"{val}\n(n={cnt})" for val, cnt in zip(order, counts)], fontsize=TICK_SIZE)
        ax.set_xlabel("")
        ax.set_ylabel("MAE (SUVR)" if c == 0 else "")
        ax.set_title(label, fontsize=LABEL_SIZE)
        ax.grid(axis="y", alpha=0.25)

    hide_unused_axes(axes, len(present_cols), n_rows * n_cols, n_cols)
    add_panel_labels(axes, [chr(ord("A") + i) for i in range(len(present_cols))], xytext=(-10, 6), fontsize=13)

    y_max = float(np.nanmax(subj["mae"].to_numpy(dtype=float))) if len(subj) else np.nan
    if np.isfinite(y_max) and y_max > 0:
        for ax_row in axes:
            for ax_ in ax_row:
                if ax_.get_visible():
                    ax_.set_ylim(-0.05 * y_max, y_max * 1.08)

    finalize_figure(fig, rect=(0.0, 0.0, 0.995, 0.995))
    save_figure(fig, out_dir / filename, dpi=300)
    plt.close(fig)


# ------------------------------
# Classification (visual-read) figure builders
# ------------------------------
CLASS_LABELS = {0: "Negative", 1: "Positive"}
CLASS_PALETTE = {"Negative": CUSTOM_PALETTE[0], "Positive": CUSTOM_PALETTE[1]}


def make_class_balance_panel(
    df_demo: pd.DataFrame,
    out_dir: Path,
    filename: str = "visual_read_class_balance.png",
    target_col: str = "visual_read",
    group_col: str = "site",
):
    """Overall positive/negative counts, plus a breakdown by `group_col` if present."""
    df = df_demo.copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])
    if df.empty:
        return
    df["_label"] = df[target_col].round().astype(int).map(CLASS_LABELS)
    order = [label for label in ("Negative", "Positive") if label in set(df["_label"])]

    has_group = group_col in df.columns and df[group_col].notna().any()
    n_cols = 2 if has_group else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(5.6 * n_cols, 5.0), squeeze=False)
    axes = axes[0]

    ax = axes[0]
    counts = df["_label"].value_counts().reindex(order)
    sns.barplot(x=order, y=counts.to_numpy(), hue=order, order=order, hue_order=order,
                palette=[CLASS_PALETTE[o] for o in order], ax=ax, legend=False)
    for i, v in enumerate(counts.to_numpy()):
        ax.text(i, v, f"n={int(v)}", ha="center", va="bottom", fontsize=TICK_SIZE)
    style_axes(ax, "Visual read distribution", "Visual read", "Count")

    if has_group:
        ax = axes[1]
        tmp = df[[group_col, "_label"]].copy()
        tmp[group_col] = tmp[group_col].astype("string").fillna("NA")
        group_order = get_display_order(tmp[group_col], group_col)
        sns.countplot(data=tmp, x=group_col, hue="_label", order=group_order, hue_order=order,
                      palette=[CLASS_PALETTE[o] for o in order], ax=ax)
        style_axes(ax, f"Visual read by {group_col}", group_col, "Count", xrotation=20)
        style_legend(ax, title="Visual read", loc="best")

    finalize_figure(fig, rect=(0.0, 0.0, 0.99, 0.99))
    save_figure(fig, out_dir / filename, dpi=300)
    plt.close(fig)


def compute_classification_stats(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    n = len(y_true)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    accuracy = (tp + tn) / n if n > 0 else np.nan
    balanced_accuracy = np.nanmean([sensitivity, specificity])
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    f1 = (2 * precision * sensitivity / (precision + sensitivity)
          if precision and sensitivity and (precision + sensitivity) > 0 else np.nan)
    mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / mcc_denom if mcc_denom > 0 else np.nan

    return {
        "threshold": threshold, "n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": accuracy, "sensitivity": sensitivity, "specificity": specificity,
        "balanced_accuracy": balanced_accuracy, "f1": f1, "mcc": mcc,
    }


def make_roc_confusion_panel(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_dir: Path,
    filename: str = "visual_read_roc_confusion_panel.png",
    stats_csv_path: Path | None = None,
):
    """Two-panel figure: (A) ROC curve with AUC, (B) confusion matrix at the Youden-optimal threshold."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true, y_prob = y_true[mask].astype(int), y_prob[mask]
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = float(roc_auc_score(y_true, y_prob))
    best_idx = int(np.argmax(tpr - fpr))
    best_thr = float(thresholds[best_idx])

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2), squeeze=False)

    ax = axes[0][0]
    ax.plot(fpr, tpr, color=SEABORN_COLORS[0], linewidth=2.2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.2, alpha=0.6)
    ax.scatter([fpr[best_idx]], [tpr[best_idx]], color=SEABORN_COLORS[1], zorder=5, s=55,
               label=f"Best threshold = {best_thr:.2f}")
    style_axes(ax, "ROC curve: agreement with clinical read", "False positive rate", "True positive rate")
    style_legend(ax, loc="lower right")

    ax = axes[0][1]
    stats_best = compute_classification_stats(y_true, y_prob, best_thr)
    cm = np.array([[stats_best["tn"], stats_best["fp"]], [stats_best["fn"], stats_best["tp"]]])
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Negative", "Positive"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Negative", "Positive"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=LABEL_SIZE,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    style_axes(ax, f"Confusion matrix (thr={best_thr:.2f})", "Predicted", "Reference (clinical read)")
    ax.grid(False)

    add_panel_labels(axes, ["A", "B"])
    finalize_figure(fig, rect=(0.0, 0.0, 0.99, 0.99))
    save_figure(fig, out_dir / filename, dpi=300)
    plt.close(fig)

    if stats_csv_path is not None:
        rows = [
            {"threshold_type": "0.5", "auc": auc, **compute_classification_stats(y_true, y_prob, 0.5)},
            {"threshold_type": "youden_optimal", "auc": auc, **compute_classification_stats(y_true, y_prob, best_thr)},
        ]
        pd.DataFrame(rows).to_csv(stats_csv_path, index=False)
