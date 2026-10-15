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
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve

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


def resolve_path(path_value: str | None, fallback_fn) -> Path | None:
    """A CLI-supplied path wins; otherwise fall back to auto-detection (e.g. find_demo_csv).
    Used for demographics CSVs that don't live at the default data/demo.csv (e.g. a
    dataset-specific file like demo_test_subset_tau_raw.csv on the cluster)."""
    if path_value:
        return Path(path_value).expanduser().resolve()
    return fallback_fn()


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
    title: str = "Visual read distribution",
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
    style_axes(ax, title, "Visual read", "Count")

    if has_group:
        ax = axes[1]
        tmp = df[[group_col, "_label"]].copy()
        tmp[group_col] = tmp[group_col].astype("string").fillna("NA")
        group_order = get_display_order(tmp[group_col], group_col)
        sns.countplot(data=tmp, x=group_col, hue="_label", order=group_order, hue_order=order,
                      palette=[CLASS_PALETTE[o] for o in order], ax=ax)
        style_axes(ax, f"{title} by {group_col}", group_col, "Count", xrotation=20)
        style_legend(ax, title="Visual read", loc="best")

    finalize_figure(fig, rect=(0.0, 0.0, 0.99, 0.99))
    save_figure(fig, out_dir / filename, dpi=300)
    plt.close(fig)


def compute_classification_stats(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {"threshold": threshold, **compute_classification_stats_from_preds(y_true, y_pred)}


def compute_classification_stats_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    n = len(y_true)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    accuracy = (tp + tn) / n if n > 0 else np.nan
    balanced_accuracy = np.nanmean([sensitivity, specificity])
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan  # == PPV
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    f1 = (2 * precision * sensitivity / (precision + sensitivity)
          if precision and sensitivity and (precision + sensitivity) > 0 else np.nan)
    mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / mcc_denom if mcc_denom > 0 else np.nan

    return {
        "n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": accuracy, "sensitivity": sensitivity, "specificity": specificity,
        "ppv": precision, "npv": npv,
        "balanced_accuracy": balanced_accuracy, "f1": f1, "mcc": mcc,
    }


def make_roc_confusion_panel(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    out_dir: Path,
    filename: str = "visual_read_roc_confusion_panel.png",
    stats_csv_path: Path | None = None,
):
    """Two-panel figure: (A) ROC curve with AUC, (B) confusion matrix from the model's
    actual predicted visual_read (0/1, already decided at inference time), not a
    threshold searched post-hoc on this same set."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = np.asarray(y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_prob) & np.isfinite(y_pred)
    y_true, y_prob, y_pred = y_true[mask].astype(int), y_prob[mask], y_pred[mask].astype(int)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = float(roc_auc_score(y_true, y_prob))
    auprc = float(average_precision_score(y_true, y_prob))

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2), squeeze=False)

    ax = axes[0][0]
    ax.plot(fpr, tpr, color=SEABORN_COLORS[0], linewidth=2.2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.2, alpha=0.6)
    style_axes(ax, "ROC curve: agreement with clinical read", "False positive rate", "True positive rate")
    style_legend(ax, loc="lower right")

    ax = axes[0][1]
    stats_pred = compute_classification_stats_from_preds(y_true, y_pred)
    cm = np.array([[stats_pred["tn"], stats_pred["fp"]], [stats_pred["fn"], stats_pred["tp"]]])
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Negative", "Positive"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Negative", "Positive"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=LABEL_SIZE,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    style_axes(ax, "Confusion matrix (model prediction)", "Predicted", "Reference (clinical read)")
    ax.grid(False)

    add_panel_labels(axes, ["A", "B"])
    finalize_figure(fig, rect=(0.0, 0.0, 0.99, 0.99))
    save_figure(fig, out_dir / filename, dpi=300)
    plt.close(fig)

    if stats_csv_path is not None:
        rows = [{"auc": auc, "auprc": auprc, **stats_pred}]
        pd.DataFrame(rows).to_csv(stats_csv_path, index=False)


def make_pr_curve_panel(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_dir: Path,
    filename: str = "visual_read_pr_curve.png",
):
    """Precision-recall curve with AUPRC, the threshold-free companion to the ROC panel.
    Under class imbalance, ROC can look deceptively good (true-negative rate dominates),
    while PR curves show the real operating-point tradeoffs. The no-skill baseline plotted
    is the actual positive-class prevalence, not 0.5 - the correct PR chance level."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true, y_prob = y_true[mask].astype(int), y_prob[mask]
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auprc = float(average_precision_score(y_true, y_prob))
    prevalence = float(y_true.mean())  # correct PR no-skill baseline, not 0.5

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot(recall, precision, color=SEABORN_COLORS[0], linewidth=2.2, label=f"AUPRC = {auprc:.3f}")
    ax.axhline(prevalence, linestyle="--", color="gray", linewidth=1.2, alpha=0.6,
               label=f"No-skill (prevalence) = {prevalence:.3f}")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(-0.02, 1.02)
    style_axes(ax, "Precision-recall curve: agreement with clinical read", "Recall (sensitivity)", "Precision (PPV)")
    style_legend(ax, loc="best")

    finalize_figure(fig, rect=(0.0, 0.0, 0.99, 0.99))
    save_figure(fig, out_dir / filename, dpi=300)
    plt.close(fig)


def make_probability_by_class_panel(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    out_dir: Path,
    filename: str = "visual_read_probability_by_class.png",
):
    """KDE of predicted probability, split by true clinical-read class - a sanity check for
    whether class reweighting (--class_weight_cls) is already doing most of the work of
    separating the classes, and whether the chosen decision threshold sits in a sparse or
    dense region of either distribution."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true, y_prob = y_true[mask].astype(int), y_prob[mask]
    if len(y_true) == 0:
        return

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    labels = {0: "Negative (clinical read)", 1: "Positive (clinical read)"}
    colors = {0: SEABORN_COLORS[0], 1: SEABORN_COLORS[1]}
    kde_vals = {}
    for cls in (0, 1):
        vals = y_prob[y_true == cls]
        if len(vals) >= 2:
            sns.kdeplot(vals, ax=ax, color=colors[cls], linewidth=2.2, fill=True, alpha=0.15,
                        label=f"{labels[cls]} (n={len(vals)})")
            kde_vals[cls] = vals
        elif len(vals) == 1:
            ax.axvline(vals[0], color=colors[cls], linewidth=2.2, label=f"{labels[cls]} (n=1)")

    # Rug ticks (actual values) drawn after all KDE curves - each kdeplot() call autoscales
    # the y-axis, which would otherwise clip an earlier rugplot()'s bottom margin extension.
    # A small/near-constant subgroup (e.g. a tiny imbalanced CV fold) can make the KDE
    # degenerate into a razor-thin spike; the rug keeps the actual values legible regardless.
    for cls, vals in kde_vals.items():
        sns.rugplot(vals, ax=ax, color=colors[cls], height=0.05, linewidth=1.6, alpha=0.8)

    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.6,
               label=f"Decision threshold = {threshold:.3f}")
    ax.set_xlim(-0.02, 1.02)
    style_axes(ax, "Predicted probability by true class", "Predicted probability", "Density")
    style_legend(ax, loc="best")

    finalize_figure(fig, rect=(0.0, 0.0, 0.99, 0.99))
    save_figure(fig, out_dir / filename, dpi=300)
    plt.close(fig)


CONFUSION_CATEGORY_LABELS = {
    (0, 0): "True Negative",
    (1, 1): "True Positive",
    (0, 1): "False Positive",
    (1, 0): "False Negative",
}
CONFUSION_CATEGORY_ORDER = ["True Negative", "True Positive", "False Positive", "False Negative"]
CONFUSION_CATEGORY_PALETTE = {
    "True Negative": CUSTOM_PALETTE[0],   # blue
    "True Positive": CUSTOM_PALETTE[3],   # green
    "False Positive": CUSTOM_PALETTE[1],  # red
    "False Negative": CUSTOM_PALETTE[4],  # yellow
}


def make_confusion_category_distribution_panel(
    df_preds: pd.DataFrame,
    df_demo: pd.DataFrame,
    out_dir: Path,
    value_col: str = "Universal",
    value_label: str | None = None,
    filename: str = "visual_read_universal_suvr_by_outcome.png",
    id_col_preds: str = "ID_ind",
    id_col_demo: str = "ID",
):
    """Density (KDE) of a continuous variable (e.g. the Universal tau SUVR composite),
    one line per confusion-matrix category (TP/TN/FP/FN) using the model's actual
    predicted visual_read - shows whether discordant cases (FP/FN) sit near the
    decision boundary rather than being clear-cut model errors."""
    if value_col not in df_demo.columns:
        print(f"[WARNING] '{value_col}' not found in demographics table; skipping confusion-category distribution panel.")
        return

    df = pd.merge(
        df_preds[[id_col_preds, "y", "pred"]],
        df_demo[[id_col_demo, value_col]],
        left_on=id_col_preds, right_on=id_col_demo, how="left",
    )
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["pred"] = pd.to_numeric(df["pred"], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["y", "pred", value_col])
    if df.empty or df["y"].nunique() < 2:
        return

    y_true = df["y"].astype(int).to_numpy()
    y_pred = df["pred"].astype(int).to_numpy()

    df["_category"] = [CONFUSION_CATEGORY_LABELS[(t, p)] for t, p in zip(y_true, y_pred)]

    order = [c for c in CONFUSION_CATEGORY_ORDER if (df["_category"] == c).sum() >= 2]
    if not order:
        return

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for category in order:
        vals = df.loc[df["_category"] == category, value_col]
        sns.kdeplot(vals, ax=ax, color=CONFUSION_CATEGORY_PALETTE[category], linewidth=2.2,
                    label=f"{category} (n={len(vals)})")

    label = value_label or value_col
    style_axes(ax, f"{label} by classification outcome", label, "Density")
    style_legend(ax, loc="best")

    finalize_figure(fig, rect=(0.0, 0.0, 0.99, 0.99))
    save_figure(fig, out_dir / filename, dpi=300)
    plt.close(fig)


def make_confusion_category_categorical_panel(
    df_preds: pd.DataFrame,
    df_demo: pd.DataFrame,
    out_dir: Path,
    cat_col: str = "apoe",
    cat_label: str | None = None,
    filename: str | None = None,
    id_col_preds: str = "ID_ind",
    id_col_demo: str = "ID",
):
    """Within-group proportion of a categorical variable (e.g. APOE e4 carrier status),
    one group of bars per confusion-matrix category (TP/TN/FP/FN) - an AD-epidemiology
    construct-validity check: true positives should skew toward known AD risk factors
    (APOE e4 carriage, etc.) more than true negatives, and discordant cases are the
    interesting middle ground. Normalized within each category (not raw counts) since
    FP/FN groups are typically much smaller than TP/TN."""
    if cat_col not in df_demo.columns:
        print(f"[WARNING] '{cat_col}' not found in demographics table; skipping confusion-category categorical panel.")
        return

    df = pd.merge(
        df_preds[[id_col_preds, "y", "pred"]],
        df_demo[[id_col_demo, cat_col]],
        left_on=id_col_preds, right_on=id_col_demo, how="left",
    )
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["pred"] = pd.to_numeric(df["pred"], errors="coerce")
    df = df.dropna(subset=["y", "pred", cat_col])
    if df.empty or df["y"].nunique() < 2:
        return

    y_true = df["y"].astype(int).to_numpy()
    y_pred = df["pred"].astype(int).to_numpy()
    df["_category"] = [CONFUSION_CATEGORY_LABELS[(t, p)] for t, p in zip(y_true, y_pred)]
    df[cat_col] = df[cat_col].astype("string").fillna("NA")

    order = [c for c in CONFUSION_CATEGORY_ORDER if (df["_category"] == c).sum() >= 2]
    if not order:
        return

    cat_order = get_display_order(df[cat_col], cat_col)
    proportions = (
        df.groupby("_category")[cat_col]
        .value_counts(normalize=True)
        .rename("proportion")
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sns.barplot(
        data=proportions, x="_category", y="proportion", hue=cat_col,
        order=order, hue_order=cat_order, ax=ax,
    )
    label = cat_label or cat_col
    counts = df["_category"].value_counts()
    xticklabels = [f"{c}\n(n={counts.get(c, 0)})" for c in order]
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(xticklabels)
    style_axes(ax, f"{label} by classification outcome", "", "Proportion within outcome")
    style_legend(ax, title=label, loc="best")

    finalize_figure(fig, rect=(0.0, 0.0, 0.99, 0.99))
    save_figure(fig, out_dir / (filename or f"visual_read_{cat_col}_by_outcome.png"), dpi=300)
    plt.close(fig)


def make_probability_vs_suvr_scatter(
    df_preds: pd.DataFrame,
    df_demo: pd.DataFrame,
    out_dir: Path,
    decision_threshold: float,
    value_col: str = "Universal",
    value_label: str | None = None,
    filename: str | None = None,
    id_col_preds: str = "ID_ind",
    id_col_demo: str = "ID",
):
    """Model probability (continuous) vs. quantitative tau SUVR (continuous), one point
    per subject, colored by confusion-matrix category (TP/TN/FP/FN against the clinical
    visual read). Unlike the KDE-by-outcome panel (which bins into 4 discrete groups and
    shows distributions), this shows the actual joint relationship between what the model
    predicts and the quantitative signal - where the model's decision boundary sits
    relative to SUVR, and whether discordant points sit near clearly-positive/negative
    quantitative territory or in between. No SUVR cutoff is drawn (no validated per-scan
    cutoff exists for these regions) - only the model's own decision threshold, which is
    an actual value the model uses."""
    if value_col not in df_demo.columns:
        print(f"[WARNING] '{value_col}' not found in demographics table; skipping probability-vs-SUVR scatter.")
        return

    df = pd.merge(
        df_preds[[id_col_preds, "y", "pred", "prob"]],
        df_demo[[id_col_demo, value_col]],
        left_on=id_col_preds, right_on=id_col_demo, how="left",
    )
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["pred"] = pd.to_numeric(df["pred"], errors="coerce")
    df["prob"] = pd.to_numeric(df["prob"], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["y", "pred", "prob", value_col])
    if df.empty or df["y"].nunique() < 2:
        return

    y_true = df["y"].astype(int).to_numpy()
    y_pred = df["pred"].astype(int).to_numpy()
    df["_category"] = [CONFUSION_CATEGORY_LABELS[(t, p)] for t, p in zip(y_true, y_pred)]

    order = [c for c in CONFUSION_CATEGORY_ORDER if (df["_category"] == c).sum() >= 1]
    if not order:
        return

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for category in order:
        subset = df[df["_category"] == category]
        ax.scatter(
            subset[value_col], subset["prob"],
            color=CONFUSION_CATEGORY_PALETTE[category], label=f"{category} (n={len(subset)})",
            s=45, alpha=0.75, edgecolor="white", linewidth=0.5,
        )

    ax.axhline(decision_threshold, color="black", linestyle="--", linewidth=1.6,
               label=f"Decision threshold = {decision_threshold:.3f}")

    label = value_label or value_col
    ax.set_ylim(-0.02, 1.02)
    style_axes(ax, f"Model probability vs. {label}", label, "Predicted probability")
    style_legend(ax, loc="best")

    finalize_figure(fig, rect=(0.0, 0.0, 0.99, 0.99))
    save_figure(fig, out_dir / (filename or f"visual_read_probability_vs_{value_col.lower()}.png"), dpi=300)
    plt.close(fig)


CENTILOID_BIN_EDGES = [-np.inf, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, np.inf]
CENTILOID_BIN_LABELS = [
    "<-50", "-50–-40", "-40–-30", "-30–-20", "-20–-10", "-10–0",
    "0–10", "10–20", "20–30", "30–40", "40–50", "50–60",
    "60–70", "70–80", "80–90", "90–100", ">100",
]


def _plot_line(ax, df: pd.DataFrame, bin_labels, min_bin_n: int, color: str, label: str, metric_col: str = "_agree"):
    by_bin = df.groupby("_bin", observed=False)[metric_col].mean().reindex(bin_labels)
    n_by_bin = df.groupby("_bin", observed=False)[metric_col].size().reindex(bin_labels, fill_value=0)
    y = by_bin.where(n_by_bin >= min_bin_n)
    ax.plot(
        bin_labels, y, marker="o", markersize=7, linewidth=2.2,
        color=color, label=f"{label} (n={int(n_by_bin.sum())})", zorder=3,
    )


def _plot_agreement_breakdown(ax, df: pd.DataFrame, bin_labels, min_bin_n: int):
    """Splits 'agreement' into the 3 metrics that actually compose it - overall accuracy,
    sensitivity (model's hit rate on clinician-positive cases), and specificity (model's
    hit rate on clinician-negative cases) - instead of one blended line. This surfaces
    WHERE disagreement concentrates: e.g. overall accuracy can look fine while sensitivity
    specifically collapses in the equivocal mid-range, which a single accuracy line hides."""
    _plot_line(ax, df, bin_labels, min_bin_n, CUSTOM_PALETTE[1], "Overall agreement")
    _plot_line(ax, df[df["y"] == 1], bin_labels, min_bin_n, CUSTOM_PALETTE[3], "Sensitivity (clinician-positive cases)")
    _plot_line(ax, df[df["y"] == 0], bin_labels, min_bin_n, CUSTOM_PALETTE[5], "Specificity (clinician-negative cases)")


def make_agreement_by_centiloid_panel(
    df_preds: pd.DataFrame,
    df_demo: pd.DataFrame,
    out_dir: Path,
    value_col: str = "CL",
    value_label: str = "Centiloid",
    group_col: str | None = None,
    group_label: str | None = None,
    min_bin_n: int = 2,
    filename: str | None = None,
    id_col_preds: str = "ID_ind",
    id_col_demo: str = "ID",
):
    """Model/clinical-read agreement binned over quantitative amyloid burden, styled after
    Moscoso et al. 2025 (JAMA, https://doi.org/10.1001/jama.2025.7817) Fig 2: a line of
    binned agreement against a background histogram (right axis) of what fraction of scans
    fall in each bin. Makes the usual finding visible at a glance - agreement is highest at
    the clearly-negative and clearly-positive extremes and dips in the clinically-equivocal
    mid-range, which is also where most scans actually sit - something a raw per-subject
    scatter doesn't show directly. Bin edges match the source figure's Centiloid bins.
    By default draws 3 lines - overall agreement, sensitivity, specificity (see
    _plot_agreement_breakdown) - matching the source figure's 3-line look. Pass group_col
    (e.g. site) instead to compare overall agreement across subgroups; the two modes are
    mutually exclusive (group_col disables the sensitivity/specificity breakdown)."""
    if value_col not in df_demo.columns:
        print(f"[WARNING] '{value_col}' not found in demographics table; skipping agreement-by-{value_col} panel.")
        return

    merge_cols = [id_col_demo, value_col]
    has_group = group_col is not None and group_col in df_demo.columns
    if group_col is not None and not has_group:
        print(f"[WARNING] group_col '{group_col}' not found in demographics table; plotting a single overall line.")
    if has_group:
        merge_cols.append(group_col)

    df = pd.merge(
        df_preds[[id_col_preds, "y", "pred"]],
        df_demo[merge_cols],
        left_on=id_col_preds, right_on=id_col_demo, how="left",
    )
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["pred"] = pd.to_numeric(df["pred"], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["y", "pred", value_col])
    if df.empty:
        return

    df["_bin"] = pd.cut(df[value_col], bins=CENTILOID_BIN_EDGES, labels=CENTILOID_BIN_LABELS, right=False)
    df["_agree"] = (df["y"] == df["pred"]).astype(float)

    bin_counts = df["_bin"].value_counts().reindex(CENTILOID_BIN_LABELS, fill_value=0)
    pct_of_scans = bin_counts / bin_counts.sum() * 100

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax2 = ax.twinx()
    ax2.bar(CENTILOID_BIN_LABELS, pct_of_scans.to_numpy(), color="0.85", width=0.65, zorder=1)
    ax2.set_ylabel("Percentage of scans (%)", fontsize=LABEL_SIZE)
    ax2.tick_params(axis="y", labelsize=TICK_SIZE)
    ax2.grid(False)

    if has_group:
        groups = get_display_order(df[group_col], group_col)
        for i, g in enumerate(groups):
            _plot_line(ax, df[df[group_col] == g], CENTILOID_BIN_LABELS, min_bin_n, CUSTOM_PALETTE[i % len(CUSTOM_PALETTE)], str(g))
        style_legend(ax, title=group_label or group_col, loc="lower center")
    else:
        _plot_agreement_breakdown(ax, df, CENTILOID_BIN_LABELS, min_bin_n)
        style_legend(ax, loc="lower center")

    ax.set_ylim(0, 1.05)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    style_axes(ax, f"Model-clinician agreement vs. {value_label}", value_label, "Agreement", xrotation=45)

    finalize_figure(fig, rect=(0.0, 0.0, 0.99, 0.99))
    save_figure(fig, out_dir / (filename or f"visual_read_agreement_by_{value_col.lower()}.png"), dpi=300)
    plt.close(fig)


def _quantile_bin_labels(series: pd.Series, n_bins: int):
    """Equal-count quantile bin edges/labels for a continuous column, with open-ended end
    labels ("<x" / ">x") - used where no validated per-scan clinical cutoff exists, so bins
    are data-driven rather than literature thresholds. Returns (None, None) if there isn't
    enough distinct data to form at least 2 bins."""
    n_bins = min(n_bins, series.nunique() - 1)
    if n_bins < 2:
        return None, None
    try:
        _, bin_edges = pd.qcut(series, q=n_bins, retbins=True, duplicates="drop")
    except ValueError:
        return None, None
    if len(bin_edges) < 3:
        return None, None

    labels = []
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == 0:
            labels.append(f"<{hi:.2f}")
        elif i == len(bin_edges) - 2:
            labels.append(f">{lo:.2f}")
        else:
            labels.append(f"{lo:.2f}–{hi:.2f}")
    return bin_edges, labels


def _fixed_width_bin_labels(series: pd.Series, n_bins: int):
    """Fixed-width bin edges/labels spanning this column's actual min-max range. Unlike
    equal-count quantile bins (which are ~uniform population by construction and so make
    the background histogram trivially flat), fixed-width bins reveal the true shape of
    the distribution - e.g. a right-skewed SUVR distribution shows up as most scans
    concentrated in the lower bins, which is the whole point of drawing that histogram."""
    lo, hi = series.min(), series.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return None, None
    bin_edges = np.linspace(lo, hi, n_bins + 1)
    labels = [f"{bin_edges[i]:.2f}–{bin_edges[i + 1]:.2f}" for i in range(n_bins)]
    return bin_edges, labels


def make_agreement_by_value_panel(
    df_preds: pd.DataFrame,
    df_demo: pd.DataFrame,
    out_dir: Path,
    value_col: str,
    value_label: str | None = None,
    n_bins: int = 6,
    min_bin_n: int = 2,
    group_col: str | None = None,
    group_label: str | None = None,
    filename: str | None = None,
    id_col_preds: str = "ID_ind",
    id_col_demo: str = "ID",
):
    """Same style as make_agreement_by_centiloid_panel (binned agreement line over a
    background histogram of % of scans per bin) but for an arbitrary quantitative column
    (e.g. a regional tau SUVR) using data-driven fixed-width bins instead of the
    Centiloid-specific literature bins - there's no literature SUVR cutoff per region to
    anchor fixed bins to, but fixed-width (not equal-count) still shows the real
    distribution shape. Use this for the 5 CenTauR regions; use
    make_agreement_by_centiloid_panel for Centiloid itself to keep the literature-matching
    bins. Optional group_col draws one line per subgroup (e.g. site) instead of one overall
    line, matching the source figure's multi-line comparison."""
    if value_col not in df_demo.columns:
        print(f"[WARNING] '{value_col}' not found in demographics table; skipping agreement-by-{value_col} panel.")
        return

    merge_cols = [id_col_demo, value_col]
    has_group = group_col is not None and group_col in df_demo.columns
    if group_col is not None and not has_group:
        print(f"[WARNING] group_col '{group_col}' not found in demographics table; plotting a single overall line.")
    if has_group:
        merge_cols.append(group_col)

    df = pd.merge(
        df_preds[[id_col_preds, "y", "pred"]],
        df_demo[merge_cols],
        left_on=id_col_preds, right_on=id_col_demo, how="left",
    )
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["pred"] = pd.to_numeric(df["pred"], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["y", "pred", value_col])
    if df.empty:
        return

    bin_edges, bin_labels = _fixed_width_bin_labels(df[value_col], n_bins)
    if bin_labels is None:
        return
    df["_bin"] = pd.cut(df[value_col], bins=bin_edges, labels=bin_labels, include_lowest=True)
    df["_agree"] = (df["y"] == df["pred"]).astype(float)

    bin_counts = df["_bin"].value_counts().reindex(bin_labels, fill_value=0)
    pct_of_scans = bin_counts / bin_counts.sum() * 100

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax2 = ax.twinx()
    ax2.bar(bin_labels, pct_of_scans.to_numpy(), color="0.85", width=0.65, zorder=1)
    ax2.set_ylabel("Percentage of scans (%)", fontsize=LABEL_SIZE)
    ax2.tick_params(axis="y", labelsize=TICK_SIZE)
    ax2.grid(False)

    if has_group:
        groups = get_display_order(df[group_col], group_col)
        for i, g in enumerate(groups):
            _plot_line(ax, df[df[group_col] == g], bin_labels, min_bin_n, CUSTOM_PALETTE[i % len(CUSTOM_PALETTE)], str(g))
        style_legend(ax, title=group_label or group_col, loc="lower center")
    else:
        _plot_agreement_breakdown(ax, df, bin_labels, min_bin_n)
        style_legend(ax, loc="lower center")

    label_x = value_label or value_col

    ax.set_ylim(0, 1.05)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    style_axes(ax, f"Model-clinician agreement vs. {label_x}", label_x, "Agreement", xrotation=30)

    finalize_figure(fig, rect=(0.0, 0.0, 0.99, 0.99))
    save_figure(fig, out_dir / (filename or f"visual_read_agreement_by_{value_col.lower()}.png"), dpi=300)
    plt.close(fig)


def make_classification_rate_by_value_panel(
    df_preds: pd.DataFrame,
    df_demo: pd.DataFrame,
    out_dir: Path,
    value_col: str,
    value_label: str | None = None,
    class_col: str = "y",
    class_labels: tuple[str, str] = ("Negative (clinical read)", "Positive (clinical read)"),
    n_bins: int = 6,
    filename: str | None = None,
    id_col_preds: str = "ID_ind",
    id_col_demo: str = "ID",
):
    """Percentage classified negative vs. positive, binned over a quantitative SUVR column -
    a calibration-style curve (same spirit as Moscoso et al. 2025's SUVR-vs-visual-read
    figures) showing where classification rate transitions as quantitative tau burden
    increases. Bins are data-driven equal-count quantiles of this run's own cohort, not a
    literature/clinical cutoff (none validated for individual scans in these regions) -
    end bins are shown open-ended ("<x" / ">x") since they're the sparsest, most
    extreme-tail portions of the distribution. class_col defaults to the clinical read
    (y); pass class_col="pred" to show the model's own call rate instead."""
    if value_col not in df_demo.columns:
        print(f"[WARNING] '{value_col}' not found in demographics table; skipping classification-rate-by-{value_col} panel.")
        return

    df = pd.merge(
        df_preds[[id_col_preds, class_col]],
        df_demo[[id_col_demo, value_col]],
        left_on=id_col_preds, right_on=id_col_demo, how="left",
    )
    df[class_col] = pd.to_numeric(df[class_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[class_col, value_col])
    if df.empty or df[class_col].nunique() < 2:
        return

    n_bins = min(n_bins, df[value_col].nunique() - 1)
    if n_bins < 2:
        return
    try:
        _, bin_edges = pd.qcut(df[value_col], q=n_bins, retbins=True, duplicates="drop")
    except ValueError:
        return
    if len(bin_edges) < 3:
        return

    labels = []
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == 0:
            labels.append(f"<{hi:.2f}")
        elif i == len(bin_edges) - 2:
            labels.append(f">{lo:.2f}")
        else:
            labels.append(f"{lo:.2f}–{hi:.2f}")
    df["_bin"] = pd.cut(df[value_col], bins=bin_edges, labels=labels, include_lowest=True)

    rates = pd.crosstab(df["_bin"], df[class_col].astype(int), normalize="index") * 100
    rates = rates.reindex(index=labels, columns=[0, 1], fill_value=0)
    n_by_bin = df["_bin"].value_counts().reindex(labels, fill_value=0)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for cls, color, label in [(0, SEABORN_COLORS[0], class_labels[0]), (1, SEABORN_COLORS[1], class_labels[1])]:
        ax.plot(labels, rates[cls], marker="o", markersize=7, linewidth=2.2, color=color, label=label)

    ax.set_ylim(-2, 102)
    label_x = value_label or value_col
    style_axes(ax, f"Classification rate vs. {label_x}", label_x, "Percentage (%)", xrotation=30)
    xticklabels = [f"{lab}\n(n={n_by_bin[lab]})" for lab in labels]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(xticklabels)
    style_legend(ax, loc="center left")

    finalize_figure(fig, rect=(0.0, 0.0, 0.99, 0.99))
    save_figure(fig, out_dir / (filename or f"visual_read_classification_rate_by_{value_col.lower()}.png"), dpi=300)
    plt.close(fig)


def make_discordant_probability_violin_panel(
    df_preds: pd.DataFrame,
    out_dir: Path,
    decision_threshold: float | None = None,
    filename: str = "visual_read_discordant_probability_violin.png",
):
    """Predicted-probability distribution among discordant cases only (False Negatives:
    model=0/human=1, and False Positives: model=1/human=0), as a violin+stripplot - shows
    whether misses cluster near the decision threshold (borderline) or sit confidently on
    the wrong side (a more striking model/rater disagreement), without the much larger
    correctly-classified majority diluting the view the way the full-population
    probability-by-class panel does."""
    df = df_preds[["y", "pred", "prob"]].copy()
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["pred"] = pd.to_numeric(df["pred"], errors="coerce")
    df["prob"] = pd.to_numeric(df["prob"], errors="coerce")
    df = df.dropna(subset=["y", "pred", "prob"])
    if df.empty:
        return

    y_int = df["y"].astype(int)
    pred_int = df["pred"].astype(int)
    df["_category"] = [CONFUSION_CATEGORY_LABELS[(t, p)] for t, p in zip(y_int, pred_int)]

    order = [c for c in ["False Negative", "False Positive"] if (df["_category"] == c).sum() >= 1]
    if not order:
        return
    plot_df = df[df["_category"].isin(order)]

    fig, ax = plt.subplots(figsize=(6.5, 6))
    sns.violinplot(
        data=plot_df, x="_category", y="prob", order=order, ax=ax,
        color=SEABORN_COLORS[0], inner="box", cut=0,
    )
    sns.stripplot(
        data=plot_df, x="_category", y="prob", order=order, ax=ax,
        color=SEABORN_COLORS[5], size=5, alpha=0.85, jitter=0.12, linewidth=0,
    )
    if decision_threshold is not None:
        ax.axhline(decision_threshold, color="black", linestyle="--", linewidth=1.4,
                   label=f"Decision threshold = {decision_threshold:.3f}")
        style_legend(ax, loc="upper center")

    model_vals = {"False Negative": 0, "False Positive": 1}
    human_vals = {"False Negative": 1, "False Positive": 0}
    counts = plot_df["_category"].value_counts()
    xticklabels = [f"model: {model_vals[c]}\nhuman: {human_vals[c]}\n(n={counts.get(c, 0)})" for c in order]
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(xticklabels)
    ax.set_ylim(-0.02, 1.02)
    style_axes(ax, "Predicted probability among discordant cases", "", "Predicted probability")

    finalize_figure(fig, rect=(0.0, 0.0, 0.99, 0.99))
    save_figure(fig, out_dir / filename, dpi=300)
    plt.close(fig)
