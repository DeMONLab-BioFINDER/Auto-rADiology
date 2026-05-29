#!/usr/bin/env python3

import argparse
from pathlib import Path
import math

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

import plot_metatemporal_results as mod


def save_figure(fig, out_path: Path, **kwargs) -> bool:
    fig.savefig(out_path, **kwargs)
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create single-target style plots for a multi-regression run."
    )
    parser.add_argument("--run_dir", required=True, help="Path to the run directory.")
    parser.add_argument(
        "--dedup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop exact duplicate prediction rows before plotting.",
    )
    return parser.parse_args()


def infer_targets(df_preds: pd.DataFrame) -> list[str]:
    targets = []
    for col in df_preds.columns:
        if col.endswith("_y"):
            target = col[:-2]
            pred_col = f"{target}_pred"
            if pred_col in df_preds.columns:
                targets.append(target)
    return targets


def resolve_existing_path(candidates, label: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Missing {label}. Checked:\n{searched}")


def compute_metrics(df: pd.DataFrame) -> dict:
    y = pd.to_numeric(df["y"], errors="coerce").to_numpy()
    pred = pd.to_numeric(df["pred"], errors="coerce").to_numpy()
    finite = np.isfinite(y) & np.isfinite(pred)
    y = y[finite]
    pred = pred[finite]

    mae = float(mean_absolute_error(y, pred))
    rmse = float(root_mean_squared_error(y, pred))
    r2 = float(r2_score(y, pred)) if len(y) > 1 else np.nan
    corr = float(np.corrcoef(y, pred)[0, 1]) if len(y) > 1 else np.nan
    bias = float(np.mean(pred - y))
    medae = float(np.median(np.abs(pred - y)))

    return {
        "n": int(len(y)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson_r": corr,
        "bias": bias,
        "median_abs_error": medae,
    }


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


def find_demo_csv():
    script_path = Path(__file__).resolve()
    demo_candidates = [
        script_path.parents[1] / "data" / "demo.csv",
        script_path.parents[2] / "data" / "demo.csv",
    ]
    return next((p for p in demo_candidates if p.exists()), None)


def get_valid_target_values(df: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray]:
    y_true = pd.to_numeric(df[f"{target}_y"], errors="coerce").to_numpy()
    y_pred = pd.to_numeric(df[f"{target}_pred"], errors="coerce").to_numpy()
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[valid], y_pred[valid]


def get_plot_limits(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    lo = float(np.nanmin([y_true.min(), y_pred.min()]))
    hi = float(np.nanmax([y_true.max(), y_pred.max()]))
    return lo, hi


def get_target_label(target: str, mod) -> str:
    display_target = mod.pretty_region_name(target)
    return mod.format_suvr_label(display_target)


def hide_unused_axes(axes, start_idx: int, total_slots: int, n_cols: int):
    for idx in range(start_idx, total_slots):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        axes[row_idx][col_idx].set_visible(False)


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


def make_true_vs_predicted_panel(
    df_preds_wide: pd.DataFrame,
    out_dir: Path,
    targets: list[str],
    mod,
    title: str,
    filename: str,
):
    targets = order_targets(targets)
    if not targets:
        return

    n_cols = 2 if len(targets) > 1 else 1
    n_rows = math.ceil(len(targets) / n_cols)
    fig, axes = mod.plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 5.2 * n_rows), squeeze=False)

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

        display_target = mod.pretty_region_name(target)
        mod.style_axes(
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

    mod.finalize_figure(fig, rect=(0.0, 0.0, 0.99, 0.99))
    save_figure(fig, out_dir / filename, dpi=300)
    mod.plt.close(fig)


def make_suvr_distribution_panel(
    demo_path: Path,
    out_dir: Path,
    targets: list[str],
    mod,
    title: str,
    filename: str,
):
    df_demo = pd.read_csv(demo_path)
    targets = [t for t in order_targets(targets) if t in df_demo.columns]
    if not targets:
        return

    n_cols = 2 if len(targets) > 1 else 1
    n_rows = math.ceil(len(targets) / n_cols)
    fig, axes = mod.plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.6 * n_rows), squeeze=False)

    for idx, target in enumerate(targets):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r][c]

        values = pd.to_numeric(df_demo[target], errors="coerce").dropna()
        if values.empty:
            ax.set_visible(False)
            continue

        mod.sns.histplot(values, bins=22, kde=True, ax=ax, color=mod.SEABORN_COLORS[0])
        target_label = get_target_label(target, mod)
        mod.style_axes(
            ax,
            target_label.replace(" SUVR", ""),
            "SUVR",
            "Count",
        )

    hide_unused_axes(axes, len(targets), n_rows * n_cols, n_cols)
    add_panel_labels(axes, [chr(ord("A") + i) for i in range(len(targets))], xytext=(-3, 2), fontsize=13)

    mod.finalize_figure(fig, rect=(0.0, 0.0, 0.99, 0.99))
    save_figure(fig, out_dir / filename, dpi=300)
    mod.plt.close(fig)


def make_mae_boxstrip_panel(
    base_dir: Path,
    targets: list[str],
    mod,
    group: str,
    title: str,
    filename: str,
):
    targets = order_targets(targets)
    if not targets:
        return

    n_cols = 2 if len(targets) > 1 else 1
    n_rows = math.ceil(len(targets) / n_cols)
    fig, axes = mod.plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 5.4 * n_rows), squeeze=False)
    fig.subplots_adjust(wspace=0.01, hspace=0.02)

    for idx, target in enumerate(targets):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r][c]
        fig_path = base_dir / target / "figures_error_analysis" / f"mae_boxstrip_by_{group}.png"
        if not fig_path.exists():
            ax.set_visible(False)
            continue

        img = mod.plt.imread(fig_path)
        ax.imshow(img)
        ax.axis("off")

    hide_unused_axes(axes, len(targets), n_rows * n_cols, n_cols)

    add_panel_labels(axes, [chr(ord("A") + i) for i in range(len(targets))])
    mod.finalize_figure(fig, rect=(0.0, 0.0, 0.995, 0.995))
    save_figure(fig, base_dir / filename, dpi=300)
    mod.plt.close(fig)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    validation_dir = run_dir / "validation" / "Gothenburg"
    evaluation_dir = run_dir / "evaluation" / "Gothenburg"
    preds_path = resolve_existing_path(
        [
            validation_dir / "Test_Gothenburg_results.csv",
            evaluation_dir / "Eval_Gothenburg_results.csv",
        ],
        "predictions csv",
    )
    train_curve_path = resolve_existing_path(
        [
            run_dir / "train-test-split" / "trainning_metrics_per_epoch.csv",
            run_dir / "train-test-split" / "metrics" / "trainning_metrics_per_epoch.csv",
        ],
        "training curve csv",
    )
    test_split_path = resolve_existing_path(
        [
            run_dir / "train-test-split" / "train-test-split_testing-set.csv",
            run_dir / "train-test-split" / "preds" / "train-test-split_testing-set.csv",
            run_dir / "splits" / "Hold-out_testing-set.csv",
        ],
        "test split csv",
    )

    mod.sns.set_theme(style="whitegrid", palette=mod.CUSTOM_PALETTE)

    df_preds_wide = pd.read_csv(preds_path)
    if args.dedup:
        df_preds_wide = df_preds_wide.drop_duplicates().reset_index(drop=True)

    df_curve = pd.read_csv(train_curve_path)
    df_curve = df_curve.loc[:, ~df_curve.columns.str.contains(r"^Unnamed")]

    df_demo = pd.read_csv(test_split_path).copy()
    if "ID" not in df_demo.columns and "Unnamed: 0" in df_demo.columns:
        df_demo = df_demo.rename(columns={"Unnamed: 0": "ID"})
    df_demo = df_demo.loc[:, ~df_demo.columns.str.contains(r"^Unnamed")]
    if args.dedup:
        df_demo = df_demo.drop_duplicates().reset_index(drop=True)

    targets = infer_targets(df_preds_wide)
    if not targets:
        raise ValueError("Could not infer target names from prediction csv columns.")

    summary_rows = []
    shared_fig_dir = run_dir / "figures_multi" / "_shared"
    shared_fig_dir.mkdir(parents=True, exist_ok=True)
    has_validation_run = validation_dir.exists()
    shared_panel_name = "true_vs_predicted_panel_validation.png" if has_validation_run else "true_vs_predicted_panel_test.png"
    shared_panel_title = "Validation Set" if has_validation_run else "Test Set"

    mod.make_training_curve(df_curve, shared_fig_dir)
    mod.make_training_loss_curve(df_curve, shared_fig_dir)

    make_true_vs_predicted_panel(
        df_preds_wide,
        shared_fig_dir,
        targets,
        mod,
        title=f"{shared_panel_title}: Reference vs Predicted {mod.format_suvr_label()}",
        filename=shared_panel_name,
    )

    demo_path = find_demo_csv()
    if demo_path is not None:
        try:
            make_suvr_distribution_panel(
                demo_path,
                shared_fig_dir,
                targets,
                mod,
                title=f"Reference {mod.format_suvr_label()} Distributions",
                filename="suvr_distribution_panel.png",
            )
        except Exception as e:
            print(f"[WARNING] Could not create SUVR distribution panel from demo script: {e}")
            # Fallback to local implementation to avoid breaking existing output.
            make_suvr_distribution_panel(
                demo_path,
                shared_fig_dir,
                targets,
                mod,
                title=f"Reference {mod.format_suvr_label()} Distributions",
                filename="suvr_distribution_panel.png",
            )
    else:
        print("[WARNING] Could not find demo.csv for SUVR distribution panel.")

    # Create a shared (aggregate across targets) long-form dataframe to produce
    # combined MAE boxstrip plots for each demographic group.
    try:
        df_demo_for_merge = pd.read_csv(test_split_path).copy()
        if "ID" not in df_demo_for_merge.columns and "Unnamed: 0" in df_demo_for_merge.columns:
            df_demo_for_merge = df_demo_for_merge.rename(columns={"Unnamed: 0": "ID"})
        df_demo_for_merge = df_demo_for_merge.loc[:, ~df_demo_for_merge.columns.str.contains(r"^Unnamed")]
        if args.dedup:
            df_demo_for_merge = df_demo_for_merge.drop_duplicates().reset_index(drop=True)
    except Exception:
        df_demo_for_merge = pd.DataFrame()

    rows = []
    for target in targets:
        if f"{target}_y" not in df_preds_wide.columns or f"{target}_pred" not in df_preds_wide.columns:
            continue
        tmp = df_preds_wide[["ID_ind", f"{target}_y", f"{target}_pred"]].copy()
        tmp = tmp.rename(columns={f"{target}_y": "y", f"{target}_pred": "pred"})
        tmp["target"] = target
        if "ID_ind" in tmp.columns and "ID" in df_demo_for_merge.columns:
            merged = pd.merge(tmp, df_demo_for_merge, left_on="ID_ind", right_on="ID", how="left")
        else:
            merged = tmp.copy()
        rows.append(merged)

    if rows:
        df_all_long = pd.concat(rows, ignore_index=True)
        if "dx" in df_all_long.columns:
            df_all_long["dx_grouped"] = mod.combine_dx_groups(df_all_long["dx"])
        # Shared group columns to plot (same as per-target)
        shared_group_cols = ["dx_grouped", "amyloid_status", "apoe", "CDR", "site", "gender", "sex", "age"]
        # Generate shared MAE boxstrip plots in the shared figures directory
        # Use default title inside the plotting function ("Absolute Error") instead of "All targets"
        mod.make_group_mae_boxstrip_plots(df_all_long, shared_fig_dir, shared_group_cols, target_name=None)
        # Also create a concise two-panel figure (A: DX, B: Site) for main text
        try:
            mod.make_dx_site_two_panel_mae(df_all_long, shared_fig_dir)
        except Exception as e:
            print(f"[warning] could not create two-panel MAE figure: {e}")
        print(f"[done] shared MAE boxstrip plots saved to {shared_fig_dir}")

    for target in targets:
        display_target = mod.pretty_region_name(target)
        out_dir = run_dir / "figures_multi" / target
        demo_fig_dir = out_dir / "figures_demographics"
        error_out_dir = out_dir / "figures_error_analysis"
        out_dir.mkdir(parents=True, exist_ok=True)
        demo_fig_dir.mkdir(parents=True, exist_ok=True)
        error_out_dir.mkdir(parents=True, exist_ok=True)

        target_df = df_preds_wide[["ID_ind", f"{target}_y", f"{target}_pred"]].copy()
        target_df = target_df.rename(columns={f"{target}_y": "y", f"{target}_pred": "pred"})

        target_metrics = compute_metrics(target_df)
        summary_rows.append({"target": target, **target_metrics})
        df_metrics = pd.DataFrame([target_metrics])

        if "ID_ind" in target_df.columns and "ID" in df_demo.columns:
            df_merged = pd.merge(target_df, df_demo, left_on="ID_ind", right_on="ID", how="left")
        else:
            df_merged = target_df.copy()

        if "dx" in df_merged.columns:
            df_merged["dx_grouped"] = mod.combine_dx_groups(df_merged["dx"])

        demo_cols = [
            col
            for col in ["site", "age", "gender", "sex", "dx_grouped", "apoe", "amyloid_status", "CDR"]
            if col in df_merged.columns
        ]

        n_demo_plots = 0
        for col in demo_cols:
            vals = df_merged[col]
            if vals.isnull().all() or vals.nunique() <= 1:
                continue

            fig, ax = mod.plt.subplots(figsize=(6, 6))
            is_categorical = (col != "age") and (
                (not pd.api.types.is_numeric_dtype(vals)) or (vals.nunique(dropna=True) <= 12)
            )

            if is_categorical:
                cat_vals = vals.astype("string").fillna("NA")
                categories = mod.get_display_order(cat_vals, col)
                cat_to_code = {cat: i for i, cat in enumerate(categories)}
                color_codes = cat_vals.map(cat_to_code).astype(float)
                palette_map = mod.get_category_palette(categories)
                palette = [palette_map[cat] for cat in categories]
                cmap = mod.ListedColormap(palette)
                scatter = ax.scatter(
                    df_merged["y"], df_merged["pred"], c=color_codes, cmap=cmap,
                    alpha=0.82, s=36, edgecolor="white", linewidth=0.3
                )
            else:
                num_vals = pd.to_numeric(vals, errors="coerce")
                if num_vals.isnull().all():
                    mod.plt.close(fig)
                    continue
                scatter = ax.scatter(
                    df_merged["y"], df_merged["pred"], c=num_vals, cmap=mod.SEABORN_CMAP,
                    alpha=0.84, s=36, edgecolor="white", linewidth=0.3
                )

            lo = np.nanmin([df_merged["y"].min(), df_merged["pred"].min()])
            hi = np.nanmax([df_merged["y"].max(), df_merged["pred"].max()])
            ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1.2, alpha=0.5)
            z = np.polyfit(df_merged["y"].dropna(), df_merged["pred"].dropna(), 1)
            p = np.poly1d(z)
            ax.plot([lo, hi], p([lo, hi]), linestyle="-", color="black", linewidth=1.8)
            col_display = "dx" if col == "dx_grouped" else col
            label_map = {
                "apoe": "APOE",
                "amyloid_status": "Amyloid Status",
                "cdr": "CDR",
            }
            if col_display == "dx":
                display_group = "DX"
            else:
                key = col_display.lower()
                display_group = label_map.get(key, col_display.replace("_", " ").title())
            target_label = mod.format_suvr_label(display_target)
            mod.style_axes(
                ax,
                f"Reference vs Predicted\n{target_label}\nColored by {display_group}",
                f"Reference {target_label}",
                f"Predicted {target_label}",
            )

            if is_categorical:
                handles = [
                    mod.plt.Line2D(
                        [0], [0], marker="o", color="w", label=str(cat),
                        markerfacecolor=palette[i], markersize=8
                    )
                    for i, cat in enumerate(categories)
                ]
                ncol = 2 if len(categories) > 4 else 1
                ax.legend(handles=handles, loc="upper left" if col == "dx_grouped" else "lower right",
                          ncol=ncol, frameon=True, facecolor="white", framealpha=0.88,
                          fontsize=9 if col == "dx_grouped" else mod.LEGEND_SIZE)
            else:
                cbar = fig.colorbar(scatter, ax=ax, label=col)
                cbar.ax.tick_params(labelsize=mod.TICK_SIZE)
                cbar.set_label(col, fontsize=mod.LABEL_SIZE)

            mod.finalize_figure(fig)
            save_figure(fig, demo_fig_dir / f"true_vs_predicted_by_{col}.png", dpi=300)
            mod.plt.close(fig)
            n_demo_plots += 1

        mod.make_scatter_plot(target_df, out_dir, target)
        mod.make_residual_plot(target_df, out_dir, target)
        mod.make_residual_histogram(target_df, out_dir)
        mod.make_true_value_histogram(target_df, out_dir, target)

        df_err = mod.add_error_bins(df_merged.copy())
        bin_summary = mod.summarize_error_by_bin(df_err)
        bin_summary.to_csv(error_out_dir / "global_error_by_true_bin.csv", index=False)
        mod.make_mae_by_bin_plot(bin_summary, error_out_dir, target)
        # Intentionally disabled: residual_trend_vs_true.png
        mod.save_subgroup_bin_summary(
            df_err,
            error_out_dir,
            subgroup_cols=["dx_grouped", "amyloid_status", "apoe", "CDR", "site", "gender", "sex"],
        )
        group_cols = ["dx_grouped", "amyloid_status", "apoe", "CDR", "site", "gender", "sex", "age"]
        mod.make_group_mae_boxstrip_plots(df_merged, error_out_dir, group_cols, target)
        mod.run_group_significance_tests(df_merged, error_out_dir, group_cols)
        # Intentionally disabled: site_raw_value_distribution.png
        mod.make_site_error_correlation_plot(df_merged, error_out_dir, target)
        mod.make_training_distribution_plot(run_dir, error_out_dir, target)

        print(f"[done] {target}: saved plots to {out_dir}")
        print(f"[done] {target}: saved demographic plots to {demo_fig_dir} (count: {n_demo_plots})")
        print(f"[done] {target}: saved error analysis to {error_out_dir}")

    make_mae_boxstrip_panel(
        run_dir / "figures_multi",
        targets,
        mod,
        group="dx_grouped",
        title=f"{mod.format_suvr_label()} Absolute Error by DX",
        filename="mae_boxstrip_by_dx_panel.png",
    )
    make_mae_boxstrip_panel(
        run_dir / "figures_multi",
        targets,
        mod,
        group="gender",
        title=f"{mod.format_suvr_label()} Absolute Error by Gender",
        filename="mae_boxstrip_by_gender_panel.png",
    )
    make_mae_boxstrip_panel(
        run_dir / "figures_multi",
        targets,
        mod,
        group="site",
        title=f"{mod.format_suvr_label()} Absolute Error by Site",
        filename="mae_boxstrip_by_site_panel.png",
    )
    make_mae_boxstrip_panel(
        run_dir / "figures_multi",
        targets,
        mod,
        group="age",
        title=f"{mod.format_suvr_label()} Absolute Error by Age",
        filename="mae_boxstrip_by_age_panel.png",
    )
    make_mae_boxstrip_panel(
        run_dir / "figures_multi",
        targets,
        mod,
        group="apoe",
        title=f"{mod.format_suvr_label()} Absolute Error by APOE",
        filename="mae_boxstrip_by_apoe_panel.png",
    )

    print(f"[done] shared training-curve plots saved to {shared_fig_dir}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(run_dir / "figures_multi" / "summary_metrics_by_target.csv", index=False)
    print(f"\nSaved summary metrics to: {run_dir / 'figures_multi' / 'summary_metrics_by_target.csv'}")


if __name__ == "__main__":
    main()
