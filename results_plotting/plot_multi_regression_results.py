#!/usr/bin/env python3

import argparse
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def load_single_target_plot_module():
    script_path = Path(__file__).resolve().parent / "plot_metatemporal_results.py"
    spec = importlib.util.spec_from_file_location("plot_metatemporal_results", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    validation_dir = run_dir / "validation" / "Gothenburg"
    preds_path = validation_dir / "Test_Gothenburg_results.csv"
    train_curve_path = run_dir / "train-test-split" / "trainning_metrics_per_epoch.csv"
    test_split_path = run_dir / "train-test-split" / "train-test-split_testing-set.csv"

    if not preds_path.exists():
        raise FileNotFoundError(f"Missing predictions csv: {preds_path}")
    if not train_curve_path.exists():
        raise FileNotFoundError(f"Missing training curve csv: {train_curve_path}")
    if not test_split_path.exists():
        raise FileNotFoundError(f"Missing test split csv: {test_split_path}")

    mod = load_single_target_plot_module()
    mod.sns.set_theme(style="whitegrid", palette=mod.CUSTOM_PALETTE)

    df_preds_wide = pd.read_csv(preds_path)
    if args.dedup:
        df_preds_wide = df_preds_wide.drop_duplicates().reset_index(drop=True)

    df_curve = pd.read_csv(train_curve_path)
    df_curve = df_curve.loc[:, ~df_curve.columns.str.contains(r"^Unnamed")]

    df_demo = pd.read_csv(test_split_path, index_col=0).copy()
    if args.dedup:
        df_demo = df_demo.drop_duplicates().reset_index(drop=True)

    targets = infer_targets(df_preds_wide)
    if not targets:
        raise ValueError("Could not infer target names from prediction csv columns.")

    summary_rows = []
    shared_fig_dir = run_dir / "figures_multi" / "_shared"
    shared_fig_dir.mkdir(parents=True, exist_ok=True)

    mod.make_training_curve(df_curve, shared_fig_dir)
    mod.make_training_curve_cut_first4(df_curve, shared_fig_dir)
    mod.make_training_curve_cut_first4_panel(df_curve, shared_fig_dir)

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
            mod.style_axes(
                ax,
                f"Reference vs Predicted\n{display_target} SUVR\ncolored by {col_display}",
                f"Reference {display_target} SUVR",
                f"Predicted {display_target} SUVR",
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
            fig.savefig(demo_fig_dir / f"true_vs_predicted_by_{col}.png", dpi=300)
            mod.plt.close(fig)
            n_demo_plots += 1

        mod.make_scatter_plot(target_df, out_dir, target)
        mod.make_residual_plot(target_df, out_dir, target)
        mod.make_residual_histogram(target_df, out_dir)
        mod.make_true_value_histogram(target_df, out_dir, target)

        df_err = mod.add_error_bins(df_merged.copy())
        bin_summary = mod.summarize_error_by_bin(df_err)
        bin_summary.to_csv(error_out_dir / "global_error_by_true_bin.csv", index=False)
        mod.make_mae_by_bin_plot(bin_summary, error_out_dir)
        mod.make_residual_trend_plot(df_err, error_out_dir, target)
        mod.save_subgroup_bin_summary(
            df_err,
            error_out_dir,
            subgroup_cols=["dx_grouped", "amyloid_status", "apoe", "CDR", "site", "gender", "sex"],
        )
        group_cols = ["dx_grouped", "amyloid_status", "apoe", "CDR", "site", "gender", "sex", "age"]
        mod.make_group_mae_boxstrip_plots(df_merged, error_out_dir, group_cols)
        mod.run_group_significance_tests(df_merged, error_out_dir, group_cols)
        mod.make_site_raw_value_plot(df_merged, error_out_dir, target)
        mod.make_site_error_correlation_plot(df_merged, error_out_dir, target)
        mod.make_training_distribution_plot(run_dir, error_out_dir, target)

        print(f"[done] {target}: saved plots to {out_dir}")
        print(f"[done] {target}: saved demographic plots to {demo_fig_dir} (count: {n_demo_plots})")
        print(f"[done] {target}: saved error analysis to {error_out_dir}")

    print(f"[done] shared training-curve plots saved to {shared_fig_dir}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(run_dir / "figures_multi" / "summary_metrics_by_target.csv", index=False)
    print(f"\nSaved summary metrics to: {run_dir / 'figures_multi' / 'summary_metrics_by_target.csv'}")


if __name__ == "__main__":
    main()
