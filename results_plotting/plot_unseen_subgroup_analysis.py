#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_heldout_test_results as mod


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create per-subject subgroup MAE boxplots for AVID unseen validation results."
    )
    parser.add_argument(
        "--run_dir",
        required=True,
        help="Run directory containing External_validation_AVID_unseen__*_zeroshot_results.csv.",
    )
    parser.add_argument(
        "--results_csv",
        default=None,
        help="Optional path to the AVID unseen results CSV.",
    )
    parser.add_argument(
        "--demo_csv",
        default=None,
        help="Optional path to demo_AVID_unseen.csv.",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory for subgroup panels.",
    )
    return parser.parse_args()


def find_results_csv(run_dir: Path) -> Path:
    matches = sorted(run_dir.glob("External_validation_AVID_unseen__*_zeroshot_results.csv"))
    if not matches:
        raise FileNotFoundError(f"Could not find validation results CSV in {run_dir}")
    return matches[0]


def find_demo_csv() -> Path | None:
    script_path = Path(__file__).resolve()
    demo_candidates = [
        script_path.parents[1] / "data" / "demo_AVID_unseen.csv",
        script_path.parents[2] / "data" / "demo_AVID_unseen.csv",
    ]
    return next((p for p in demo_candidates if p.exists()), None)


def infer_targets(df: pd.DataFrame) -> list[str]:
    targets = []
    for col in df.columns:
        if col.endswith("_y") and f"{col[:-2]}_pred" in df.columns:
            targets.append(col[:-2])
    return targets


def select_id_column(df: pd.DataFrame) -> str | None:
    for col in ["ID_ind", "ID"]:
        if col in df.columns:
            return col
    return None


def merge_demo(df_preds: pd.DataFrame, df_demo: pd.DataFrame, id_col: str | None) -> pd.DataFrame:
    if id_col is None:
        return pd.concat([df_preds.reset_index(drop=True), df_demo.reset_index(drop=True)], axis=1)
    demo_id_col = "ID" if "ID" in df_demo.columns else id_col
    if demo_id_col not in df_demo.columns:
        return pd.concat([df_preds.reset_index(drop=True), df_demo.reset_index(drop=True)], axis=1)
    return pd.merge(df_preds, df_demo, left_on=id_col, right_on=demo_id_col, how="left")


def build_subject_mae(df_results: pd.DataFrame, id_col: str) -> pd.DataFrame:
    targets = infer_targets(df_results)
    if not targets:
        raise ValueError("Could not infer target columns from results CSV.")

    rows = []
    for target in targets:
        y_col = f"{target}_y"
        pred_col = f"{target}_pred"
        tmp = df_results[[id_col, y_col, pred_col]].copy()
        tmp = tmp.rename(columns={y_col: "y", pred_col: "pred"})
        tmp["abs_error"] = (pd.to_numeric(tmp["pred"], errors="coerce") - pd.to_numeric(tmp["y"], errors="coerce")).abs()
        tmp = tmp.dropna(subset=["abs_error"])
        rows.append(tmp[[id_col, "abs_error"]])

    df_long = pd.concat(rows, ignore_index=True)
    df_subject = df_long.groupby(id_col, observed=False)["abs_error"].mean().reset_index()
    return df_subject.rename(columns={"abs_error": "mae"})


def make_group_boxplot(
    df: pd.DataFrame,
    group_col: str,
    display_label: str,
    out_dir: Path,
    filename: str,
    *,
    order: list[str] | None = None,
):
    if group_col not in df.columns:
        print(f"[skip] Missing column: {group_col}")
        return

    plot_df = df[[group_col, "mae"]].copy()
    plot_df[group_col] = plot_df[group_col].astype("string").fillna("NA")

    if order is None:
        order = mod.get_display_order(plot_df[group_col], group_col)

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    mod.sns.boxplot(
        data=plot_df,
        x=group_col,
        y="mae",
        order=order,
        ax=ax,
        palette=mod.get_category_palette(order),
        width=0.6,
        fliersize=0,
    )
    mod.sns.stripplot(
        data=plot_df,
        x=group_col,
        y="mae",
        order=order,
        ax=ax,
        color="white",
        edgecolor="black",
        linewidth=0.4,
        size=3.6,
        jitter=0.2,
        alpha=0.75,
    )

    counts = plot_df[group_col].value_counts().reindex(order).fillna(0).astype(int)
    xtick_labels = [f"{val}\n(n={counts[val]})" for val in order]
    ax.set_xticklabels(xtick_labels, rotation=0, fontsize=mod.TICK_SIZE)

    mod.style_axes(
        ax,
        f"Per-subject MAE by {display_label}",
        display_label,
        "MAE (SUVR)",
        xrotation=0,
    )
    mod.finalize_figure(fig)
    mod.save_figure(fig, out_dir / filename, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main():
    args = parse_args()
    mod.sns.set_theme(style="whitegrid", palette=mod.CUSTOM_PALETTE)

    run_dir = Path(args.run_dir).resolve()
    results_path = Path(args.results_csv).resolve() if args.results_csv else find_results_csv(run_dir)
    demo_path = Path(args.demo_csv).resolve() if args.demo_csv else find_demo_csv()

    if demo_path is None:
        raise FileNotFoundError("Could not find demo_AVID_unseen.csv. Provide --demo_csv.")

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        out_dir = run_dir / "figures_unseen_subgroups"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_results = pd.read_csv(results_path)
    df_results = df_results.loc[:, ~df_results.columns.astype(str).str.contains(r"^Unnamed")]
    df_demo = pd.read_csv(demo_path)
    df_demo = df_demo.loc[:, ~df_demo.columns.astype(str).str.contains(r"^Unnamed")]

    id_col = select_id_column(df_results)
    if id_col is None:
        raise ValueError("Could not find ID column (ID_ind or ID) in results CSV.")

    df_subject = build_subject_mae(df_results, id_col)
    df_merged = merge_demo(df_subject, df_demo, id_col)

    if "dx" in df_merged.columns and "dx_grouped" not in df_merged.columns:
        df_merged["dx_grouped"] = mod.combine_dx_groups(df_merged["dx"])

    if "age" in df_merged.columns:
        age = pd.to_numeric(df_merged["age"], errors="coerce")
        df_merged["age_group"] = pd.cut(
            age,
            bins=[50, 60, 70, 80, 90, 110],
            labels=["50-60", "60-70", "70-80", "80-90", "90+"],
            include_lowest=True,
            right=False,
        )

    sex_col = "gender" if "gender" in df_merged.columns else "sex"
    make_group_boxplot(
        df_merged,
        sex_col,
        "Sex",
        out_dir,
        "mae_by_sex.png",
        order=mod.get_display_order(df_merged[sex_col].astype("string"), sex_col) if sex_col in df_merged.columns else None,
    )

    if "age_group" in df_merged.columns:
        age_order = ["50-60", "60-70", "70-80", "80-90", "90+"]
        make_group_boxplot(
            df_merged,
            "age_group",
            "Age group",
            out_dir,
            "mae_by_age.png",
            order=[x for x in age_order if x in df_merged["age_group"].astype("string").unique()],
        )
    else:
        print("[skip] Missing column: age")

    dx_col = "dx" if "dx" in df_merged.columns else "dx_grouped"
    make_group_boxplot(
        df_merged,
        dx_col,
        "dx",
        out_dir,
        "mae_by_dx.png",
    )

    make_group_boxplot(
        df_merged,
        "apoe",
        "apoe",
        out_dir,
        "mae_by_apoe.png",
    )

    print(f"[done] per-subject subgroup plots saved to {out_dir}")


if __name__ == "__main__":
    main()
