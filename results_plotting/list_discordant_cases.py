#!/usr/bin/env python3
"""List discordant cases (false positives / false negatives) for a visual_read run.

A discordant case is one where the model's prediction disagrees with the expert
clinical read (visual_read), at whatever decision_threshold the run actually used
(saved in the sibling *_metrics.csv; defaults to 0.5 if absent - see
plot_visual_read_results.py's load_decision_threshold).

For each discordant case, this pulls together everything needed to inspect it:
predicted probability, distance from the decision threshold (a confidently-wrong
case near 0 or 1 is a more interesting interpretability target than a borderline
one near the threshold), demographics, and quantitative SUVR values (so you can
see e.g. whether a false positive sits near the quantitative boundary too, or is
a clear model error against a clearly-normal scan).

There is no validated individual-scan SUVR cutoff for these regions (confirmed with
Nadine's supervisor) - CenTauR's own CTR>2 threshold (Villemagne et al. 2023,
https://doi.org/10.1002/dad2.12454) is a group-level validation statistic, not a
clinical per-scan cutoff. So instead of thresholding against an invented number,
discordant cases are ranked by COHORT-RELATIVE PERCENTILE: where each case's SUVR
falls within the full set of scans for this run, for --quantitative_col (default
Universal) and --mtl_col (default MesialTemporal). This is descriptive positioning
within the sample, not a clinical claim, and will shift if the cohort changes (e.g.
adding more validation scans) - fine for a one-off analysis, worth re-deriving if
reused later on a different sample.

Using the cohort median (50th percentile) as a neutral, non-arbitrary split point
(not a diagnostic cutoff - just "above/below the middle of this sample"), discordant
cases are split three ways (visual_read is graded against flortaucipir's separate
FDA/EMA visual-read criterion, Moscoso et al. 2025,
https://doi.org/10.1001/jama.2025.7817, which keys on posterolateral temporal/
parietal signal and deliberately NOT on isolated mesial temporal signal, due to
flortaucipir's known off-target binding there):
  - model agrees with quantitative --quantitative_col (above/below the cohort
    median), disagrees with the clinical visual read -> the model may be siding
    with an independent quantitative signal against the human reader, not just
    "wrong."
  - isolated --mtl_col (MesialTemporal) elevation: --mtl_col's percentile sits
    at least --mtl_margin points above --quantitative_col's percentile for the
    SAME case (a within-subject, relative comparison - no cohort-wide cutoff
    needed), while --quantitative_col is still below the cohort median -> a
    distinct pattern: possible early/isolated tau the visual-read criterion is
    structurally not designed to credit, not necessarily a model or quantitative
    disagreement in the usual sense.
  - model disagrees with quantitative too, and no isolated MTL pattern -> a more
    conventional, unexplained model error.

--baseline_n also prints the most-confidently-CORRECT TP/TN cases, for a
comparison panel showing what attribution maps look like when the model is right.

Intended next step: feed the printed comma-separated ID list(s) into run_vis.py's
--vis_img_list to generate GradCAM/occlusion maps for exactly these subjects.

Usage:
    python results_plotting/list_discordant_cases.py /path/to/run_dir --dataset Gothenburg
"""

import argparse
from pathlib import Path

import pandas as pd

from plot_utils import (
    CONFUSION_CATEGORY_LABELS,
    confusion_display_label,
    find_demo_csv,
    resolve_existing_path,
    resolve_path,
)

DEFAULT_SUVR_COLS = ["Universal", "MetaTemporal", "MesialTemporal", "TemporoParietal", "Frontal"]
DEFAULT_DEMO_COLS = ["site", "age", "gender", "dx", "apoe", "amyloid_status", "CDR", "CDR_SB"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="Path to a results run directory trained with --targets visual_read.")
    parser.add_argument("--dataset", default="Gothenburg", help="Dataset name used in the result filenames.")
    parser.add_argument(
        "--demo_csv", default=None,
        help="Path to the demographics csv (defaults to data/demo_<dataset>.csv, then data/demo.csv).",
    )
    parser.add_argument(
        "--out_csv", default=None,
        help="Where to save the discordant-cases csv (defaults next to the run's other figures).",
    )
    parser.add_argument(
        "--top_n", type=int, default=10,
        help="How many of the most-confidently-discordant cases to print to the console, per direction (Model +, Visual read - and Model -, Visual read +).",
    )
    parser.add_argument(
        "--quantitative_col", default="Universal",
        help="Demographics column to use as the quantitative positivity reference (default: Universal tau SUVR composite). "
             "No validated cutoff exists, so this is ranked by cohort-relative percentile, not thresholded.",
    )
    parser.add_argument(
        "--mtl_col", default="MesialTemporal",
        help="Column to check for isolated elevation relative to --quantitative_col (same case, higher "
             "percentile) - a pattern the FDA/EMA visual-read criterion is not designed to credit "
             "(default: MesialTemporal).",
    )
    parser.add_argument(
        "--mtl_margin", type=float, default=15.0,
        help="Percentile-point margin by which --mtl_col's cohort-relative percentile must exceed "
             "--quantitative_col's for the same case to count as 'isolated MTL elevation' (default: 15.0).",
    )
    parser.add_argument(
        "--skip_quant_split", action="store_true",
        help="Skip the quantitative-percentile three-way split entirely (just list FP/FN with raw "
             "SUVR values and percentiles shown, no automatic bucketing).",
    )
    parser.add_argument(
        "--baseline_n", type=int, default=0,
        help="Also list this many of the most-confidently-CORRECT TP and TN cases (for an "
             "interpretability comparison panel against the discordant cases). 0 = skip.",
    )
    return parser.parse_args()


def find_zeroshot_results_csv(run_dir: Path, dataset: str) -> Path | None:
    matches = sorted((run_dir / "validation").glob(f"External_validation_{dataset}__*_zeroshot_results.csv"))
    return matches[0] if matches else None


def find_dataset_demo_csv(dataset: str) -> Path | None:
    script_path = Path(__file__).resolve()
    demo_candidates = [
        script_path.parents[1] / "data" / f"demo_{dataset}.csv",
        script_path.parents[2] / "data" / f"demo_{dataset}.csv",
    ]
    return next((p for p in demo_candidates if p.exists()), None)


def load_decision_threshold(preds_path: Path) -> float:
    metrics_path = preds_path.parent / preds_path.name.replace("_results.csv", "_metrics.csv")
    if metrics_path.exists():
        df_metrics = pd.read_csv(metrics_path)
        if "decision_threshold" in df_metrics.columns and len(df_metrics) > 0:
            return float(df_metrics["decision_threshold"].iloc[0])
    print(f"[WARNING] Could not find decision_threshold in {metrics_path}; defaulting to 0.5.")
    return 0.5


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()

    zeroshot_path = find_zeroshot_results_csv(run_dir, args.dataset)
    if zeroshot_path is not None:
        preds_path = zeroshot_path
        out_dir = run_dir / "validation" / "figures"
    else:
        evaluation_dir = run_dir / "evaluation" / args.dataset
        preds_path = resolve_existing_path(
            [
                run_dir / "validation" / args.dataset / f"Test_{args.dataset}_results.csv",
                evaluation_dir / f"Eval_{args.dataset}_results.csv",
            ],
            "predictions csv",
        )
        out_dir = run_dir / "figures"

    df_preds = pd.read_csv(preds_path)
    required_cols = {"ID_ind", "y", "prob", "pred"}
    missing = required_cols - set(df_preds.columns)
    if missing:
        raise ValueError(f"{preds_path} is missing columns {sorted(missing)} - not a visual_read results csv.")

    threshold = load_decision_threshold(preds_path)

    demo_path = resolve_path(args.demo_csv, lambda: find_dataset_demo_csv(args.dataset) or find_demo_csv())
    if demo_path is None:
        raise FileNotFoundError("Could not find a demographics csv - pass --demo_csv explicitly.")
    df_demo = pd.read_csv(demo_path)

    demo_cols = [c for c in DEFAULT_DEMO_COLS + DEFAULT_SUVR_COLS if c in df_demo.columns]
    df = pd.merge(
        df_preds[["ID_ind", "y", "prob", "pred"]],
        df_demo[["ID"] + demo_cols],
        left_on="ID_ind", right_on="ID", how="left",
    )
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["pred"] = pd.to_numeric(df["pred"], errors="coerce")
    df["prob"] = pd.to_numeric(df["prob"], errors="coerce")
    df = df.dropna(subset=["y", "pred", "prob"])
    df["y"] = df["y"].astype(int)
    df["pred"] = df["pred"].astype(int)

    df["category"] = [CONFUSION_CATEGORY_LABELS[(t, p)] for t, p in zip(df["y"], df["pred"])]
    df["dist_from_threshold"] = (df["prob"] - threshold).abs()

    if not args.skip_quant_split:
        if args.quantitative_col not in df.columns:
            print(f"[WARNING] '{args.quantitative_col}' not found; skipping quantitative-agreement split.")
        else:
            # Cohort-relative percentile rank (0-100), computed across every subject in
            # this run - not a clinical cutoff, just "where does this case's SUVR sit
            # relative to everyone else here."
            quant_pct = df[args.quantitative_col].rank(pct=True) * 100
            df["quant_percentile"] = quant_pct
            quant_above_median = (quant_pct >= 50).astype("Int64")
            df["quant_above_median"] = quant_above_median
            df["model_agrees_with_quant"] = (df["pred"] == quant_above_median)

            if args.mtl_col in df.columns:
                mtl_pct = df[args.mtl_col].rank(pct=True) * 100
                df["mtl_percentile"] = mtl_pct
                # Within-subject comparison (this case's MTL percentile vs. its own
                # quantitative-composite percentile) - no cohort-wide cutoff needed.
                df["isolated_mtl"] = (
                    (mtl_pct - quant_pct >= args.mtl_margin) & (quant_above_median == 0)
                )
            else:
                print(f"[WARNING] '{args.mtl_col}' not found; skipping isolated-MTL split.")

    df_discordant = df[df["category"].isin(["False Positive", "False Negative"])].copy()
    df_discordant = df_discordant.sort_values(["category", "dist_from_threshold"], ascending=[True, False])
    df_discordant = df_discordant.drop(columns=["ID"]).rename(columns={"ID_ind": "ID"})

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_csv) if args.out_csv else out_dir / "visual_read_discordant_cases.csv"
    df_discordant.to_csv(out_csv, index=False)

    n_fp = (df_discordant["category"] == "False Positive").sum()
    n_fn = (df_discordant["category"] == "False Negative").sum()
    print(
        f"[done] {len(df_discordant)} discordant cases "
        f"({n_fp} x [{confusion_display_label('False Positive')}], "
        f"{n_fn} x [{confusion_display_label('False Negative')}]) at threshold={threshold:.3f}"
    )
    print(f"       saved to {out_csv}")

    display_cols = ["ID", "prob", "dist_from_threshold"] + demo_cols
    percentile_cols = [c for c in ["quant_percentile", "mtl_percentile"] if c in df_discordant.columns]
    display_cols = display_cols + percentile_cols

    if "model_agrees_with_quant" in df_discordant.columns:
        has_mtl_split = "isolated_mtl" in df_discordant.columns
        is_isolated_mtl = df_discordant["isolated_mtl"] if has_mtl_split else pd.Series(False, index=df_discordant.index)

        human_read_questionable_all = df_discordant[(df_discordant["model_agrees_with_quant"] == True) & ~is_isolated_mtl]  # noqa: E712
        mtl_driven_all = df_discordant[is_isolated_mtl]
        model_error_all = df_discordant[(df_discordant["model_agrees_with_quant"] == False) & ~is_isolated_mtl]  # noqa: E712

        buckets = [
            (f"Model agrees with quantitative {args.quantitative_col} "
             "(above/below this run's cohort median), disagrees with clinical read",
             human_read_questionable_all),
        ]
        if has_mtl_split:
            buckets.append((
                f"Isolated {args.mtl_col} elevation (percentile >= {args.mtl_margin:g} pts above "
                f"{args.quantitative_col}'s, which itself is below the cohort median) - possible "
                "early tau not credited by the visual-read criterion",
                mtl_driven_all,
            ))
        buckets.append(("Model disagrees with quantitative too (unexplained model error)", model_error_all))

        for label, bucket_all in buckets:
            bucket = bucket_all.head(args.top_n)
            print(f"\n=== {label} (showing {len(bucket)} of {len(bucket_all)} total) ===")
            if not bucket.empty:
                print(bucket[display_cols].to_string(index=False))
                print(f"--vis_img_list {','.join(str(i) for i in bucket['ID'])}")
            else:
                print("(none)")
    else:
        for category in ["False Positive", "False Negative"]:
            subset = df_discordant[df_discordant["category"] == category].head(args.top_n)
            if subset.empty:
                continue
            print(f"\nTop {len(subset)} most-confidently-discordant [{confusion_display_label(category)}] cases (for --vis_img_list):")
            print(subset[display_cols].to_string(index=False))
            print(f"--vis_img_list {','.join(str(i) for i in subset['ID'])}")

    all_ids = ",".join(str(i) for i in df_discordant["ID"])
    print(f"\nAll {len(df_discordant)} discordant IDs: --vis_img_list {all_ids}")

    if args.baseline_n > 0:
        df_correct = df[df["category"].isin(["True Positive", "True Negative"])].copy()
        df_correct["dist_from_threshold"] = (df_correct["prob"] - threshold).abs()
        df_correct = df_correct.sort_values(["category", "dist_from_threshold"], ascending=[True, False])
        df_correct = df_correct.drop(columns=["ID"]).rename(columns={"ID_ind": "ID"})

        baseline_ids = []
        for category in ["True Positive", "True Negative"]:
            subset = df_correct[df_correct["category"] == category].head(args.baseline_n)
            if subset.empty:
                continue
            print(f"\nBaseline: top {len(subset)} most-confidently-agreeing [{confusion_display_label(category)}] cases:")
            print(subset[display_cols].to_string(index=False))
            baseline_ids += [str(i) for i in subset["ID"]]

        if baseline_ids:
            print(f"\nBaseline (correct) IDs: --vis_img_list {','.join(baseline_ids)}")


if __name__ == "__main__":
    main()
