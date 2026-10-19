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

"Quantitatively positive" means the region's SUVR is at or above the SUVR equivalent of
CenTauRz = 2 (Villemagne et al. 2023, https://doi.org/10.1002/dad2.12454; per-region
values in plot_utils.CENTAUR_FTP_CTRZ2_SUVR, ~1.34 for MetaTemporal, ~1.35 for
MesialTemporal). CTRz 2 is a group-level CenTauR convention, not a validated per-scan
clinical cutoff (confirmed with Nadine's supervisor), so treat the split as descriptive.
The cohort-relative percentile of each case (quant_percentile / mtl_percentile) is also
saved, for orientation only - it does not drive the split.

The quantitative reference defaults to MetaTemporal, not Universal: Karagianni et al.
2026 (J Nucl Med) found Meta Temporal gave slightly the best visual-read/CenTauRz
concordance, and it matches the posterolateral temporal/parietal territory the visual
read keys on (Universal also covers mesial and frontal signal the read does not credit).

Discordant cases are split three ways (visual_read is graded against flortaucipir's separate
FDA/EMA visual-read criterion, Moscoso et al. 2025,
https://doi.org/10.1001/jama.2025.7817, which keys on posterolateral temporal/
parietal signal and deliberately NOT on isolated mesial temporal signal). The split is
a triage aid for picking cases to inspect, not a result. It is saved as the `bucket`
column:
  - "quant_agrees": model agrees with quantitative --quantitative_col (at/above or
    below its CTRz 2 SUVR equivalent), disagrees with the clinical visual read -> the model may be siding
    with an independent quantitative signal against the human reader, not just
    "wrong."
  - "isolated_mesial": --mtl_col is at/above its own CTRz 2 SUVR equivalent while
    --quantitative_col is below its -> possible early mesial tau the visual-read criterion is not
    designed to credit. CenTauR's masks exclude known off-target regions, but the
    mesial temporal lobe remains the most contamination-prone area (partial volume,
    flortaucipir-specific limits; Villemagne 2023, Karagianni 2026), so this is
    uncertain, not "real tau". Checked first, so a case lands in one bucket only.
  - "unexplained": model disagrees with quantitative too, and no isolated mesial
    pattern -> a more conventional, unexplained model error.

Alongside `bucket`, the csv carries flags to cross-tabulate against it: amyloid_status
(Karagianni: Vis+/CTRz- cases were 75-90% Abeta+, Vis-/CTRz+ only 50-67%),
`confident_error` (|prob - threshold| > --confident_dist) and, when the quantitative
column has a CenTauR equivalent, `near_ctrz2` (within --near_cutoff_margin SUVR of the
CTRz 2 equivalent, i.e. a borderline case for the split).

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
    CENTAUR_FTP_CTRZ2_SUVR,
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
        "--quantitative_col", default="MetaTemporal",
        help="Demographics column to use as the quantitative positivity reference (default: MetaTemporal tau SUVR; "
             "best visual-read concordance in Karagianni 2026). Positive = at/above its CTRz 2 SUVR equivalent.",
    )
    parser.add_argument(
        "--confident_dist", type=float, default=0.35,
        help="|prob - threshold| above this marks a discordant case as a confident error (default: 0.35).",
    )
    parser.add_argument(
        "--near_cutoff_margin", type=float, default=0.1,
        help="SUVR distance from the CTRz 2 equivalent within which a case is flagged near_ctrz2 (default: 0.1).",
    )
    parser.add_argument(
        "--mtl_col", default="MesialTemporal",
        help="Column to check for isolated elevation (at/above its CTRz 2 equivalent while "
             "--quantitative_col is below its) - a pattern the FDA/EMA visual-read criterion is not "
             "designed to credit (default: MesialTemporal).",
    )
    parser.add_argument(
        "--skip_quant_split", action="store_true",
        help="Skip the quantitative three-way split entirely (just list FP/FN with raw "
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
            quant_cutoff = CENTAUR_FTP_CTRZ2_SUVR.get(args.quantitative_col)
            if quant_cutoff is None:
                print(f"[WARNING] No CTRz 2 equivalent for '{args.quantitative_col}'; skipping quantitative-agreement split.")
            else:
                # Cohort percentile is descriptive only (printed/saved for orientation); the split
                # itself uses the CenTauR CTRz 2 SUVR equivalent per region.
                df["quant_percentile"] = df[args.quantitative_col].rank(pct=True) * 100
                quant_values = df[args.quantitative_col]
                quant_positive = (quant_values >= quant_cutoff).where(quant_values.notna()).astype("Int64")
                df["quant_positive"] = quant_positive
                df["model_agrees_with_quant"] = (df["pred"] == quant_positive)

                mtl_cutoff = CENTAUR_FTP_CTRZ2_SUVR.get(args.mtl_col)
                if args.mtl_col in df.columns and mtl_cutoff is not None:
                    df["mtl_percentile"] = df[args.mtl_col].rank(pct=True) * 100
                    # Mesial at/above its own CTRz 2 equivalent while the quantitative column is below its.
                    df["isolated_mtl"] = (df[args.mtl_col] >= mtl_cutoff) & (quant_positive == 0)
                else:
                    print(f"[WARNING] '{args.mtl_col}' not found or has no CTRz 2 equivalent; skipping isolated-mesial split.")

    df_discordant = df[df["category"].isin(["False Positive", "False Negative"])].copy()
    df_discordant = df_discordant.sort_values(["category", "dist_from_threshold"], ascending=[True, False])
    df_discordant = df_discordant.drop(columns=["ID"]).rename(columns={"ID_ind": "ID"})

    df_discordant["confident_error"] = df_discordant["dist_from_threshold"] > args.confident_dist
    cutoff = CENTAUR_FTP_CTRZ2_SUVR.get(args.quantitative_col)
    if cutoff is not None and args.quantitative_col in df_discordant.columns:
        df_discordant["near_ctrz2"] = (df_discordant[args.quantitative_col] - cutoff).abs() <= args.near_cutoff_margin

    if "model_agrees_with_quant" in df_discordant.columns:
        is_isolated_mesial = (
            df_discordant["isolated_mtl"].fillna(False).astype(bool)
            if "isolated_mtl" in df_discordant.columns else pd.Series(False, index=df_discordant.index)
        )
        agrees = (df_discordant["model_agrees_with_quant"] == True).fillna(False).astype(bool)  # noqa: E712
        # Isolated mesial is checked first, so each case lands in exactly one bucket.
        df_discordant["bucket"] = "unexplained"
        df_discordant.loc[agrees, "bucket"] = "quant_agrees"
        df_discordant.loc[is_isolated_mesial, "bucket"] = "isolated_mesial"

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
    flag_cols = [c for c in ["confident_error", "near_ctrz2", "bucket"] if c in df_discordant.columns]
    display_cols = display_cols + percentile_cols + flag_cols

    if "bucket" in df_discordant.columns:
        bucket_labels = [
            ("quant_agrees",
             f"Model agrees with quantitative {args.quantitative_col} "
             "(at/above or below its CTRz 2 SUVR equivalent), disagrees with clinical read"),
            ("isolated_mesial",
             f"Isolated {args.mtl_col} elevation (>= its CTRz 2 equivalent while "
             f"{args.quantitative_col} is below its) - possible "
             "early mesial tau not credited by the visual-read criterion (uncertain: mesial SUVR is "
             "partial-volume prone)"),
            ("unexplained", "Model disagrees with quantitative too (unexplained model error)"),
        ]
        buckets = [(label, df_discordant[df_discordant["bucket"] == name]) for name, label in bucket_labels]
        if "isolated_mtl" not in df_discordant.columns:
            buckets = [b for b in buckets if not b[0].startswith("Isolated")]

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
            # bucket / confident_error / near_ctrz2 are only computed for the discordant table.
            print(subset[[c for c in display_cols if c in subset.columns]].to_string(index=False))
            baseline_ids += [str(i) for i in subset["ID"]]

        if baseline_ids:
            print(f"\nBaseline (correct) IDs: --vis_img_list {','.join(baseline_ids)}")


if __name__ == "__main__":
    main()
