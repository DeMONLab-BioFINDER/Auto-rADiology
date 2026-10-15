#!/usr/bin/env python3
"""Plots for the visual-read (binary classification) model: class balance and
agreement between model predictions and the expert clinical read."""

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns

from plot_utils import (
    CUSTOM_PALETTE,
    find_demo_csv,
    make_agreement_by_centiloid_panel,
    make_agreement_by_value_panel,
    make_class_balance_panel,
    make_classification_breakdown_panel,
    make_classification_breakdown_violin_panel,
    make_confusion_category_categorical_panel,
    make_confusion_category_distribution_panel,
    make_discordant_probability_violin_panel,
    make_pr_curve_panel,
    make_probability_by_class_panel,
    make_probability_vs_suvr_scatter,
    make_roc_confusion_panel,
    resolve_existing_path,
    resolve_path,
)

# AD construct-validity check: true positives should skew toward known AD risk/severity
# markers (higher amyloid burden, lower cognition) more than true negatives.
CONSTRUCT_VALIDITY_CONTINUOUS = [
    ("CL", "Amyloid Centiloid"),
    ("CDR", "CDR"),
    ("CDR_SB", "CDR Sum of Boxes"),
]

# Regional tau SUVR vs. classification outcome (model agreement/discordance by region).
# Universal/MesialTemporal/MetaTemporal/TemporoParietal/Frontal are CenTauR's own
# standardized, tracer-harmonized masks (Villemagne et al. 2023, Alzheimer's & Dementia,
# https://doi.org/10.1002/dad2.12454): a single "universal" cortical mask (intersection of
# 6 tracers' AD-signature difference maps) with 4 subregions nested inside it. CenTauR's
# own validation (CTR > 2, AD A-beta+ vs CU A-beta-) found Mesial Temporal, Meta Temporal,
# and Temporo-Parietal comparably strong discriminators (accuracy 0.94-0.96, sensitivity
# 0.88-0.91); Frontal was the weakest (accuracy 0.91, sensitivity only 0.73) - ordered
# last here for that reason. Universal is first (the overall composite, closest to
# "is this scan tau-positive at all"); Temporo-Parietal second - CenTauR's authors
# recommend it as the region to use "for a global transformation" when only one region is
# wanted, and note it's the most sensitive to longitudinal tau accumulation.
#
# Separately, and specific to flortaucipir: the FDA/EMA-approved *visual-read* criterion
# that visual_read is graded against (Moscoso et al. 2025, JAMA,
# https://doi.org/10.1001/jama.2025.7817) calls a scan positive based on posterolateral
# temporal/occipital/parietal-precuneus signal, deliberately NOT on isolated mesial
# temporal signal - flortaucipir has known non-specific/off-target binding there, making
# an MTL-only pattern unreliable for a human rater to call, even though MesialTemporal
# SUVR discriminates AD well in aggregate (above). So a false positive driven mainly by
# elevated MesialTemporal may reflect early tau the visual-read label is structurally
# blind to at the individual-scan level, not necessarily a model error - worth flagging
# specifically on that panel, not a reason to deprioritize the region generally.
# This regional mapping (and the visual-read criterion) is specific to flortaucipir and
# does not transfer as-is to other tau tracers (RO948, MK6240, PI2620, ...), which have
# different off-target binding profiles (CenTauR paper, same point; Moscoso paper limits
# itself to flortaucipir for the same reason).
REGIONAL_SUVR_BY_OUTCOME = [
    ("Universal", "Universal tau SUVR"),
    ("TemporoParietal", "TemporoParietal tau SUVR"),
    ("MetaTemporal", "MetaTemporal tau SUVR"),
    ("MesialTemporal", "MesialTemporal tau SUVR (strong discriminator in aggregate; not a sole visual-read-positivity trigger)"),
    ("Frontal", "Frontal tau SUVR (weakest of the 4 CenTauR subregions for AD discrimination)"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create class-balance and clinical-read-agreement plots for a visual_read run."
    )
    parser.add_argument("run_dir", help="Path to a results run directory trained with --targets visual_read.")
    parser.add_argument("--dataset", default="Gothenburg", help="Dataset name used in the result filenames.")
    parser.add_argument("--group_col", default="site", help="Demographic column to break class balance down by.")
    parser.add_argument(
        "--demo_csv",
        default=None,
        help="Path to the demographics csv (defaults to data/demo.csv if found). Set this "
             "explicitly for a dataset with its own demo file, e.g. demo_test_subset_tau_raw.csv.",
    )
    return parser.parse_args()


def find_zeroshot_results_csv(run_dir: Path, dataset: str) -> Path | None:
    """Results from a run_val.py --few_shot 0 external validation (e.g. reusing
    test_subset as a stand-in unseen dataset), saved under <run_dir>/validation/."""
    matches = sorted((run_dir / "validation").glob(f"External_validation_{dataset}__*_zeroshot_results.csv"))
    return matches[0] if matches else None


def find_dataset_demo_csv(dataset: str) -> Path | None:
    """demo_{dataset}.csv, the same naming convention run_val.py's load_validation_data
    uses (e.g. demo_AVID_unseen.csv) - checked before falling back to the generic
    data/demo.csv, since a run on a non-Gothenburg dataset (AVID, a subset smoke test,
    ...) needs its own demographics for the confusion-category merge to find any
    matching IDs at all."""
    script_path = Path(__file__).resolve()
    demo_candidates = [
        script_path.parents[1] / "data" / f"demo_{dataset}.csv",
        script_path.parents[2] / "data" / f"demo_{dataset}.csv",
    ]
    return next((p for p in demo_candidates if p.exists()), None)


def load_predictions(preds_path: Path) -> pd.DataFrame:
    df = pd.read_csv(preds_path)
    required_cols = {"y", "prob", "pred"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"{preds_path} is missing columns {sorted(missing)} — "
            "this run may not have been trained with a visual_read classification target."
        )
    return df


def load_decision_threshold(preds_path: Path) -> float:
    """The results csv (y/prob/pred) doesn't carry the threshold used to build 'pred' -
    it's saved in the sibling *_metrics.csv (see decision_threshold in src/cv.py's
    run_fold / run_val.py). Falls back to 0.5 (argparse default) if not found, e.g. for
    results from a run predating this field."""
    metrics_path = preds_path.parent / preds_path.name.replace("_results.csv", "_metrics.csv")
    if metrics_path.exists():
        df_metrics = pd.read_csv(metrics_path)
        if "decision_threshold" in df_metrics.columns and len(df_metrics) > 0:
            return float(df_metrics["decision_threshold"].iloc[0])
    print(f"[WARNING] Could not find decision_threshold in {metrics_path}; defaulting to 0.5.")
    return 0.5


def main():
    sns.set_theme(style="whitegrid", palette=CUSTOM_PALETTE)

    args = parse_args()
    run_dir = Path(args.run_dir).resolve()

    # A zero-shot external validation result (run_val.py) takes priority if present -
    # its figures live under validation/figures/, separate from the run's own held-out
    # test figures in figures/, so the two never overwrite each other.
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
    out_dir.mkdir(parents=True, exist_ok=True)

    demo_path = resolve_path(args.demo_csv, lambda: find_dataset_demo_csv(args.dataset) or find_demo_csv())
    df_demo = None
    if demo_path is not None:
        df_demo = pd.read_csv(demo_path)
        make_class_balance_panel(df_demo, out_dir, group_col=args.group_col)
    else:
        print("[WARNING] Could not find demographics csv for class balance panel.")

    df_preds = load_predictions(preds_path)
    decision_threshold = load_decision_threshold(preds_path)
    make_roc_confusion_panel(
        df_preds["y"].to_numpy(),
        df_preds["prob"].to_numpy(),
        df_preds["pred"].to_numpy(),
        out_dir,
        stats_csv_path=out_dir / "visual_read_performance_stats.csv",
    )
    make_pr_curve_panel(
        df_preds["y"].to_numpy(),
        df_preds["prob"].to_numpy(),
        out_dir,
    )
    make_probability_by_class_panel(
        df_preds["y"].to_numpy(),
        df_preds["prob"].to_numpy(),
        decision_threshold,
        out_dir,
    )
    make_discordant_probability_violin_panel(df_preds, out_dir, decision_threshold)
    make_classification_breakdown_panel(df_preds, out_dir)
    make_classification_breakdown_violin_panel(df_preds, out_dir, decision_threshold)

    if df_demo is not None:
        for value_col, value_label in REGIONAL_SUVR_BY_OUTCOME:
            make_agreement_by_value_panel(
                df_preds, df_demo, out_dir,
                value_col=value_col, value_label=value_label,
            )
        for value_col, value_label in REGIONAL_SUVR_BY_OUTCOME + CONSTRUCT_VALIDITY_CONTINUOUS:
            # Keep the original documented filename for Universal (README's
            # visual_read_universal_suvr_by_outcome.png); every other column gets the
            # generic <col>_by_outcome.png naming.
            kwargs = {} if value_col == "Universal" else {
                "filename": f"visual_read_{value_col.lower()}_by_outcome.png"
            }
            make_confusion_category_distribution_panel(
                df_preds, df_demo, out_dir,
                value_col=value_col, value_label=value_label, **kwargs,
            )
        make_confusion_category_categorical_panel(
            df_preds, df_demo, out_dir, cat_col="apoe", cat_label="APOE e4 status",
        )
        # Headline readability panel: binned agreement vs. Centiloid, styled after the
        # Moscoso et al. 2025 source figure (background histogram + agreement line) -
        # easier to read at a glance than the per-subject scatter/distribution panels above.
        make_agreement_by_centiloid_panel(df_preds, df_demo, out_dir)
        # Continuous model-probability-vs-quantitative-SUVR scatter, for the CenTauR
        # regions only (not CONSTRUCT_VALIDITY_CONTINUOUS - CL/CDR aren't the tau-SUVR
        # quantitative reference the model is being compared against).
        for value_col, value_label in REGIONAL_SUVR_BY_OUTCOME:
            make_probability_vs_suvr_scatter(
                df_preds, df_demo, out_dir, decision_threshold,
                value_col=value_col, value_label=value_label,
            )

    print(f"[done] figures saved to {out_dir}")


if __name__ == "__main__":
    main()
