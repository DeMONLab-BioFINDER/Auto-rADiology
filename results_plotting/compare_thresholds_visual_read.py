#!/usr/bin/env python3
"""Pick visual_read decision thresholds on CV out-of-fold predictions, then apply them once to
the held-out test set (and AVID, if run) for several models side by side.

Rules (each threshold is chosen on the CV pool only, never on test/AVID):
    fixed 0.5     - the current default
    balanced      - Youden (max sensitivity + specificity - 1) on the out-of-fold predictions
    spec >= X     - lowest threshold whose out-of-fold specificity is at least X, i.e. the most
                    sensitive threshold that still meets the specificity target
Prediction = prob >= threshold.

The out-of-fold predictions come from the 5 fold models, while test/AVID are scored by the
final model, so the specificity reached on test/AVID is only approximately the target. That
gap is itself worth reporting.

Usage (one --model per model, in the order you want them shown, e.g. raw, 2mm, MNI):
    python results_plotting/compare_thresholds_visual_read.py \\
        --model raw  <cv_run_dir> <final_run_dir> \\
        --model 2mm  <cv_run_dir> <final_run_dir> \\
        --model MNI  <cv_run_dir> <final_run_dir>
cv_run_dir has oof_predictions.csv; final_run_dir has evaluation/Gothenburg/Eval_Gothenburg_results.csv
and, after submit_val_AVID_unseen_visual-read.sh, validation/External_validation_AVID_unseen_*_zeroshot_results.csv.
Relative run dirs are looked up under <proj_path>/results.

Outputs (default <proj_path>/results/thresholds, outside the repo):
    thresholds.csv          - the threshold each rule picked per model
    threshold_metrics.csv   - fp/fn/specificity/sensitivity/PPV/... per model, rule and dataset
    roc_pr_<dataset>.png    - ROC + precision-recall curves of all models, with the 0.5 and
                              main-target operating points marked
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from plot_utils import (
    add_panel_labels,
    compute_classification_stats,
    finalize_figure,
    save_figure,
    style_axes,
    style_legend,
)

MODEL_COLORS = ["#8c8c8c", "#1f77b4", "#d62728", "#2ca02c", "#9467bd"]  # in --model order


def parse_args():
    repo = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", nargs=3, action="append", required=True, metavar=("LABEL", "CV_DIR", "FINAL_DIR"))
    p.add_argument("--spec_targets", default="0.95,0.98,0.99", help="Comma-separated specificity targets")
    p.add_argument("--main_spec", type=float, default=0.98, help="Target marked on the curves (must be in --spec_targets)")
    p.add_argument("--dataset", default="Gothenburg")
    p.add_argument("--external", default="AVID_unseen", help="External dataset name used by run_val.py")
    p.add_argument("--results_root", default=str(repo.parent / "results"))
    p.add_argument("--out_dir", default=str(repo.parent / "results" / "thresholds"))
    return p.parse_args()


def load_preds(path):
    """y/prob from a results csv, or None if the file isn't there (e.g. AVID not run yet)."""
    if path is None or not Path(path).exists():
        return None
    df = pd.read_csv(path)
    df = df[np.isfinite(df["y"]) & np.isfinite(df["prob"])]
    return df["y"].to_numpy().astype(int), df["prob"].to_numpy(dtype=float)


def pick_thresholds(y, prob, spec_targets):
    """{rule name: threshold} chosen on the out-of-fold predictions."""
    fpr, tpr, thr = roc_curve(y, prob, drop_intermediate=False)  # pred = prob >= thr
    ok = np.isfinite(thr)
    fpr, tpr, thr = fpr[ok], tpr[ok], thr[ok]
    rules = {"fixed 0.5": 0.5, "balanced": float(thr[np.argmax(tpr - fpr)])}
    for s in spec_targets:
        meets = fpr <= 1 - s + 1e-12
        # most sensitive threshold that meets the target; ties -> the higher (stricter) one
        i = np.flatnonzero(meets)[np.argmax(tpr[meets])] if meets.any() else int(np.argmin(fpr))
        rules[f"spec >= {s:.0%}"] = float(thr[i])
    return rules


def main():
    args = parse_args()
    spec_targets = [float(s) for s in args.spec_targets.split(",") if s.strip()]
    main_rule = f"spec >= {args.main_spec:.0%}"
    results_root, out_dir = Path(args.results_root), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def run_dir(d):
        d = Path(d)
        return d if d.is_absolute() else results_root / d

    thr_rows, metric_rows, curves = [], [], {}
    for k, (label, cv_dir, final_dir) in enumerate(args.model):
        cv_dir, final_dir = run_dir(cv_dir), run_dir(final_dir)
        oof = load_preds(cv_dir / "oof_predictions.csv")
        if oof is None:
            print(f"[{label}] no oof_predictions.csv in {cv_dir} - skipped")
            continue
        rules = pick_thresholds(*oof, spec_targets)
        thr_rows += [{"model": label, "rule": r, "threshold": t} for r, t in rules.items()]

        avid = sorted((final_dir / "validation").glob(f"External_validation_{args.external}_*_zeroshot_results.csv"))
        sets = {
            "CV out-of-fold": oof,
            "test": load_preds(final_dir / "evaluation" / args.dataset / f"Eval_{args.dataset}_results.csv"),
            args.external: load_preds(avid[0] if avid else None),
        }
        for name, preds in sets.items():
            if preds is None:
                print(f"[{label}] no {name} predictions - skipped")
                continue
            y, prob = preds
            auc, auprc = roc_auc_score(y, prob), average_precision_score(y, prob)
            for rule, t in rules.items():
                metric_rows.append({"model": label, "dataset": name, "rule": rule, "auc": auc, "auprc": auprc,
                                    **compute_classification_stats(y, prob, t)})
            if name != "CV out-of-fold":
                curves.setdefault(name, []).append((label, MODEL_COLORS[k % len(MODEL_COLORS)], y, prob, rules))

    if not metric_rows:
        print("Nothing to compare.")
        return
    thr_df = pd.DataFrame(thr_rows)
    met_df = pd.DataFrame(metric_rows)
    thr_df.to_csv(out_dir / "thresholds.csv", index=False)
    met_df.to_csv(out_dir / "threshold_metrics.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_columns", None, "display.max_rows", None):
        print("\n===== thresholds (chosen on CV out-of-fold predictions) =====")
        print(thr_df.pivot(index="model", columns="rule", values="threshold").round(3).to_string())
        cols = ["model", "rule", "threshold", "n", "fp", "fn", "specificity", "sensitivity", "ppv", "npv",
                "balanced_accuracy", "mcc"]
        for name, g in met_df.groupby("dataset", sort=False):
            print(f"\n===== {name} (AUC / AUPRC per model: "
                  + ", ".join(f"{m} {a:.3f}/{p:.3f}" for m, a, p in
                              g.drop_duplicates("model")[["model", "auc", "auprc"]].itertuples(index=False))
                  + ") =====")
            print(g[cols].round(3).to_string(index=False))

    for name, models in curves.items():
        fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2), squeeze=False)
        ax_roc, ax_pr = axes[0]
        for label, color, y, prob, rules in models:
            fpr, tpr, _ = roc_curve(y, prob)
            prec, rec, _ = precision_recall_curve(y, prob)
            ax_roc.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC {roc_auc_score(y, prob):.3f})")
            ax_pr.plot(rec, prec, color=color, lw=2, label=f"{label} (AUPRC {average_precision_score(y, prob):.3f})")
            for rule, marker in (("fixed 0.5", "o"), (main_rule, "*")):
                if rule not in rules:
                    continue
                s = compute_classification_stats(y, prob, rules[rule])
                ax_roc.plot(1 - s["specificity"], s["sensitivity"], marker, color=color, ms=11, mec="black")
                if np.isfinite(s["ppv"]):
                    ax_pr.plot(s["sensitivity"], s["ppv"], marker, color=color, ms=11, mec="black")
        ax_roc.plot([0, 1], [0, 1], ls="--", color="gray", lw=1, alpha=0.6)
        prevalence = float(models[0][2].mean())
        ax_pr.axhline(prevalence, ls="--", color="gray", lw=1, alpha=0.6, label=f"No skill ({prevalence:.2f})")
        for ax in (ax_roc, ax_pr):
            ax.plot([], [], "o", color="white", mec="black", label="threshold 0.5")
            ax.plot([], [], "*", color="white", mec="black", ms=11, label=f"{main_rule} (from CV)")
        ax_pr.set_xlim(-0.02, 1.02); ax_pr.set_ylim(-0.02, 1.02)
        style_axes(ax_roc, f"ROC: {name}", "False positive rate (1 - specificity)", "Sensitivity")
        style_axes(ax_pr, f"Precision-recall: {name}", "Sensitivity", "PPV")
        style_legend(ax_roc, loc="lower right")
        style_legend(ax_pr, loc="lower left")
        add_panel_labels(axes, ["A", "B"])
        finalize_figure(fig, rect=(0.0, 0.0, 0.99, 0.99))
        save_figure(fig, out_dir / f"roc_pr_{name.replace(' ', '_')}.png", dpi=300)
        plt.close(fig)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
