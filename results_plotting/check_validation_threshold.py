#!/usr/bin/env python3
"""Would setting each model's threshold on its own held-out validation scans make its operating point
stable? Checked on CV runs that already exist - no new training, the test set is not touched.

Every CV fold model has a validation fold (~365 scans) it did not train on, which works as a stand-in
for a validation split held out from the final model's training pool. For each fold model:
    fixed 0.5            - specificity / sensitivity on its whole validation fold
    threshold from val   - split the validation fold into two random halves (stratified by read, each
                           ~10% of the training pool), pick the threshold on one half with the rule,
                           measure on the other half; repeated --n_repeats times (both directions)
If the fold models land at similar specificity with the validation threshold but spread out at 0.5,
the approach fixes the run-to-run drift; the spread over repeats shows how noisy a threshold picked on
~180 scans is (strict targets like 98% rest on only a few read-negative scans).

Caveat: the fold models picked their best epoch on the same validation fold, so these numbers are a
little optimistic - the same for every rule.

Usage:
    python results_plotting/check_validation_threshold.py --cv raw <raw_cv_run> --cv 2mm <2mm_cv_run> \\
        --cv MNI <MNI_cv_run>
Relative run dirs are looked up under <proj_path>/results.

Outputs (default <proj_path>/results/thresholds/validation_check, outside the repo):
    validation_threshold_per_fold.csv  - per model, fold and rule: threshold and specificity / sensitivity
                                         (median and 5th-95th percentile over the repeats)
    validation_threshold_check.png     - specificity and sensitivity per fold model: 0.5 vs threshold
                                         from validation (--main_rule)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compare_thresholds_visual_read import pick_thresholds
from plot_utils import CUSTOM_PALETTE, LABEL_SIZE, LEGEND_SIZE, TICK_SIZE, TITLE_SIZE, finalize_figure, save_figure


def parse_args():
    repo = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cv", nargs=2, action="append", required=True, metavar=("LABEL", "CV_DIR"))
    p.add_argument("--spec_targets", default="0.95,0.98", help="Comma-separated specificity targets")
    p.add_argument("--main_rule", default="spec >= 95%", help="Rule shown in the figure")
    p.add_argument("--n_repeats", type=int, default=200)
    p.add_argument("--results_root", default=str(repo.parent / "results"))
    p.add_argument("--out_dir", default=str(repo.parent / "results" / "thresholds" / "validation_check"))
    return p.parse_args()


def spec_sens(y, prob, thr):
    pred = prob >= thr
    return (~pred[y == 0]).mean(), pred[y == 1].mean()


def halves(y, rng):
    """Random half of the indices, stratified by y."""
    idx = [rng.permutation(np.flatnonzero(y == c)) for c in (0, 1)]
    a = np.concatenate([i[: len(i) // 2] for i in idx])
    return np.isin(np.arange(len(y)), a)


def check_fold(y, prob, spec_targets, n_repeats, rng):
    rows = {"fixed 0.5": [(0.5, *spec_sens(y, prob, 0.5))]}
    for _ in range(n_repeats):
        in_a = halves(y, rng)
        for pick, measure in [(in_a, ~in_a), (~in_a, in_a)]:
            for rule, thr in pick_thresholds(y[pick], prob[pick], spec_targets).items():
                if rule != "fixed 0.5":
                    rows.setdefault(rule, []).append((thr, *spec_sens(y[measure], prob[measure], thr)))
    out = []
    for rule, vals in rows.items():
        v = np.array(vals)
        row = {"rule": rule, "n_val": len(y), "n_read_neg": int((y == 0).sum()), "n_read_pos": int((y == 1).sum())}
        for j, name in enumerate(["threshold", "specificity", "sensitivity"]):
            row[name] = np.median(v[:, j])
            row[f"{name}_p5"], row[f"{name}_p95"] = np.percentile(v[:, j], [5, 95])
        out.append(row)
    return out


def make_figure(table, main_rule, out_path):
    models = list(dict.fromkeys(table["model"]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, metric in zip(axes, ["specificity", "sensitivity"]):
        for m_i, model in enumerate(models):
            for k, (rule, label, color, dx) in enumerate([("fixed 0.5", "Fixed 0.5", CUSTOM_PALETTE[0], -0.15),
                                                         (main_rule, f"Threshold from validation ({main_rule})", CUSTOM_PALETTE[1], 0.15)]):
                sub = table[(table["model"] == model) & (table["rule"] == rule)]
                x = m_i + dx + np.linspace(-0.08, 0.08, len(sub))
                if rule != "fixed 0.5":
                    ax.vlines(x, sub[f"{metric}_p5"], sub[f"{metric}_p95"], color=color, alpha=0.5, linewidth=1.5)
                ax.scatter(x, sub[metric], color=color, s=36, zorder=3, edgecolor="white", linewidth=0.6,
                           label=label if m_i == 0 else None)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, fontsize=TICK_SIZE - 2)
        ax.set_ylabel(metric.capitalize(), fontsize=LABEL_SIZE - 4)
        ax.tick_params(axis="y", labelsize=TICK_SIZE - 3)
        ax.grid(axis="y", alpha=0.28)
    axes[0].legend(fontsize=LEGEND_SIZE - 3, loc="lower left", frameon=True, framealpha=0.9)
    fig.suptitle("Each CV fold model (one dot each): fixed 0.5 vs threshold set on half its validation fold\n"
                 "(whiskers: 5th–95th percentile over random halves)", fontsize=TITLE_SIZE - 4)
    finalize_figure(fig, rect=(0, 0, 1, 1.0))
    save_figure(fig, out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")


def main():
    args = parse_args()
    spec_targets = [float(s) for s in args.spec_targets.split(",") if s.strip()]
    results_root, out_dir = Path(args.results_root), Path(args.out_dir)
    rng = np.random.default_rng(0)
    rows = []
    for label, cv_dir in args.cv:
        cv_dir = Path(cv_dir) if Path(cv_dir).is_absolute() else results_root / cv_dir
        oof = pd.read_csv(cv_dir / "oof_predictions.csv")
        oof = oof[np.isfinite(oof["y"]) & np.isfinite(oof["prob"])]
        for fold, g in oof.groupby("fold"):
            y, prob = g["y"].to_numpy().astype(int), g["prob"].to_numpy(dtype=float)
            rows += [{"model": label, "fold": fold, **r} for r in check_fold(y, prob, spec_targets, args.n_repeats, rng)]
    table = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(out_dir / "validation_threshold_per_fold.csv", index=False)
    make_figure(table, args.main_rule, out_dir / "validation_threshold_check.png")

    print("\n[validation threshold] per fold model: specificity / sensitivity (threshold), median [5-95%] over halves")
    for (model, rule), g in table.groupby(["model", "rule"], sort=False):
        cells = "  ".join(f"{r['specificity']:.3f}/{r['sensitivity']:.3f} ({r['threshold']:.2f})" for _, r in g.iterrows())
        spread = f"spec range across folds {g['specificity'].min():.3f}-{g['specificity'].max():.3f}"
        noise = "" if rule == "fixed 0.5" else f", typical within-fold 5-95% {(g['specificity_p95'] - g['specificity_p5']).median():.3f}"
        print(f"  {model:<6} {rule:<12} {cells}\n         {spread}{noise}")
    print(f"\n[done] saved to {out_dir}")


if __name__ == "__main__":
    main()
