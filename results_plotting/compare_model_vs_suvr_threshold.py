#!/usr/bin/env python3
"""Does the visual-read model do better than simply thresholding a regional tau SUVR?

The baseline an examiner will ask about. On the held-out test scans, the model's own calls
(probability >= 0.5) are compared with two SUVR rules, each predicting the visual read:
    CTRz 2           - the published CenTauR cut-off (no fitting at all)
    Youden, training - the SUVR threshold that best reproduces the visual reads on the TRAINING
                       scans (max sensitivity + specificity), then applied unchanged to the test scans
Both rules see only a regional SUVR, which needs an MRI-based pipeline (CenTauR standard, SPM8);
the model sees only the native-space PET.

Compared, on the same test scans:
    Sensitivity / specificity  - Wilson 95% CIs; model vs rule tested with an exact McNemar test
                                 (read-positive scans for sensitivity, read-negative for specificity)
    AUC                        - threshold-free: model probability vs continuous SUVR, paired
                                 bootstrap 95% CI of the difference
For all test scans and separately for cognitively unimpaired (CU) and impaired (MCI + AD dementia),
because SUVR agrees with the visual read much less well in CU (Karagianni 2026).

Outputs (to <run_dir>/figures, next to the other visual-read figures):
    visual_read_model_vs_suvr_threshold.png  - ROC curves on the test scans (model vs Meta-temporal
                                               SUVR), with each rule's operating point; one panel
                                               per group
    visual_read_model_vs_suvr_threshold.csv  - every region x group x rule: n, sensitivity,
                                               specificity (CIs), McNemar p vs the model, AUCs

Usage:
    python results_plotting/compare_model_vs_suvr_threshold.py /path/to/run_dir --dataset Gothenburg \
        --demo_csv /path/to/demo.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

from plot_utils import (
    CENTAUR_FTP_CTRZ2_SUVR,
    CUSTOM_PALETTE,
    LABEL_SIZE,
    LEGEND_SIZE,
    TICK_SIZE,
    TITLE_SIZE,
    combine_dx_groups,
    find_demo_csv,
    finalize_figure,
    pretty_region_name,
    resolve_path,
    resolve_predictions_csv,
    save_figure,
    wilson_interval,
)

REGIONS = ["MetaTemporal", "MesialTemporal", "TemporoParietal", "Frontal", "Universal"]
FIGURE_REGION = "MetaTemporal"
N_BOOT = 2000
GROUPS = ["All", "CU", "Impaired"]
RULE_COLORS = {"Model": CUSTOM_PALETTE[0], "CTRz 2": "black", "Youden, training": CUSTOM_PALETTE[4]}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="Visual-read run directory (needs splits/train_subjects.csv).")
    parser.add_argument("--dataset", default="Gothenburg", help="Dataset name used in the result filenames.")
    parser.add_argument("--demo_csv", default=None,
                        help="Demographics csv with the regional SUVRs (default: data/demo.csv).")
    return parser.parse_args()


def youden_threshold(y: np.ndarray, x: np.ndarray) -> float:
    """SUVR threshold (positive if SUVR >= t) maximising sensitivity + specificity."""
    fpr, tpr, thr = roc_curve(y, x)
    keep = np.isfinite(thr)
    return float(thr[keep][np.argmax((tpr - fpr)[keep])])


def sens_spec(y: np.ndarray, call: np.ndarray) -> dict:
    pos, neg = y == 1, y == 0
    k_se, k_sp = int((call[pos] == 1).sum()), int((call[neg] == 0).sum())
    se_lo, se_hi = wilson_interval(k_se, int(pos.sum()))
    sp_lo, sp_hi = wilson_interval(k_sp, int(neg.sum()))
    return {"sensitivity": k_se / max(pos.sum(), 1), "sens_lo": se_lo, "sens_hi": se_hi,
            "specificity": k_sp / max(neg.sum(), 1), "spec_lo": sp_lo, "spec_hi": sp_hi}


def mcnemar_p(a: np.ndarray, b: np.ndarray) -> float:
    """Exact McNemar test on paired 0/1 'correct' indicators."""
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    return float(stats.binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0


def paired_auc(y: np.ndarray, prob: np.ndarray, x: np.ndarray, rng) -> dict:
    out = {"auc_model": roc_auc_score(y, prob), "auc_suvr": roc_auc_score(y, x)}
    diffs = []
    for idx in rng.integers(0, len(y), (N_BOOT, len(y))):
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], prob[idx]) - roc_auc_score(y[idx], x[idx]))
    out["auc_diff"] = out["auc_model"] - out["auc_suvr"]
    out["auc_diff_lo"], out["auc_diff_hi"] = np.percentile(diffs, [2.5, 97.5])
    return out


def group_masks(df: pd.DataFrame) -> dict:
    masks = {"All": pd.Series(True, index=df.index)}
    if "dx" in df.columns:
        dx = combine_dx_groups(df["dx"])
        masks["CU"] = (dx == "CU").fillna(False)
        masks["Impaired"] = dx.isin(["MCI", "AlzCS dem"]).fillna(False)
    return masks


def make_figure(test: pd.DataFrame, thresholds: dict, rows: pd.DataFrame, out_path: Path):
    masks = group_masks(test)
    groups = [g for g in GROUPS if g in masks]
    fig, axes = plt.subplots(1, len(groups), figsize=(5.4 * len(groups), 5.6), squeeze=False)
    region = FIGURE_REGION
    for ax, g in zip(axes[0], groups):
        d = test[masks[g]].dropna(subset=[region])
        y = d["y"].to_numpy(int)
        if len(np.unique(y)) < 2:
            ax.set_visible(False)
            continue
        sub = rows[(rows["region"] == region) & (rows["group"] == g)].set_index("rule")
        for score, label, color in [(d["prob"], "Model", RULE_COLORS["Model"]),
                                    (d[region], f"{pretty_region_name(region)} SUVR", "#888888")]:
            fpr, tpr, _ = roc_curve(y, score)
            auc = roc_auc_score(y, score)
            ax.plot(fpr, tpr, color=color, linewidth=2.2 if label == "Model" else 1.8,
                    label=f"{label} (AUC {auc:.2f})")
        for rule, marker in [("Model", "o"), ("CTRz 2", "D"), ("Youden, training", "s")]:
            if rule in sub.index:
                r = sub.loc[rule]
                extra = "" if rule == "Model" else f" ({thresholds[region][rule]:.2f})"
                ax.scatter([1 - r["specificity"]], [r["sensitivity"]], s=80, marker=marker,
                           color=RULE_COLORS[rule], edgecolor="white", linewidth=0.9, zorder=5,
                           label=f"{rule}{extra}: {r['sensitivity']:.0%} / {r['specificity']:.0%}")
        ax.plot([0, 1], [0, 1], color="#cccccc", linestyle=":", linewidth=1)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        n_pos = int(y.sum())
        ax.set_title(f"{g} (n={len(y)}, {n_pos} read +)", fontsize=TITLE_SIZE - 3)
        ax.set_xlabel("1 − Specificity", fontsize=LABEL_SIZE - 3)
        ax.set_ylabel("Sensitivity", fontsize=LABEL_SIZE - 3)
        ax.tick_params(labelsize=TICK_SIZE - 2)
        ax.grid(alpha=0.28)
        ax.legend(loc="lower right", fontsize=LEGEND_SIZE - 2, frameon=True, framealpha=0.9,
                  title="Sensitivity / specificity", title_fontsize=LEGEND_SIZE - 2)
    fig.suptitle(f"Model vs {pretty_region_name(region)} SUVR threshold (test set)", fontsize=TITLE_SIZE)
    finalize_figure(fig, rect=(0, 0, 1, 1.0))
    save_figure(fig, out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    preds_path, out_dir = resolve_predictions_csv(run_dir, args.dataset)
    preds = pd.read_csv(preds_path)
    if "prob" not in preds.columns:
        raise ValueError(f"{preds_path} has no 'prob' column - is this a visual_read run?")
    preds = preds[["ID_ind", "y", "prob", "pred"]].apply(pd.to_numeric, errors="coerce").dropna()
    print(f"[load] {len(preds)} test predictions from {preds_path}")

    demo_path = resolve_path(args.demo_csv, find_demo_csv)
    if demo_path is None:
        raise FileNotFoundError("No demographics csv found - pass --demo_csv.")
    demo = pd.read_csv(demo_path)
    demo = demo.loc[:, ~demo.columns.astype(str).str.startswith("Unnamed")]
    for c in REGIONS + ["visual_read"]:
        if c in demo.columns:
            demo[c] = pd.to_numeric(demo[c], errors="coerce")
    regions = [r for r in REGIONS if r in demo.columns]

    split_path = run_dir / "splits" / "train_subjects.csv"
    if not split_path.exists():
        raise FileNotFoundError(f"{split_path} not found - the training scans are needed to fit the Youden threshold.")
    train_ids = pd.read_csv(split_path)["ID"]
    train = demo[demo["ID"].isin(train_ids)]
    test = preds.merge(demo, left_on="ID_ind", right_on="ID", how="inner")
    overlap = set(train["ID"]) & set(test["ID"])
    print(f"[split] {len(train)} training scans, {len(test)} test scans with demographics")
    if overlap:
        raise ValueError(f"{len(overlap)} scans are in both the training list and the test predictions.")
    if len(test) < len(preds):
        print(f"[WARNING] {len(preds) - len(test)} test scans have no row in {demo_path.name}.")

    thresholds = {}
    for region in regions:
        tr = train[[region, "visual_read"]].dropna()
        thresholds[region] = {"CTRz 2": CENTAUR_FTP_CTRZ2_SUVR[region],
                              "Youden, training": youden_threshold(tr["visual_read"].to_numpy(int),
                                                                   tr[region].to_numpy(float))}

    rng = np.random.default_rng(0)
    rows = []
    masks = group_masks(test)
    for region in regions:
        for g in [g for g in GROUPS if g in masks]:
            d = test[masks[g]].dropna(subset=[region])
            y = d["y"].to_numpy(int)
            if len(np.unique(y)) < 2:
                continue
            x, prob, pred = d[region].to_numpy(float), d["prob"].to_numpy(float), d["pred"].to_numpy(int)
            auc = paired_auc(y, prob, x, rng)
            model_ok = pred == y
            base = {"region": region, "group": g, "n": len(y), "n_read_positive": int(y.sum())}
            rows.append({**base, "rule": "Model", "threshold": 0.5, **sens_spec(y, pred), **auc})
            for rule, t in thresholds[region].items():
                call = (x >= t).astype(int)
                rule_ok = call == y
                rows.append({**base, "rule": rule, "threshold": t, **sens_spec(y, call),
                             "mcnemar_p_sensitivity": mcnemar_p(model_ok[y == 1], rule_ok[y == 1]),
                             "mcnemar_p_specificity": mcnemar_p(model_ok[y == 0], rule_ok[y == 0])})
    table = pd.DataFrame(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(out_dir / "visual_read_model_vs_suvr_threshold.csv", index=False)
    if FIGURE_REGION in regions:
        make_figure(test, thresholds, table, out_dir / "visual_read_model_vs_suvr_threshold.png")

    print("\n[model vs SUVR threshold] test set; sensitivity / specificity [95% CI]; McNemar p vs model")
    for region in regions:
        print(f"  {pretty_region_name(region)}  (Youden threshold from training: "
              f"{thresholds[region]['Youden, training']:.2f}; CTRz 2: {thresholds[region]['CTRz 2']:.2f})")
        for g in GROUPS:
            sub = table[(table["region"] == region) & (table["group"] == g)]
            if sub.empty:
                continue
            m = sub[sub["rule"] == "Model"].iloc[0]
            print(f"    {g} (n={int(m['n'])}, {int(m['n_read_positive'])} read +): AUC model {m['auc_model']:.3f} vs "
                  f"SUVR {m['auc_suvr']:.3f}, difference {m['auc_diff']:+.3f} [{m['auc_diff_lo']:+.3f} to {m['auc_diff_hi']:+.3f}]")
            for _, r in sub.iterrows():
                p = "" if r["rule"] == "Model" else (f"   p sens={r['mcnemar_p_sensitivity']:.2g}, "
                                                    f"p spec={r['mcnemar_p_specificity']:.2g}")
                print(f"      {r['rule']:<17} sens {r['sensitivity']:.1%} [{r['sens_lo']:.1%}-{r['sens_hi']:.1%}]  "
                      f"spec {r['specificity']:.1%} [{r['spec_lo']:.1%}-{r['spec_hi']:.1%}]{p}")
    print(f"\n[done] saved to {out_dir}")


if __name__ == "__main__":
    main()
