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

Is the model just an SUVR threshold? (Meta-temporal only, all test scans, descriptive)
    Scan by scan: does the model disagree with the read on the same scans as an SUVR rule? Besides
    the two rules above, "matched to model" is the SUVR threshold that calls as many test scans
    positive as the model does, so a model that only follows SUVR would disagree on exactly its scans.
    Plus the Spearman correlation of model probability with SUVR (all, read +, read - scans).

Is the model better than ALL FIVE regional SUVRs together? (all / CU / impaired)
    A single region is a weak baseline; a combination of regions is the strongest SUVR one.
    SUVR model       - logistic regression of the read on the five SUVRs (standardised, plus their
                       squares, so curved relations count), fitted on the TRAINING scans and applied
                       unchanged to the test scans. Its test AUC vs the model's, paired bootstrap CI.
    Explained by SUVR - R^2 of the model's log-odds on the same SUVR terms (test scans): how much of
                       what the model says is regional tau amount.
    Beyond SUVR      - on the test scans, read ~ SUVR terms vs read ~ SUVR terms + model log-odds:
                       likelihood-ratio test (does the model know something about the read the SUVRs
                       don't?) and the odds ratio per SD of model log-odds.
    The extra part   - the model's log-odds minus what the SUVRs explain (the OLS residual): its AUC
                       for the read (0.5 = says nothing about the read) and a Kruskal-Wallis test across
                       sites (a small p would mean part of it follows site/scanner).

Outputs (to <run_dir>/figures, next to the other visual-read figures):
    visual_read_model_vs_suvr_threshold.png  - ROC curves on the test scans (model vs Meta-temporal
                                               SUVR), with each rule's operating point; one panel
                                               per group
    visual_read_model_vs_suvr_threshold.csv  - every region x group x rule: n, sensitivity,
                                               specificity (CIs), McNemar p vs the model, AUCs
    visual_read_model_vs_suvr_overlap.csv    - per rule and direction: scans the model / the rule
                                               disagree on, both, only one (counts, no IDs)
    visual_read_model_vs_all_suvr.csv        - per group: the all-regions comparison above

Usage:
    python results_plotting/compare_model_vs_suvr_threshold.py /path/to/run_dir --dataset Gothenburg \
        --demo_csv /path/to/demo.csv
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
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
    confusion_display_label,
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


def suvr_overlap(test: pd.DataFrame, region: str, thresholds: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scan-level overlap of the model's and each SUVR rule's disagreements with the read, plus
    Spearman correlations of model probability with SUVR."""
    d = test.dropna(subset=[region])
    y, x, pred = d["y"].to_numpy(int), d[region].to_numpy(float), d["pred"].to_numpy(int)
    n_pos_calls = int(pred.sum())
    # SUVR threshold calling as many scans positive as the model (ties at the threshold count as positive)
    matched = float(np.sort(x)[::-1][n_pos_calls - 1]) if n_pos_calls else np.inf
    rules = {**thresholds[region], "matched to model": matched}
    model_dis = pred != y
    rows = []
    for rule, t in rules.items():
        rule_dis = (x >= t).astype(int) != y
        for read, label in [(0, confusion_display_label("False Positive")),
                            (1, confusion_display_label("False Negative"))]:
            m, r = model_dis[y == read], rule_dis[y == read]
            rows.append({"rule": rule, "threshold": round(t, 3), "direction": label,
                         "model disagrees": int(m.sum()), "rule disagrees": int(r.sum()),
                         "both": int((m & r).sum()), "only model": int((m & ~r).sum()),
                         "only rule": int((~m & r).sum())})
    corr = []
    for name, keep in [("all", np.ones_like(y, bool)), ("read +", y == 1), ("read -", y == 0)]:
        rho, p = stats.spearmanr(d["prob"].to_numpy(float)[keep], x[keep])
        corr.append({"scans": name, "n": int(keep.sum()), "spearman_rho": round(float(rho), 3), "p": float(p)})
    return pd.DataFrame(rows), pd.DataFrame(corr)


def suvr_terms(df: pd.DataFrame, regions: list[str], mu: pd.Series, sd: pd.Series) -> pd.DataFrame:
    """Design matrix: constant, standardised SUVRs and their squares (mu/sd from the training scans)."""
    z = (df[regions] - mu) / sd
    return sm.add_constant(pd.concat([z, (z ** 2).add_suffix("_sq")], axis=1), has_constant="add")


def fit_logit(y, X):
    """statsmodels Logit, or None if it does not converge (e.g. a group the SUVRs separate perfectly)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = sm.Logit(y, X).fit(disp=0, maxiter=500)
        except Exception:
            return None
    return res if res.mle_retvals.get("converged", False) else None


def all_suvr_comparison(train: pd.DataFrame, test: pd.DataFrame, regions: list[str], rng) -> pd.DataFrame:
    tr = train.dropna(subset=regions + ["visual_read"])
    mu, sd = tr[regions].mean(), tr[regions].std()
    base = fit_logit(tr["visual_read"].astype(int), suvr_terms(tr, regions, mu, sd))
    rows = []
    masks = group_masks(test)
    for g in [g for g in GROUPS if g in masks]:
        d = test[masks[g]].dropna(subset=regions)
        y = d["y"].to_numpy(int)
        if len(np.unique(y)) < 2:
            continue
        X = suvr_terms(d, regions, mu, sd)
        p = np.clip(d["prob"].to_numpy(float), 1e-6, 1 - 1e-6)
        model_logodds = np.log(p / (1 - p))
        row = {"group": g, "n": len(y), "n_read_positive": int(y.sum())}
        if base is not None:
            row.update(paired_auc(y, d["prob"].to_numpy(float), base.predict(X).to_numpy(), rng))
        ols = sm.OLS(model_logodds, X).fit()
        row["r2_model_by_suvr"] = ols.rsquared
        resid = ols.resid
        row["auc_extra_part"] = roc_auc_score(y, resid)
        if "site" in d.columns:
            by_site = [resid[(d["site"] == s).to_numpy()] for s in d["site"].dropna().unique()]
            by_site = [r for r in by_site if len(r) >= 5]
            row["site_kruskal_p"] = stats.kruskal(*by_site).pvalue if len(by_site) >= 2 else np.nan
        m0 = fit_logit(y, X)
        m1 = fit_logit(y, X.assign(model_logodds=(model_logodds - model_logodds.mean()) / model_logodds.std()))
        if m0 is not None and m1 is not None:
            lr = 2 * (m1.llf - m0.llf)
            row["lr_p_beyond_suvr"] = stats.chi2.sf(lr, 1)
            lo, hi = m1.conf_int().loc["model_logodds"]
            row["or_per_sd_model"], row["or_lo"], row["or_hi"] = np.exp([m1.params["model_logodds"], lo, hi])
        rows.append(row)
    return pd.DataFrame(rows)


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

    if FIGURE_REGION in regions:
        overlap_df, corr_df = suvr_overlap(test, FIGURE_REGION, thresholds)
        overlap_df.to_csv(out_dir / "visual_read_model_vs_suvr_overlap.csv", index=False)
        print(f"\n[same scans?] Does the model disagree with the read on the same scans as a "
              f"{pretty_region_name(FIGURE_REGION)} SUVR rule? (all test scans)")
        print(overlap_df.to_string(index=False))
        print(f"\n[Spearman] model probability vs {pretty_region_name(FIGURE_REGION)} SUVR")
        print(corr_df.to_string(index=False))

    allr = all_suvr_comparison(train, test, regions, rng)
    allr.round(4).to_csv(out_dir / "visual_read_model_vs_all_suvr.csv", index=False)
    print(f"\n[all regions] Model vs all {len(regions)} regional SUVRs together (SUVR model fitted on the training scans)")
    for _, r in allr.iterrows():
        print(f"  {r['group']} (n={int(r['n'])}, {int(r['n_read_positive'])} read +)")
        if pd.notna(r.get("auc_suvr")):
            print(f"    AUC: model {r['auc_model']:.3f} vs all-SUVR model {r['auc_suvr']:.3f}, "
                  f"difference {r['auc_diff']:+.3f} [{r['auc_diff_lo']:+.3f} to {r['auc_diff_hi']:+.3f}]")
        print(f"    Model's log-odds explained by the SUVRs: R^2 = {r['r2_model_by_suvr']:.2f}")
        if pd.notna(r.get("lr_p_beyond_suvr")):
            print(f"    Beyond SUVR: the model still predicts the read once the SUVRs are known: "
                  f"LR p = {r['lr_p_beyond_suvr']:.2g}, OR per SD {r['or_per_sd_model']:.1f} [{r['or_lo']:.1f}-{r['or_hi']:.1f}]")
        else:
            print("    Beyond SUVR: not estimable (the logistic fit did not converge in this group)")
        site = f", differs by site p = {r['site_kruskal_p']:.2g}" if pd.notna(r.get("site_kruskal_p")) else ""
        print(f"    The part not explained by SUVR: AUC for the read {r['auc_extra_part']:.2f}{site}")
    print(f"\n[done] saved to {out_dir}")


if __name__ == "__main__":
    main()
