#!/usr/bin/env python3
"""Do the subgroup differences in where the model disagrees with the visual read survive at the same tau level?

The subgroup analysis found more Model + / Visual read − scans in Aβ+, APOE ε4 carriers and older
people, and more Model − / Visual read + scans in Aβ− and CU people. Two explanations predict that:
    Closer to the line - e.g. read-negative Aβ+ scans have more sub-threshold tau, so they sit closer
                    to the boundary and the model and the read split more often
    Something else in the image - the model reacts to something in those groups' scans beyond tau level
Comparing at the same Meta-temporal SUVR separates them: if a group difference disappears once SUVR
is accounted for, it was closeness to the line; if it stays, it is something else.

For each difference that was significant in the subgroup analysis, within the visual-read stratum it
was found in (read-negative for specificity, read-positive for sensitivity), a logistic regression of
"model disagrees with the read" (statsmodels), reported as an odds ratio with 95% CI and a
likelihood-ratio p-value:
    None                     - the original difference
    Meta-temporal (main)     - the same, at the same Meta-temporal SUVR. FDR-corrected across the 5 checks
    Meta-temporal, curved    - SUVR + SUVR², in case risk does not rise in a straight line with SUVR
Model + / Visual read − only (~30 scans; the ~11 Model − / Visual read + scans cannot afford more):
    Mesial temporal, Meta + mesial temporal
                             - is the effect medial-temporal signal (early tau or off-target, which the
                               reading rules ignore)? If so the odds ratio shrinks
    Meta-temporal + temporo-parietal / frontal / Universal
                             - is the effect signal in another region? If so the odds ratio shrinks
    Meta-temporal, complete-data subset
                             - the main model on the scans with amyloid, APOE, age and Centiloid all known
                               (drops most of NACC / Geneva), so the next rows compare like with like
    Meta-temporal + the other two of amyloid / APOE / age
                             - does each survive the others (e.g. is APOE just amyloid)?
    Meta-temporal + Centiloid instead of amyloid yes/no
                             - (APOE, age) the amount of amyloid rather than positive/negative
Everything except the main row is a sensitivity analysis: reported, not tested as a family.

Plus a descriptive look at the regions: among Visual read − scans, regional SUVR of Model + vs Model −
scans, for all of them and for ε4 carriers and the older half (age split for this table only). The
difference is in SD units of that region in the Model − scans, so regions on different scales compare.
Model + scans have more tau everywhere, so the question is which regions are *disproportionately* higher.

Outputs (to <run_dir>/figures):
    visual_read_errors_at_same_suvr.csv  - every check x variant: n, disagreements, odds ratio [95% CI], p
    visual_read_errors_at_same_suvr.png  - main analysis: probability the model disagrees vs Meta-temporal SUVR,
                                           one fitted curve per group (curves on top of each other =
                                           harder cases; apart = something else at the same SUVR)
    visual_read_errors_robustness.png    - forest plot of every variant per check
    visual_read_regions_model_pos_vs_neg.csv / .png
                                         - regional SUVR of Model + vs Model − among Visual read − scans

Usage:
    python results_plotting/analyze_errors_at_same_suvr.py /path/to/run_dir --dataset Gothenburg \
        --demo_csv /path/to/demo.csv
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from scipy import stats

from plot_utils import (
    CUSTOM_PALETTE,
    LABEL_SIZE,
    LEGEND_SIZE,
    TICK_SIZE,
    TITLE_SIZE,
    add_subgroup_columns,
    fdr_bh,
    find_demo_csv,
    finalize_figure,
    resolve_path,
    resolve_predictions_csv,
    save_figure,
)

# (metric, display name, column of the group variable, label for 0 / 1)
CHECKS = [
    ("specificity", "Amyloid status", "amy", "Aβ−", "Aβ+"),
    ("specificity", "APOE ε4", "apoe", "Non-carrier", "Carrier"),
    ("specificity", "Age", "age10", "65 years", "80 years"),
    ("sensitivity", "Amyloid status", "amy", "Aβ−", "Aβ+"),
    ("sensitivity", "Diagnosis", "impaired", "CU", "Impaired"),
]
METRIC_READ = {"sensitivity": 1, "specificity": 0}
AGE_SHOWN = (6.5, 8.0)  # age10 values drawn in the figure (65 and 80 years)
COMPLETE = ["amy", "apoe", "age10", "cl"]
NAMES = {"amy": "amyloid", "apoe": "APOE", "age10": "age", "cl": "Centiloid"}
MAIN = "Meta-temporal (main)"
# (csv column, column in the regression data, display name)
REGIONS = [("MesialTemporal", "mesial", "Mesial temporal"), ("MetaTemporal", "meta", "Meta-temporal"),
           ("TemporoParietal", "tp", "Temporo-parietal"), ("Frontal", "frontal", "Frontal"),
           ("Universal", "universal", "Universal")]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="Visual-read run directory.")
    parser.add_argument("--dataset", default="Gothenburg", help="Dataset name used in the result filenames.")
    parser.add_argument("--demo_csv", default=None, help="Demographics csv (default: data/demo.csv).")
    parser.add_argument("--female_code", type=float, default=2, help="Value of `gender` meaning female.")
    return parser.parse_args()


def variants(metric: str, var: str) -> list[tuple[str, list[str], list[str]]]:
    """(label, adjustment terms, columns that must be known) for one check."""
    out = [("None", [], []), (MAIN, ["meta"], []), ("Meta-temporal, curved", ["meta", "meta_sq"], [])]
    if metric == "specificity":
        others = [o for o in ("amy", "apoe", "age10") if o != var]
        out += [("Mesial temporal", ["mesial"], []),
                ("Meta + mesial temporal", ["meta", "mesial"], []),
                ("Meta + temporo-parietal", ["meta", "tp"], []),
                ("Meta + frontal", ["meta", "frontal"], []),
                ("Meta + Universal", ["meta", "universal"], []),
                ("Meta-temporal, complete-data subset", ["meta"], COMPLETE),
                (f"Meta-temporal + {' + '.join(NAMES[o] for o in others)}", ["meta", *others], COMPLETE)]
        if var != "amy":
            with_cl = ["cl" if o == "amy" else o for o in others]
            out.append((f"Meta-temporal + {' + '.join(NAMES[o] for o in with_cl)}", ["meta", *with_cl], COMPLETE))
    return out


def stratum(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """The scans one metric is defined on, with every variable the models use."""
    sub = df[df["y"] == METRIC_READ[metric]].copy()
    sub["wrong"] = (sub["pred"] != sub["y"]).astype(int)
    for csv_col, col, _ in REGIONS:
        sub[col] = pd.to_numeric(sub[csv_col], errors="coerce") if csv_col in sub else np.nan
    sub["meta_sq"] = (sub["meta"] - sub["meta"].mean()) ** 2
    sub["amy"] = sub["_g_amy"].map({"Aβ negative": 0.0, "Aβ positive": 1.0})
    sub["apoe"] = sub["_g_apoe"].map({"Non-carrier": 0.0, "Carrier": 1.0})
    sub["impaired"] = sub["_g_dx"].map({"CU": 0.0, "MCI": 1.0, "AD dementia": 1.0})  # "Other" stays missing
    sub["age10"] = pd.to_numeric(sub["age"], errors="coerce") / 10  # odds ratio per 10 years
    sub["cl"] = pd.to_numeric(sub["CL"], errors="coerce") if "CL" in sub else np.nan
    return sub


def fit(d: pd.DataFrame, terms: list[str]):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return smf.logit("wrong ~ " + (" + ".join(terms) or "1"), data=d).fit(disp=0)


def run_check(df: pd.DataFrame, metric: str, name: str, var: str) -> tuple[list[dict], object]:
    sub = stratum(df, metric)
    rows, main_fit = [], None
    for label, adjust, require in variants(metric, var):
        d = sub.dropna(subset=[var, "wrong", *adjust, *require])
        row = {"metric": metric, "variable": name, "adjusted_for": label, "n": len(d), "n_disagree": int(d["wrong"].sum())}
        if d["wrong"].sum() < 2 or d[var].nunique() < 2:
            rows.append({**row, "note": "too few disagreements / one group only"})
            continue
        try:
            full, reduced = fit(d, [var, *adjust]), fit(d, adjust)
        except Exception as err:  # e.g. perfect separation: a group where the model never disagrees
            rows.append({**row, "note": f"could not fit ({type(err).__name__})"})
            continue
        lo, hi = full.conf_int().loc[var]
        if not np.isfinite(hi) or abs(full.params[var]) > np.log(1000):  # quasi-separation: estimate meaningless
            rows.append({**row, "note": "could not fit (one group (almost) never disagrees)"})
            continue
        row.update({"or": np.exp(full.params[var]), "or_lo": np.exp(lo), "or_hi": np.exp(hi),
                    "p": stats.chi2.sf(2 * (full.llf - reduced.llf), 1)})
        high = d[var] >= d[var].median() if var == "age10" else d[var] == 1
        for col, key in [("meta", "median_meta"), ("mesial", "median_mesial")]:
            if col in adjust or (label == "None" and col == "meta"):
                row[f"{key}_group0_group1"] = f"{d.loc[~high, col].median():.2f} / {d.loc[high, col].median():.2f}"
        if label == MAIN:
            main_fit = (full, d)
        rows.append(row)
    return rows, main_fit


def regional_table(df: pd.DataFrame) -> pd.DataFrame:
    """Among Visual read − scans: regional SUVR of Model + vs Model −, overall / ε4 carriers / older half."""
    sub = stratum(df, "specificity")
    groups = [("All Visual read −", np.ones(len(sub), bool)), ("ε4 carriers", sub["apoe"] == 1),
              (f"Older half (age ≥ {sub['age10'].median() * 10:.0f})", sub["age10"] >= sub["age10"].median())]
    rows = []
    for group, mask in groups:
        g = sub[np.asarray(mask, bool)]
        for _, col, region in REGIONS:
            neg, pos = g.loc[g["wrong"] == 0, col].dropna(), g.loc[g["wrong"] == 1, col].dropna()
            if len(pos) < 2 or len(neg) < 2:
                continue
            rows.append({"group": group, "region": region, "n_model_neg": len(neg), "n_model_pos": len(pos),
                         "median_model_neg": neg.median(), "median_model_pos": pos.median(),
                         "diff_in_sd": (pos.mean() - neg.mean()) / neg.std(),
                         "p_mann_whitney": stats.mannwhitneyu(pos, neg).pvalue})
    return pd.DataFrame(rows)


# ------------------------------
# Figures
# ------------------------------
def make_regions_figure(df: pd.DataFrame, out_path: Path):
    """Box plots of each region's SUVR, Model − vs Model +, among Visual read − scans."""
    sub = stratum(df, "specificity")
    regions = [(col, name) for _, col, name in REGIONS if sub[col].notna().any()]
    fig, ax = plt.subplots(figsize=(1.9 * len(regions) + 2, 4.8))
    rng = np.random.default_rng(0)
    for i, (col, _) in enumerate(regions):
        for offset, wrong, label, color in [(-0.2, 0, "Model −", CUSTOM_PALETTE[0]), (0.2, 1, "Model +", CUSTOM_PALETTE[1])]:
            vals = sub.loc[sub["wrong"] == wrong, col].dropna()
            ax.boxplot(vals, positions=[i + offset], widths=0.32, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor=color, alpha=0.35, edgecolor=color), medianprops=dict(color=color, lw=2),
                       whiskerprops=dict(color=color), capprops=dict(color=color))
            ax.scatter(i + offset + rng.uniform(-0.1, 0.1, len(vals)), vals, s=8 if wrong == 0 else 16,
                       alpha=0.25 if wrong == 0 else 0.8, color=color, lw=0, label=label if i == 0 else None)
    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels([name for _, name in regions], fontsize=TICK_SIZE - 3)
    ax.set_ylabel("SUVR", fontsize=LABEL_SIZE - 4)
    ax.set_title("Visual read − scans: regional SUVR when the model says − vs +", fontsize=TITLE_SIZE - 4)
    ax.tick_params(axis="y", labelsize=TICK_SIZE - 3)
    ax.grid(axis="y", alpha=0.28)
    ax.legend(fontsize=LEGEND_SIZE - 2, frameon=True, framealpha=0.9)
    finalize_figure(fig, rect=(0, 0, 1, 1.0))
    save_figure(fig, out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")


def make_curves_figure(main_fits: list, out_path: Path):
    ncols = min(3, len(main_fits))
    nrows = int(np.ceil(len(main_fits) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.4 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(main_fits):]:
        ax.axis("off")
    rng = np.random.default_rng(0)
    for ax, ((metric, name, var, lab0, lab1), (res, d), row) in zip(axes, main_fits):
        grid = np.linspace(d["meta"].quantile(0.01), d["meta"].quantile(0.99), 200)
        levels = list(zip(AGE_SHOWN if var == "age10" else (0.0, 1.0), (lab0, lab1), CUSTOM_PALETTE[:2]))
        if var == "age10":
            sc = ax.scatter(d["meta"], d["wrong"] + rng.normal(0, 0.025, len(d)), c=d["age10"] * 10, cmap="viridis",
                            s=12, alpha=0.5, lw=0)
            plt.colorbar(sc, ax=ax, pad=0.01).set_label("Age", fontsize=TICK_SIZE - 2)
        for value, label, color in levels:
            if var != "age10":
                pts = d[d[var] == value]
                ax.scatter(pts["meta"], pts["wrong"] + rng.normal(0, 0.025, len(pts)), s=12, alpha=0.35, color=color, lw=0)
            ax.plot(grid, res.predict(pd.DataFrame({"meta": grid, var: value})), color=color, linewidth=2.2, label=label)
        ax.set_xlim(grid[0], grid[-1])
        ax.set_ylim(-0.08, 1.38)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel("Meta-temporal SUVR", fontsize=LABEL_SIZE - 4)
        ax.set_ylabel("Model calls positive" if metric == "specificity" else "Model calls negative", fontsize=LABEL_SIZE - 4)
        ax.set_title(f"Visual read {'−' if metric == 'specificity' else '+'}: {name}", fontsize=TITLE_SIZE - 4)
        ax.text(0.02, 0.97, f"OR at same SUVR {row['or']:.2f} [{row['or_lo']:.2f}–{row['or_hi']:.2f}], p = {row['p']:.2g}",
                transform=ax.transAxes, va="top", fontsize=TICK_SIZE - 4)
        ax.tick_params(labelsize=TICK_SIZE - 3)
        ax.grid(alpha=0.28)
        ax.legend(loc="center right", fontsize=LEGEND_SIZE - 2, frameon=True, framealpha=0.9)
    fig.suptitle("Model disagreeing with the visual read, at the same Meta-temporal SUVR", fontsize=TITLE_SIZE)
    finalize_figure(fig, rect=(0, 0, 1, 1.0))
    save_figure(fig, out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")


def make_forest_figure(table: pd.DataFrame, out_path: Path):
    checks = list(dict.fromkeys(zip(table["metric"], table["variable"])))
    ncols = 2
    nrows = int(np.ceil(len(checks) / ncols))
    max_rows = table.groupby(["metric", "variable"]).size().max()
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 0.42 * max_rows * nrows + 1.6 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(checks):]:
        ax.axis("off")
    for ax, (metric, var) in zip(axes, checks):
        sub = table[(table["metric"] == metric) & (table["variable"] == var)].reset_index(drop=True)
        y = np.arange(len(sub))[::-1]
        ax.axvline(1, color="#888888", linestyle="--", linewidth=1)
        for yi, (_, r) in zip(y, sub.iterrows()):
            if not np.isfinite(r.get("or", np.nan)):
                continue
            color = "#999999" if r["adjusted_for"] == "None" else (CUSTOM_PALETTE[1] if r["adjusted_for"] == MAIN else CUSTOM_PALETTE[0])
            ax.hlines(yi, r["or_lo"], r["or_hi"], color=color, linewidth=2.5)
            ax.scatter([r["or"]], [yi], s=55, color=color, zorder=3, edgecolor="white", linewidth=0.8)
            ax.annotate(f"{r['or']:.2f}", (r["or"], yi), textcoords="offset points", xytext=(0, 6), ha="center",
                        fontsize=TICK_SIZE - 4)
        ax.set_xscale("log")
        lo_x, hi_x = ax.get_xlim()
        ax.xaxis.set_major_locator(FixedLocator([t for t in (0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50) if lo_x <= t <= hi_x]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_yticks(y)
        ax.set_yticklabels([f"{a} (n={n})" for a, n in zip(sub["adjusted_for"], sub["n"])], fontsize=TICK_SIZE - 4)
        ax.set_ylim(-0.7, len(sub) - 0.3)
        ax.set_title(f"Visual read {'−' if metric == 'specificity' else '+'}: {var}{' per 10 years' if var == 'Age' else ''}",
                     fontsize=TITLE_SIZE - 5)
        ax.set_xlabel("Odds ratio of the model disagreeing with the read (log scale)", fontsize=LABEL_SIZE - 5)
        ax.tick_params(axis="x", labelsize=TICK_SIZE - 3)
        ax.grid(axis="x", alpha=0.28)
    fig.suptitle("Does the answer depend on how tau level and the other variables are accounted for?", fontsize=TITLE_SIZE - 1)
    finalize_figure(fig, rect=(0, 0, 1, 1.0))
    save_figure(fig, out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")


# ------------------------------
# Main
# ------------------------------
def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    preds_path, out_dir = resolve_predictions_csv(run_dir, args.dataset)
    preds = pd.read_csv(preds_path)[["ID_ind", "y", "pred"]].apply(pd.to_numeric, errors="coerce").dropna()
    demo_path = resolve_path(args.demo_csv, find_demo_csv)
    if demo_path is None:
        raise FileNotFoundError("No demographics csv found - pass --demo_csv.")
    demo = pd.read_csv(demo_path)
    demo = demo.loc[:, ~demo.columns.astype(str).str.startswith("Unnamed")]
    df = preds.merge(demo, left_on="ID_ind", right_on="ID", how="inner")
    if len(df) < len(preds):
        print(f"[WARNING] {len(preds) - len(df)} test scans have no row in {demo_path.name}.")
    add_subgroup_columns(df, female_code=args.female_code)
    print(f"[load] {len(df)} test scans")

    rows, main_fits = [], []
    for check in CHECKS:
        check_rows, main_fit = run_check(df, *check[:3])
        rows += check_rows
        main_row = next((r for r in check_rows if r["adjusted_for"] == MAIN and "or" in r), None)
        if main_fit is not None and main_row is not None:
            main_fits.append((check, main_fit, main_row))
    table = pd.DataFrame(rows)
    is_main = table["adjusted_for"] == MAIN
    table.loc[is_main, "p_fdr_main"] = fdr_bh(table.loc[is_main, "p"])

    out_dir.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(out_dir / "visual_read_errors_at_same_suvr.csv", index=False)
    if main_fits:
        make_curves_figure(main_fits, out_dir / "visual_read_errors_at_same_suvr.png")
    make_forest_figure(table, out_dir / "visual_read_errors_robustness.png")
    regions = regional_table(df)
    regions.round(4).to_csv(out_dir / "visual_read_regions_model_pos_vs_neg.csv", index=False)
    make_regions_figure(df, out_dir / "visual_read_regions_model_pos_vs_neg.png")

    print("\n[errors at the same SUVR] odds ratio of the model disagreeing with the visual read (main row FDR-corrected across the 5 checks)")
    for (metric, var), sub in table.groupby(["metric", "variable"], sort=False):
        print(f"  {metric} {var}{' (per 10 years)' if var == 'Age' else ''}")
        for _, r in sub.iterrows():
            if not np.isfinite(r.get("or", np.nan)):
                print(f"      {r['adjusted_for']:<42} {r.get('note', '')}")
                continue
            extra = "".join(f"   {k.replace('_group0_group1', '').replace('_', ' ')} {r[k]}"
                            for k in ("median_meta_group0_group1", "median_mesial_group0_group1") if isinstance(r.get(k), str))
            fdr = f" (FDR {r['p_fdr_main']:.3g})" if r["adjusted_for"] == MAIN else ""
            print(f"      {r['adjusted_for']:<42} OR {r['or']:.2f} [{r['or_lo']:.2f}-{r['or_hi']:.2f}], p = {r['p']:.3g}{fdr}, "
                  f"n={r['n']} ({r['n_disagree']} disagree){extra}")
    print("\n[regions] Visual read − scans: median SUVR Model − / Model +, difference in SD units (Model − SD)")
    for group, sub in regions.groupby("group", sort=False):
        print(f"  {group} (Model − n={sub['n_model_neg'].iloc[0]}, Model + n={sub['n_model_pos'].iloc[0]})")
        for _, r in sub.iterrows():
            print(f"      {r['region']:<18} {r['median_model_neg']:.2f} / {r['median_model_pos']:.2f}   "
                  f"diff {r['diff_in_sd']:+.2f} SD   Mann-Whitney p = {r['p_mann_whitney']:.2g}")
    print(f"\n[done] saved to {out_dir}")


if __name__ == "__main__":
    main()
