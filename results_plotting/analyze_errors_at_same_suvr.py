#!/usr/bin/env python3
"""Do the subgroup differences in the visual-read model's mistakes survive at the same tau level?

The subgroup analysis found more false positives in Aβ+, APOE ε4 carriers and older people, and
more missed positives in Aβ− and CU people. Two explanations predict that:
    Harder cases  - e.g. read-negative Aβ+ scans have more sub-threshold tau, so they sit closer to
                    the boundary and any reader / model gets more of them wrong
    Learned lean  - the model leans positive for scans that look like the groups where positives were
                    common in training, even at the same tau level
Comparing at the same Meta-temporal SUVR separates them: if a group difference disappears once SUVR
is accounted for, it was harder cases; if it stays, the model treats that group differently.

For each difference that was significant in the subgroup analysis, within the visual-read stratum it
was found in (read-negative for specificity, read-positive for sensitivity):
    outcome  = model wrong (false positive, or missed positive)
    model 1  = outcome ~ group                      (the original difference, as an odds ratio)
    model 2  = outcome ~ group + Meta-temporal SUVR (the same, at the same tau level)
Logistic regression; odds ratio with 95% Wald CI, p from a likelihood-ratio test. Both models use the
same scans. With 31 false positives / 11 missed positives only one adjustment variable is affordable,
so nothing else is added. These are follow-up tests, FDR-corrected as their own family.

Robustness (does the answer depend on how "same tau level" is defined?), per check:
    Meta-temporal, Firth     - Firth's penalised regression, less biased with this few mistakes
    Meta-temporal, curved    - SUVR + SUVR², in case the risk does not rise in a straight line on the log-odds scale
    Mesial temporal          - (false positives only) is the group effect medial-temporal signal (early tau or
                               off-target, which the reading rules ignore)? If the odds ratio shrinks when
                               mesial SUVR is accounted for, the effect goes through medial-temporal signal
    Meta + mesial temporal   - both together
    Meta-temporal, complete-data subset
                             - (false positives only) the main model refitted on the scans the next two use, so a
                               change can be split into "different people" vs "adjustment"
    Meta-temporal + the other two of amyloid / APOE / age, Firth
                             - (false positives only) does each survive the others, e.g. is APOE just amyloid?
                               Uses only scans with amyloid, APOE, age and Centiloid known (APOE is missing at
                               NACC / Geneva), so n is smaller
    Meta-temporal + Centiloid instead of amyloid yes/no, Firth
                             - (APOE and age only) the amount of amyloid, not just positive/negative
These are sensitivity analyses: reported, not tested as a family.

Outputs (to <run_dir>/figures):
    visual_read_errors_robustness.csv/.png - every variant: odds ratio [95% CI], p, n; forest plot per check
    visual_read_errors_at_same_suvr.csv  - per check: n, mistakes, median SUVR per group, odds ratio
                                           before / after SUVR, p-values
    visual_read_errors_at_same_suvr.png  - per check: probability of a mistake vs Meta-temporal SUVR,
                                           one fitted curve per group (curves on top of each other =
                                           harder cases; side by side = the model treats the group
                                           differently at the same SUVR)

Usage:
    python results_plotting/analyze_errors_at_same_suvr.py /path/to/run_dir --dataset Gothenburg \
        --demo_csv /path/to/demo.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np
import pandas as pd
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

SUVR_REGION = "MetaTemporal"
# (metric, name, how to build the 0/1 or numeric group variable, label for 0, label for 1)
CHECKS = [
    ("specificity", "Amyloid status", "amy", "Aβ−", "Aβ+"),
    ("specificity", "APOE ε4", "apoe", "Non-carrier", "Carrier"),
    ("specificity", "Age", "age", "65 years", "80 years"),
    ("sensitivity", "Amyloid status", "amy", "Aβ−", "Aβ+"),
    ("sensitivity", "Diagnosis", "dx", "CU", "Impaired"),
]
METRIC_READ = {"sensitivity": 1, "specificity": 0}
AGE_UNIT = 10  # odds ratio for age is per 10 years
AGE_SHOWN = (65, 80)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="Visual-read run directory.")
    parser.add_argument("--dataset", default="Gothenburg", help="Dataset name used in the result filenames.")
    parser.add_argument("--demo_csv", default=None, help="Demographics csv (default: data/demo.csv).")
    parser.add_argument("--female_code", type=float, default=2, help="Value of `gender` meaning female.")
    return parser.parse_args()


def fit_logistic(X: np.ndarray, y: np.ndarray, firth: bool = False, fix_first: bool = False) -> dict:
    """Logistic regression by Newton-Raphson (X without intercept). Returns coefficients and
    covariance on the original scale (intercept first), log-likelihood, and whether it converged
    without separation (a coefficient running off to infinity).

    firth=True: Firth's penalised likelihood, which removes most of the small-sample bias of
    ordinary logistic regression (odds ratios pulled away from 1 when there are few events) and
    stays finite under separation. The log-likelihood returned is then the penalised one, so
    likelihood-ratio tests compare penalised fits, as is standard for Firth regression.

    fix_first=True: the first column's coefficient is held at 0 (the null model for testing it),
    while the penalty still uses the full design - the correct Firth likelihood-ratio test
    (refitting a smaller design would change the penalty and give wrong p-values)."""
    m, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1
    Z = np.column_stack([np.ones(len(y)), (X - m) / sd])
    b = np.zeros(Z.shape[1])
    converged = False
    for _ in range(200):
        p = 1 / (1 + np.exp(-np.clip(Z @ b, -30, 30)))
        w = p * (1 - p)
        H = Z.T @ (Z * w[:, None]) + 1e-10 * np.eye(len(b))
        resid = y - p
        if firth:
            hat = w * np.einsum("ij,jk,ik->i", Z, np.linalg.inv(H), Z)
            resid = resid + hat * (0.5 - p)
        score = Z.T @ resid
        free = np.ones(len(b), dtype=bool)
        if fix_first:
            free[1] = False
        step = np.zeros(len(b))
        step[free] = np.linalg.solve(H[np.ix_(free, free)], score[free])
        if firth:
            step = np.clip(step, -5, 5)
        b += step
        if np.abs(step).max() < 1e-9:
            converged = True
            break
    p = np.clip(1 / (1 + np.exp(-np.clip(Z @ b, -30, 30))), 1e-12, 1 - 1e-12)
    loglik = float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum())
    info = Z.T @ (Z * (p * (1 - p))[:, None])
    if firth:
        loglik += 0.5 * float(np.linalg.slogdet(info)[1])
    cov_z = np.linalg.inv(info)
    # back to the original scale: beta_j = b_j / sd_j, intercept = b_0 - sum(b_j m_j / sd_j)
    T = np.eye(len(b))
    T[0, 1:] = -m / sd
    T[1:, 1:] = np.diag(1 / sd)
    beta, cov = T @ b, T @ cov_z @ T.T
    ok = converged and (firth or np.abs(b[1:]).max(initial=0) < 15)
    return {"beta": beta, "cov": cov, "loglik": loglik, "ok": ok}


def lr_test(full: dict, reduced: dict) -> float:
    return float(stats.chi2.sf(max(2 * (full["loglik"] - reduced["loglik"]), 0), 1))


def odds_ratio(fit: dict, idx: int, unit: float = 1.0) -> tuple[float, float, float]:
    b, se = fit["beta"][idx] * unit, np.sqrt(fit["cov"][idx, idx]) * unit
    lo, hi = np.clip([b - 1.96 * se, b + 1.96 * se], -50, 50)
    return float(np.exp(np.clip(b, -50, 50))), float(np.exp(lo)), float(np.exp(hi))


def group_variable(df: pd.DataFrame, kind: str) -> pd.Series:
    if kind == "amy":
        return df["_g_amy"].map({"Aβ negative": 0.0, "Aβ positive": 1.0})
    if kind == "apoe":
        return df["_g_apoe"].map({"Non-carrier": 0.0, "Carrier": 1.0})
    if kind == "dx":
        return df["_g_dx"].map({"CU": 0.0, "MCI": 1.0, "AD dementia": 1.0})  # "Other" stays missing
    return pd.to_numeric(df["age"], errors="coerce")


def run_check(df: pd.DataFrame, metric: str, name: str, kind: str, lab0: str, lab1: str) -> dict:
    read = METRIC_READ[metric]
    sub = df[df["y"] == read].copy()
    sub["_x"] = group_variable(sub, kind)
    sub = sub.dropna(subset=["_x", "_suvr"])
    wrong = (sub["pred"] != sub["y"]).to_numpy(float)
    x, s = sub["_x"].to_numpy(float), sub["_suvr"].to_numpy(float)
    unit = AGE_UNIT if kind == "age" else 1.0
    out = {"metric": metric, "variable": name, "comparison": f"{lab1} vs {lab0}" if kind != "age" else "per 10 years",
           "visual_read": "+" if read else "−", "n": len(sub), "n_mistakes": int(wrong.sum())}
    if kind == "age":
        out["median_suvr_group0"] = float(np.median(s[x < np.median(x)])) if len(x) else np.nan
        out["median_suvr_group1"] = float(np.median(s[x >= np.median(x)])) if len(x) else np.nan
        out["groups_for_suvr"] = "below / above median age"
    else:
        out["median_suvr_group0"] = float(np.median(s[x == 0])) if (x == 0).any() else np.nan
        out["median_suvr_group1"] = float(np.median(s[x == 1])) if (x == 1).any() else np.nan
        out["groups_for_suvr"] = f"{lab0} / {lab1}"
    if wrong.sum() < 2 or len(np.unique(x)) < 2:
        out["note"] = "too few mistakes to fit"
        return out
    m_null = fit_logistic(np.zeros((len(x), 0)), wrong)
    m_x = fit_logistic(x[:, None], wrong)
    m_s = fit_logistic(s[:, None], wrong)
    m_xs = fit_logistic(np.column_stack([x, s]), wrong)
    out["or_unadjusted"], out["or_unadj_lo"], out["or_unadj_hi"] = odds_ratio(m_x, 1, unit)
    out["p_unadjusted"] = lr_test(m_x, m_null)
    out["or_same_suvr"], out["or_same_suvr_lo"], out["or_same_suvr_hi"] = odds_ratio(m_xs, 1, unit)
    out["p_same_suvr"] = lr_test(m_xs, m_s)
    out["or_suvr_per_0.1"] = odds_ratio(m_xs, 2, 0.1)[0]
    out["fit_ok"] = bool(m_x["ok"] and m_xs["ok"])
    if not out["fit_ok"]:
        out["note"] = "unstable fit (a group with almost no mistakes) - treat with caution"
    out["_fit"], out["_data"] = m_xs, (x, s, wrong)
    return out


# ------------------------------
# Robustness: does the main answer depend on how "same tau level" is defined?
# ------------------------------
COMPLETE = ["amy", "apoe", "age", "cl"]  # the mutually adjusted models all use scans with these known


def robustness_variants(metric: str, kind: str) -> list[tuple[str, list[str], bool, list[str]]]:
    """(label, adjustment columns, Firth?, extra columns that must be known) for one check. Main
    analysis first. Read-negative checks (~30 mistakes) can afford the extra adjustments;
    read-positive checks (~11 mistakes) only get the ones that add no or one parameter.

    The mutually adjusted models need amyloid, APOE, age and Centiloid all known, which drops
    scans (mostly NACC / Geneva). "Meta-temporal, complete-data subset" refits the main model on
    exactly those scans, so a change in the odds ratio can be split into "different people" (main
    vs subset) and "adjustment" (subset vs mutually adjusted). The Centiloid version replaces
    amyloid yes/no by the amount of amyloid, which the yes/no label cannot capture."""
    variants = [("Meta-temporal (main)", ["meta"], False, []),
                ("Meta-temporal, Firth", ["meta"], True, []),
                ("Meta-temporal, curved", ["meta", "meta_sq"], False, [])]
    if metric == "specificity":
        names = {"amy": "amyloid", "apoe": "APOE", "age": "age"}
        others = [o for o in ("amy", "apoe", "age") if o != kind]
        variants += [("Mesial temporal", ["mesial"], False, []),
                     ("Meta + mesial temporal", ["meta", "mesial"], False, []),
                     ("Meta-temporal, complete-data subset", ["meta"], False, COMPLETE),
                     (f"Meta-temporal + {' + '.join(names[o] for o in others)}, Firth", ["meta", *others], True, COMPLETE)]
        if kind != "amy":
            cl_others = ["cl" if o == "amy" else o for o in others]
            label = " + ".join("Centiloid" if o == "cl" else names[o] for o in cl_others)
            variants.append((f"Meta-temporal + {label}, Firth", ["meta", *cl_others], True, COMPLETE))
    return variants


def run_robustness(df: pd.DataFrame, metric: str, name: str, kind: str, lab0: str, lab1: str) -> list[dict]:
    read = METRIC_READ[metric]
    sub = df[df["y"] == read].copy()
    sub["_x"] = group_variable(sub, kind)
    sub["meta"] = sub["_suvr"]
    sub["meta_sq"] = (sub["_suvr"] - sub["_suvr"].mean()) ** 2
    sub["mesial"] = pd.to_numeric(sub.get("MesialTemporal"), errors="coerce") if "MesialTemporal" in sub.columns else np.nan
    for k in ("amy", "apoe", "age"):
        sub[k] = group_variable(sub, k)
    sub["cl"] = pd.to_numeric(sub["CL"], errors="coerce") if "CL" in sub.columns else np.nan
    unit = AGE_UNIT if kind == "age" else 1.0
    rows = []
    for label, adjust, firth, require in [("None", [], False, [])] + robustness_variants(metric, kind):
        d = sub.dropna(subset=["_x", *adjust, *require])
        wrong = (d["pred"] != d["y"]).to_numpy(float)
        row = {"metric": metric, "variable": name, "adjusted_for": label, "firth": firth,
               "n": len(d), "n_mistakes": int(wrong.sum())}
        if wrong.sum() < 2 or d["_x"].nunique() < 2:
            rows.append({**row, "note": "too few mistakes / one group only"})
            continue
        A = d[adjust].to_numpy(float) if adjust else np.zeros((len(d), 0))
        design = np.column_stack([d["_x"].to_numpy(float), A])
        full = fit_logistic(design, wrong, firth)
        reduced = fit_logistic(design, wrong, firth, fix_first=True)
        row["or"], row["or_lo"], row["or_hi"] = odds_ratio(full, 1, unit)
        row["p"] = lr_test(full, reduced)
        if not full["ok"]:
            row["note"] = "unstable fit"
        if "mesial" in adjust and not np.isnan(d["mesial"]).all():
            g = d["_x"] >= d["_x"].median() if kind == "age" else d["_x"] == 1
            row["median_mesial_group0_group1"] = f"{d.loc[~g, 'mesial'].median():.2f} / {d.loc[g, 'mesial'].median():.2f}"
        rows.append(row)
    return rows


def make_forest_figure(rob: pd.DataFrame, out_path: Path):
    checks = list(dict.fromkeys(zip(rob["metric"], rob["variable"])))
    ncols = 2
    nrows = int(np.ceil(len(checks) / ncols))
    max_rows = rob.groupby(["metric", "variable"]).size().max()
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 0.42 * max_rows * nrows + 1.6 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(checks):]:
        ax.axis("off")
    for ax, (metric, var) in zip(axes, checks):
        sub = rob[(rob["metric"] == metric) & (rob["variable"] == var)].reset_index(drop=True)
        y = np.arange(len(sub))[::-1]
        ax.axvline(1, color="#888888", linestyle="--", linewidth=1)
        for yi, (_, r) in zip(y, sub.iterrows()):
            if not np.isfinite(r.get("or", np.nan)):
                continue
            color = "#999999" if r["adjusted_for"] == "None" else (CUSTOM_PALETTE[1] if "(main)" in r["adjusted_for"] else CUSTOM_PALETTE[0])
            ax.hlines(yi, r["or_lo"], r["or_hi"], color=color, linewidth=2.5)
            ax.scatter([r["or"]], [yi], s=55, color=color, zorder=3, edgecolor="white", linewidth=0.8,
                       marker="D" if r["firth"] else "o")
            ax.annotate(f"{r['or']:.2f}", (r["or"], yi), textcoords="offset points", xytext=(0, 6), ha="center",
                        fontsize=TICK_SIZE - 4)
        ax.set_xscale("log")
        lo_x, hi_x = ax.get_xlim()
        ticks = [t for t in (0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50) if lo_x <= t <= hi_x]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_yticks(y)
        ax.set_yticklabels([f"{a} (n={n})" for a, n in zip(sub["adjusted_for"], sub["n"])], fontsize=TICK_SIZE - 4)
        ax.set_ylim(-0.7, len(sub) - 0.3)
        read = "−" if metric == "specificity" else "+"
        unit = " per 10 years" if var == "Age" else ""
        ax.set_title(f"Visual read {read}: {var}{unit}", fontsize=TITLE_SIZE - 5)
        ax.set_xlabel("Odds ratio of a model mistake (log scale)", fontsize=LABEL_SIZE - 5)
        ax.tick_params(axis="x", labelsize=TICK_SIZE - 3)
        ax.grid(axis="x", alpha=0.28)
    fig.suptitle("Does the answer depend on how the tau level is accounted for?", fontsize=TITLE_SIZE - 1)
    finalize_figure(fig, rect=(0, 0, 1, 1.0))
    save_figure(fig, out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")


def make_figure(results: list, out_path: Path):
    shown = [r for r in results if "_fit" in r]
    if not shown:
        return
    ncols = min(3, len(shown))
    nrows = int(np.ceil(len(shown) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.4 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(shown):]:
        ax.axis("off")
    rng = np.random.default_rng(0)
    for ax, r in zip(axes, shown):
        x, s, wrong = r["_data"]
        beta = r["_fit"]["beta"]
        is_age = r["variable"] == "Age"
        check = next(c for c in CHECKS if c[0] == r["metric"] and c[1] == r["variable"])
        lab0, lab1 = check[3], check[4]
        grid = np.linspace(np.percentile(s, 1), np.percentile(s, 99), 200)
        levels = [(AGE_SHOWN[0], lab0, CUSTOM_PALETTE[0]), (AGE_SHOWN[1], lab1, CUSTOM_PALETTE[1])] if is_age \
            else [(0.0, lab0, CUSTOM_PALETTE[0]), (1.0, lab1, CUSTOM_PALETTE[1])]
        for value, label, color in levels:
            if not is_age:
                pts = x == value
                ax.scatter(s[pts], wrong[pts] + rng.normal(0, 0.025, pts.sum()), s=12, alpha=0.35, color=color, lw=0)
            prob = 1 / (1 + np.exp(-(beta[0] + beta[1] * value + beta[2] * grid)))
            ax.plot(grid, prob, color=color, linewidth=2.2, label=label)
        if is_age:
            sc = ax.scatter(s, wrong + rng.normal(0, 0.025, len(s)), c=x, cmap="viridis", s=12, alpha=0.5, lw=0)
            plt.colorbar(sc, ax=ax, pad=0.01).set_label("Age", fontsize=TICK_SIZE - 2)
        ax.set_xlim(grid[0], grid[-1])
        ax.set_ylim(-0.08, 1.38)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel("Meta-temporal SUVR", fontsize=LABEL_SIZE - 4)
        ax.set_ylabel("Model calls positive" if r["metric"] == "specificity" else "Model calls negative",
                      fontsize=LABEL_SIZE - 4)
        ax.set_title(f"Visual read {r['visual_read']}: {r['variable']}", fontsize=TITLE_SIZE - 4)
        ax.text(0.02, 0.97, f"OR at same SUVR {r['or_same_suvr']:.2f} [{r['or_same_suvr_lo']:.2f}–{r['or_same_suvr_hi']:.2f}]\n"
                            f"p = {r['p_same_suvr']:.2g} (before SUVR: OR {r['or_unadjusted']:.2f}, p = {r['p_unadjusted']:.2g})",
                transform=ax.transAxes, va="top", fontsize=TICK_SIZE - 4)
        ax.tick_params(labelsize=TICK_SIZE - 3)
        ax.grid(alpha=0.28)
        ax.legend(loc="center right", fontsize=LEGEND_SIZE - 2, frameon=True, framealpha=0.9)
    fig.suptitle("Model mistakes at the same Meta-temporal SUVR", fontsize=TITLE_SIZE)
    finalize_figure(fig, rect=(0, 0, 1, 1.0))
    save_figure(fig, out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    preds_path, out_dir = resolve_predictions_csv(run_dir, args.dataset)
    preds = pd.read_csv(preds_path)
    preds = preds[["ID_ind", "y", "pred"]].apply(pd.to_numeric, errors="coerce").dropna()
    demo_path = resolve_path(args.demo_csv, find_demo_csv)
    if demo_path is None:
        raise FileNotFoundError("No demographics csv found - pass --demo_csv.")
    demo = pd.read_csv(demo_path)
    demo = demo.loc[:, ~demo.columns.astype(str).str.startswith("Unnamed")]
    df = preds.merge(demo, left_on="ID_ind", right_on="ID", how="inner")
    if len(df) < len(preds):
        print(f"[WARNING] {len(preds) - len(df)} test scans have no row in {demo_path.name}.")
    add_subgroup_columns(df, female_code=args.female_code)
    df["_suvr"] = pd.to_numeric(df[SUVR_REGION], errors="coerce")
    print(f"[load] {len(df)} test scans")

    results = [run_check(df, *c) for c in CHECKS]
    p = pd.Series([r.get("p_same_suvr", np.nan) for r in results])
    for r, q in zip(results, fdr_bh(p)):
        r["p_same_suvr_fdr"] = q

    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in results])
    table.round(4).to_csv(out_dir / "visual_read_errors_at_same_suvr.csv", index=False)
    make_figure(results, out_dir / "visual_read_errors_at_same_suvr.png")
    rob = pd.DataFrame([row for c in CHECKS for row in run_robustness(df, *c)])
    rob.round(4).to_csv(out_dir / "visual_read_errors_robustness.csv", index=False)
    make_forest_figure(rob, out_dir / "visual_read_errors_robustness.png")

    print("\n[errors at the same SUVR] odds ratio of a mistake, before -> after accounting for Meta-temporal SUVR")
    for r in results:
        head = (f"  {r['metric']:<11} {r['variable']:<15} ({r['comparison']}; read {r['visual_read']}, "
                f"n={r['n']}, {r['n_mistakes']} mistakes)")
        print(head)
        print(f"      median SUVR {r['groups_for_suvr']}: {r['median_suvr_group0']:.2f} / {r['median_suvr_group1']:.2f}")
        if "or_unadjusted" not in r:
            print(f"      {r.get('note', '')}")
            continue
        print(f"      before: OR {r['or_unadjusted']:.2f} [{r['or_unadj_lo']:.2f}-{r['or_unadj_hi']:.2f}], p = {r['p_unadjusted']:.3g}")
        print(f"      same SUVR: OR {r['or_same_suvr']:.2f} [{r['or_same_suvr_lo']:.2f}-{r['or_same_suvr_hi']:.2f}], "
              f"p = {r['p_same_suvr']:.3g} (FDR {r['p_same_suvr_fdr']:.3g})" + (f"   [{r['note']}]" if r.get("note") else ""))
    print("\n[robustness] odds ratio of a mistake under different ways of accounting for tau level and the other variables")
    for (metric, var), sub in rob.groupby(["metric", "variable"], sort=False):
        print(f"  {metric} {var}{' (per 10 years)' if var == 'Age' else ''}")
        for _, r in sub.iterrows():
            if not np.isfinite(r.get("or", np.nan)):
                print(f"      {r['adjusted_for']:<48} {r.get('note', '')}")
                continue
            extra = f"   median mesial {r['median_mesial_group0_group1']}" if isinstance(r.get("median_mesial_group0_group1"), str) else ""
            note = f"   [{r['note']}]" if isinstance(r.get("note"), str) else ""
            print(f"      {r['adjusted_for']:<48} OR {r['or']:.2f} [{r['or_lo']:.2f}-{r['or_hi']:.2f}], p = {r['p']:.3g}, "
                  f"n={r['n']} ({r['n_mistakes']} mistakes){extra}{note}")
    print(f"\n[done] saved to {out_dir}")


if __name__ == "__main__":
    main()
