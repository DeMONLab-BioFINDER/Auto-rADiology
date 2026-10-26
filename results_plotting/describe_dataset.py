#!/usr/bin/env python3
"""Plain description of the dataset's variables - no model, no significance tests.

Covers diagnosis, amyloid status, APOE e4, sex, site, age (and Centiloid / CDR-SB in the tables),
plus the visual read itself. Uses the whole dataset; with --run_dir, also the training and
held-out test sets of that run.

Outputs (default <proj_path>/results/dataset_description, outside the repo):
    dataset_associations.png / .csv      - how strongly each pair of variables goes together, 0 (unrelated)
                                           to 1 (one fully determines the other): Cramér's V for two
                                           categorical variables, the correlation ratio (eta) for age
                                           vs a categorical one. Whole dataset, and the test set if given.
                                           Shows which subgroup results could be the same effect.
    dataset_table1_by_site.csv           - "Table 1": per site and overall, n, age, sex, diagnosis, Aβ+,
                                           APOE e4, visual read +, Centiloid, CDR-SB
    dataset_read_positive_rates.png/.csv - % visual read positive (Wilson 95% CI) in each group; age in
                                           bands here for description only (the tests use age as a number)
    dataset_missing_by_site.csv          - % missing per variable per site (every test drops missing values,
                                           so a variable missing mostly at one site leaves that site out)
    dataset_train_vs_test.csv            - (with --run_dir) the same table for training vs test, with the
                                           standardised difference (|SMD| < 0.1 is the usual "balanced")

Usage:
    python results_plotting/describe_dataset.py --demo_csv /path/to/demo.csv [--run_dir /path/to/run_dir]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from plot_utils import (
    CUSTOM_PALETTE,
    LABEL_SIZE,
    TICK_SIZE,
    TITLE_SIZE,
    combine_dx_groups,
    find_demo_csv,
    finalize_figure,
    resolve_path,
    save_figure,
    wilson_interval,
)

AGE_BANDS = [-np.inf, 65, 75, 85, np.inf]
AGE_LABELS = ["<65", "65–74", "75–84", "≥85"]
CATEGORICAL = [("_dx", "Diagnosis"), ("_amy", "Amyloid"), ("_apoe", "APOE ε4"), ("_sex", "Sex"),
               ("_site", "Site"), ("_read", "Visual read")]
LEVEL_ORDER = {"_dx": ["CU", "MCI", "AD dementia", "Other"], "_amy": ["Aβ−", "Aβ+"],
               "_apoe": ["Non-carrier", "Carrier"], "_sex": ["Female", "Male"], "_read": ["Negative", "Positive"],
               "_ageband": AGE_LABELS}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demo_csv", default=None, help="Demographics csv for the whole dataset (default: data/demo.csv).")
    parser.add_argument("--run_dir", default=None, help="Run directory with splits/, for the training vs test comparison.")
    parser.add_argument("--female_code", type=float, default=2, help="Value of `gender` meaning female.")
    parser.add_argument("--out_dir", default=None, help="Output directory (default: <proj_path>/results/dataset_description).")
    return parser.parse_args()


def prepare(demo: pd.DataFrame, female_code: float) -> pd.DataFrame:
    df = demo.loc[:, ~demo.columns.astype(str).str.startswith("Unnamed")].copy()
    for c in ["age", "CL", "CDR_SB", "gender", "apoe", "amyloid_status", "visual_read"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    def cat(col, mapping):
        return df[col].map(mapping).astype("string") if col in df.columns else pd.Series(pd.NA, index=df.index, dtype="string")

    df["_dx"] = (combine_dx_groups(df["dx"]).replace({"AlzCS dem": "AD dementia"})
                 if "dx" in df.columns else pd.Series(pd.NA, index=df.index, dtype="string"))
    df["_amy"] = cat("amyloid_status", {0: "Aβ−", 1: "Aβ+"})
    df["_apoe"] = cat("apoe", {0: "Non-carrier", 1: "Carrier"})
    df["_sex"] = df["gender"].map(lambda v: pd.NA if pd.isna(v) else ("Female" if v == female_code else "Male")).astype("string") \
        if "gender" in df.columns else pd.Series(pd.NA, index=df.index, dtype="string")
    df["_site"] = df["site"].astype("string") if "site" in df.columns else pd.Series(pd.NA, index=df.index, dtype="string")
    df["_read"] = cat("visual_read", {0: "Negative", 1: "Positive"})
    df["_ageband"] = pd.cut(df["age"], AGE_BANDS, labels=AGE_LABELS, right=False).astype("string") \
        if "age" in df.columns else pd.Series(pd.NA, index=df.index, dtype="string")
    return df


# ------------------------------
# 1. Associations
# ------------------------------
def cramers_v(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    ok = a.notna() & b.notna()
    table = pd.crosstab(a[ok], b[ok])
    if min(table.shape) < 2:
        return np.nan, int(ok.sum())
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    return float(np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1)))), int(ok.sum())


def correlation_ratio(groups: pd.Series, values: pd.Series) -> tuple[float, int]:
    """eta: share of the spread in `values` explained by the groups, square-rooted (0..1)."""
    ok = groups.notna() & values.notna()
    g, v = groups[ok], values[ok].astype(float)
    if g.nunique() < 2 or v.var() == 0:
        return np.nan, int(ok.sum())
    between = sum(len(x) * (x.mean() - v.mean()) ** 2 for _, x in v.groupby(g))
    return float(np.sqrt(between / ((v - v.mean()) ** 2).sum())), int(ok.sum())


def association_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    names = [n for c, n in CATEGORICAL if df[c].notna().any()] + (["Age"] if df["age"].notna().any() else [])
    cols = {n: c for c, n in CATEGORICAL}
    mat = pd.DataFrame(np.nan, index=names, columns=names)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if "Age" in (a, b):
                other = b if a == "Age" else a
                value, n = correlation_ratio(df[cols[other]], df["age"])
                measure = "correlation ratio (eta)"
            else:
                value, n = cramers_v(df[cols[a]], df[cols[b]])
                measure = "Cramér's V"
            mat.loc[a, b] = mat.loc[b, a] = value
            rows.append({"variable_1": a, "variable_2": b, "association": value, "measure": measure, "n": n})
    return mat, pd.DataFrame(rows)


def make_association_figure(mats: dict, out_path: Path):
    fig, axes = plt.subplots(1, len(mats), figsize=(6.6 * len(mats), 4.9), squeeze=False)
    for ax, (title, mat) in zip(axes[0], mats.items()):
        shown = mat.where(~np.triu(np.ones(mat.shape, dtype=bool)))
        im = ax.imshow(shown.to_numpy(float), cmap="Blues", vmin=0, vmax=1)
        for i in range(len(mat)):
            for j in range(i):
                v = mat.iloc[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=TICK_SIZE - 1,
                            color="white" if v > 0.55 else "black")
        ax.set_xticks(range(len(mat)))
        ax.set_xticklabels(mat.columns, rotation=40, ha="right", fontsize=TICK_SIZE - 1)
        ax.set_yticks(range(len(mat)))
        ax.set_yticklabels(mat.index, fontsize=TICK_SIZE - 1)
        ax.set_xlim(-0.5, len(mat) - 1.5)
        ax.set_ylim(len(mat) - 0.5, 0.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, fontsize=TITLE_SIZE - 3)
    cbar = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, pad=0.02)
    cbar.set_label("Association (0 = unrelated, 1 = fully determined)", fontsize=LABEL_SIZE - 4)
    fig.suptitle("How strongly the variables go together", fontsize=TITLE_SIZE)
    save_figure(fig, out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")


# ------------------------------
# 2 / 5. Table 1 (by site, or training vs test)
# ------------------------------
def _median_iqr(s: pd.Series) -> str:
    s = s.dropna()
    return f"{s.median():.1f} [{s.quantile(.25):.1f}–{s.quantile(.75):.1f}]" if len(s) else "–"


def _n_pct(mask: pd.Series, known: pd.Series) -> str:
    n_known = int(known.sum())
    return f"{int((mask & known).sum())} ({(mask & known).sum() / n_known:.0%})" if n_known else "–"


def table_one(groups: dict) -> pd.DataFrame:
    rows = {}
    for name, g in groups.items():
        col = {"n": str(len(g)), "Age, median [IQR]": _median_iqr(g["age"])}
        col["Female, n (% of known)"] = _n_pct(g["_sex"] == "Female", g["_sex"].notna())
        for level in LEVEL_ORDER["_dx"]:
            col[f"Diagnosis: {level}, n (%)"] = _n_pct(g["_dx"] == level, g["_dx"].notna())
        col["Aβ+, n (% of known)"] = _n_pct(g["_amy"] == "Aβ+", g["_amy"].notna())
        col["APOE ε4 carrier, n (% of known)"] = _n_pct(g["_apoe"] == "Carrier", g["_apoe"].notna())
        col["Visual read +, n (%)"] = _n_pct(g["_read"] == "Positive", g["_read"].notna())
        if "CL" in g.columns:
            col["Centiloid, median [IQR]"] = _median_iqr(g["CL"])
        if "CDR_SB" in g.columns:
            col["CDR-SB, median [IQR]"] = _median_iqr(g["CDR_SB"])
        rows[name] = col
    return pd.DataFrame(rows)


def smd(a: pd.Series, b: pd.Series) -> float:
    """Standardised mean difference between two samples (a 0/1 indicator works too)."""
    a, b = a.dropna().astype(float), b.dropna().astype(float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt((a.var() + b.var()) / 2)
    return float(abs(a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def train_vs_test(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    table = table_one({"Training": train, "Test": test})
    checks = {"Age, median [IQR]": lambda g: g["age"],
              "Female, n (% of known)": lambda g: (g["_sex"] == "Female").where(g["_sex"].notna()),
              "Aβ+, n (% of known)": lambda g: (g["_amy"] == "Aβ+").where(g["_amy"].notna()),
              "APOE ε4 carrier, n (% of known)": lambda g: (g["_apoe"] == "Carrier").where(g["_apoe"].notna()),
              "Visual read +, n (%)": lambda g: (g["_read"] == "Positive").where(g["_read"].notna()),
              "Centiloid, median [IQR]": lambda g: g.get("CL"),
              "CDR-SB, median [IQR]": lambda g: g.get("CDR_SB")}
    for level in LEVEL_ORDER["_dx"]:
        checks[f"Diagnosis: {level}, n (%)"] = lambda g, lv=level: (g["_dx"] == lv).where(g["_dx"].notna())
    table["|SMD|"] = [round(smd(checks[r](train), checks[r](test)), 3) if r in checks and checks[r](train) is not None
                      else np.nan for r in table.index]
    return table


# ------------------------------
# 3. Read-positive rates
# ------------------------------
def read_positive_rates(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, name in [("_dx", "Diagnosis"), ("_amy", "Amyloid"), ("_apoe", "APOE ε4"), ("_sex", "Sex"),
                      ("_ageband", "Age"), ("_site", "Site")]:
        levels = LEVEL_ORDER.get(col) or sorted(df[col].dropna().unique())
        for level in levels:
            g = df[(df[col] == level) & df["_read"].notna()]
            if len(g) == 0:
                continue
            k = int((g["_read"] == "Positive").sum())
            lo, hi = wilson_interval(k, len(g))
            rows.append({"variable": name, "level": level, "n": len(g), "n_read_positive": k,
                         "share_read_positive": k / len(g), "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def make_rates_figure(rates: pd.DataFrame, out_path: Path):
    variables = list(dict.fromkeys(rates["variable"]))
    ncols = 3
    nrows = int(np.ceil(len(variables) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.3 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(variables):]:
        ax.axis("off")
    for i, (ax, var) in enumerate(zip(axes, variables)):
        sub = rates[rates["variable"] == var].reset_index(drop=True)
        x = np.arange(len(sub))
        ax.bar(x, sub["share_read_positive"] * 100, color=CUSTOM_PALETTE[1], alpha=0.8, width=0.6)
        ax.errorbar(x, sub["share_read_positive"] * 100,
                    yerr=[((sub["share_read_positive"] - sub["ci_lo"]) * 100).clip(lower=0),
                          ((sub["ci_hi"] - sub["share_read_positive"]) * 100).clip(lower=0)],
                    fmt="none", ecolor="black", elinewidth=1.2, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{lv}\n(n={n})" for lv, n in zip(sub["level"], sub["n"])], fontsize=TICK_SIZE - 3)
        ax.set_ylim(0, 100)
        ax.set_title(var, fontsize=TITLE_SIZE - 4)
        if i % ncols == 0:
            ax.set_ylabel("Visual read positive (%)", fontsize=LABEL_SIZE - 4)
        ax.tick_params(axis="y", labelsize=TICK_SIZE - 2)
        ax.grid(axis="y", alpha=0.28)
    fig.suptitle("How common a positive visual read is in each group (whole dataset)", fontsize=TITLE_SIZE)
    finalize_figure(fig, rect=(0, 0, 1, 1.0))
    save_figure(fig, out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")


# ------------------------------
# 4. Missing data
# ------------------------------
def missing_by_site(df: pd.DataFrame) -> pd.DataFrame:
    source = {"Age": "age", "Sex": "gender", "Diagnosis": "dx", "Amyloid status": "amyloid_status",
              "APOE": "apoe", "Centiloid": "CL", "CDR-SB": "CDR_SB", "Visual read": "visual_read"}
    present = {k: v for k, v in source.items() if v in df.columns}
    out = {site: {k: g[v].isna().mean() for k, v in present.items()} for site, g in df.groupby("_site")}
    out["All"] = {k: df[v].isna().mean() for k, v in present.items()}
    return (pd.DataFrame(out) * 100).round(1)


# ------------------------------
# Main
# ------------------------------
def main():
    args = parse_args()
    demo_path = resolve_path(args.demo_csv, find_demo_csv)
    if demo_path is None:
        raise FileNotFoundError("No demographics csv found - pass --demo_csv.")
    df = prepare(pd.read_csv(demo_path), args.female_code)
    print(f"[load] {len(df)} scans from {demo_path}")

    train = test = None
    if args.run_dir:
        split_dir = Path(args.run_dir).resolve() / "splits"
        ids = {k: pd.read_csv(split_dir / f"{k}_subjects.csv")["ID"] for k in ("train", "test")
               if (split_dir / f"{k}_subjects.csv").exists()}
        if len(ids) == 2:
            train, test = df[df["ID"].isin(ids["train"])], df[df["ID"].isin(ids["test"])]
            print(f"[split] {len(train)} training, {len(test)} test scans")
        else:
            print(f"[skip] training vs test: no splits/train_subjects.csv + test_subjects.csv in {args.run_dir}")

    script_path = Path(__file__).resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else script_path.parents[2] / "results" / "dataset_description"
    out_dir.mkdir(parents=True, exist_ok=True)

    mats, assoc_rows = {}, []
    for label, data in [("Whole dataset", df)] + ([("Test set", test)] if test is not None else []):
        mat, rows = association_matrix(data)
        mats[f"{label} (n={len(data)})"] = mat
        assoc_rows.append(rows.assign(data=label))
    assoc = pd.concat(assoc_rows, ignore_index=True)
    assoc.round(3).to_csv(out_dir / "dataset_associations.csv", index=False)
    make_association_figure(mats, out_dir / "dataset_associations.png")

    sites = {s: g for s, g in df.groupby("_site")}
    t1 = table_one({**sites, "All": df})
    t1.to_csv(out_dir / "dataset_table1_by_site.csv")

    rates = read_positive_rates(df)
    rates.round(4).to_csv(out_dir / "dataset_read_positive_rates.csv", index=False)
    make_rates_figure(rates, out_dir / "dataset_read_positive_rates.png")

    missing = missing_by_site(df)
    missing.to_csv(out_dir / "dataset_missing_by_site.csv")

    tvt = None
    if train is not None:
        tvt = train_vs_test(train, test)
        tvt.to_csv(out_dir / "dataset_train_vs_test.csv")

    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.max_colwidth", 30):
        print("\n[associations] whole dataset, strongest first (0 = unrelated, 1 = fully determined)")
        whole = assoc[assoc["data"] == "Whole dataset"].sort_values("association", ascending=False)
        for _, r in whole.iterrows():
            extra = ""
            if test is not None:
                t = assoc[(assoc["data"] == "Test set") & (assoc["variable_1"] == r["variable_1"]) & (assoc["variable_2"] == r["variable_2"])]
                extra = f"   (test set {t['association'].iloc[0]:.2f})" if len(t) else ""
            print(f"  {r['variable_1']} – {r['variable_2']}: {r['association']:.2f}{extra}")
        print("\n[table 1] by site")
        print(t1.to_string())
        print("\n[visual read positive] % in each group")
        for var, sub in rates.groupby("variable", sort=False):
            print(f"  {var}: " + ", ".join(f"{r['level']} {r['share_read_positive']:.0%} (n={r['n']})" for _, r in sub.iterrows()))
        print("\n[missing] % missing per site")
        print(missing.to_string())
        if tvt is not None:
            print("\n[training vs test] (|SMD| < 0.1 = balanced)")
            print(tvt.to_string())
    print(f"\n[done] saved to {out_dir}")


if __name__ == "__main__":
    main()
