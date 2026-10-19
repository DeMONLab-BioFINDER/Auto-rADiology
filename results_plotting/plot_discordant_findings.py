#!/usr/bin/env python3
"""Headline figure + slim table for the discordant-case (Model vs Visual read) analysis.

Replaces eyeballing the long demographics table: it shows the general picture (how often
Model and Visual read agree) and the key finding (what the discordant groups look like)
by comparing each discordant group against the group that shares its visual read and the
model got right - not against the overall cohort, which mixes visual-read positives and
negatives and so misrepresents both discordant groups:

    Model +, Visual read -   vs  Model -, Visual read -  (same visual read, model agreed)
    Model -, Visual read +   vs  Model +, Visual read +  (same visual read, model agreed)

Reads the run's predictions csv (every scan, with y / pred) and the demographics csv, the
same resolution as plot_discordant_demographics.py, so it does NOT need
list_discordant_cases.py to have been run first. Descriptive only - no p-values (the
discordant groups can be tens of subjects); dot-plot whiskers are Wilson 95% intervals so
small groups visibly look uncertain.

Produces, next to the run's other figures (<run_dir>/figures/, or
<run_dir>/validation/figures/ for a zero-shot external validation run):

1. visual_read_discordant_findings.png - left: 2x2 Model x Visual read agreement grid
   (counts, % agreement). Right: dot plot, one row per characteristic (proportion in each of
   the four groups, whiskers = Wilson 95% CI), rows sorted by how far Model +, Visual read -
   sits from Model -, Visual read -. n/N printed per discordant group, since the
   denominator differs per row when a variable is missing.
2. visual_read_discordant_findings_table.csv - slim supplement table: four agreement/
   discordance groups + overall cohort, one row per yes/no variable, diagnosis and CDR
   grouped, skewed variables as median [IQR], n/N shown wherever data is missing.

Usage:
    python results_plotting/plot_discordant_findings.py /path/to/run_dir --dataset Gothenburg
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from list_discordant_cases import find_dataset_demo_csv, find_zeroshot_results_csv
from plot_utils import (
    CENTAUR_FTP_CTRZ2_SUVR,
    CONFUSION_CATEGORY_LABELS,
    CONFUSION_CATEGORY_PALETTE,
    LABEL_SIZE,
    TICK_SIZE,
    TITLE_SIZE,
    combine_dx_groups,
    confusion_display_label,
    find_demo_csv,
    finalize_figure,
    resolve_existing_path,
    resolve_path,
    save_figure,
)

# Left-to-right order: each discordant group sits next to the group it is compared against.
GROUP_ORDER = ["True Negative", "False Positive", "False Negative", "True Positive"]
DISCORDANT = ["False Positive", "False Negative"]
SORT_REFERENCE = {"False Positive": "True Negative", "False Negative": "True Positive"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="Path to a results run directory trained with --targets visual_read.")
    parser.add_argument("--dataset", default="Gothenburg", help="Dataset name used in the result filenames.")
    parser.add_argument("--demo_csv", default=None, help="Path to the demographics csv (default: data/demo_<dataset>.csv, then data/demo.csv).")
    parser.add_argument("--age_cutoff", type=float, default=75.0, help="Age threshold for the 'older' row of the dot plot.")
    parser.add_argument(
        "--female_code", type=float, default=2,
        help="Value of the `gender` column that means female (2 in this dataset).",
    )
    return parser.parse_args()


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_scans(args, run_dir: Path):
    """One row per scan: confusion category + demographics. Same preds/out_dir resolution as
    plot_discordant_demographics.py (a zero-shot external validation result takes priority)."""
    zeroshot_path = find_zeroshot_results_csv(run_dir, args.dataset)
    if zeroshot_path is not None:
        preds_path, out_dir = zeroshot_path, run_dir / "validation" / "figures"
    else:
        preds_path = resolve_existing_path(
            [
                run_dir / "validation" / args.dataset / f"Test_{args.dataset}_results.csv",
                run_dir / "evaluation" / args.dataset / f"Eval_{args.dataset}_results.csv",
            ],
            "predictions csv",
        )
        out_dir = run_dir / "figures"

    demo_path = resolve_path(args.demo_csv, lambda: find_dataset_demo_csv(args.dataset) or find_demo_csv())
    if demo_path is None:
        raise FileNotFoundError("Could not find a demographics csv - pass --demo_csv explicitly.")

    df_preds = pd.read_csv(preds_path)[["ID_ind", "y", "pred"]]
    df_demo = pd.read_csv(demo_path)
    df = pd.merge(df_preds, df_demo, left_on="ID_ind", right_on="ID", how="left")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["pred"] = pd.to_numeric(df["pred"], errors="coerce")
    df = df.dropna(subset=["y", "pred"]).reset_index(drop=True)
    df["category"] = [CONFUSION_CATEGORY_LABELS[(int(t), int(p))] for t, p in zip(df["y"], df["pred"])]
    return df, out_dir


def add_indicator_columns(df: pd.DataFrame, args) -> list[tuple[str, str]]:
    """Adds boolean (NaN where missing) indicator columns; returns [(column, display label)]."""
    indicators = []

    def add(col, label, values):
        df[col] = values
        indicators.append((col, label))

    def binary(col, positive):
        raw = pd.to_numeric(df[col], errors="coerce")
        return (raw == positive).where(raw.notna())

    if "amyloid_status" in df.columns:
        add("_amyloid", "Amyloid positive", binary("amyloid_status", 1))
    if "apoe" in df.columns:
        add("_apoe", "APOE ε4 carrier", binary("apoe", 1))
    if "dx" in df.columns:
        dx = combine_dx_groups(df["dx"])
        add("_cu", "Cog. unimpaired", (dx == "CU").where(df["dx"].notna()))
    if "age" in df.columns:
        age = pd.to_numeric(df["age"], errors="coerce")
        add("_older", f"Age ≥ {args.age_cutoff:g}", (age >= args.age_cutoff).where(age.notna()))
    if "CDR" in df.columns:
        cdr = pd.to_numeric(df["CDR"], errors="coerce")
        add("_cdr", "CDR ≥ 0.5", (cdr >= 0.5).where(cdr.notna()))
    if "gender" in df.columns:
        sex = pd.to_numeric(df["gender"], errors="coerce")
        add("_sex", "Female", (sex == args.female_code).where(sex.notna()))
    if "MetaTemporal" in df.columns:
        suvr = pd.to_numeric(df["MetaTemporal"], errors="coerce")
        cutoff = CENTAUR_FTP_CTRZ2_SUVR["MetaTemporal"]
        add("_meta", "Meta SUVR ≥ CTRz 2", (suvr >= cutoff).where(suvr.notna()))
    return indicators


def make_agreement_grid(ax, df: pd.DataFrame):
    counts = df["category"].value_counts()
    n_total = len(df)
    # rows: Visual read +/-, cols: Model -/+ (so the diagonal is agreement)
    layout = [
        [("False Negative", 0, 1), ("True Positive", 1, 1)],
        [("True Negative", 0, 0), ("False Positive", 1, 0)],
    ]
    for r, row in enumerate(layout):
        for c, (cat, _, _) in enumerate(row):
            n = int(counts.get(cat, 0))
            agree = cat not in DISCORDANT
            ax.add_patch(plt.Rectangle((c, 1 - r), 1, 1, facecolor=CONFUSION_CATEGORY_PALETTE[cat],
                                       alpha=0.85 if not agree else 0.35, edgecolor="white", linewidth=3))
            ax.text(c + 0.5, 1 - r + 0.56, f"{n}", ha="center", va="center", fontsize=26, fontweight="bold")
            ax.text(c + 0.5, 1 - r + 0.25, f"{n / n_total:.1%}", ha="center", va="center", fontsize=TICK_SIZE + 1)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Model −", "Model +"], fontsize=LABEL_SIZE - 1)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Visual\nread −", "Visual\nread +"], fontsize=LABEL_SIZE - 1)
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    n_disc = int(counts.reindex(DISCORDANT).fillna(0).sum())
    ax.set_title(
        f"{n_total - n_disc}/{n_total} scans agree ({(n_total - n_disc) / n_total:.0%})\n"
        f"{n_disc} discordant", fontsize=TITLE_SIZE - 3, pad=16,
    )


def group_proportions(df: pd.DataFrame, indicators) -> pd.DataFrame:
    rows = []
    for col, label in indicators:
        for cat in GROUP_ORDER:
            vals = df.loc[df["category"] == cat, col].dropna()
            k, n = int(vals.sum()), len(vals)
            lo, hi = wilson_interval(k, n)
            rows.append({"label": label, "category": cat, "k": k, "n": n,
                         "p": k / n if n else np.nan, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def make_dot_plot(ax, df: pd.DataFrame, indicators):
    props = group_proportions(df, indicators)

    def diff(label, cat):
        p = props.set_index(["label", "category"])["p"]
        return abs(p.get((label, cat), np.nan) - p.get((label, SORT_REFERENCE[cat]), np.nan))

    # Sort by the larger of the two discordant-vs-reference gaps, biggest at the top.
    labels = [lab for _, lab in indicators]
    gap = {lab: np.nanmax([diff(lab, cat) for cat in DISCORDANT]) for lab in labels}
    labels = sorted(labels, key=lambda lab: gap[lab] if not np.isnan(gap[lab]) else -1)  # bottom -> top

    offsets = np.linspace(-0.27, 0.27, len(GROUP_ORDER))
    for cat, off in zip(GROUP_ORDER, offsets):
        sub = props.set_index("label").loc[labels].reset_index()
        sub = sub[sub["category"] == cat]
        ys = np.arange(len(labels)) + off
        discordant = cat in DISCORDANT
        ax.hlines(ys, sub["lo"], sub["hi"], color=CONFUSION_CATEGORY_PALETTE[cat], linewidth=2 if discordant else 1.4,
                  alpha=0.9 if discordant else 0.6, zorder=2)
        ax.scatter(sub["p"], ys, s=95 if discordant else 55, color=CONFUSION_CATEGORY_PALETTE[cat],
                   edgecolor="white", linewidth=0.8, zorder=3,
                   label=f"{confusion_display_label(cat)} (n={int((df['category'] == cat).sum())})")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=LABEL_SIZE - 2)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("Proportion of group (whiskers: Wilson 95% CI)", fontsize=LABEL_SIZE - 2)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_title("What the discordant groups look like", fontsize=TITLE_SIZE - 3, pad=16)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.12), ncol=2, frameon=False, fontsize=TICK_SIZE - 1)

    # n/N for the discordant groups, in the right margin (denominator differs when data is missing).
    p_idx = props.set_index(["label", "category"])
    x_text = [1.04, 1.16]
    for cat, x in zip(DISCORDANT, x_text):
        ax.text(x, len(labels) - 0.35, confusion_display_label(cat, multiline=True).replace("Visual read", "VR"),
                transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=TICK_SIZE - 3,
                color=CONFUSION_CATEGORY_PALETTE[cat], fontweight="bold")
        for i, lab in enumerate(labels):
            k, n = int(p_idx.loc[(lab, cat), "k"]), int(p_idx.loc[(lab, cat), "n"])
            ax.text(x, i, f"{k}/{n}", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=TICK_SIZE - 2)


def make_findings_figure(df: pd.DataFrame, indicators, out_dir: Path, filename: str):
    fig = plt.figure(figsize=(15, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.8, 2.0], wspace=0.55)
    make_agreement_grid(fig.add_subplot(gs[0]), df)
    make_dot_plot(fig.add_subplot(gs[1]), df, indicators)
    fig.subplots_adjust(left=0.07, right=0.86, top=0.82, bottom=0.2)
    save_figure(fig, out_dir / filename, dpi=300)
    plt.close(fig)


# ------------------------------
# Slim supplement table
# ------------------------------
def _mean_sd(s: pd.Series) -> str:
    v = pd.to_numeric(s, errors="coerce").dropna()
    return "-" if v.empty else (f"{v.iloc[0]:.1f} (n=1)" if len(v) == 1 else f"{v.mean():.1f} ± {v.std():.1f}")


def _median_iqr(s: pd.Series, d: int = 1) -> str:
    v = pd.to_numeric(s, errors="coerce").dropna()
    return "-" if v.empty else f"{v.median():.{d}f} [{v.quantile(0.25):.{d}f}, {v.quantile(0.75):.{d}f}]"


def _count_cell(mask: pd.Series, available: pd.Series) -> str:
    n = int(available.sum())
    if n == 0:
        return "-"
    k = int((mask & available).sum())
    pct = k / n * 100
    pct_str = "<1%" if 0 < pct < 1 else f"{pct:.0f}%"
    return f"{k}/{n} ({pct_str})"


def _cdr_group(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce")
    out = pd.Series(pd.NA, index=s.index, dtype="string")
    out[v == 0] = "0"
    out[v == 0.5] = "0.5"
    out[v >= 1] = "≥ 1"
    return out


def _dx_group(s: pd.Series) -> pd.Series:
    mapped = combine_dx_groups(s).replace({"AlzCS dem": "Dementia (AD)"})
    return mapped.where(s.notna())


def make_findings_table(df: pd.DataFrame, indicators, out_dir: Path, filename: str, args):
    groups = {cat: df[df["category"] == cat] for cat in GROUP_ORDER}
    groups["Overall cohort"] = df
    col_labels = ["Characteristic"] + [
        f"{confusion_display_label(name) if name in GROUP_ORDER else name} (n={len(g)})" for name, g in groups.items()
    ]
    rows = []

    def add_row(label, fn):
        rows.append([label] + [fn(g) for g in groups.values()])

    if "age" in df.columns:
        add_row("Age, mean ± SD", lambda g: _mean_sd(g["age"]))
    for col, label in indicators:
        if col in ("_older", "_meta", "_cu", "_cdr"):
            continue  # grouped rows below / imaging block cover these
        add_row(f"{label}, n/N (%)", lambda g, c=col: _count_cell(g[c].fillna(False).astype(bool), g[c].notna()))

    if "dx" in df.columns:
        rows.append(["Diagnosis, n/N (%)"] + [""] * len(groups))
        for val in ["CU", "MCI", "Dementia (AD)", "Other"]:
            add_row(f"    {val}", lambda g, v=val: _count_cell(_dx_group(g["dx"]) == v, g["dx"].notna()))
    if "CDR" in df.columns:
        rows.append(["CDR, n/N (%)"] + [""] * len(groups))
        for val in ["0", "0.5", "≥ 1"]:
            add_row(f"    {val}", lambda g, v=val: _count_cell(_cdr_group(g["CDR"]) == v, g["CDR"].notna()))
    if "CDR_SB" in df.columns:
        add_row("CDR-SB, median [IQR]", lambda g: _median_iqr(g["CDR_SB"]))
    if "site" in df.columns:
        rows.append(["Site, n/N (%)"] + [""] * len(groups))
        for val in df["site"].value_counts().index:
            add_row(f"    {val}", lambda g, v=val: _count_cell(g["site"] == v, g["site"].notna()))

    imaging = [(c, l) for c, l in [("CL", "Centiloid"), ("MetaTemporal", "Meta-temporal SUVR"),
                                   ("Universal", "Universal SUVR")] if c in df.columns]
    if imaging:
        rows.append(["Imaging, median [IQR]"] + [""] * len(groups))
        for col, label in imaging:
            add_row(f"    {label}", lambda g, c=col: _median_iqr(g[c], 1 if c == "CL" else 2))
        if "_meta" in df.columns:
            add_row(
                f"    Meta-temporal SUVR ≥ CTRz 2 ({CENTAUR_FTP_CTRZ2_SUVR['MetaTemporal']:.2f}), n/N (%)",
                lambda g: _count_cell(g["_meta"].fillna(False).astype(bool), g["_meta"].notna()),
            )

    pd.DataFrame(rows, columns=col_labels).to_csv(out_dir / filename, index=False)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    df, out_dir = load_scans(args, run_dir)
    if df.empty:
        print("[done] No scans with predictions; nothing to plot.")
        return

    indicators = add_indicator_columns(df, args)
    out_dir.mkdir(parents=True, exist_ok=True)
    make_findings_figure(df, indicators, out_dir, "visual_read_discordant_findings.png")
    make_findings_table(df, indicators, out_dir, "visual_read_discordant_findings_table.csv", args)

    counts = df["category"].value_counts()
    print("[done] " + ", ".join(f"{int(counts.get(c, 0))} x [{confusion_display_label(c)}]" for c in GROUP_ORDER)
          + f"; saved to {out_dir}")


if __name__ == "__main__":
    main()
