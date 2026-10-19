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

1. visual_read_discordant_findings.png - two-panel dot plot of the yes/no characteristics (Aβ+,
   APOE ε4, female, cognitively unimpaired, CDR ≥ 0.5): left = Model -, Visual read + vs
   Model +, Visual read +; right = Model +, Visual read - vs Model -, Visual read -. Proportions
   with Wilson 95% CI, rows sorted by the larger discordant-vs-reference gap. n/N printed per
   discordant group, since the denominator differs per row when a variable is missing.
   Continuous variables are NOT thresholded here - see 3 and 4.
3. visual_read_discordant_continuous.png - age, CDR-SB and Centiloid, every scan a dot with
   the median marked, same two comparisons (top: false negatives vs true positives, bottom:
   false positives vs true negatives).
4. visual_read_discordant_regional_pattern.png - regional SUVRs and Mesial/Meta, Frontal/Meta
   ratios across all four groups, dashed line = CTRz 2 equivalent (reference only).
2. visual_read_discordant_findings_table.csv - slim supplement table: four agreement/
   discordance groups + overall cohort, one row per yes/no variable, diagnosis and CDR
   grouped, skewed variables as median [IQR], n/N shown wherever data is missing.

Usage:
    python results_plotting/plot_discordant_findings.py /path/to/run_dir --dataset Gothenburg
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

    df_preds = pd.read_csv(preds_path)
    df_preds = df_preds[[c for c in ["ID_ind", "y", "pred", "prob"] if c in df_preds.columns]]
    df_demo = pd.read_csv(demo_path)
    df = pd.merge(df_preds, df_demo, left_on="ID_ind", right_on="ID", how="left")
    if "prob" in df.columns:
        df["prob"] = pd.to_numeric(df["prob"], errors="coerce")
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


# Left panel = false negatives (the group being looked at first), right = false positives.
PANEL_ORDER = ["False Negative", "False Positive"]


def make_dot_plot(axes, df: pd.DataFrame, indicators):
    """Two panels sharing the same rows: each plots one discordant group against the group that
    shares its visual read and that the model got right (two dots per row)."""
    props = group_proportions(df, indicators)
    p_idx = props.set_index(["label", "category"])

    def diff(label, cat):
        return abs(p_idx["p"].get((label, cat), np.nan) - p_idx["p"].get((label, SORT_REFERENCE[cat]), np.nan))

    # Sort by the larger of the two discordant-vs-reference gaps, biggest at the top.
    labels = [lab for _, lab in indicators]
    gap = {lab: np.nanmax([diff(lab, cat) for cat in DISCORDANT]) for lab in labels}
    labels = sorted(labels, key=lambda lab: gap[lab] if not np.isnan(gap[lab]) else -1)  # bottom -> top

    for panel, (ax, disc) in enumerate(zip(axes, PANEL_ORDER)):
        ref = SORT_REFERENCE[disc]
        for cat, off in ((ref, -0.17), (disc, 0.17)):
            sub = props[props["category"] == cat].set_index("label").loc[labels].reset_index()
            ys = np.arange(len(labels)) + off
            is_disc = cat == disc
            ax.hlines(ys, sub["lo"], sub["hi"], color=CONFUSION_CATEGORY_PALETTE[cat],
                      linewidth=2.2 if is_disc else 1.6, alpha=0.9 if is_disc else 0.65, zorder=2)
            ax.scatter(sub["p"], ys, s=100 if is_disc else 60, color=CONFUSION_CATEGORY_PALETTE[cat],
                       edgecolor="white", linewidth=0.8, zorder=3,
                       label=f"{confusion_display_label(cat)} (n={int((df['category'] == cat).sum())})")

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels if panel == 0 else [], fontsize=LABEL_SIZE - 2)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.6, len(labels) - 0.4)
        ax.set_xlabel("Proportion of group (whiskers: Wilson 95% CI)", fontsize=LABEL_SIZE - 3)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(axis="x", color="#DDDDDD", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.set_title(f"{confusion_display_label(disc)}\nvs {confusion_display_label(ref)}",
                     fontsize=TITLE_SIZE - 5, pad=10, color=CONFUSION_CATEGORY_PALETTE[disc])
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, frameon=False, fontsize=TICK_SIZE - 1)

        # n/N for the discordant group, in the right margin (denominator differs when data is missing).
        for i, lab in enumerate(labels):
            k, n = int(p_idx.loc[(lab, disc), "k"]), int(p_idx.loc[(lab, disc), "n"])
            ax.text(1.03, i, f"{k}/{n}", transform=ax.get_yaxis_transform(), ha="left", va="center",
                    fontsize=TICK_SIZE - 2, color=CONFUSION_CATEGORY_PALETTE[disc])


def make_findings_figure(df: pd.DataFrame, indicators, out_dir: Path, filename: str):
    fig, axes = plt.subplots(1, 2, figsize=(15, 2.6 + 0.85 * len(indicators)))
    make_dot_plot(axes, df, indicators)
    n_disc = int(df["category"].isin(DISCORDANT).sum())
    fig.suptitle(f"What the discordant groups look like ({n_disc}/{len(df)} scans discordant)",
                 fontsize=TITLE_SIZE - 3)
    fig.subplots_adjust(left=0.17, right=0.95, top=0.82, bottom=0.2, wspace=0.28)
    save_figure(fig, out_dir / filename, dpi=300)
    plt.close(fig)


# ------------------------------
# Continuous characteristics: every point shown, no cut-offs
# ------------------------------
CONTINUOUS_VARS = [("age", "Age (years)"), ("CDR_SB", "CDR-SB"), ("CL", "Centiloid")]


def make_continuous_figure(df: pd.DataFrame, out_dir: Path, filename: str):
    """Age / CDR-SB / Centiloid without dichotomising: one panel per variable, every scan a dot,
    black bar = median. Top row compares false negatives to true positives, bottom row false
    positives to true negatives (same pairs as the dot plot)."""
    variables = [(c, lab) for c, lab in CONTINUOUS_VARS if c in df.columns]
    if not variables:
        return
    fig, axes = plt.subplots(2, len(variables), figsize=(4.6 * len(variables), 8.2), squeeze=False)
    rng = np.random.default_rng(seed=0)
    for r, disc in enumerate(PANEL_ORDER):
        ref = SORT_REFERENCE[disc]
        for c, (col, lab) in enumerate(variables):
            ax = axes[r][c]
            vals = pd.to_numeric(df[col], errors="coerce")
            for x, cat in enumerate((ref, disc)):
                v = vals[(df["category"] == cat)].dropna()
                is_disc = cat == disc
                ax.scatter(x + rng.normal(0, 0.06, size=len(v)), v, s=70 if is_disc else 35,
                           color=CONFUSION_CATEGORY_PALETTE[cat], edgecolor="white", linewidth=0.7,
                           alpha=0.95 if is_disc else 0.6, zorder=3 if is_disc else 2)
                if len(v):
                    ax.hlines(v.median(), x - 0.3, x + 0.3, color="black", linewidth=1.8, zorder=4)
            ax.set_xlim(-0.6, 1.6)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(
                [f"{confusion_display_label(cat, multiline=True)}\n(n={int(vals[df['category'] == cat].notna().sum())})"
                 for cat in (ref, disc)], fontsize=TICK_SIZE - 2)
            if r == 0:
                ax.set_title(lab, fontsize=TITLE_SIZE - 4)
            ax.grid(axis="y", color="#DDDDDD", linewidth=0.8, zorder=0)
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
    fig.suptitle("Continuous characteristics by group (black bar = median)", fontsize=TITLE_SIZE - 3)
    finalize_figure(fig, rect=(0.0, 0.0, 1.0, 0.95))
    save_figure(fig, out_dir / filename, dpi=300)
    plt.close(fig)


# ------------------------------
# Regional SUVR: are the errors borderline, and do they share a regional pattern?
# ------------------------------
SUVR_REGIONS = ["Universal", "MetaTemporal", "MesialTemporal", "TemporoParietal", "Frontal"]
# (numerator, denominator) - Mesial/Meta high with low Meta = medial-temporal-only uptake;
# Frontal/Meta high = non-AD-like spread.
SUVR_RATIOS = [("MesialTemporal", "MetaTemporal"), ("Frontal", "MetaTemporal")]
CONFIDENT_DIST = 0.35  # |prob - 0.5| above this = "confident" error (outlined in black)


def make_regional_pattern_figure(df: pd.DataFrame, out_dir: Path, filename: str):
    """One strip panel per region SUVR plus the two ratios, all four groups side by side.
    Dashed line = SUVR equivalent of CTRz 2 - a group-level CenTauR convention, drawn only as
    a reference (list_discordant_cases.py explains why it is not a per-scan cutoff). Discordant
    points with |prob - 0.5| > CONFIDENT_DIST are outlined in black."""
    regions = [r for r in SUVR_REGIONS if r in df.columns]
    ratios = [(a, b) for a, b in SUVR_RATIOS if a in df.columns and b in df.columns]
    if not regions:
        return
    panels = [(r, pd.to_numeric(df[r], errors="coerce"), CENTAUR_FTP_CTRZ2_SUVR.get(r), f"{r} SUVR") for r in regions]
    for a, b in ratios:
        panels.append((f"{a}/{b}", pd.to_numeric(df[a], errors="coerce") / pd.to_numeric(df[b], errors="coerce"),
                       None, f"{a} / {b}"))

    n_cols = 3
    n_rows = int(np.ceil(len(panels) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 4.6 * n_rows), squeeze=False)
    rng = np.random.default_rng(seed=0)
    confident = (df["prob"] - 0.5).abs() > CONFIDENT_DIST if "prob" in df.columns else pd.Series(False, index=df.index)

    for ax, (name, vals, cutoff, ylabel) in zip(axes.flat, panels):
        for i, cat in enumerate(GROUP_ORDER):
            m = (df["category"] == cat) & vals.notna()
            discordant = cat in DISCORDANT
            xs = i + rng.normal(0, 0.07, size=int(m.sum()))
            edge = np.where(confident[m] & discordant, "black", "white")
            ax.scatter(xs, vals[m], s=60 if discordant else 22, color=CONFUSION_CATEGORY_PALETTE[cat],
                       edgecolor=edge, linewidth=1.2 if discordant else 0.4,
                       alpha=0.95 if discordant else 0.45, zorder=3 if discordant else 2)
            if m.any():
                ax.hlines(vals[m].median(), i - 0.3, i + 0.3, color="black", linewidth=1.6, zorder=4)
        if cutoff is not None:
            ax.axhline(cutoff, color="#555555", linestyle="--", linewidth=1.1, zorder=1)
        ax.set_xticks(range(len(GROUP_ORDER)))
        ax.set_xticklabels([confusion_display_label(c, multiline=True) for c in GROUP_ORDER], fontsize=TICK_SIZE - 3)
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE - 3)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)

    fig.suptitle("Regional tau pattern by agreement group (black bar = median; dashed = CTRz 2 equivalent; "
                 "black outline = confident error)", fontsize=TITLE_SIZE - 5)
    finalize_figure(fig, rect=(0.0, 0.0, 1.0, 0.95))
    save_figure(fig, out_dir / filename, dpi=300)
    plt.close(fig)


def make_meta_amyloid_figure(df: pd.DataFrame, out_dir: Path, filename: str):
    """Headline picture: who does the SUVR side with? One column per Model/Visual read group, every
    scan a dot at its Meta-temporal SUVR, filled = amyloid positive, hollow = amyloid negative,
    grey = amyloid unknown. Dashed line = CTRz 2 SUVR equivalent (reference only). Counts above each
    column: how many sit above the line, and how many are amyloid positive."""
    if "MetaTemporal" not in df.columns:
        return
    suvr = pd.to_numeric(df["MetaTemporal"], errors="coerce")
    amyloid = pd.to_numeric(df["amyloid_status"], errors="coerce") if "amyloid_status" in df.columns \
        else pd.Series(np.nan, index=df.index)
    cutoff = CENTAUR_FTP_CTRZ2_SUVR["MetaTemporal"]
    rng = np.random.default_rng(seed=0)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    y_top = np.nanmax(suvr.to_numpy()) if suvr.notna().any() else cutoff
    for x, cat in enumerate(GROUP_ORDER):
        in_cat = (df["category"] == cat) & suvr.notna()
        color = CONFUSION_CATEGORY_PALETTE[cat]
        for status, face, edge in ((1, color, "white"), (0, "white", color), (np.nan, "#BBBBBB", "white")):
            m = in_cat & (amyloid.isna() if np.isnan(status) else (amyloid == status))
            if not m.any():
                continue
            ax.scatter(x + rng.uniform(-0.28, 0.28, size=int(m.sum())), suvr[m], s=34,
                       facecolor=face, edgecolor=edge, linewidth=1.1, alpha=0.85, zorder=3)
        if in_cat.any():
            ax.hlines(suvr[in_cat].median(), x - 0.34, x + 0.34, color="black", linewidth=2, zorder=4)
        n = int(in_cat.sum())
        above = int((suvr[in_cat] >= cutoff).sum())
        amy = amyloid[df["category"] == cat].dropna()
        ax.text(x, 1.02, f"above line: {above}/{n}\nAβ+: {int(amy.sum())}/{len(amy)}",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=TICK_SIZE - 1)

    ax.axhline(cutoff, color="#555555", linestyle="--", linewidth=1.2, zorder=1)
    ax.text(len(GROUP_ORDER) - 0.5, cutoff, f" CTRz 2 ≈ {cutoff:.2f}", va="bottom", ha="right",
            fontsize=TICK_SIZE - 2, color="#555555")
    ax.set_xticks(range(len(GROUP_ORDER)))
    ax.set_xticklabels([f"{confusion_display_label(c, multiline=True)}\n(n={int((df['category'] == c).sum())})"
                        for c in GROUP_ORDER], fontsize=TICK_SIZE)
    ax.set_xlim(-0.6, len(GROUP_ORDER) - 0.4)
    ax.set_ylabel("Meta-temporal SUVR", fontsize=LABEL_SIZE - 2)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=7, markerfacecolor="#666666",
               markeredgecolor="white", label="Amyloid positive"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=7, markerfacecolor="white",
               markeredgecolor="#666666", markeredgewidth=1.2, label="Amyloid negative"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=7, markerfacecolor="#BBBBBB",
               markeredgecolor="white", label="Amyloid unknown"),
        Line2D([0], [0], color="black", linewidth=2, label="Median"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False,
              fontsize=TICK_SIZE - 1)
    fig.suptitle("Where each group sits on Meta-temporal SUVR, and who has amyloid", fontsize=TITLE_SIZE - 3)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.8, bottom=0.25)
    save_figure(fig, out_dir / filename, dpi=300)
    plt.close(fig)


# ------------------------------
# Agreement between the three "judges": model, visual read, Meta-temporal SUVR (CTRz 2)
# ------------------------------
KAPPA_PAIRS = [("model", "human", "Model vs visual read"),
               ("human", "suvr", "Visual read vs SUVR"),
               ("model", "suvr", "Model vs SUVR")]
# CU vs impaired follows Karagianni 2026 (agreement differs a lot by clinical stage, partly through
# prevalence). Amyloid positive only matches their Abeta+-restricted analysis and is the population a
# tau-PET trial screen would actually be applied to.
KAPPA_STRATA = [("All", "#333333"), ("Cog. unimpaired", "#4C78A8"), ("Cog. impaired", "#E4572E"),
                ("Amyloid positive", "#7B4FA0")]
# Inter-reader kappa for the FDA/EMA flortaucipir visual read vs fully manual external reads
# (Moscoso et al. 2025), drawn as a reference band.
HUMAN_HUMAN_KAPPA = (0.78, 0.94)
N_BOOT = 2000


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0:
        return np.nan
    po = np.mean(a == b)
    pe = np.mean(a) * np.mean(b) + np.mean(1 - a) * np.mean(1 - b)
    return np.nan if pe == 1 else (po - pe) / (1 - pe)


def kappa_votes(df: pd.DataFrame) -> pd.DataFrame:
    """One row per scan with a Meta-temporal SUVR: the model's, the visual read's and the SUVR's
    positive/negative call. All pairs use these same scans, so the kappas are comparable."""
    suvr = pd.to_numeric(df.get("MetaTemporal"), errors="coerce")
    return pd.DataFrame({
        "model": df["pred"].astype(int),
        "human": df["y"].astype(int),
        "suvr": (suvr >= CENTAUR_FTP_CTRZ2_SUVR["MetaTemporal"]).astype(int),
    })[suvr.notna()]


def kappa_strata(df: pd.DataFrame, votes: pd.DataFrame) -> dict:
    dx = combine_dx_groups(df["dx"]).where(df["dx"].notna()) if "dx" in df.columns else pd.Series(np.nan, index=df.index)
    amyloid = pd.to_numeric(df["amyloid_status"], errors="coerce") if "amyloid_status" in df.columns \
        else pd.Series(np.nan, index=df.index)
    return {"All": votes,
            "Cog. unimpaired": votes[(dx == "CU").reindex(votes.index, fill_value=False)],
            "Cog. impaired": votes[(dx.notna() & (dx != "CU")).reindex(votes.index, fill_value=False)],
            "Amyloid positive": votes[(amyloid == 1).reindex(votes.index, fill_value=False)]}


def compute_kappa_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Cohen's kappa with bootstrap 95% CI, overall and per KAPPA_STRATA group."""
    strata = kappa_strata(df, kappa_votes(df))

    rng = np.random.default_rng(seed=0)
    rows = []
    for stratum, v in strata.items():
        idx = rng.integers(0, len(v), size=(N_BOOT, len(v))) if len(v) else None
        for a, b, label in KAPPA_PAIRS:
            x, y = v[a].to_numpy(), v[b].to_numpy()
            k = cohen_kappa(x, y)
            boot = [cohen_kappa(x[i], y[i]) for i in idx] if idx is not None else []
            lo, hi = (np.nanpercentile(boot, [2.5, 97.5]) if len(boot) else (np.nan, np.nan))
            rows.append({"pair": label, "group": stratum, "n": len(v),
                         "agree_pct": round(100 * np.mean(x == y), 1) if len(v) else np.nan,
                         "kappa": round(k, 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3)})
    return pd.DataFrame(rows)


def make_kappa_figure(kappa: pd.DataFrame, out_dir: Path, filename: str):
    """One row per pair, one dot per group (all / cog. unimpaired / cog. impaired), whiskers =
    bootstrap 95% CI. Shaded band = human-vs-human agreement range for the same visual read."""
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.axvspan(*HUMAN_HUMAN_KAPPA, color="#DDDDDD", alpha=0.6, zorder=0)
    ax.text(np.mean(HUMAN_HUMAN_KAPPA), len(KAPPA_PAIRS) - 0.45, "two human readers\n(Moscoso 2025)",
            ha="center", va="bottom", fontsize=TICK_SIZE - 3, color="#555555")

    labels = [label for _, _, label in KAPPA_PAIRS][::-1]  # first pair at the top
    offsets = np.linspace(0.27, -0.27, len(KAPPA_STRATA))
    for (stratum, color), off in zip(KAPPA_STRATA, offsets):
        sub = kappa[kappa["group"] == stratum].set_index("pair").reindex(labels)
        ys = np.arange(len(labels)) + off
        ax.hlines(ys, sub["ci_low"], sub["ci_high"], color=color, linewidth=1.8, zorder=2)
        n = int(sub["n"].dropna().iloc[0]) if sub["n"].notna().any() else 0
        ax.scatter(sub["kappa"], ys, s=70, color=color, edgecolor="white", linewidth=0.8, zorder=3,
                   label=f"{stratum} (n={n})")
        # Value printed just right of each whisker, so it never sits on top of another dot.
        for y_, k, hi in zip(ys, sub["kappa"], sub["ci_high"]):
            if pd.notna(k):
                ax.text((hi if pd.notna(hi) else k) + 0.015, y_, f"{k:.2f}", ha="left", va="center",
                        fontsize=TICK_SIZE - 3, color=color)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=LABEL_SIZE - 2)
    ax.set_ylim(-0.6, len(labels) - 0.1)
    # Extend left of 0 when a CI dips below it (small strata), so no whisker is clipped.
    ax.set_xlim(min(-0.05, np.nanmin(kappa[["kappa", "ci_low"]].to_numpy()) - 0.05), 1.1)
    x_left = ax.get_xlim()[0]
    ax.set_xticks(np.round(np.arange(np.ceil(x_left * 5) / 5, 1.001, 0.2), 1) + 0.0)  # 0.2 grid, inside the axis
    ax.set_xlim(x_left, 1.1)
    ax.axvline(0, color="#999999", linewidth=0.8, zorder=1)
    ax.set_xlabel("Cohen's kappa (whiskers: bootstrap 95% CI)", fontsize=LABEL_SIZE - 3)
    ax.grid(axis="x", color="#EEEEEE", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2, frameon=False, fontsize=TICK_SIZE - 2)
    ax.set_title(f"How often the model, the visual read and the SUVR agree\n"
                 f"(SUVR positive = Meta-temporal ≥ {CENTAUR_FTP_CTRZ2_SUVR['MetaTemporal']:.2f})",
                 fontsize=TITLE_SIZE - 5, pad=12)
    fig.subplots_adjust(left=0.24, right=0.97, top=0.84, bottom=0.28)
    save_figure(fig, out_dir / filename, dpi=300)
    plt.close(fig)


def make_kappa_grid_figure(df: pd.DataFrame, kappa: pd.DataFrame, out_dir: Path, filename: str):
    """Three 2x2 agreement tables (everyone), the format Karagianni 2026 used: counts and % of all
    scans in each cell, diagonal = agreement, kappa with bootstrap 95% CI in each title."""
    votes = kappa_votes(df)
    names = {"model": "Model", "human": "Visual read", "suvr": "SUVR"}
    fig, axes = plt.subplots(1, len(KAPPA_PAIRS), figsize=(5.0 * len(KAPPA_PAIRS), 4.8))
    for ax, (a, b, label) in zip(axes, KAPPA_PAIRS):
        counts = np.array([[int(((votes[a] == i) & (votes[b] == j)).sum()) for j in (0, 1)] for i in (0, 1)])
        frac = counts / max(counts.sum(), 1)
        ax.imshow(frac, cmap="Greys", vmin=0, vmax=max(frac.max(), 1e-9) * 1.6)
        for i in (0, 1):
            for j in (0, 1):
                agree = i == j
                ax.text(j, i, f"{counts[i, j]}\n({100 * frac[i, j]:.0f}%)", ha="center", va="center",
                        fontsize=LABEL_SIZE - 1, fontweight="bold" if agree else "normal",
                        color="#B2182B" if not agree else "black")
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"{names[b]} −", f"{names[b]} +"], fontsize=TICK_SIZE)
        ax.set_yticks([0, 1])
        ax.set_yticklabels([f"{names[a]} −", f"{names[a]} +"], fontsize=TICK_SIZE)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        row = kappa[(kappa["pair"] == label) & (kappa["group"] == "All")].iloc[0]
        ax.set_title(f"{label}\nκ = {row['kappa']:.2f} [{row['ci_low']:.2f}, {row['ci_high']:.2f}], "
                     f"agree {row['agree_pct']:.0f}%", fontsize=TITLE_SIZE - 5)
    fig.suptitle(f"Agreement between model, visual read and SUVR (n={len(votes)}; red = disagreement; "
                 f"SUVR + = Meta-temporal ≥ {CENTAUR_FTP_CTRZ2_SUVR['MetaTemporal']:.2f})",
                 fontsize=TITLE_SIZE - 5)
    finalize_figure(fig, rect=(0.0, 0.0, 1.0, 0.9))
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
        if col in ("_meta", "_cu", "_cdr"):
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
    # The dot plot is for yes/no characteristics; Meta SUVR >= CTRz 2 is a thresholded continuous
    # variable (kept in the table), so it is shown in the regional-pattern figure instead.
    dot_indicators = [i for i in indicators if i[0] != "_meta"]
    make_findings_figure(df, dot_indicators, out_dir, "visual_read_discordant_findings.png")
    make_continuous_figure(df, out_dir, "visual_read_discordant_continuous.png")
    make_findings_table(df, indicators, out_dir, "visual_read_discordant_findings_table.csv", args)
    make_regional_pattern_figure(df, out_dir, "visual_read_discordant_regional_pattern.png")
    make_meta_amyloid_figure(df, out_dir, "visual_read_discordant_meta_amyloid.png")
    if "MetaTemporal" in df.columns:
        kappa = compute_kappa_table(df)
        kappa.to_csv(out_dir / "visual_read_kappa_table.csv", index=False)
        make_kappa_figure(kappa, out_dir, "visual_read_kappa.png")
        make_kappa_grid_figure(df, kappa, out_dir, "visual_read_kappa_grids.png")

    counts = df["category"].value_counts()
    print("[done] " + ", ".join(f"{int(counts.get(c, 0))} x [{confusion_display_label(c)}]" for c in GROUP_ORDER)
          + f"; saved to {out_dir}")


if __name__ == "__main__":
    main()
