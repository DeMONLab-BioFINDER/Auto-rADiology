#!/usr/bin/env python3
"""How confident is the visual-read model, can its confidence be trusted, and does its
uncertainty follow the biology?

Confidence levels, fixed before looking at any results, on the model's raw probability of
"positive". Together they cover every scan:
    Confident             - probability <= 0.1 or >= 0.9, split into
                              Confidently right (agrees with the visual read) and
                              Confidently wrong (disagrees with it)
    Moderately confident  - probability 0.1-0.3 or 0.7-0.9
    Uncertain             - probability 0.3-0.7, right or wrong
Fallback, also fixed in advance: if "Confidently wrong" has fewer than 5 scans the confident
cut-offs widen to 0.2 / 0.8; if "Uncertain" has fewer than 5 the band widens to 0.2-0.8. The
script says when it does either.

Raw probabilities, not recalibrated ones: the question is about the model's own sense of
confidence. The calibration check says how far to trust that confidence; it does not change
the levels.

Step 1 - confidence and calibration (predictions only):
    visual_read_confidence_histogram.png  - probabilities, split by visual read, bars coloured by level
    visual_read_calibration.png           - predicted probability vs how often the visual read was
                                            positive (on the diagonal = probabilities mean what they say)
    visual_read_calibration_bins.csv
    visual_read_confidence_triage.csv     - per level: share of scans, agreement with the visual read
                                            (what an "auto-decide if confident, refer the rest" rule gives)

Step 2 - does the uncertainty follow the biology? (needs the demographics csv, MetaTemporal):
    Every scan, no groups: confidence = |probability - 0.5| (0 = coin flip, 0.5 = certain) against
    how borderline its Meta-temporal SUVR is = |SUVR - CTRz 2 equivalent|. Spearman correlation,
    for all scans and within each visual read. Positive = the model is least sure where the SUVR
    is closest to the cut-off, i.e. its uncertainty tracks real biological ambiguity.
    visual_read_probability_vs_suvr.png   - every scan: Meta-temporal SUVR vs model probability
    visual_read_confidence_by_level.csv   - per level: SUVR, diagnosis, amyloid, site (descriptive)

    Because CTRz 2 is itself an uncertain choice, the script also lets the data say where the cut-offs
    are: the Meta-temporal SUVR at which the VISUAL READ becomes more likely positive than not, and the
    SUVR at which the MODEL does (each a plain logistic fit of the yes/no call on SUVR, crossing 0.5,
    with a bootstrap 95% CI, and the paired difference model - readers). A lower model cut-off is the
    model's lean toward positive, in SUVR units. The confidence-vs-SUVR correlation is then repeated
    with distance from the model's own cut-off, which needs no externally chosen cut-off at all.
    visual_read_suvr_cutoffs.png          - the two fitted curves, their 0.5 crossings, and CTRz 2
    visual_read_suvr_cutoffs.csv

Step 3 - the confidently wrong scans, one by one:
    visual_read_confidently_wrong_cases.csv - each case: probability, every regional SUVR (and which
                                              are at or above their own CTRz 2 equivalent), Centiloid,
                                              diagnosis, amyloid, APOE, age, sex, CDR-SB, site - plus
                                              median rows for the confidently right scans to read them against

Always:
    visual_read_confidence_groups.csv     - one row per scan: ID, visual read, probability, model call, level
    visual_read_confidence_summary.csv    - cut-offs used, level sizes, Brier score, calibration error,
                                            confidence-vs-SUVR correlations

Usage:
    python results_plotting/analyze_model_confidence_visual_read.py /path/to/run_dir --dataset Gothenburg \
        --demo_csv /path/to/demo.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from list_discordant_cases import find_dataset_demo_csv
from plot_utils import (
    CENTAUR_FTP_CTRZ2_SUVR,
    CUSTOM_PALETTE,
    LABEL_SIZE,
    TICK_SIZE,
    TITLE_SIZE,
    combine_dx_groups,
    confusion_display_label,
    find_demo_csv,
    finalize_figure,
    resolve_path,
    resolve_predictions_csv,
    save_figure,
    wilson_interval,
)

PRIMARY = {"confident": (0.1, 0.9), "uncertain": (0.3, 0.7)}
FALLBACK = {"confident": (0.2, 0.8), "uncertain": (0.2, 0.8)}
MIN_GROUP_N = 5
SUVR_REGION = "MetaTemporal"
REGIONS = ["Universal", "MetaTemporal", "MesialTemporal", "TemporoParietal", "Frontal"]
N_BOOT = 2000

GROUP_ORDER = ["Confidently right", "Moderately confident", "Uncertain", "Confidently wrong"]
GROUP_COLORS = {"Confidently right": CUSTOM_PALETTE[0], "Moderately confident": "#BBBBBB",
                "Uncertain": CUSTOM_PALETTE[4], "Confidently wrong": CUSTOM_PALETTE[1]}
# What a viewer sees: the visual read is not ground truth, so figures say agrees/disagrees, not right/wrong.
GROUP_DISPLAY_LABELS = {"Confidently right": "Confident, agrees with visual read",
                        "Moderately confident": "Moderately confident", "Uncertain": "Uncertain",
                        "Confidently wrong": "Confident, disagrees with visual read"}
READ_COLORS = {0: CUSTOM_PALETTE[0], 1: CUSTOM_PALETTE[1]}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="Path to a results run directory trained with --targets visual_read.")
    parser.add_argument("--dataset", default="Gothenburg", help="Dataset name used in the result filenames.")
    parser.add_argument("--demo_csv", default=None,
                        help="Demographics csv, for step 2 (default: data/demo_<dataset>.csv, then data/demo.csv).")
    return parser.parse_args()


# ------------------------------
# Confidence levels
# ------------------------------
def assign_groups(prob: np.ndarray, correct: np.ndarray, confident: tuple, uncertain: tuple) -> np.ndarray:
    lo, hi = confident
    u_lo, u_hi = uncertain
    is_confident = (prob <= lo) | (prob >= hi)
    groups = np.full(len(prob), "Moderately confident", dtype=object)
    groups[is_confident & correct] = "Confidently right"
    groups[is_confident & ~correct] = "Confidently wrong"
    # Strictly inside the band, so the fallback band (0.2-0.8) never overlaps the confident levels.
    groups[(prob > u_lo) & (prob < u_hi)] = "Uncertain"
    return groups


def choose_cutoffs(prob: np.ndarray, correct: np.ndarray) -> tuple[dict, list[str]]:
    """The pre-specified cut-offs, widened independently for each group that comes out too small."""
    cutoffs, notes = dict(PRIMARY), []
    groups = assign_groups(prob, correct, cutoffs["confident"], cutoffs["uncertain"])
    if (groups == "Confidently wrong").sum() < MIN_GROUP_N:
        cutoffs["confident"] = FALLBACK["confident"]
        notes.append(f"'Confidently wrong' had fewer than {MIN_GROUP_N} scans at 0.1 / 0.9 - "
                     f"confident cut-offs widened to 0.2 / 0.8 (pre-specified fallback).")
    if (groups == "Uncertain").sum() < MIN_GROUP_N:
        cutoffs["uncertain"] = FALLBACK["uncertain"]
        notes.append(f"'Uncertain' had fewer than {MIN_GROUP_N} scans at 0.3-0.7 - "
                     f"band widened to 0.2-0.8 (pre-specified fallback).")
    return cutoffs, notes


def triage_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per confidence level (confident right + wrong counted together): share of all scans and how
    often the model agreed with the visual read there."""
    level = df["group"].replace({"Confidently right": "Confident", "Confidently wrong": "Confident"})
    rows = []
    for name in ["Confident", "Moderately confident", "Uncertain"]:
        g = df[level == name]
        k, n = int((g["y"] == g["pred"]).sum()), len(g)
        lo, hi = wilson_interval(k, n)
        rows.append({"level": name, "n": n, "share_of_scans": n / len(df) if len(df) else np.nan,
                     "agree_k": k, "agreement": k / n if n else np.nan, "agreement_lo": lo, "agreement_hi": hi})
    return pd.DataFrame(rows)


# ------------------------------
# Step 1 figures
# ------------------------------
def make_histogram(df: pd.DataFrame, cutoffs: dict, out_path: Path):
    """One row per visual read, probability on x, every bar coloured by the level its scans are in
    (bin edges fall on every cut-off, so a bar never mixes levels). Most scans pile up near 0 and 1,
    which would flatten every other bar, so the y-axis is capped and a bar that runs past the cap is
    cut off with its true count printed at the top. Counts are printed on every bar."""
    edges = np.linspace(0, 1, 21)
    centres = (edges[:-1] + edges[1:]) / 2
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.6), sharex=True)
    for ax, read in zip(axes, [0, 1]):
        sub = df[df["y"] == read]
        counts, _ = np.histogram(sub["prob"], bins=edges)
        bar_groups = assign_groups(centres, (centres > 0.5) == bool(read), cutoffs["confident"], cutoffs["uncertain"])
        inner = np.sort(counts)[-3] if len(counts) >= 3 else counts.max()  # ignore the two tallest bars
        cap = max(10, int(np.ceil(inner * 1.6)))
        if counts.max() <= cap * 1.5:
            cap = max(int(np.ceil(counts.max() * 1.15)), 1)
        ax.bar(centres, np.minimum(counts, cap), width=0.05, color=[GROUP_COLORS[g] for g in bar_groups],
               edgecolor="white", linewidth=0.8, zorder=2)
        for x, c in zip(centres, counts):
            if c:
                cut = c > cap
                ax.text(x, min(c, cap), f"{c}" + (" ↑" if cut else ""), ha="center", va="bottom",
                        fontsize=TICK_SIZE - 3, color="black" if cut else "#555555",
                        fontweight="bold" if cut else "normal")
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1.2, zorder=3)
        ax.set_ylim(0, cap * 1.12)
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_title(f"Visual read {'+' if read else '−'} (n={len(sub)})", fontsize=LABEL_SIZE)
        ax.set_ylabel("Scans", fontsize=LABEL_SIZE - 2)
        ax.grid(axis="y", alpha=0.25)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[-1].set_xlabel("Model probability of tau positive (dashed line = 0.5 threshold)", fontsize=LABEL_SIZE - 2)
    axes[-1].set_xlim(0, 1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[g]) for g in GROUP_ORDER]
    fig.legend(handles, [GROUP_DISPLAY_LABELS[g] for g in GROUP_ORDER], loc="lower center", ncol=2,
               frameon=False, fontsize=TICK_SIZE - 1)
    fig.suptitle("Model confidence", fontsize=TITLE_SIZE)
    finalize_figure(fig, rect=(0.0, 0.09, 1.0, 1.0))
    save_figure(fig, out_path, dpi=300)
    plt.close(fig)


def calibration_table(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Ten equal-width probability bins: mean predicted probability vs the share of scans the
    visual read called positive (Wilson 95% CI), empty bins dropped."""
    bins = np.clip((df["prob"] * n_bins).astype(int), 0, n_bins - 1)
    rows = []
    for b, g in df.groupby(bins):
        k, n = int(g["y"].sum()), len(g)
        lo, hi = wilson_interval(k, n)
        rows.append({"bin": f"{b / n_bins:.1f}-{(b + 1) / n_bins:.1f}", "n": n,
                     "mean_probability": g["prob"].mean(), "share_visual_read_positive": k / n,
                     "share_lo": lo, "share_hi": hi})
    return pd.DataFrame(rows)


def make_calibration_figure(table: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.2, zorder=1)
    ax.vlines(table["mean_probability"], table["share_lo"], table["share_hi"], color="#999999", linewidth=2.2, zorder=2)
    ax.scatter(table["mean_probability"], table["share_visual_read_positive"], s=90, color=CUSTOM_PALETTE[0],
               edgecolor="white", linewidth=0.8, zorder=3)
    for r in table.itertuples():
        ax.annotate(f"n={r.n}", (r.mean_probability, r.share_hi), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=TICK_SIZE - 3, color="#555555")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlabel("Model probability (mean in bin)", fontsize=LABEL_SIZE - 2)
    ax.set_ylabel("Share called positive by visual read", fontsize=LABEL_SIZE - 2)
    ax.set_title("Calibration (dashed line = perfect)", fontsize=TITLE_SIZE - 3)
    ax.grid(alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    finalize_figure(fig, rect=(0.0, 0.0, 1.0, 1.0))
    save_figure(fig, out_path, dpi=300)
    plt.close(fig)


# ------------------------------
# Step 2: does the uncertainty follow the biology?
# ------------------------------
def confidence_vs_suvr(df: pd.DataFrame, cutoff: float) -> list[dict]:
    """Spearman correlation between the model's confidence and how far the scan's SUVR is from the
    cut-off - for all scans, and within each visual read (so the result is not just "positive
    scans have high SUVR and confident probabilities")."""
    rows = []
    for label, sub in [("all scans", df), ("visual read −", df[df["y"] == 0]), ("visual read +", df[df["y"] == 1])]:
        sub = sub.dropna(subset=["_suvr"])
        conf = (sub["prob"] - 0.5).abs()
        dist = (sub["_suvr"] - cutoff).abs()
        rho, p = stats.spearmanr(conf, dist) if len(sub) >= 3 else (np.nan, np.nan)
        rows.append({"scans": label, "n": len(sub), "spearman_rho": rho, "p_value": p})
    return rows


def make_probability_vs_suvr_figure(df: pd.DataFrame, cutoff: float, cutoffs: dict, out_path: Path):
    """Every scan: Meta-temporal SUVR on x, model probability on y, coloured by visual read. The
    vertical line is the CTRz 2 SUVR equivalent, the shaded band the 'uncertain' probability range.
    If the model's uncertainty follows the biology, the scans inside the band sit near the line."""
    sub = df.dropna(subset=["_suvr"])
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.axhspan(*cutoffs["uncertain"], color=GROUP_COLORS["Uncertain"], alpha=0.15, zorder=0, lw=0)
    ax.axhline(0.5, color="#777777", linestyle=":", linewidth=1.1, zorder=1)
    ax.axvline(cutoff, color="black", linestyle="--", linewidth=1.2, zorder=1)
    for read in (0, 1):
        r = sub[sub["y"] == read]
        ax.scatter(r["_suvr"], r["prob"], s=20, alpha=0.65, color=READ_COLORS[read], edgecolor="white",
                   linewidth=0.3, zorder=3, label=f"Visual read {'+' if read else '−'} (n={len(r)})")
    ax.annotate(f"CTRz 2 ({cutoff:.2f})", (cutoff, 1.03), textcoords="offset points", xytext=(5, 0),
                ha="left", va="bottom", fontsize=TICK_SIZE - 2)
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel("Meta-temporal SUVR", fontsize=LABEL_SIZE - 2)
    ax.set_ylabel("Model probability of tau positive", fontsize=LABEL_SIZE - 2)
    ax.set_title("Model probability vs Meta-temporal SUVR", fontsize=TITLE_SIZE - 3, pad=22)
    ax.legend(loc="lower right", fontsize=TICK_SIZE - 2, frameon=False)
    ax.grid(alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    finalize_figure(fig, rect=(0.0, 0.0, 1.0, 1.0))
    save_figure(fig, out_path, dpi=300)
    plt.close(fig)


def _logistic_crossing(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Plain (unpenalised) logistic regression y ~ x by Newton-Raphson on standardised x. Returns
    (intercept, slope) on the original scale and the x at which the fitted probability is 0.5 (NaN
    if the slope is not positive, i.e. no sensible crossing)."""
    m, sd = x.mean(), x.std()
    if sd == 0 or len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan
    X = np.column_stack([np.ones_like(x), (x - m) / sd])
    b = np.zeros(2)
    for _ in range(100):
        prob = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        w = prob * (1 - prob)
        step = np.linalg.solve(X.T @ (X * w[:, None]) + 1e-9 * np.eye(2), X.T @ (y - prob))
        b += step
        if np.abs(step).max() < 1e-8:
            break
    slope, intercept = b[1] / sd, b[0] - b[1] * m / sd
    return intercept, slope, (-intercept / slope if slope > 0 else np.nan)


def suvr_cutoffs(df: pd.DataFrame, seed: int = 0) -> dict:
    """The Meta-temporal SUVR at which the visual read, and the model's call, become more likely
    positive than not - with bootstrap 95% CIs and the paired difference (model - readers)."""
    sub = df.dropna(subset=["_suvr"])
    x = sub["_suvr"].to_numpy(float)
    calls = {"Visual read": sub["y"].to_numpy(float), "Model": sub["pred"].to_numpy(float)}
    fits = {k: _logistic_crossing(x, v) for k, v in calls.items()}
    rng = np.random.default_rng(seed)
    boot = {k: [] for k in calls}
    for idx in rng.integers(0, len(x), size=(N_BOOT, len(x))):
        for k, v in calls.items():
            boot[k].append(_logistic_crossing(x[idx], v[idx]))
    boot = {k: np.array(v) for k, v in boot.items()}
    out = {"n": len(x), "fits": fits, "boot": boot}
    for k in calls:
        out[k] = (fits[k][2], *np.nanpercentile(boot[k][:, 2], [2.5, 97.5]))
    diff = boot["Model"][:, 2] - boot["Visual read"][:, 2]
    out["difference"] = (fits["Model"][2] - fits["Visual read"][2], *np.nanpercentile(diff, [2.5, 97.5]))
    return out


def make_suvr_cutoffs_figure(df: pd.DataFrame, cut: dict, ctrz2: float, out_path: Path):
    """Two fitted curves against Meta-temporal SUVR: probability that the visual read calls a scan
    positive, and that the model does. Where each crosses 0.5 is that judge's SUVR cut-off (dot,
    with its 95% CI as a horizontal bar); dashed line = CTRz 2 for reference."""
    x = df["_suvr"].dropna()
    grid = np.linspace(x.quantile(0.01), x.quantile(0.97), 300)
    colors = {"Visual read": CUSTOM_PALETTE[1], "Model": CUSTOM_PALETTE[0]}
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.axvline(ctrz2, color="black", linestyle="--", linewidth=1.2, zorder=1)
    ax.annotate(f"CTRz 2 ({ctrz2:.2f})", (ctrz2, 1.03), textcoords="offset points", xytext=(5, 0),
                ha="left", va="bottom", fontsize=TICK_SIZE - 2)
    ax.axhline(0.5, color="#999999", linestyle=":", linewidth=1.0, zorder=1)
    for k, color in colors.items():
        b0, b1, c = cut["fits"][k]
        if not np.isfinite(c):
            continue
        boot = cut["boot"][k]
        boot = boot[np.isfinite(boot[:, 1])]
        curves = 1 / (1 + np.exp(-(boot[:, [0]] + boot[:, [1]] * grid[None, :])))
        lo, hi = np.percentile(curves, [2.5, 97.5], axis=0)
        ax.fill_between(grid, lo, hi, color=color, alpha=0.15, lw=0, zorder=2)
        ax.plot(grid, 1 / (1 + np.exp(-(b0 + b1 * grid))), color=color, linewidth=2.2, zorder=3,
                label=f"{k}: {c:.2f} [{cut[k][1]:.2f}-{cut[k][2]:.2f}]")
        ax.hlines(0.5, cut[k][1], cut[k][2], color=color, linewidth=3, zorder=4)
        ax.scatter([c], [0.5], s=70, color=color, edgecolor="white", linewidth=0.8, zorder=5)
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel("Meta-temporal SUVR", fontsize=LABEL_SIZE - 2)
    ax.set_ylabel("Probability of being called positive", fontsize=LABEL_SIZE - 2)
    ax.set_title("SUVR cut-off of the visual read and the model", fontsize=TITLE_SIZE - 3, pad=22)
    ax.legend(loc="lower right", fontsize=TICK_SIZE - 2, frameon=False, title="Cut-off [95% CI]",
              title_fontsize=TICK_SIZE - 2)
    ax.grid(alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    finalize_figure(fig, rect=(0.0, 0.0, 1.0, 1.0))
    save_figure(fig, out_path, dpi=300)
    plt.close(fig)


def confidently_wrong_cases(df: pd.DataFrame) -> pd.DataFrame:
    """Step 3: one row per confidently wrong scan, plus median rows for the confidently right scans
    of each visual read to read them against. "Regions >= CTRz 2" names every region whose SUVR is
    at or above that region's own CTRz 2 equivalent."""
    regions = [r for r in REGIONS if r in df.columns]
    extra = [c for c in ["CL", "amyloid_status", "apoe", "age", "gender", "CDR_SB", "dx", "site"] if c in df.columns]

    def row(r):
        out = {"ID_ind": r["ID_ind"],
               "case": confusion_display_label({(1, 0): "False Positive", (0, 1): "False Negative"}[(r["pred"], r["y"])]),
               "probability": round(r["prob"], 3)}
        for reg in regions:
            out[f"{reg} SUVR"] = round(float(r[reg]), 2) if pd.notna(r[reg]) else np.nan
        out["Regions >= CTRz 2"] = ", ".join(reg for reg in regions
                                             if pd.notna(r[reg]) and r[reg] >= CENTAUR_FTP_CTRZ2_SUVR[reg]) or "none"
        for c in extra:
            out[c] = round(float(r[c]), 1) if c in ("CL", "age", "CDR_SB") and pd.notna(r[c]) else r[c]
        return out

    wrong = df[df["group"] == "Confidently wrong"].sort_values(["y", "prob"], ascending=[True, False])
    rows = [row(r) for _, r in wrong.iterrows()]
    for read in (0, 1):
        ref = df[(df["group"] == "Confidently right") & (df["y"] == read)]
        ref_row = {"ID_ind": "median", "case": f"Confidently right, Visual read {'+' if read else '−'} (n={len(ref)})",
                   "probability": round(ref["prob"].median(), 3)}
        for reg in regions:
            ref_row[f"{reg} SUVR"] = round(pd.to_numeric(ref[reg], errors="coerce").median(), 2)
        for c in ("CL", "age", "CDR_SB"):
            if c in ref.columns:
                ref_row[c] = round(pd.to_numeric(ref[c], errors="coerce").median(), 1)
        if "amyloid_status" in ref.columns:
            amy = pd.to_numeric(ref["amyloid_status"], errors="coerce").dropna()
            ref_row["amyloid_status"] = f"{int(amy.sum())}/{len(amy)} positive"
        rows.append(ref_row)
    return pd.DataFrame(rows)


def level_description(df: pd.DataFrame, cutoff: float, model_cutoff: float = np.nan) -> pd.DataFrame:
    """Descriptive table, one column per confidence level (all scans appear in exactly one): SUVR,
    distance from the cut-off, amyloid, diagnosis, site. Counts as k/n so missing data stays visible."""
    cols = {}
    for g in GROUP_ORDER:
        sub = df[df["group"] == g]
        col = {"n": str(len(sub))}
        suvr = sub["_suvr"].dropna()
        if len(suvr):
            col["Meta-temporal SUVR, median [IQR]"] = f"{suvr.median():.2f} [{suvr.quantile(.25):.2f}-{suvr.quantile(.75):.2f}]"
            col["Distance from CTRz 2, median"] = f"{(suvr - cutoff).abs().median():.2f}"
            if np.isfinite(model_cutoff):
                col["Distance from model's own cut-off, median"] = f"{(suvr - model_cutoff).abs().median():.2f}"
        if "amyloid_status" in sub.columns:
            amy = pd.to_numeric(sub["amyloid_status"], errors="coerce").dropna()
            col["Aβ positive"] = f"{int(amy.sum())}/{len(amy)}"
        if "dx" in sub.columns:
            dx = combine_dx_groups(sub["dx"]).dropna()
            for level in ["CU", "MCI", "AlzCS dem", "Other"]:
                col[f"Diagnosis: {'AD dementia' if level == 'AlzCS dem' else level}"] = f"{int((dx == level).sum())}/{len(dx)}"
        if "site" in sub.columns:
            site = sub["site"].astype("string").dropna()
            for level in sorted(df["site"].astype("string").dropna().unique()):
                col[f"Site: {level}"] = f"{int((site == level).sum())}/{len(site)}"
        cols[g] = col
    return pd.DataFrame(cols)


# ------------------------------
def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    preds_path, out_dir = resolve_predictions_csv(run_dir, args.dataset)
    df = pd.read_csv(preds_path)
    if "prob" not in df.columns:
        raise ValueError(f"{preds_path} has no 'prob' column - is this a visual_read run?")
    df = df[[c for c in ["ID_ind", "y", "prob", "pred"] if c in df.columns]].copy()
    for col in ["y", "prob", "pred"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["y", "prob", "pred"]).reset_index(drop=True)
    df["y"], df["pred"] = df["y"].astype(int), df["pred"].astype(int)
    print(f"[load] {len(df)} scans from {preds_path}")

    correct = (df["y"] == df["pred"]).to_numpy()
    cutoffs, notes = choose_cutoffs(df["prob"].to_numpy(), correct)
    df["group"] = assign_groups(df["prob"].to_numpy(), correct, cutoffs["confident"], cutoffs["uncertain"])

    # Step 1
    table = calibration_table(df)
    brier = float(((df["prob"] - df["y"]) ** 2).mean())
    ece = float((table["n"] / len(df) * (table["share_visual_read_positive"] - table["mean_probability"]).abs()).sum())
    triage = triage_table(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    make_histogram(df, cutoffs, out_dir / "visual_read_confidence_histogram.png")
    make_calibration_figure(table, out_dir / "visual_read_calibration.png")
    table.to_csv(out_dir / "visual_read_calibration_bins.csv", index=False)
    triage.to_csv(out_dir / "visual_read_confidence_triage.csv", index=False)

    # Step 2 (needs SUVR from the demographics)
    demo_path = resolve_path(args.demo_csv, lambda: find_dataset_demo_csv(args.dataset) or find_demo_csv())
    corr_rows, model_corr_rows, cut, wrong = [], [], None, None
    cutoff = CENTAUR_FTP_CTRZ2_SUVR[SUVR_REGION]
    if demo_path is None:
        print("[skip] step 2: no demographics csv found - pass --demo_csv.")
    else:
        demo = pd.read_csv(demo_path)
        demo = demo.loc[:, ~demo.columns.astype(str).str.startswith("Unnamed")]
        df = df.merge(demo, left_on="ID_ind", right_on="ID", how="left")
        unmatched = df["ID"].isna().sum()
        if unmatched:
            print(f"[WARNING] {unmatched}/{len(df)} scans have no row in {demo_path.name} - check --demo_csv.")
        if SUVR_REGION in df.columns:
            df["_suvr"] = pd.to_numeric(df[SUVR_REGION], errors="coerce")
            corr_rows = confidence_vs_suvr(df, cutoff)
            make_probability_vs_suvr_figure(df, cutoff, cutoffs, out_dir / "visual_read_probability_vs_suvr.png")
            cut = suvr_cutoffs(df)
            model_cut = cut["Model"][0]
            if np.isfinite(model_cut):
                model_corr_rows = confidence_vs_suvr(df, model_cut)
            make_suvr_cutoffs_figure(df, cut, cutoff, out_dir / "visual_read_suvr_cutoffs.png")
            pd.DataFrame([{"judge": k, "suvr_cutoff": cut[k][0], "ci_lo": cut[k][1], "ci_hi": cut[k][2]}
                          for k in ("Visual read", "Model", "difference")]
                         + [{"judge": "CTRz 2 (reference)", "suvr_cutoff": cutoff}]
                         ).to_csv(out_dir / "visual_read_suvr_cutoffs.csv", index=False)
            level_description(df, cutoff, model_cut).to_csv(out_dir / "visual_read_confidence_by_level.csv")
        else:
            print(f"[skip] step 2: no {SUVR_REGION} column in {demo_path.name}.")
        # Step 3 - works with whatever regional SUVR / demographic columns are present
        wrong = confidently_wrong_cases(df)
        wrong.to_csv(out_dir / "visual_read_confidently_wrong_cases.csv", index=False)

    keep = [c for c in ["ID_ind", "y", "prob", "pred", "group", "_suvr"] if c in df.columns]
    df[keep].rename(columns={"_suvr": f"{SUVR_REGION}_SUVR"}).to_csv(out_dir / "visual_read_confidence_groups.csv", index=False)

    counts = df.groupby(["group", "y"]).size().unstack(fill_value=0).reindex(GROUP_ORDER, fill_value=0)
    summary = [("confident_cutoffs", f"<= {cutoffs['confident'][0]} or >= {cutoffs['confident'][1]}"),
               ("uncertain_band", f"{cutoffs['uncertain'][0]}-{cutoffs['uncertain'][1]}"),
               ("fallback_used", "; ".join(notes) or "no")]
    group_lines = []
    for g in GROUP_ORDER:
        value = (f"{int(counts.loc[g].sum())} (visual read +: {int(counts.loc[g].get(1, 0))}, "
                 f"visual read −: {int(counts.loc[g].get(0, 0))})")
        summary.append((f"n_{g.lower().replace(' ', '_')}", value))
        group_lines.append(f"    {g.lower()}: {value}")
    summary += [("brier_score", f"{brier:.3f}"), ("expected_calibration_error", f"{ece:.3f}")]
    for r in corr_rows:
        summary.append((f"confidence_vs_suvr_distance_rho, CTRz 2 ({r['scans']})",
                        f"{r['spearman_rho']:.2f} (p={r['p_value']:.2g}, n={r['n']})"))
    if cut is not None:
        for k in ("Visual read", "Model", "difference"):
            summary.append((f"suvr_cutoff ({k})", f"{cut[k][0]:.3f} [{cut[k][1]:.3f}-{cut[k][2]:.3f}]"))
    for r in model_corr_rows:
        summary.append((f"confidence_vs_suvr_distance_rho, model's own cut-off ({r['scans']})",
                        f"{r['spearman_rho']:.2f} (p={r['p_value']:.2g}, n={r['n']})"))
    pd.DataFrame(summary, columns=["item", "value"]).to_csv(out_dir / "visual_read_confidence_summary.csv", index=False)

    for note in notes:
        print(f"[fallback] {note}")
    for g in ("Confidently wrong", "Uncertain"):
        if (df["group"] == g).sum() < MIN_GROUP_N:
            print(f"[note] '{g}' still has fewer than {MIN_GROUP_N} scans after the fallback - "
                  "describe those cases one by one rather than as a group.")
    print("[levels]")
    print("\n".join(group_lines))
    print(f"[calibration] Brier score {brier:.3f}, expected calibration error {ece:.3f} "
          "(0 = probabilities mean exactly what they say)")
    print("[triage] auto-decide confident scans, refer the rest:")
    for r in triage.itertuples():
        print(f"    {r.level}: {r.n} scans ({r.share_of_scans:.0%}), agrees with visual read "
              f"{r.agree_k}/{r.n} ({r.agreement:.0%})")
    if corr_rows:
        print(f"[step 2] model confidence vs distance of {SUVR_REGION} SUVR from CTRz 2 ({cutoff:.2f}), "
              "Spearman (positive = least sure where SUVR is borderline):")
        for r in corr_rows:
            print(f"    {r['scans']}: rho {r['spearman_rho']:.2f} (p={r['p_value']:.2g}, n={r['n']})")
    if cut is not None:
        print(f"[step 2] {SUVR_REGION} SUVR at which each judge becomes more likely positive than not (95% CI):")
        for k in ("Visual read", "Model"):
            print(f"    {k}: {cut[k][0]:.2f} [{cut[k][1]:.2f}-{cut[k][2]:.2f}]")
        d = cut["difference"]
        print(f"    model - visual read: {d[0]:+.2f} [{d[1]:+.2f} to {d[2]:+.2f}]  (CTRz 2 = {cutoff:.2f})")
    if model_corr_rows:
        print(f"[step 2] same correlation, distance from the model's own cut-off ({cut['Model'][0]:.2f}):")
        for r in model_corr_rows:
            print(f"    {r['scans']}: rho {r['spearman_rho']:.2f} (p={r['p_value']:.2g}, n={r['n']})")
    if wrong is not None:
        cases = wrong[wrong["ID_ind"] != "median"]
        print(f"[step 3] {len(cases)} confidently wrong scans -> visual_read_confidently_wrong_cases.csv")
        show = [c for c in ["ID_ind", "case", "probability", "MetaTemporal SUVR", "TemporoParietal SUVR",
                            "Regions >= CTRz 2", "CL", "dx", "amyloid_status", "site"] if c in wrong.columns]
        print(wrong[show].to_string(index=False))
    print(f"[done] saved to {out_dir}")


if __name__ == "__main__":
    main()
