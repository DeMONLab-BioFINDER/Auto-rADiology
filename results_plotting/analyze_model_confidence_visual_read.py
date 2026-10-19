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

GROUP_ORDER = ["Confidently right", "Moderately confident", "Uncertain", "Confidently wrong"]
GROUP_COLORS = {"Confidently right": CUSTOM_PALETTE[0], "Moderately confident": "#BBBBBB",
                "Uncertain": CUSTOM_PALETTE[4], "Confidently wrong": CUSTOM_PALETTE[1]}
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
    fig.legend(handles, GROUP_ORDER, loc="lower center", ncol=4, frameon=False, fontsize=TICK_SIZE - 1)
    fig.suptitle("Model confidence", fontsize=TITLE_SIZE)
    finalize_figure(fig, rect=(0.0, 0.05, 1.0, 1.0))
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


def level_description(df: pd.DataFrame, cutoff: float) -> pd.DataFrame:
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
    corr_rows = []
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
            level_description(df, cutoff).to_csv(out_dir / "visual_read_confidence_by_level.csv")
        else:
            print(f"[skip] step 2: no {SUVR_REGION} column in {demo_path.name}.")

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
        summary.append((f"confidence_vs_suvr_distance_rho ({r['scans']})",
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
    print(f"[done] saved to {out_dir}")


if __name__ == "__main__":
    main()
