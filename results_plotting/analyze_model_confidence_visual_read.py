#!/usr/bin/env python3
"""How confident is the visual-read model, and can its confidence be trusted? Step 1 of the
confidence analysis: sorts scans into the pre-specified groups below, and checks calibration.

Groups, fixed before looking at any results, on the model's raw probability of "positive":
    Confidently wrong  - disagrees with the visual read, probability >= 0.9 or <= 0.1
    Uncertain          - probability between 0.3 and 0.7, right or wrong
    Confidently right  - agrees with the visual read, probability >= 0.9 or <= 0.1
Scans at 0.1-0.3 or 0.7-0.9 are in none of them, on purpose, so each group is clearly what its
name says. Fallback, also fixed in advance: if "Confidently wrong" has fewer than 5 scans the
confident cut-offs widen to 0.2 / 0.8; if "Uncertain" has fewer than 5 the band widens to 0.2-0.8.
The script says when it does either.

Raw probabilities, not recalibrated ones: the question is about the model's own sense of
confidence. The calibration check says how far to trust that confidence; it does not change
the groups.

Outputs, next to the run's other figures:
    visual_read_confidence_histogram.png  - probabilities, split by visual read, bars coloured by group
    visual_read_calibration.png           - predicted probability vs how often the visual read was
                                            positive (on the diagonal = probabilities mean what they say)
    visual_read_confidence_groups.csv     - one row per scan: ID, visual read, probability, model call, group
    visual_read_confidence_summary.csv    - cut-offs used, group sizes, Brier score, calibration error

Usage:
    python results_plotting/analyze_model_confidence_visual_read.py /path/to/run_dir --dataset Gothenburg
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_utils import (
    CUSTOM_PALETTE,
    LABEL_SIZE,
    TICK_SIZE,
    TITLE_SIZE,
    finalize_figure,
    resolve_predictions_csv,
    save_figure,
    wilson_interval,
)

PRIMARY = {"confident": (0.1, 0.9), "uncertain": (0.3, 0.7)}
FALLBACK = {"confident": (0.2, 0.8), "uncertain": (0.2, 0.8)}
MIN_GROUP_N = 5

GROUP_ORDER = ["Confidently right", "Uncertain", "Confidently wrong", "Neither"]
GROUP_COLORS = {"Confidently right": CUSTOM_PALETTE[0], "Uncertain": CUSTOM_PALETTE[4],
                "Confidently wrong": CUSTOM_PALETTE[1], "Neither": "#BBBBBB"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="Path to a results run directory trained with --targets visual_read.")
    parser.add_argument("--dataset", default="Gothenburg", help="Dataset name used in the result filenames.")
    return parser.parse_args()


def assign_groups(prob: np.ndarray, correct: np.ndarray, confident: tuple, uncertain: tuple) -> np.ndarray:
    lo, hi = confident
    u_lo, u_hi = uncertain
    is_confident = (prob <= lo) | (prob >= hi)
    groups = np.full(len(prob), "Neither", dtype=object)
    groups[is_confident & correct] = "Confidently right"
    groups[is_confident & ~correct] = "Confidently wrong"
    # Strictly inside the band, so the fallback band (0.2-0.8) never overlaps the confident zones.
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


def make_histogram(df: pd.DataFrame, cutoffs: dict, out_path: Path):
    """One row per visual read, probability on x, every bar coloured by the group its scans are in
    (bin edges fall on every cut-off, so a bar never mixes groups). Most scans pile up near 0 and 1,
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

    table = calibration_table(df)
    brier = float(((df["prob"] - df["y"]) ** 2).mean())
    ece = float((table["n"] / len(df) * (table["share_visual_read_positive"] - table["mean_probability"]).abs()).sum())

    out_dir.mkdir(parents=True, exist_ok=True)
    make_histogram(df, cutoffs, out_dir / "visual_read_confidence_histogram.png")
    make_calibration_figure(table, out_dir / "visual_read_calibration.png")
    df.to_csv(out_dir / "visual_read_confidence_groups.csv", index=False)

    counts = df.groupby(["group", "y"]).size().unstack(fill_value=0).reindex(GROUP_ORDER, fill_value=0)
    summary = [("confident_cutoffs", f"<= {cutoffs['confident'][0]} or >= {cutoffs['confident'][1]}"),
               ("uncertain_band", f"{cutoffs['uncertain'][0]}-{cutoffs['uncertain'][1]}"),
               ("fallback_used", "; ".join(notes) or "no")]
    for g in GROUP_ORDER:
        summary.append((f"n_{g.lower().replace(' ', '_')}",
                        f"{int(counts.loc[g].sum())} (visual read +: {int(counts.loc[g].get(1, 0))}, "
                        f"visual read −: {int(counts.loc[g].get(0, 0))})"))
    summary += [("brier_score", f"{brier:.3f}"), ("expected_calibration_error", f"{ece:.3f}")]
    pd.DataFrame(summary, columns=["item", "value"]).to_csv(out_dir / "visual_read_confidence_summary.csv", index=False)
    table.to_csv(out_dir / "visual_read_calibration_bins.csv", index=False)

    for note in notes:
        print(f"[fallback] {note}")
    for g in ("Confidently wrong", "Uncertain"):
        if (df["group"] == g).sum() < MIN_GROUP_N:
            print(f"[note] '{g}' still has fewer than {MIN_GROUP_N} scans after the fallback - "
                  "describe those cases one by one rather than as a group.")
    print("[groups]")
    for item, value in summary[3:3 + len(GROUP_ORDER)]:
        print(f"    {item.removeprefix('n_').replace('_', ' ')}: {value}")
    print(f"[calibration] Brier score {brier:.3f}, expected calibration error {ece:.3f} "
          "(0 = probabilities mean exactly what they say)")
    print(f"[done] saved to {out_dir}")


if __name__ == "__main__":
    main()
