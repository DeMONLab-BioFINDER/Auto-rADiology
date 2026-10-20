#!/usr/bin/env python3
"""Where is the tau-positive SUVR cut-off? Four ways of estimating it, per region, side by side.

    1. CTRz 2 (published)      - the CenTauR CTRz 2 FTP SUVR equivalent, for reference (no data used).
    2. Reference group         - mean + 2 SD of the cognitively unimpaired, amyloid-negative people
                                 (whole dataset). Same logic as CTRz, but on this cohort.
    3. Gaussian mixture        - assume the SUVRs come from two overlapping bell curves (tau-negative,
                                 tau-positive), fit both (whole dataset, no labels) and take the SUVR
                                 between their means where a scan is equally likely to belong to either.
                                 Tau SUVR is often one skewed hump rather than two, so this can be
                                 unstable; the csv reports whether two curves fit better than one (BIC)
                                 and how often the bootstrap found a crossing at all.
    4. Implied by a judge      - logistic fit of a yes/no call on SUVR; the cut-off is where the fitted
                                 probability of "positive" crosses 0.5:
                                   Visual read, all scans  (whole dataset - the labels, no model involved)
                                   Visual read, test set   (like-for-like with the model)
                                   Model, test set         (held-out only: on training scans the model has
                                                            partly memorised the readers' answers)
                                 plus the paired difference Model - Visual read on the test set.

All CIs are bootstrap 95% (resampling scans; one scan per person). Methods 1-3 and the whole-dataset
visual read need only the demographics csv; the test-set rows need a visual-read run directory.

Outputs (default <proj_path>/results/suvr_cutoffs, outside the repo):
    suvr_cutoffs_summary.png        - per region, every method's cut-off with its 95% CI
    suvr_cutoffs_distributions.png  - per region, SUVR histogram by visual read, the two fitted
                                      Gaussian-mixture curves, and the cut-offs as lines
    suvr_cutoffs_logistic.png       - per region, probability of a positive call vs SUVR for the
                                      visual read (all scans) and the model (test set)
    suvr_cutoffs.csv                - region, method, data, n, cut-off, 95% CI
    suvr_cutoffs_gmm_details.csv    - per region: the two curves' means, SDs, weights, BIC 1 vs 2 curves

Usage:
    python results_plotting/analyze_suvr_cutoffs.py --demo_csv /path/to/demo.csv \
        --run_dir /path/to/visual_read_run_dir --dataset Gothenburg
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from analyze_model_confidence_visual_read import _logistic_crossing
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
)

REGIONS = ["MetaTemporal", "MesialTemporal", "TemporoParietal", "Frontal", "Universal"]
N_BOOT = 2000
N_BOOT_GMM = 500

METHODS = ["CTRz 2 (published)", "Reference: CU Aβ− mean + 2 SD", "Gaussian mixture",
           "Visual read, all scans", "Visual read, test set", "Model, test set"]
METHOD_COLORS = dict(zip(METHODS, ["black", CUSTOM_PALETTE[3], CUSTOM_PALETTE[4],
                                   CUSTOM_PALETTE[1], CUSTOM_PALETTE[6], CUSTOM_PALETTE[0]]))
READ_COLORS = {0: CUSTOM_PALETTE[0], 1: CUSTOM_PALETTE[1]}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demo_csv", default=None, help="Demographics csv for the whole dataset (default: data/demo.csv).")
    parser.add_argument("--run_dir", default=None,
                        help="Visual-read run directory, for the test-set rows (optional; skipped if absent).")
    parser.add_argument("--dataset", default="Gothenburg", help="Dataset name used in the result filenames.")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory (default: <proj_path>/results/suvr_cutoffs).")
    return parser.parse_args()


# ------------------------------
# Estimators - each returns (cut-off, 95% CI low, 95% CI high)
# ------------------------------
def _ci(point: float, boot: np.ndarray) -> tuple[float, float, float]:
    boot = boot[np.isfinite(boot)]
    if not np.isfinite(point) or len(boot) == 0:
        return point, np.nan, np.nan
    return point, *np.percentile(boot, [2.5, 97.5])


def reference_cutoff(x: np.ndarray, rng) -> tuple[float, float, float]:
    point = x.mean() + 2 * x.std(ddof=1)
    boot = np.array([(s.mean() + 2 * s.std(ddof=1)) for s in (x[i] for i in rng.integers(0, len(x), (N_BOOT, len(x))))])
    return _ci(point, boot)


def fit_gmm(x: np.ndarray, seed: int = 0) -> dict:
    """Two-component 1-D Gaussian mixture. The cut-off is where the posterior of the higher-mean
    component crosses 0.5 between the two means (NaN if it never does there)."""
    X = x.reshape(-1, 1)
    gm = GaussianMixture(2, n_init=3, random_state=seed).fit(X)
    order = np.argsort(gm.means_.ravel())
    means, sds = gm.means_.ravel()[order], np.sqrt(gm.covariances_.ravel()[order])
    weights = gm.weights_[order]
    grid = np.linspace(means[0], means[1], 2000)
    post_hi = gm.predict_proba(grid.reshape(-1, 1))[:, order[1]]
    above = np.nonzero(post_hi >= 0.5)[0]
    cut = grid[above[0]] if len(above) and above[0] > 0 else np.nan
    bic1 = GaussianMixture(1).fit(X).bic(X)
    return {"cut": cut, "means": means, "sds": sds, "weights": weights,
            "bic_1": bic1, "bic_2": gm.bic(X), "converged": gm.converged_}


def gmm_cutoff(x: np.ndarray, rng) -> tuple[tuple[float, float, float], dict, float]:
    fit = fit_gmm(x)
    boot = np.array([fit_gmm(x[i], seed=k)["cut"]
                     for k, i in enumerate(rng.integers(0, len(x), (N_BOOT_GMM, len(x))))])
    return _ci(fit["cut"], boot), fit, float(np.isfinite(boot).mean())


def judge_cutoffs(x: np.ndarray, calls: dict, rng) -> dict:
    """Logistic 0.5 crossing for each judge, on the same bootstrap resamples (so the difference
    between two judges has a fair paired CI). Returns {judge: (cut, lo, hi)}, plus fitted curves."""
    fits = {k: _logistic_crossing(x, v) for k, v in calls.items()}
    boot = {k: [] for k in calls}
    for idx in rng.integers(0, len(x), (N_BOOT, len(x))):
        for k, v in calls.items():
            boot[k].append(_logistic_crossing(x[idx], v[idx]))
    boot = {k: np.array(v) for k, v in boot.items()}
    out = {k: _ci(fits[k][2], boot[k][:, 2]) for k in calls}
    if len(calls) == 2:
        a, b = list(calls)
        out["difference"] = _ci(fits[b][2] - fits[a][2], boot[b][:, 2] - boot[a][:, 2])
    out["_fits"], out["_boot"] = fits, boot
    return out


# ------------------------------
# Figures
# ------------------------------
def _region_grid(n: int, panel_w: float, panel_h: float):
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_w * ncols, panel_h * nrows), squeeze=False)
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.axis("off")
    return fig, axes[:n], axes[n:]


def make_summary_figure(table: pd.DataFrame, regions: list, out_path: Path):
    fig, axes, spare = _region_grid(len(regions), 6.4, 3.9)
    methods = [m for m in METHODS if m in set(table["method"])]
    ys = np.arange(len(methods))[::-1]
    for i, (ax, region) in enumerate(zip(axes, regions)):
        sub = table[table["region"] == region].set_index("method")
        ctrz = CENTAUR_FTP_CTRZ2_SUVR.get(region)
        if ctrz is not None:
            ax.axvline(ctrz, color="black", linestyle="--", linewidth=1.0, zorder=1)
        for y, m in zip(ys, methods):
            if m not in sub.index or not np.isfinite(sub.loc[m, "cutoff"]):
                ax.text(0.02, y, "no estimate", transform=ax.get_yaxis_transform(), va="center",
                        fontsize=TICK_SIZE - 3, color="#888888")
                continue
            c, lo, hi = sub.loc[m, ["cutoff", "ci_lo", "ci_hi"]]
            color = METHOD_COLORS[m]
            if np.isfinite(lo):
                ax.hlines(y, lo, hi, color=color, linewidth=3, zorder=2)
            ax.scatter([c], [y], s=70, color=color, marker="D" if m == METHODS[0] else "o",
                       edgecolor="white", linewidth=0.8, zorder=3)
            ax.annotate(f"{c:.2f}", (c, y), textcoords="offset points", xytext=(0, 7), ha="center",
                        fontsize=TICK_SIZE - 3)
        ax.set_yticks(ys)
        ax.set_yticklabels(methods if i % 2 == 0 else [], fontsize=TICK_SIZE - 2)
        ax.set_ylim(-0.6, len(methods) - 0.3)
        ax.set_title(pretty_region_name(region), fontsize=TITLE_SIZE - 3)
        ax.set_xlabel("SUVR", fontsize=LABEL_SIZE - 3)
        ax.tick_params(axis="x", labelsize=TICK_SIZE - 2)
        ax.grid(axis="x", alpha=0.28)
    fig.suptitle("Tau-positive SUVR cut-off, by method", fontsize=TITLE_SIZE)
    finalize_figure(fig, rect=(0, 0, 1, 1.0))
    save_figure(fig, out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")


def make_distribution_figure(demo: pd.DataFrame, table: pd.DataFrame, gmm_fits: dict, regions: list, out_path: Path):
    fig, axes, spare = _region_grid(len(regions), 6.4, 4.2)
    shown = ["CTRz 2 (published)", "Reference: CU Aβ− mean + 2 SD", "Gaussian mixture", "Visual read, all scans"]
    for ax, region in zip(axes, regions):
        d = demo[[region, "visual_read"]].dropna()
        x = d[region].to_numpy(float)
        lo, hi = np.percentile(x, [0.5, 99])
        bins = np.linspace(lo, hi, 45)
        width = bins[1] - bins[0]
        for read in (0, 1):
            ax.hist(d.loc[d["visual_read"] == read, region], bins=bins, color=READ_COLORS[read],
                    alpha=0.55, label=f"Visual read {'+' if read else '−'}")
        fit = gmm_fits.get(region)
        if fit is not None:
            grid = np.linspace(lo, hi, 400)
            total = np.zeros_like(grid)
            for mu, sd, w in zip(fit["means"], fit["sds"], fit["weights"]):
                curve = len(x) * width * w * np.exp(-0.5 * ((grid - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
                total += curve
                ax.plot(grid, curve, color="#555555", linewidth=1.2, linestyle=":")
            ax.plot(grid, total, color="#555555", linewidth=1.5, label="Gaussian mixture: fitted curves")
        sub = table[table["region"] == region].set_index("method")
        for m in shown:
            if m in sub.index and np.isfinite(sub.loc[m, "cutoff"]):
                ax.axvline(sub.loc[m, "cutoff"], color=METHOD_COLORS[m], linewidth=1.6,
                           linestyle="--" if m == METHODS[0] else "-", label=f"Cut-off: {m}")
        ax.set_xlim(lo, hi)
        ax.set_title(pretty_region_name(region), fontsize=TITLE_SIZE - 3)
        ax.set_xlabel("SUVR", fontsize=LABEL_SIZE - 3)
        ax.set_ylabel("Scans", fontsize=LABEL_SIZE - 3)
        ax.tick_params(labelsize=TICK_SIZE - 2)
        ax.grid(alpha=0.28)
    handles, labels = axes[0].get_legend_handles_labels()
    if spare:
        spare[0].legend(handles, labels, loc="center", fontsize=LEGEND_SIZE, frameon=False)
    else:
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=LEGEND_SIZE, frameon=False,
                   bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("SUVR distributions and cut-offs (all scans)", fontsize=TITLE_SIZE)
    finalize_figure(fig, rect=(0, 0, 1, 1.0))
    save_figure(fig, out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")


def make_logistic_figure(curves: dict, table: pd.DataFrame, regions: list, out_path: Path):
    """curves[region][judge] = (x values, fits tuple, bootstrap array)."""
    fig, axes, spare = _region_grid(len(regions), 6.4, 4.2)
    for ax, region in zip(axes, regions):
        ctrz = CENTAUR_FTP_CTRZ2_SUVR.get(region)
        if ctrz is not None:
            ax.axvline(ctrz, color="black", linestyle="--", linewidth=1.0, zorder=1, label="CTRz 2 (published)")
        ax.axhline(0.5, color="#999999", linestyle=":", linewidth=1.0, zorder=1)
        xs = np.concatenate([v[0] for v in curves[region].values()])
        grid = np.linspace(*np.percentile(xs, [1, 97]), 300)
        sub = table[table["region"] == region].set_index("method")
        for judge, (x, fit, boot) in curves[region].items():
            b0, b1, c = fit
            if not np.isfinite(c):
                continue
            color = METHOD_COLORS[judge]
            ok = boot[np.isfinite(boot[:, 1])]
            band = 1 / (1 + np.exp(-(ok[:, [0]] + ok[:, [1]] * grid[None, :])))
            ax.fill_between(grid, *np.percentile(band, [2.5, 97.5], axis=0), color=color, alpha=0.15, lw=0)
            ax.plot(grid, 1 / (1 + np.exp(-(b0 + b1 * grid))), color=color, linewidth=2.2,
                    label=f"{judge} (n={len(x)})")
            lo, hi = sub.loc[judge, ["ci_lo", "ci_hi"]]
            ax.hlines(0.5, lo, hi, color=color, linewidth=3, zorder=4)
            ax.scatter([c], [0.5], s=60, color=color, edgecolor="white", linewidth=0.8, zorder=5)
        ax.set_xlim(grid[0], grid[-1])
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(pretty_region_name(region), fontsize=TITLE_SIZE - 3)
        ax.set_xlabel("SUVR", fontsize=LABEL_SIZE - 3)
        ax.set_ylabel("Probability of a positive call", fontsize=LABEL_SIZE - 3)
        ax.tick_params(labelsize=TICK_SIZE - 2)
        ax.grid(alpha=0.28)
    handles, labels = axes[0].get_legend_handles_labels()
    if spare:
        spare[0].legend(handles, labels, loc="center", fontsize=LEGEND_SIZE, frameon=False)
    else:
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=LEGEND_SIZE, frameon=False,
                   bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("SUVR at which the visual read and the model call positive", fontsize=TITLE_SIZE)
    finalize_figure(fig, rect=(0, 0, 1, 1.0))
    save_figure(fig, out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")


# ------------------------------
# Main
# ------------------------------
def load_test_predictions(run_dir: Path, dataset: str) -> pd.DataFrame | None:
    preds_path, _ = resolve_predictions_csv(run_dir, dataset)
    preds = pd.read_csv(preds_path)
    if "pred" not in preds.columns or "ID_ind" not in preds.columns:
        print(f"[skip] test-set rows: {preds_path.name} has no 'pred'/'ID_ind' column - is this a visual_read run?")
        return None
    preds = preds[["ID_ind", "pred"]].copy()
    preds["pred"] = pd.to_numeric(preds["pred"], errors="coerce")
    print(f"[load] {preds['pred'].notna().sum()} test-set predictions from {preds_path}")
    return preds.dropna()


def main():
    args = parse_args()
    demo_path = resolve_path(args.demo_csv, find_demo_csv)
    if demo_path is None:
        raise FileNotFoundError("No demographics csv found - pass --demo_csv.")
    demo = pd.read_csv(demo_path)
    demo = demo.loc[:, ~demo.columns.astype(str).str.startswith("Unnamed")]
    for c in REGIONS + ["visual_read", "amyloid_status"]:
        if c in demo.columns:
            demo[c] = pd.to_numeric(demo[c], errors="coerce")
    regions = [r for r in REGIONS if r in demo.columns and demo[r].notna().any()]
    print(f"[load] {len(demo)} scans from {demo_path}; regions: {', '.join(regions)}")
    if "ID" in demo.columns and "sID" in demo.columns and demo["sID"].duplicated().any():
        print(f"[WARNING] {demo['sID'].duplicated().sum()} repeat scans of the same person - CIs assume one scan per person.")

    is_ref = pd.Series(False, index=demo.index)
    if "dx" in demo.columns and "amyloid_status" in demo.columns:
        is_ref = (combine_dx_groups(demo["dx"]) == "CU").fillna(False) & (demo["amyloid_status"] == 0)
        print(f"[reference] {int(is_ref.sum())} cognitively unimpaired, amyloid-negative scans")
    else:
        print("[skip] reference group: needs 'dx' and 'amyloid_status' columns.")

    test = None
    if args.run_dir:
        preds = load_test_predictions(Path(args.run_dir).resolve(), args.dataset)
        if preds is not None:
            test = preds.merge(demo, left_on="ID_ind", right_on="ID", how="inner")
            missing = len(preds) - len(test)
            if missing:
                print(f"[WARNING] {missing}/{len(preds)} test scans have no row in {demo_path.name}.")
    else:
        print("[skip] test-set rows (visual read and model on the held-out set): pass --run_dir.")

    rng = np.random.default_rng(0)
    rows, gmm_rows, gmm_fits, curves = [], [], {}, {}

    def add(region, method, data, n, est, note=""):
        rows.append({"region": region, "method": method, "data": data, "n": n,
                     "cutoff": est[0], "ci_lo": est[1], "ci_hi": est[2], "note": note})

    for region in regions:
        print(f"[region] {pretty_region_name(region)}")
        curves[region] = {}
        if region in CENTAUR_FTP_CTRZ2_SUVR:
            add(region, METHODS[0], "published equation", np.nan, (CENTAUR_FTP_CTRZ2_SUVR[region], np.nan, np.nan))

        x_ref = demo.loc[is_ref, region].dropna().to_numpy(float)
        if len(x_ref) >= 10:
            add(region, METHODS[1], "whole dataset, CU Aβ−", len(x_ref), reference_cutoff(x_ref, rng))

        x_all = demo[region].dropna().to_numpy(float)
        est, fit, found = gmm_cutoff(x_all, rng)
        gmm_fits[region] = fit
        add(region, METHODS[2], "whole dataset, no labels", len(x_all), est,
            f"crossing found in {found:.0%} of bootstrap fits")
        gmm_rows.append({"region": region, "n": len(x_all),
                         "mean_low": fit["means"][0], "sd_low": fit["sds"][0], "weight_low": fit["weights"][0],
                         "mean_high": fit["means"][1], "sd_high": fit["sds"][1], "weight_high": fit["weights"][1],
                         "bic_1_curve": fit["bic_1"], "bic_2_curves": fit["bic_2"],
                         "two_curves_fit_better": fit["bic_2"] < fit["bic_1"],
                         "bootstrap_crossing_found": found, "converged": fit["converged"]})

        d = demo[[region, "visual_read"]].dropna()
        x = d[region].to_numpy(float)
        res = judge_cutoffs(x, {METHODS[3]: d["visual_read"].to_numpy(float)}, rng)
        add(region, METHODS[3], "whole dataset", len(x), res[METHODS[3]])
        curves[region][METHODS[3]] = (x, res["_fits"][METHODS[3]], res["_boot"][METHODS[3]])

        if test is not None:
            t = test[[region, "visual_read", "pred"]].dropna()
            xt = t[region].to_numpy(float)
            res = judge_cutoffs(xt, {METHODS[4]: t["visual_read"].to_numpy(float),
                                     METHODS[5]: t["pred"].to_numpy(float)}, rng)
            add(region, METHODS[4], "held-out test set", len(xt), res[METHODS[4]])
            add(region, METHODS[5], "held-out test set", len(xt), res[METHODS[5]])
            add(region, "Difference: Model − Visual read", "held-out test set, paired", len(xt), res["difference"])
            curves[region][METHODS[5]] = (xt, res["_fits"][METHODS[5]], res["_boot"][METHODS[5]])

    table = pd.DataFrame(rows)
    script_path = Path(__file__).resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else script_path.parents[2] / "results" / "suvr_cutoffs"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(out_dir / "suvr_cutoffs.csv", index=False)
    pd.DataFrame(gmm_rows).round(4).to_csv(out_dir / "suvr_cutoffs_gmm_details.csv", index=False)
    make_summary_figure(table, regions, out_dir / "suvr_cutoffs_summary.png")
    make_distribution_figure(demo, table, gmm_fits, regions, out_dir / "suvr_cutoffs_distributions.png")
    make_logistic_figure(curves, table, regions, out_dir / "suvr_cutoffs_logistic.png")

    print("\n[cut-offs] SUVR [95% CI]")
    for region in regions:
        print(f"  {pretty_region_name(region)}")
        for _, r in table[table["region"] == region].iterrows():
            ci = f" [{r['ci_lo']:.2f}-{r['ci_hi']:.2f}]" if np.isfinite(r["ci_lo"]) else ""
            n = f", n={int(r['n'])}" if np.isfinite(r["n"]) else ""
            note = f"  ({r['note']})" if r["note"] else ""
            print(f"    {r['method']:<32} {r['cutoff']:.2f}{ci}{n}{note}")
    print("\n[gaussian mixture] two curves fit better than one (lower BIC):")
    for g in gmm_rows:
        print(f"  {pretty_region_name(g['region'])}: {'yes' if g['two_curves_fit_better'] else 'no'}; "
              f"curves at {g['mean_low']:.2f} (weight {g['weight_low']:.0%}) and {g['mean_high']:.2f} "
              f"(weight {g['weight_high']:.0%})")
    print(f"\n[done] saved to {out_dir}")


if __name__ == "__main__":
    main()
