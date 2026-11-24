#!/usr/bin/env python3
"""What do the per-reader votes say about the model's disagreements with the visual read?

Each scan gets a reader-agreement group from the per-reader votes:
    unanimous     - 3 readers, all the same call (3-0)
    split 2-1     - 3 readers, one disagreed with the majority
    single reader - 1 reader (all NACC, part of ADNI) - agreement unknown, never counted as unanimous
Positive reads are also split by the readers' tau score (visual_read_3_cat: 1 = moderate, 2 = frequent;
both count as positive). Exploratory only: the binary read is the label (advice from the data provider).

Checks (all at the run's own `pred`, i.e. its 0.5 threshold):
  1. Disagreement by group: per read (- and +), scans, disagreements and median probability per
     agreement group and tau score. Fisher's exact tests: unanimous vs split 2-1; 3 readers vs 1 reader
     (all sites, and within ADNI only, since single-reader scans are otherwise all NACC = a site
     difference); moderate vs frequent (read + only, exploratory).
  2. Unanimous-only rescoring: agreement with read + / read - and AUC on all scans vs on scans with
     3 readers who all agreed.
  3. Model vs one human reader: for 3-reader scans, each reader is compared with the other two whenever
     those two agree; the model is compared with the same two-reader consensus on the same scan-reader
     pairs. Descriptive (pairs from one scan are not independent).
  4. Probability vs number of positive votes (3-reader scans): per 0-3 positive votes, median probability,
     % called positive by the model and % in the uncertain band (0.3-0.7, the pre-set confidence cut-off).
     Spearman of probability vs votes within each read; Fisher's exact test of the uncertain-band share in
     split vs unanimous scans.
  5. Case list: every scan the model disagrees with, with its votes, site, diagnosis, amyloid, CDR-SB and
     regional SUVRs from the demographics csv (has IDs - not for slides). Printed: counts per vote group
     with median MetaTemporal SUVR.
  6. Reader tau score at the same tau level (read + only, needs the demographics csv): is "moderate" just
     "less tau"? Logistic regression (statsmodels) of Model - on moderate vs frequent, alone and with
     MetaTemporal SUVR, as odds ratio [95% CI] with a likelihood-ratio p. If the odds ratio stays high at the
     same SUVR, the readers' "moderate" grade marks something the regional average misses.
Groups are small: read the counts first and the p-values second.

Usage:
    python results_plotting/analyze_reader_votes.py <run_dir> --votes IDs_reads.csv
    python results_plotting/analyze_reader_votes.py <run_dir> --votes AVIDIDs_reads.csv --dataset AVID_unseen
    python results_plotting/analyze_reader_votes.py <cv_run_dir> --votes IDs_reads.csv --oof
The demographics csv defaults to data/demo_<dataset>.csv, then data/demo.csv (in the repo or next to it).
<run_dir> uses the run's held-out test (or zero-shot validation) predictions; with --oof, a CV run's
out-of-fold predictions (every training-pool scan once, predicted by the fold model that did not train on it).
Relative run dirs are looked up under <proj_path>/results.

Output (default <proj_path>/results/reader_votes/<run>[_oof]_<dataset>/, outside the repo):
    1_disagreement_by_group.csv, 1_tests.csv, 2_unanimous_rescoring.csv, 3_model_vs_reader.csv,
    4_prob_by_votes.csv, 4_tests.csv, 6_grade_at_same_suvr.csv (no IDs)
    5_disagreement_cases.csv (has IDs - keep off slides)
    Figures (no IDs): fig_reader_grade.png (% Model - among read + scans, moderate vs frequent, Wilson 95% CI),
    fig_reader_grade_same_suvr.png (probability of Model - against MetaTemporal SUVR, one fitted curve per tau score;
    curves on top of each other = the grade adds nothing at the same tau level), fig_reader_agreement.png
    (disagreement by unanimous / split 2-1 / single reader, read + and read - panels)
"""
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2, fisher_exact, spearmanr
from sklearn.metrics import roc_auc_score

from plot_utils import CUSTOM_PALETTE, external_name, resolve_path, resolve_predictions_csv, wilson_interval

GROUP_ORDER = ["unanimous", "split 2-1", "single reader"]
READ_COLS = ["read_1", "read_2", "read_3"]
UNCERTAIN = (0.3, 0.7)
DEMO_COLS = ["site", "dx", "age", "amyloid_status", "CL", "CDR_SB",
             "MetaTemporal", "MesialTemporal", "TemporoParietal", "Frontal", "Universal"]


def parse_args():
    repo = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="Run dir (final / final-val model), or a CV run dir with --oof.")
    p.add_argument("--votes", required=True, help="Per-reader vote csv (ID, cohort, read_1..3, visual_read_binary, visual_read_3_cat).")
    p.add_argument("--dataset", default="Gothenburg", help="Dataset name used in the result filenames.")
    p.add_argument("--oof", action="store_true", help="Use the CV run's oof_predictions.csv.")
    p.add_argument("--demo_csv", default=None, help="Demographics csv for the case list (default: see above).")
    p.add_argument("--results_root", default=str(repo.parent / "results"))
    p.add_argument("--out_dir", default=str(repo.parent / "results" / "reader_votes"))
    return p.parse_args()


def load_votes(path: Path) -> pd.DataFrame:
    v = pd.read_csv(path)
    missing = {"ID", "cohort", "visual_read_binary", "visual_read_3_cat", *READ_COLS} - set(v.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    reads = v[READ_COLS]
    v["n_readers"] = reads.notna().sum(axis=1)
    if not v["n_readers"].isin([1, 3]).all():
        raise ValueError(f"{path}: expected 1 or 3 reads per scan, found {sorted(v['n_readers'].unique())}")
    v["positive_votes"] = (reads > 0).where(reads.notna()).sum(axis=1).astype(int)
    if ((v["positive_votes"] > v["n_readers"] / 2).astype(int) != v["visual_read_binary"]).any():
        raise ValueError(f"{path}: visual_read_binary is not the majority of the reads for some scans")
    unanimous = (v["positive_votes"] == 0) | (v["positive_votes"] == v["n_readers"])
    v["agreement"] = np.select([v["n_readers"] == 1, unanimous], ["single reader", "unanimous"], "split 2-1")
    return v[["ID", "cohort", "visual_read_binary", "visual_read_3_cat", *READ_COLS,
              "n_readers", "positive_votes", "agreement"]]


def find_dataset_demo_csv(dataset: str) -> Path | None:
    repo = Path(__file__).resolve().parent.parent
    candidates = [repo / "data" / f"demo_{dataset}.csv", repo.parent / "data" / f"demo_{dataset}.csv",
                  repo / "data" / "demo.csv", repo.parent / "data" / "demo.csv"]
    return next((p for p in candidates if p.exists()), None)


def load_preds(args) -> pd.DataFrame:
    run_dir = Path(args.run)
    if not (run_dir.is_absolute() or run_dir.exists()):
        run_dir = Path(args.results_root) / run_dir
    path = run_dir / "oof_predictions.csv" if args.oof else resolve_predictions_csv(run_dir, args.dataset)[0]
    d = pd.read_csv(path)[["ID_ind", "y", "prob", "pred"]]
    print(f"{path}: {len(d)} scans")
    return d


def fisher(sub: pd.DataFrame, col: str, a, b, outcome: str = "disagrees") -> tuple[str, float]:
    """Fisher's exact test of the rate of `outcome` between groups a and b of `col`."""
    x, y = sub[sub[col] == a][outcome], sub[sub[col] == b][outcome]
    if len(x) == 0 or len(y) == 0:
        return f"{a} vs {b}", np.nan
    table = [[x.sum(), len(x) - x.sum()], [y.sum(), len(y) - y.sum()]]
    return f"{a} vs {b}", fisher_exact(table)[1]


def pct(s: pd.Series) -> float:
    return round(100 * s.mean(), 1) if len(s) else np.nan


def disagreement_by_group(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for read, sub in df.groupby("read"):
        direction = "Model + / Read -" if read == "Read -" else "Model - / Read +"
        groups = [("agreement", g) for g in GROUP_ORDER]
        if read == "Read +":
            groups += [("read category", c) for c in ["moderate", "frequent"]]
        for col, g in groups:
            s = sub[sub[col] == g]
            if len(s):
                rows.append({"read": read, "group": g, "scans": len(s), direction: int(s["disagrees"].sum()),
                             "% disagree": pct(s["disagrees"]), "median prob": round(s["prob"].median(), 3)})
    table = pd.DataFrame(rows)
    for c in ["Model + / Read -", "Model - / Read +"]:
        if c in table:
            table[c] = table[c].astype("Int64").astype(object).where(table[c].notna(), "")

    tests = []
    for read, sub in df.groupby("read"):
        sub = sub.assign(readers=np.where(sub["n_readers"] == 1, "1 reader", "3 readers"))
        adni = sub[sub["cohort"] == "ADNI"]
        pairs = [("all", *fisher(sub, "agreement", "unanimous", "split 2-1")),
                 ("all", *fisher(sub, "readers", "3 readers", "1 reader")),
                 ("ADNI only", *fisher(adni, "readers", "3 readers", "1 reader"))]
        if read == "Read +":
            pairs.append(("all", *fisher(sub, "read category", "moderate", "frequent")))
        tests += [{"read": read, "scans": who, "test": t, "Fisher p": round(p, 4)} for who, t, p in pairs]
    return table, pd.DataFrame(tests)


def unanimous_rescoring(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, s in [("all scans", df), ("3 readers, unanimous", df[df["agreement"] == "unanimous"])]:
        auc = roc_auc_score(s["y"], s["prob"]) if s["y"].nunique() == 2 else np.nan
        rows.append({"scans": name, "n": len(s), "read +": int((s["y"] == 1).sum()),
                     "% agree, read +": pct(~s.loc[s["y"] == 1, "disagrees"]),
                     "% agree, read -": pct(~s.loc[s["y"] == 0, "disagrees"]),
                     "AUC": round(auc, 3)})
    return pd.DataFrame(rows)


def model_vs_reader(df: pd.DataFrame) -> pd.DataFrame:
    tri = df[df["n_readers"] == 3]
    votes = (tri[READ_COLS].to_numpy() > 0).astype(int)
    pred = tri["pred"].to_numpy()
    pairs = []
    for i in range(3):
        j, k = [x for x in range(3) if x != i]
        both = votes[:, j] == votes[:, k]
        consensus = votes[both, j]
        pairs.append(pd.DataFrame({"consensus": consensus, "reader agrees": votes[both, i] == consensus,
                                   "model agrees": pred[both] == consensus}))
    pairs = pd.concat(pairs)
    rows = []
    for name, s in [("other two say +", pairs[pairs["consensus"] == 1]),
                    ("other two say -", pairs[pairs["consensus"] == 0]), ("all", pairs)]:
        rows.append({"consensus": name, "scan-reader pairs": len(s),
                     "% one reader agrees": pct(s["reader agrees"]), "% model agrees": pct(s["model agrees"])})
    return pd.DataFrame(rows)


def prob_by_votes(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tri = df[df["n_readers"] == 3].assign(uncertain=lambda d: d["prob"].between(*UNCERTAIN))
    table = (tri.groupby("positive_votes")
             .agg(scans=("prob", "size"), median_prob=("prob", "median"),
                  pct_model_positive=("pred", lambda s: pct(s == 1)), pct_uncertain=("uncertain", pct))
             .round(3).reset_index().rename(columns={"positive_votes": "positive votes (of 3)"}))
    tests = []
    for read, sub in tri.groupby("read"):
        rho, p = spearmanr(sub["positive_votes"], sub["prob"]) if sub["positive_votes"].nunique() > 1 and sub["prob"].nunique() > 1 else (np.nan, np.nan)
        tests.append({"read": read, "test": "Spearman prob vs positive votes", "rho": round(rho, 3), "p": round(p, 4)})
    t, p = fisher(tri, "agreement", "split 2-1", "unanimous", outcome="uncertain")
    tests.append({"read": "both", "test": f"uncertain band share, {t} (Fisher)", "rho": np.nan, "p": round(p, 4)})
    return table, pd.DataFrame(tests)


def disagreement_cases(df: pd.DataFrame, demo: pd.DataFrame | None) -> pd.DataFrame:
    cases = df[df["disagrees"]].copy()
    cases["direction"] = np.where(cases["y"] == 0, "Model + / Read -", "Model - / Read +")
    cols = ["ID_ind", "direction", "prob", "cohort", "agreement", "positive_votes", "n_readers",
            "read category", *READ_COLS]
    cases = cases[cols]
    if demo is not None:
        demo_cols = [c for c in DEMO_COLS if c in demo.columns]
        cases = cases.merge(demo[["ID"] + demo_cols], left_on="ID_ind", right_on="ID", how="left").drop(columns="ID")
        n_missing = int(cases[demo_cols].isna().all(axis=1).sum())
        if n_missing:
            print(f"[WARNING] {n_missing} disagreement scans are not in the demographics csv.")
    return cases.sort_values(["direction", "agreement", "read category", "prob"])


def grade_at_same_suvr(df: pd.DataFrame, demo: pd.DataFrame | None) -> pd.DataFrame:
    """Read + scans: odds of Model - for moderate vs frequent tau score, alone and at the same MetaTemporal SUVR."""
    if demo is None or "MetaTemporal" not in demo.columns:
        return pd.DataFrame([{"adjusted_for": "-", "note": "no demographics csv with MetaTemporal"}])
    d = df[df["y"] == 1].merge(demo[["ID", "MetaTemporal"]].rename(columns={"ID": "_id"}),
                               left_on="ID_ind", right_on="_id", how="left")
    d = d.assign(model_neg=d["disagrees"].astype(int),
                 moderate=d["read category"].map({"moderate": 1.0, "frequent": 0.0}),
                 meta=pd.to_numeric(d["MetaTemporal"], errors="coerce")).dropna(subset=["moderate", "meta"])
    rows = []
    for label, adjust in [("None", []), ("MetaTemporal SUVR", ["meta"])]:
        row = {"adjusted_for": label, "n read +": len(d), "n moderate": int(d["moderate"].sum()),
               "Model - / Read +": int(d["model_neg"].sum()),
               "median MetaTemporal frequent / moderate":
                   f"{d.loc[d['moderate'] == 0, 'meta'].median():.2f} / {d.loc[d['moderate'] == 1, 'meta'].median():.2f}"}
        if d["model_neg"].sum() < 2 or d["moderate"].nunique() < 2:
            rows.append({**row, "note": "too few disagreements / one group only"})
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                full = smf.logit("model_neg ~ " + " + ".join(["moderate", *adjust]), data=d).fit(disp=0)
                reduced = smf.logit("model_neg ~ " + (" + ".join(adjust) or "1"), data=d).fit(disp=0)
        except Exception as err:  # e.g. perfect separation
            rows.append({**row, "note": f"could not fit ({type(err).__name__})"})
            continue
        lo, hi = full.conf_int().loc["moderate"]
        if not np.isfinite(np.exp(hi)) or abs(full.params["moderate"]) > np.log(1000):  # quasi-separation
            rows.append({**row, "note": "could not fit (one group (almost) never disagrees)"})
            continue
        rows.append({**row, "OR moderate vs frequent": round(np.exp(full.params["moderate"]), 2),
                     "OR lo": round(np.exp(lo), 2), "OR hi": round(np.exp(hi), 2),
                     "LR p": round(chi2.sf(2 * (full.llf - reduced.llf), 1), 4)})
    return pd.DataFrame(rows)


GRADE_COLOURS = {"frequent": CUSTOM_PALETTE[0], "moderate": CUSTOM_PALETTE[1]}


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.9")
    ax.set_axisbelow(True)


def _bars(ax, labels, ks, ns, colours, ylabel):
    """Bars of k/n with Wilson 95% CI and the count written above each bar."""
    rates = [k / n if n else np.nan for k, n in zip(ks, ns)]
    cis = [wilson_interval(k, n) if n else (np.nan, np.nan) for k, n in zip(ks, ns)]
    x = np.arange(len(labels))
    ax.bar(x, [100 * r for r in rates], color=colours, width=0.6)
    ax.errorbar(x, [100 * r for r in rates], yerr=[[100 * (r - lo) for r, (lo, _) in zip(rates, cis)],
                                                  [100 * (hi - r) for r, (_, hi) in zip(rates, cis)]],
                fmt="none", ecolor="0.25", capsize=4, lw=1)
    for xi, k, n, (_, hi) in zip(x, ks, ns, cis):
        if n:
            ax.text(xi, 100 * hi + 2, f"{k}/{n}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 112)
    ax.set_ylabel(ylabel)
    _clean(ax)


def make_figures(df: pd.DataFrame, demo: pd.DataFrame | None, set_name: str, out_dir: Path):
    rp = df[df["y"] == 1]
    # 1. tau score: % Model - among read + scans
    grades = ["frequent", "moderate"]
    fig, ax = plt.subplots(figsize=(4.2, 4.0), dpi=180)
    _bars(ax, [g.capitalize() for g in grades],
          [int(rp.loc[rp["read category"] == g, "disagrees"].sum()) for g in grades],
          [int((rp["read category"] == g).sum()) for g in grades],
          [GRADE_COLOURS[g] for g in grades], "Read + scans called negative by the model (%)")
    ax.set_xlabel("Readers' tau score")
    ax.set_title(f"Model − / Read + by readers' tau score\n{set_name}", fontsize=10)
    fig.tight_layout(); fig.savefig(out_dir / "fig_reader_grade.png"); plt.close(fig)

    # 2. the same at equal meta-temporal SUVR: one logistic model, Model - ~ moderate + SUVR, drawn per tau score
    if demo is not None and "MetaTemporal" in demo.columns:
        d = rp.merge(demo[["ID", "MetaTemporal"]].rename(columns={"ID": "_id"}), left_on="ID_ind", right_on="_id", how="left")
        d = d.assign(model_neg=d["disagrees"].astype(int), moderate=d["read category"].map({"moderate": 1.0, "frequent": 0.0}),
                     meta=pd.to_numeric(d["MetaTemporal"], errors="coerce")).dropna(subset=["moderate", "meta"])
        if d["model_neg"].sum() >= 2 and d["moderate"].nunique() == 2:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = smf.logit("model_neg ~ moderate + meta", data=d).fit(disp=0)
            except Exception:
                fit = None
            fig, ax = plt.subplots(figsize=(5.4, 4.0), dpi=180)
            rng = np.random.default_rng(0)
            for g, code in [("frequent", 0.0), ("moderate", 1.0)]:
                s = d[d["moderate"] == code]
                y = s["model_neg"] + rng.uniform(-0.03, 0.03, len(s)) + (0.02 if code else -0.02)
                ax.scatter(s["meta"], y, s=14, alpha=0.6, color=GRADE_COLOURS[g], edgecolor="white", lw=0.3)
                if fit is not None:
                    grid = np.linspace(s["meta"].min(), s["meta"].max(), 100)
                    ax.plot(grid, fit.predict(pd.DataFrame({"moderate": code, "meta": grid})), color=GRADE_COLOURS[g],
                            lw=2.2, label=f"{g.capitalize()} (n = {len(s)})")
            ax.set_xlabel("Meta-temporal SUVR")
            ax.set_ylabel("Probability the model calls the scan negative")
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_ylim(-0.08, 1.08)
            ax.legend(title="Readers' tau score", fontsize=8, title_fontsize=8, frameon=False, loc="upper right")
            grade = grade_at_same_suvr(df, demo)
            row = grade[grade["adjusted_for"] == "MetaTemporal SUVR"]
            if len(row) and "OR moderate vs frequent" in row and pd.notna(row.iloc[0].get("OR moderate vs frequent")):
                r = row.iloc[0]
                p = r["LR p"]
                ax.text(0.98, 0.55, f"At the same SUVR: odds ratio moderate vs frequent\n"
                        f"{r['OR moderate vs frequent']:.2f} [{r['OR lo']:.2f}, {r['OR hi']:.2f}], "
                        f"p {'< .001' if p < 0.001 else '= ' + f'{p:.2f}'.lstrip('0')}",
                        transform=ax.transAxes, ha="right", va="center", fontsize=7.5, color="0.25")
            ax.set_title(f"Read + scans: does the tau score matter at the same SUVR?\n{set_name}", fontsize=10)
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout(); fig.savefig(out_dir / "fig_reader_grade_same_suvr.png"); plt.close(fig)

    # 3. reader agreement: disagreement by unanimous / split 2-1 / single reader, per read
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.0), dpi=180, sharey=True)
    for ax, read, title in [(axes[0], 1, "Read + scans called negative"), (axes[1], 0, "Read − scans called positive")]:
        sub = df[df["y"] == read]
        groups = [g for g in GROUP_ORDER if (sub["agreement"] == g).any()]
        _bars(ax, [g.replace("split 2-1", "split 2–1").capitalize() for g in groups],
              [int(sub.loc[sub["agreement"] == g, "disagrees"].sum()) for g in groups],
              [int((sub["agreement"] == g).sum()) for g in groups],
              [CUSTOM_PALETTE[3]] * len(groups), "Scans where the model disagrees (%)" if read == 1 else "")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Readers")
    fig.suptitle(f"Model disagreement by reader agreement — {set_name}", fontsize=10)
    fig.tight_layout(); fig.savefig(out_dir / "fig_reader_agreement.png"); plt.close(fig)


def main():
    args = parse_args()
    preds, votes = load_preds(args), load_votes(Path(args.votes).expanduser())
    df = preds.merge(votes, left_on="ID_ind", right_on="ID", how="left")
    n_missing = int(df["agreement"].isna().sum())
    if n_missing:
        print(f"[WARNING] {n_missing} scans have no votes in {args.votes} - left out.")
        df = df.dropna(subset=["agreement"])
    bad = int((df["y"] != df["visual_read_binary"]).sum())
    if bad:
        raise ValueError(f"{bad} scans have a different visual read in the predictions and the vote file - "
                         "the IDs do not refer to the same scans.")
    df["disagrees"] = df["y"] != df["pred"]
    df["read"] = df["y"].map({0: "Read -", 1: "Read +"})
    df["read category"] = df["visual_read_3_cat"].map({0: "-", 1: "moderate", 2: "frequent"})

    out = {}
    out["1_disagreement_by_group"], out["1_tests"] = disagreement_by_group(df)
    out["2_unanimous_rescoring"] = unanimous_rescoring(df)
    out["3_model_vs_reader"] = model_vs_reader(df)
    out["4_prob_by_votes"], out["4_tests"] = prob_by_votes(df)
    demo_path = resolve_path(args.demo_csv, lambda: find_dataset_demo_csv(args.dataset))
    if demo_path is None:
        print("[WARNING] no demographics csv found - case list without site / SUVR (pass --demo_csv).")
    else:
        print(f"Demographics: {demo_path}")
    demo = pd.read_csv(demo_path) if demo_path else None
    cases = disagreement_cases(df, demo)
    out["6_grade_at_same_suvr"] = grade_at_same_suvr(df, demo)

    titles = {
        "1_disagreement_by_group": "1. Disagreement with the read, by reader agreement and tau score",
        "1_tests": "1. Disagreement rate, Fisher's exact test",
        "2_unanimous_rescoring": "2. All scans vs scans with 3 readers who all agreed",
        "3_model_vs_reader": "3. Agreement with the other two readers' consensus: one reader vs the model",
        "4_prob_by_votes": f"4. Model probability by number of positive votes (3-reader scans; uncertain = {UNCERTAIN[0]}-{UNCERTAIN[1]})",
        "4_tests": "4. Tests",
        "6_grade_at_same_suvr": "6. Read + only: Model - for moderate vs frequent tau score, alone and at the same MetaTemporal SUVR",
    }
    print(f"\n{args.dataset}{' (out-of-fold)' if args.oof else ''}: {len(df)} scans")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        for key, table in out.items():
            print(f"\n{titles[key]}:\n{table.to_string(index=False)}")
        print("\n5. Votes of the scans the model disagrees with:")
        g = cases.groupby(["direction", "agreement", "read category"])
        summary = g.size().rename("scans").to_frame()
        summary["median prob"] = g["prob"].median().round(3)
        if "MetaTemporal" in cases:
            summary["median MetaTemporal"] = g["MetaTemporal"].median().round(2)
        if "site" in cases:
            summary["sites"] = g["site"].agg(lambda s: ", ".join(f"{k} {v}" for k, v in s.value_counts().items()))
        print(summary.to_string())

    out_dir = Path(args.out_dir) / f"{Path(args.run).name}{'_oof' if args.oof else ''}_{args.dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, table in out.items():
        table.to_csv(out_dir / f"{key}.csv", index=False)
    cases.to_csv(out_dir / "5_disagreement_cases.csv", index=False)
    set_name = "Held-out test" if args.dataset == "Gothenburg" else external_name(args.dataset)
    make_figures(df, demo, set_name + (" (out-of-fold)" if args.oof else ""), out_dir)
    print(f"\nSaved to {out_dir} (5_disagreement_cases.csv has IDs - keep it off slides).")


if __name__ == "__main__":
    main()
