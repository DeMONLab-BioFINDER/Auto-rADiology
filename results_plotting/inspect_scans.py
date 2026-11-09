#!/usr/bin/env python3
"""Look at a handful of scans by ID: everything known about them, and pictures.

For each ID:
  Info (printed and saved): site, diagnosis, age, sex, APOE, amyloid status and Centiloid,
      CDR / CDR-SB, visual read, the five regional SUVRs with their CenTauRz (CTRz) and whether
      that is above 2, and where the Meta-temporal SUVR sits among read + and read - scans
      (e.g. "lower than 85% of read + scans"). With --runs, each run's probability and call.
  Picture (one png per ID): the scan the model sees (the cached model input in
      <input_path>/<data_type>, tau_batch_*.pt), next to a "twin": the scan with the opposite
      visual read and the closest Meta-temporal SUVR (same site if one is within --twin_tol).
      Same amount of tau, different read, so what differs between the two rows is what the
      read reacted to. Rows: 5 axial slices from low to high, then a left and a right lateral
      sagittal slice (temporal and parietal cortex). Each scan is shown on its own model-input
      scale (0 to 1 after the percentile scaling), so brightness is relative within each scan.

The pngs and csv have IDs in them: for looking at, not for slides.

Usage (Berzelius, from the repo):
    python results_plotting/inspect_scans.py 54 235 16 \\
        --input_path /proj/berzelius-2024-156/users/x_nadpi/data --data_type tau_T1MNI_pts \\
        --runs <run_dir> [<run_dir> ...] --labels s42 s7
Outputs (default <proj_path>/results/scan_inspection/<data_type>, outside the repo).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from plot_utils import CENTAUR_FTP_EQUATIONS, find_demo_csv, resolve_path, resolve_predictions_csv

REGIONS = ["MetaTemporal", "TemporoParietal", "MesialTemporal", "Frontal", "Universal"]
INFO_COLS = ["site", "visual_read", "dx", "age", "gender", "apoe", "amyloid_status", "CL", "CDR", "CDR_SB"]
AXIAL_LEVELS = [0.30, 0.40, 0.50, 0.60, 0.70]  # share of the head's height, bottom -> top
SAGITTAL_LEVELS = [0.25, 0.75]                 # share of the head's width: left / right lateral


def parse_args():
    repo = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ids", nargs="+", type=int, help="Scan IDs (the ID column of the demographics csv).")
    p.add_argument("--input_path", default=str(repo / "data"), help="Folder that holds <data_type>.")
    p.add_argument("--data_type", default="tau_T1MNI_pts", help="Cached model inputs to show (e.g. tau_raw_2mm_pts).")
    p.add_argument("--dataset", default="Gothenburg", help="Dataset name used in the result filenames.")
    p.add_argument("--demo_csv", default=None, help="Demographics csv (default data/demo_<dataset>.csv, then data/demo.csv).")
    p.add_argument("--runs", nargs="*", default=[],
                   help="Run dirs whose test probabilities to add (relative ones are looked up under --results_root).")
    p.add_argument("--results_root", default=str(repo.parent / "results"))
    p.add_argument("--labels", nargs="*", default=None, help="One short name per run.")
    p.add_argument("--twin_tol", type=float, default=0.03,
                   help="Prefer a same-site twin if its Meta-temporal SUVR is within this of the scan's.")
    p.add_argument("--no_twin", action="store_true", help="Show only the scan itself.")
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()
    args.labels = args.labels or [f"run{i + 1}" for i in range(len(args.runs))]
    if len(args.labels) != len(args.runs):
        p.error("--labels needs one name per run")
    if args.out_dir is None:
        args.out_dir = str(repo.parent / "results" / "scan_inspection" / args.data_type)
    return args


def find_demo(args) -> Path:
    repo = Path(__file__).resolve().parent.parent
    dataset_csvs = [repo / "data" / f"demo_{args.dataset}.csv", repo.parent / "data" / f"demo_{args.dataset}.csv"]
    path = resolve_path(args.demo_csv, lambda: next((c for c in dataset_csvs if c.exists()), None) or find_demo_csv())
    if path is None:
        raise FileNotFoundError("No demographics csv found - pass --demo_csv.")
    return path


class ScanCache:
    """Reads model inputs from tau_batch_*.pt (scan ID // batch size = batch, remainder = index)."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        if not (data_dir / "tau_batch_000.pt").exists():
            raise FileNotFoundError(f"No tau_batch_000.pt in {data_dir}")
        manifest = data_dir / "tau_pt_manifest.json"
        self.batch_size = json.loads(manifest.read_text()).get("batch_size") if manifest.exists() else None
        self._batches = {}
        if self.batch_size is None:
            self.batch_size = self._batch(0).shape[0]

    def _batch(self, i: int):
        if i not in self._batches:
            self._batches[i] = torch.load(self.data_dir / f"tau_batch_{i:03d}.pt", map_location="cpu", weights_only=True)
        return self._batches[i]

    def get(self, scan_id: int) -> np.ndarray:
        x = self._batch(scan_id // self.batch_size)[scan_id % self.batch_size]
        x = x.numpy() if torch.is_tensor(x) else np.asarray(x)
        return x[0] if x.ndim == 4 else x  # [D, H, W] = right-left, back-front, bottom-top


def ctrz(region: str, suvr: float) -> float:
    slope, intercept = CENTAUR_FTP_EQUATIONS[region]
    return slope * suvr - intercept


def describe(demo: pd.DataFrame, scan_id: int, regions: list[str]) -> dict:
    row = demo.loc[demo["ID"] == scan_id].iloc[0]
    out = {"ID": scan_id, **{c: row[c] for c in INFO_COLS if c in demo.columns}}
    for r in regions:
        z = ctrz(r, row[r])
        out[f"{r} SUVR"] = round(float(row[r]), 3)
        out[f"{r} CTRz"] = f"{z:.1f}{' (>2)' if z > 2 else ''}"
    if "MetaTemporal" in regions:
        for read, name in [(1, "read +"), (0, "read -")]:
            others = demo.loc[demo["visual_read"] == read, "MetaTemporal"].dropna()
            out[f"Meta SUVR vs {name} scans"] = f"higher than {(others < row['MetaTemporal']).mean():.0%}"
    return out


def find_twin(demo: pd.DataFrame, scan_id: int, tol: float, exclude: set) -> int | None:
    row = demo.loc[demo["ID"] == scan_id].iloc[0]
    pool = demo[(demo["visual_read"] == 1 - row["visual_read"]) & ~demo["ID"].isin(exclude)].dropna(subset=["MetaTemporal"])
    if pool.empty:
        return None
    dist = (pool["MetaTemporal"] - row["MetaTemporal"]).abs()
    same_site = pool[(pool["site"] == row["site"]) & (dist <= tol)] if "site" in pool else pool.iloc[:0]
    pick = same_site if not same_site.empty else pool
    return int(pick.loc[dist[pick.index].idxmin(), "ID"])


def head_extent(v: np.ndarray, axis: int) -> tuple[int, int]:
    other = tuple(a for a in range(3) if a != axis)
    idx = np.nonzero((v > 0).any(axis=other))[0]
    return (int(idx.min()), int(idx.max())) if idx.size else (0, v.shape[axis] - 1)


def draw_row(axes, v: np.ndarray, title: str):
    lo_z, hi_z = head_extent(v, 2)
    lo_x, hi_x = head_extent(v, 0)
    panels = [(f"axial {int(f * 100)}%", v[:, :, int(lo_z + f * (hi_z - lo_z))].T) for f in AXIAL_LEVELS]
    panels += [(f"sagittal {'left' if f < 0.5 else 'right'}", v[int(lo_x + f * (hi_x - lo_x)), :, :].T)
               for f in SAGITTAL_LEVELS]
    for ax, (name, img) in zip(axes, panels):
        ax.imshow(img, cmap="inferno", origin="lower", vmin=0, vmax=1)
        ax.set_title(name, fontsize=8)
        ax.axis("off")
    axes[0].text(-0.08, 0.5, title, transform=axes[0].transAxes, rotation=90, ha="right", va="center", fontsize=8)


def label(info: dict, probs: dict) -> str:
    read = "Read +" if info.get("visual_read") == 1 else "Read -"
    bits = [f"ID {info['ID']}", read, str(info.get("site", "")), str(info.get("dx", "")),
            f"Meta SUVR {info.get('MetaTemporal SUVR', np.nan):.2f}"]
    if "amyloid_status" in info:
        bits.append("Aβ+" if info["amyloid_status"] == 1 else "Aβ−")
    text = " | ".join(bits[:3]) + "\n" + " | ".join(bits[3:])
    if probs:
        text += "\n" + ", ".join(f"{k} {v:.2f}" for k, v in probs.items())
    return text


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    demo = pd.read_csv(find_demo(args))
    demo = demo.loc[:, ~demo.columns.astype(str).str.startswith("Unnamed")]
    regions = [r for r in REGIONS if r in demo.columns]
    missing = [i for i in args.ids if i not in set(demo["ID"])]
    if missing:
        raise SystemExit(f"IDs not in the demographics csv: {missing}")

    probs = pd.DataFrame(index=demo["ID"])
    for run, lab in zip(args.runs, args.labels):
        run_dir = Path(run)
        if not (run_dir.is_absolute() or run_dir.exists()):
            run_dir = Path(args.results_root) / run_dir
        preds_path, _ = resolve_predictions_csv(run_dir, args.dataset)
        p = pd.read_csv(preds_path).set_index("ID_ind")
        probs[f"prob {lab}"] = p["prob"]
        probs[f"call {lab}"] = p["pred"].map({1: "+", 0: "-"})

    rows = []
    for scan_id in args.ids:
        info = describe(demo, scan_id, regions)
        for c in probs.columns:
            info[c] = probs.at[scan_id, c] if scan_id in probs.index else np.nan
        rows.append(info)
    table = pd.DataFrame(rows).set_index("ID")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 250):
        print(table.T.to_string())
    table.to_csv(out_dir / "scan_info.csv")

    cache = ScanCache(Path(args.input_path) / args.data_type)
    exclude = set(args.ids)
    for scan_id, info in zip(args.ids, rows):
        twin = None if args.no_twin else find_twin(demo, scan_id, args.twin_tol, exclude)
        scans = [(scan_id, info)] + ([(twin, describe(demo, twin, regions))] if twin is not None else [])
        fig, axes = plt.subplots(len(scans), len(AXIAL_LEVELS) + len(SAGITTAL_LEVELS),
                                 figsize=(15, 2.6 * len(scans) + 0.6), squeeze=False)
        for ax_row, (sid, inf) in zip(axes, scans):
            run_probs = {lab: probs.at[sid, f"prob {lab}"] for lab in args.labels
                         if sid in probs.index and pd.notna(probs.at[sid, f"prob {lab}"])}
            draw_row(ax_row, cache.get(sid), label(inf, run_probs))
        title = f"Scan {scan_id}" + (" (top) and a twin with the opposite read and similar Meta-temporal SUVR (bottom)"
                                     if twin is not None else "")
        fig.suptitle(f"{title}\n{args.data_type}, model input, brightness relative within each scan", fontsize=10)
        fig.tight_layout(rect=(0.03, 0, 1, 1))
        fig.savefig(out_dir / f"scan_{scan_id}.png", dpi=130)
        plt.close(fig)
        print(f"[save] {out_dir / f'scan_{scan_id}.png'}" + (f" (twin {twin})" if twin is not None else ""))


if __name__ == "__main__":
    main()
