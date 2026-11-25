#!/usr/bin/env python3
"""Perturbation experiment A: does off-target signal in the middle of the brain push the model towards negative?

For every scan of a set, the striatum + thalamus signal (FreeSurfer aseg of the MNI template; classic flortaucipir
off-target) is set to a fixed level, keeping its texture (the region is multiplied by one factor, with soft edges),
and the frozen model is run again:
    typical   the median striatum + thalamus SUVR of the set's Model - / Read - scans
    high      the 90th percentile over all scans of the set
Each change is applied two ways:
    rescaled  the full preprocessing again, so the per-scan 1st-99th percentile -> 0-1 scaling is recomputed
              (what would really happen if the scan had that much off-target signal)
    fixed     the scan's original scaling kept
The difference rescaled - fixed is the part that works through the scaling (bright off-target areas setting the
top of the scale and squeezing the cortex); 'fixed' alone is what the network itself does with the signal.

--region chp: the same experiment on the choroid plexus stand-in instead (atlas_utils.choroid_plexus_frac: lateral
ventricle + temporal horn behind the foramen of Monro, grown 2 mm into white matter). Does the model use or discount
signal there - the off-target that readers may confuse with mesial-temporal tau? Soft edges = the voxel's fraction
inside the region (so nothing outside it, e.g. the hippocampus, is changed). Outputs go to perturb_chp_<set>/.

Built-in check: the unchanged scan through the same preprocessing and model must reproduce the model's saved
probabilities (printed; a warning if they do not).

Images: the original pkls (the same files the cached inputs were made from) and the cache's own recipe
(tau_pt_manifest.json), so the preprocessing is exactly the training preprocessing. SUVR for the levels =
whole-cerebellum reference (as the focality analysis).

Usage (Berzelius, GPU; see sbatch_scripts/submit_perturb-offtarget.sh):
    python run_perturb_offtarget.py --best_model_folder <run> --dataset Gothenburg --pool attention \\
        --data_type tau_T1MNI_pts --input_path <data> --pkl_dir <tau_T1MNI pkls> \\
        --ref_mask <masks>/voi_WhlCbl_2mm.nii [--region deep|chp] [--atlas_dir <lotter2025_mni_freesurfer>] [--out_dir <folder>]
Outputs (<run>/figures/perturb_offtarget_<set>/): perturb_by_group.csv, perturb_tests.csv, perturb_change.png
    (no IDs); perturb_per_scan.csv (HAS IDs - keep local)
"""
import argparse
import glob
import json
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "results_plotting"))

from src.warnings import ignore_warnings  # noqa: E402
ignore_warnings()
from monai.data import MetaTensor  # noqa: E402
from monai.transforms import Compose  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

from src.checkpoints import load_state_dict_checked  # noqa: E402
from src.model_factory import build_model_from_args, classifier_out_dim  # noqa: E402
from src.params import parse_arguments  # noqa: E402
from src.utils import get_device  # noqa: E402

GROUPS = ["Model + / Read +", "Model − / Read −", "Model − / Read + (confident)", "Model − / Read + (other)",
          "Model + / Read −"]
CONFIDENT = 0.1
DEEP_LABELS = [10, 11, 12, 13, 49, 50, 51, 52]  # FreeSurfer aseg: thalamus, caudate, putamen, pallidum
CONDITIONS = ["typical_rescaled", "typical_fixed", "high_rescaled", "high_fixed"]
# recipe of data/tau_T1MNI_pts (submit_convert-tau-T1MNI.sh), used if the manifest has no recipe
MNI_RECIPE = dict(image_shape=[91, 109, 91], pct_lo=1.0, pct_hi=99.0, crop_foreground=False, ras=True,
                  interp="trilinear", out_min=0.0, out_max=1.0, apply_brain_mask=True, head_crop=True, crop_margin=0,
                  spacing_mm=2.0, pct_inside_head=False)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--best_model_folder", required=True)
    p.add_argument("--dataset", default="Gothenburg", help="Gothenburg (held-out test) or AVID_unseen")
    p.add_argument("--pool", default="attention")
    p.add_argument("--data_type", default="tau_T1MNI_pts")
    p.add_argument("--input_path", required=True)
    p.add_argument("--pkl_dir", required=True)
    p.add_argument("--ref_mask", required=True, help="Whole-cerebellum mask (voi_WhlCbl_2mm.nii).")
    p.add_argument("--atlas_dir", default=None)
    p.add_argument("--region", default="deep", choices=["deep", "chp"],
                   help="deep = striatum + thalamus (default); chp = choroid plexus stand-in")
    p.add_argument("--max_scans", type=int, default=0, help="Only the first N scans (quick check).")
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


def load_model(a, device):
    argv = ["x", "--best_model_folder", a.best_model_folder, "--model", "CNN3D", "--data_type", a.data_type,
            "--input_path", a.input_path, "--targets", "visual_read", "--cls_loss", "softmax", "--dataset", a.dataset]
    if a.pool != "avg":
        argv += ["--model_kwargs", json.dumps({"pool": a.pool})]
    old, sys.argv = sys.argv, argv
    try:
        margs = parse_arguments()
    finally:
        sys.argv = old
    margs.device = device
    model = build_model_from_args(margs, device=device, n_classes=classifier_out_dim(2, "softmax"))
    ckpt_dir = Path(a.best_model_folder) / "final_model" / "checkpoints"
    ckpt = ckpt_dir / "final_model_last.pt"
    ckpt = ckpt if ckpt.exists() else ckpt_dir / "final_model_best.pt"
    sd = torch.load(ckpt, map_location=device, weights_only=True)
    load_state_dict_checked(model, sd.get("model", sd) if isinstance(sd, dict) else sd, str(ckpt))
    model.eval()
    print(f"[model] {ckpt}")
    return model


def predictions(a) -> pd.DataFrame:
    run = Path(a.best_model_folder)
    if a.dataset == "Gothenburg":
        f = run / "evaluation" / a.dataset / f"Eval_{a.dataset}_results.csv"
    else:
        hits = sorted(glob.glob(str(run / "validation" / f"External_validation_{a.dataset}__*_zeroshot_results.csv")))
        if not hits:
            raise SystemExit(f"No {a.dataset} zero-shot result in {run}/validation")
        f = Path(hits[0])
    d = pd.read_csv(f)[["ID_ind", "y", "pred", "prob"]].rename(columns={"ID_ind": "ID"})
    d["group"] = [GROUPS[0] if (y == 1 and c == 1) else GROUPS[1] if (y == 0 and c == 0) else
                  (GROUPS[2] if pr <= CONFIDENT else GROUPS[3]) if y == 1 else GROUPS[4]
                  for y, c, pr in zip(d["y"], d["pred"], d["prob"])]
    return d.set_index("ID")


def pipeline(a):
    """(pre, scale, post) of the cache's own recipe, split at the intensity scaling (as convert_tau_pkl_to_pt.py)."""
    from scripts.convert_tau_pkl_to_pt import _build_transform
    man = Path(a.input_path) / a.data_type / "tau_pt_manifest.json"
    recipe = json.loads(man.read_text()).get("recipe") if man.exists() else None
    if not recipe:
        print(f"[recipe] {man} has no recipe - using the tau_T1MNI_pts recipe of submit_convert-tau-T1MNI.sh")
        recipe = dict(MNI_RECIPE)
    print("[recipe] " + ", ".join(f"{k}={recipe[k]}" for k in MNI_RECIPE if k in recipe))
    return _build_transform(SimpleNamespace(**{**MNI_RECIPE, **recipe}))


REGION_NAME = {"deep": "striatum + thalamus", "chp": "choroid plexus region"}


def masks_on(shape, affine, a):
    """The region (--region; mask + soft weight 0..1) and whole-cerebellum mask on the pkl grid."""
    import nibabel as nib
    from nilearn import image
    from atlas_utils import DEFAULT_ATLAS_DIR
    from plot_fda_regions import FS_ASEG, fetch
    atlas = Path(a.atlas_dir) if a.atlas_dir else DEFAULT_ATLAS_DIR
    if not (atlas / FS_ASEG).exists():
        fetch(atlas)
    fs = nib.load(atlas / FS_ASEG)
    target = nib.Nifti1Image(np.zeros(shape, np.int8), affine)
    frac = image.resample_to_img(nib.Nifti1Image(np.isin(np.asarray(fs.dataobj).astype(int), DEEP_LABELS)
                                                 .astype(np.float32), fs.affine), target, interpolation="linear",
                                 force_resample=True, copy_header=True).get_fdata()
    deep = frac >= 0.5
    weight = np.clip(gaussian_filter(deep.astype(np.float32), 1.0), 0, 1)  # soft edges (about 2 mm)
    if a.region == "chp":
        from atlas_utils import choroid_plexus_frac
        frac = choroid_plexus_frac(shape, affine, a.atlas_dir)
        deep, weight = frac >= 0.5, np.clip(frac, 0, 1).astype(np.float32)  # soft edges stay inside the region
    cbl = image.resample_to_img(nib.load(a.ref_mask), target, interpolation="nearest", force_resample=True,
                                copy_header=True).get_fdata() > 0
    print(f"[masks] {REGION_NAME[a.region]} {int(deep.sum())} voxels, cerebellum {int(cbl.sum())} voxels")
    return deep, weight, cbl


def scans(a, ids):
    """(ID, raw image, meta) from the pkls in the same order as the cache (ID = batch * batch length + index)."""
    batch_len = None
    for f in sorted(Path(a.pkl_dir).glob("tau_batch_*.pkl"), key=lambda p: int(p.stem.split("_")[-1])):
        b = int(f.stem.split("_")[-1])
        if batch_len is not None and not any(b * batch_len <= i < (b + 1) * batch_len for i in ids):
            continue
        with open(f, "rb") as fh:
            data = pickle.load(fh)
        batch_len = batch_len or len(data)
        for idx, s in enumerate(data):
            sid = b * batch_len + idx
            if sid in ids:
                # exactly as convert_tau_pkl_to_pt.py (not squeezed: EnsureChannelFirst reads the meta's channel flag)
                img = np.clip(np.nan_to_num(np.asarray(s["image"], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0),
                              0, None)
                meta = dict(s.get("meta") or {})
                meta.setdefault("original_channel_dim", 0 if img.ndim == 4 else "no_channel")
                yield sid, img, meta


def run_model(model, x, device) -> float:
    """Log-odds of positive (logit(pos) - logit(neg)) for one preprocessed volume (1, D, H, W)."""
    t = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    t = t.reshape(1, 1, *t.shape[-3:]).to(device)
    with torch.no_grad():
        out = model(t, extra=None)
    return float(out[0, 1] - out[0, 0])


def main():
    a = parse_args()
    device = get_device()
    set_tag = "test" if a.dataset == "Gothenburg" else a.dataset.removesuffix("_unseen").lower()
    out = Path(a.out_dir) if a.out_dir else Path(a.best_model_folder) / "figures" / f"perturb_{'offtarget' if a.region == 'deep' else 'chp'}_{set_tag}"
    out.mkdir(parents=True, exist_ok=True)
    preds = predictions(a)
    if a.max_scans:
        preds = preds.iloc[:a.max_scans]
    model = load_model(a, device)
    pre, scale, post = pipeline(a)
    post_rest = Compose(post.transforms[1:])  # what comes after the scaling
    lo_pct, hi_pct = float(scale.scaler.lower if hasattr(scale, "scaler") else scale.lower), \
        float(scale.scaler.upper if hasattr(scale, "scaler") else scale.upper)
    b_min, b_max = float(scale.b_min), float(scale.b_max)

    # pass 1: every scan's striatum + thalamus SUVR -> the two target levels
    deep = weight = cbl = None
    raw = {}
    for sid, img, meta in scans(a, set(preds.index)):
        vol = np.squeeze(img)  # the 3D volume, for the masks and SUVR
        if deep is None:
            deep, weight, cbl = masks_on(vol.shape, np.asarray(meta["affine"], dtype=float), a)
            weight_img = weight.reshape(img.shape)  # same shape as the image as stored
        ref_v = vol[cbl & (vol > 0)]
        if ref_v.size < 0.5 * cbl.sum():
            print(f"[skip] {sid}: cerebellum not covered")
            continue
        ref = float(ref_v.mean())
        raw[sid] = (img, meta, ref, float(vol[deep].mean()) / ref)
    deep_suvr = pd.Series({k: v[3] for k, v in raw.items()})
    neg = [k for k in deep_suvr.index if preds.loc[k, "group"] == GROUPS[1]]
    target = {"typical": float(deep_suvr[neg].median()), "high": float(deep_suvr.quantile(0.9))}
    print(f"[levels] {REGION_NAME[a.region]} SUVR: typical (median Model − / Read −) {target['typical']:.3f}, "
          f"high (90th percentile) {target['high']:.3f}")

    # pass 2: perturb and run the model
    rows = []
    for k, (sid, (img, meta, ref, d_suvr)) in enumerate(raw.items()):
        x_pre = pre(MetaTensor(torch.as_tensor(img), meta=meta))
        x0 = post(x_pre)
        lo, hi = np.percentile(np.asarray(x_pre), [lo_pct, hi_pct])
        row = {"ID": sid, "group": preds.loc[sid, "group"], "read": int(preds.loc[sid, "y"]),
               "saved_prob": float(preds.loc[sid, "prob"]), "deep_suvr": d_suvr, "scale_hi": float(hi),
               "original": run_model(model, x0, device)}
        for level, tgt in target.items():
            f = tgt / d_suvr
            img_mod = img * (1 + (f - 1) * weight_img)
            xp = pre(MetaTensor(torch.as_tensor(img_mod.astype(np.float32)), meta=meta))
            row[f"{level}_rescaled"] = run_model(model, post(xp), device)
            fixed = (np.asarray(xp) - lo) / (hi - lo) * (b_max - b_min) + b_min
            fixed = MetaTensor(torch.as_tensor(np.clip(fixed, b_min, b_max), dtype=torch.float32), meta=xp.meta)
            row[f"{level}_fixed"] = run_model(model, post_rest(fixed), device)
            row[f"{level}_scale_hi"] = float(np.percentile(np.asarray(xp), hi_pct))
        rows.append(row)
        if k % 50 == 0:
            print(f"[run] {k + 1} / {len(raw)} scans")
    d = pd.DataFrame(rows)
    d["original_prob"] = 1 / (1 + np.exp(-d["original"]))
    for c in CONDITIONS:
        d[f"change_{c}"] = d[c] - d["original"]
        d[f"prob_{c}"] = 1 / (1 + np.exp(-d[c]))
        d[f"flip_{c}"] = (d[f"prob_{c}"] >= 0.5) != (d["original_prob"] >= 0.5)

    # built-in check: does the unchanged scan reproduce the saved probabilities?
    diff = (d["original_prob"] - d["saved_prob"]).abs()
    print(f"\n[check] unchanged scans vs the model's saved probabilities: median |difference| {diff.median():.4f}, "
          f"max {diff.max():.4f}; same call in {int(((d['original_prob'] >= .5) == (d['saved_prob'] >= .5)).sum())} "
          f"of {len(d)}")
    if diff.median() > 0.01:
        print("[check] WARNING: the probabilities do not match - wrong model settings, recipe or IDs; "
              "do not trust the results below")

    d.round(5).to_csv(out / "perturb_per_scan.csv", index=False)
    summ = []
    for g in GROUPS:
        s = d[d["group"] == g]
        if s.empty:
            continue
        r = {"group": g, "n": len(s), "deep_suvr": s["deep_suvr"].median(), "original_prob": s["original_prob"].median()}
        for c in CONDITIONS:
            r[f"median_change_{c}"] = s[f"change_{c}"].median()
            r[f"flips_{c}"] = int(s[f"flip_{c}"].sum())
        summ.append(r)
    summ = pd.DataFrame(summ)
    summ.round(4).to_csv(out / "perturb_by_group.csv", index=False)
    tests = []
    for g in GROUPS[:4] + ["Model − / Read + (all)"]:
        s = d[d["group"].str.startswith("Model − / Read +")] if g.endswith("(all)") else d[d["group"] == g]
        for c in CONDITIONS:
            if len(s) >= 5:
                tests.append({"group": g, "condition": c, "n": len(s), "median_change": s[f"change_{c}"].median(),
                              "p_wilcoxon": stats.wilcoxon(s[f"change_{c}"]).pvalue})
        if len(s) >= 5:  # rescaled vs fixed: the part through the scaling
            for level in ("typical", "high"):
                dd = s[f"change_{level}_rescaled"] - s[f"change_{level}_fixed"]
                tests.append({"group": g, "condition": f"{level}: rescaled − fixed (scaling part)", "n": len(s),
                              "median_change": dd.median(), "p_wilcoxon": stats.wilcoxon(dd).pvalue})
    tests = pd.DataFrame(tests)
    tests.round(4).to_csv(out / "perturb_tests.csv", index=False)
    figure(d, target, set_tag, out / "perturb_change.png", REGION_NAME[a.region])
    with pd.option_context("display.width", 230, "display.max_columns", None):
        print("\nPer group (change = log-odds of positive after - before; flips = calls that change at 0.5):\n"
              + summ.round(3).to_string(index=False))
        print("\nPaired tests (Wilcoxon signed-rank):\n" + tests.round(4).to_string(index=False))
    print(f"\nSaved to {out} (perturb_per_scan.csv has IDs - keep it local)")


def figure(d: pd.DataFrame, target: dict, set_tag: str, out: Path, region: str = "striatum + thalamus"):
    """Change in log-odds when the region is set to the typical level, per group, rescaled vs fixed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_utils import CUSTOM_PALETTE
    groups = [g for g in GROUPS[:4] if (d["group"] == g).any()]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    rng = np.random.default_rng(0)
    for ax, level in zip(axes, ("typical", "high")):
        for i, g in enumerate(groups):
            s = d[d["group"] == g]
            for j, (mode, col) in enumerate((("rescaled", CUSTOM_PALETTE[1]), ("fixed", CUSTOM_PALETTE[0]))):
                v = s[f"change_{level}_{mode}"]
                xs = i + (j - 0.5) * 0.36 + rng.uniform(-0.1, 0.1, len(v))
                ax.scatter(xs, v, s=12 if len(v) > 50 else 28, alpha=0.5, color=col, edgecolor="none",
                           label=("Full preprocessing (scaling recomputed)" if mode == "rescaled"
                                  else "Original scaling kept") if i == 0 else None)
                ax.hlines(v.median(), i + (j - 0.5) * 0.36 - 0.14, i + (j - 0.5) * 0.36 + 0.14, color="0.1", lw=2)
        ax.axhline(0, color="0.5", ls="--", lw=1)
        ax.set_xticks(range(len(groups)), [g.replace(" (", "\n(") for g in groups], fontsize=10)
        ax.set_title(f"{region[0].upper() + region[1:]} set to {level} (SUVR {target[level]:.2f})", fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Change in the model's log-odds of positive", fontsize=11)
    axes[0].legend(frameon=False, fontsize=10, loc="upper left")
    fig.suptitle(f"Changing the off-target signal, {'held-out test' if set_tag == 'test' else set_tag.upper()}",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
