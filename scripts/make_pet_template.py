"""Build a tau-PET template for MRI-free rigid alignment: the average of the MNI-space tau scans.

Each warped scan (wTAU_coregT1MNI, the tau_T1MNI pkls; all on the same 91x109x91 2 mm MNI grid) is
divided by the 99th percentile of its own non-zero voxels, then the scans are averaged voxel by
voxel. No labels are used. By default every scan in the pkls is averaged - that includes AVID and
the held-out test scans (the 11 Nov 2026 template, 2,497 scans). With --ids_csv only the listed
scans are used: give it a run's splits/train_subjects.csv so the template comes from training scans
only (no AVID, no test scans), so neither is "seen" by the alignment either.

The template is only used as a target for the head pose (scripts/convert_tau_pkl_to_pt.py
--rigid_template): rotation and shift, no scaling or warping, so the scans keep their own shape.

Outputs: <out>.nii.gz and <out>.png (three middle slices, to check it looks like a head).

Usage (Berzelius):
    python scripts/make_pet_template.py --pkl_dir /proj/berzelius-2024-156/dataset/tau-PET/tau_T1MNI \
        --out /proj/berzelius-2024-156/users/x_nadpi/data/templates/tau_pet_template_mni2mm
"""
import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd


def main(args):
    keep = None
    if args.ids_csv:  # ID = position of the scan in the pkl batches (same convention as demo.csv / run.py)
        ids = pd.read_csv(args.ids_csv)
        if args.exclude_sites:
            if "site" not in ids:
                sys.exit(f"--exclude_sites needs a site column in {args.ids_csv}")
            ids = ids[~ids["site"].isin(args.exclude_sites)]
        keep = set(ids["ID"].astype(int))
        print(f"Averaging only the {len(keep)} scans in {args.ids_csv}"
              f"{' minus sites ' + ', '.join(args.exclude_sites) if args.exclude_sites else ''}")
    pkl_files = sorted(Path(args.pkl_dir).glob("tau_batch_*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No tau_batch_*.pkl in {args.pkl_dir}")
    total, n, affine, shape = None, 0, None, None
    pos, used = -1, []
    for pkl in pkl_files:
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        for sample in data:
            pos += 1
            if keep is not None and pos not in keep:
                continue
            img = np.squeeze(np.asarray(sample["image"], dtype=np.float32))
            img = np.clip(np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0), 0, None)
            aff = np.asarray(sample["meta"]["affine"], dtype=float)
            if shape is None:
                shape, affine, total = img.shape, aff, np.zeros(img.shape, dtype=np.float64)
            if img.shape != shape or not np.allclose(aff, affine, atol=1e-3):
                print(f"Skipping a scan in {pkl.name}: grid {img.shape} differs from {shape}")
                continue
            v = img[img > 0]
            if not v.size:
                continue
            total += img / max(float(np.percentile(v, 99)), 1e-6)
            n += 1
            used.append(pos)
    if keep is not None and len(keep - set(used)):
        print(f"WARNING: {len(keep - set(used))} listed IDs were not found / not usable in the pkls")
    template = (total / max(n, 1)).astype(np.float32)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(template, affine), str(out) + ".nii.gz")
    Path(str(out) + "_ids.txt").write_text(" ".join(map(str, used)) + "\n")  # record of the scans averaged

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    cx, cy, cz = (s // 2 for s in template.shape)
    for ax, sl, title in zip(axes, (template[cx, :, :], template[:, cy, :], template[:, :, cz]),
                             ("Sagittal", "Coronal", "Axial")):
        ax.imshow(np.rot90(sl), cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"Tau-PET template, mean of {n} MNI scans" + (" (training scans only)" if keep is not None else ""))
    fig.tight_layout()
    fig.savefig(str(out) + ".png", dpi=120)
    print(f"Template from {n} scans: {out}.nii.gz (grid {shape}), picture {out}.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pkl_dir", required=True, help="Folder with the MNI-space tau_batch_*.pkl files (tau_T1MNI).")
    p.add_argument("--out", required=True, help="Output path without extension.")
    p.add_argument("--ids_csv", default="", help="Use only the scans in this CSV's ID column (e.g. a run's "
                                                 "splits/train_subjects.csv). Default: every scan.")
    p.add_argument("--exclude_sites", nargs="*", default=[], help="With --ids_csv: also leave out these sites "
                                                                   "(e.g. OASIS when it is the external set).")
    main(p.parse_args())
