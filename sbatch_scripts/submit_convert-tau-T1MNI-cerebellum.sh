#!/bin/bash
# SLURM batch job script for Berzelius HOPPER - submit from a berzelius-hopper login node.
# One-off data onboarding: the MNI scans scaled relative to the cerebellum, as for the visual read,
# instead of each scan's own 1st-99th percentile. Same pkls and recipe as submit_convert-tau-T1MNI.sh
# (91x109x91 MNI grid, head mask, no crop), only the brightness scaling differs:
#
#   data/tau_T1MNI_cbl_pts   each scan divided by its whole-cerebellum mean (SUVR), then one fixed window
#                            for all scans, SUVR 0-5.89 -> 0-1, so 1.65 x cerebellum sits at 0.28.
#
# Where 5.89 comes from: FDA TAUVID label, UCV = (MCC x 1.65) x (100% / % level of colour transition),
# transition anywhere in 25-60%. The scale the labels were read on (Moscoso 2025 supplement, example figures
# of their automated pipeline, pp. 19 and 21) has the transition at 28% of the colour bar, so the top is
# 1.65 / 0.28 = 5.89 x cerebellum. The readers' colour bands are not copied (no new information for a CNN).
#
# The cerebellum mask is the Centiloid whole-cerebellum VOI (voi_WhlCbl_2mm.nii from GAAIN's Centiloid
# project, MNI 2 mm); the conversion stops if it is not on the scans' grid. Eroded by 1 voxel.
# Units differ by site (ADNI/NACC arrive as SUVR, the others as raw activity); dividing by the
# cerebellum makes them the same. Check at the end, per site: ADNI/NACC cerebellum means should be
# near 1 (their SUVR used the inferior cerebellar cortex, so not exactly 1), and coverage near 1
# (lower = cerebellum partly outside the scanner's field of view).
# The pkls also hold AVID, so AVID is converted too. Same demo.csv/IDs, so the same split and folds
# as tau_T1MNI_pts: compare CV runs scan by scan with results_plotting/compare_cv_runs_oof.py.
#
# Refuses to overwrite an output folder that already has .pt batches.
# A GPU is requested only because Berzelius jobs need one; the work is CPU.
#
# Usage:
#   sbatch submit_convert-tau-T1MNI-cerebellum.sh <path to voi_WhlCbl_2mm.nii>
# then e.g. sbatch submit_visual-read-cv5.sh tau_T1MNI_cbl_pts

#SBATCH -A berzelius-2026-231
#SBATCH --gpus=1
#SBATCH -t 04:00:00
#SBATCH -J convert-tau-T1MNI-cbl
#SBATCH -o ../../logs/convert-tau-T1MNI-cbl-slurm-%j.out
#SBATCH -e ../../logs/convert-tau-T1MNI-cbl-slurm-%j.err

# Load environment (Hopper module name)
module load Miniforge3/25.3.1-0
mamba activate ai-pet

# The mask path may be relative to where sbatch was run: make it absolute before changing directory
MASK="${1:+$(realpath -m "$1")}"

# Determine project root based on current directory
if [[ "$(basename "$PWD")" == "sbatch_scripts" ]]; then
  cd .. || exit 1
elif [[ ! -d "sbatch_scripts" ]]; then
  echo "Error: Must be run from sbatch_scripts or Auto-rADiology root directory"
  exit 1
fi

if [[ -z "$MASK" || ! -f "$MASK" ]]; then
  echo "Error: give the whole-cerebellum mask (voi_WhlCbl_2mm.nii) as the first argument"
  exit 1
fi

PKL_DIR=/proj/berzelius-2024-156/dataset/tau-PET/tau_T1MNI
DATA=/proj/berzelius-2024-156/users/x_nadpi/data

convert() {
  local out_dir="$1"; shift
  if compgen -G "$out_dir/tau_batch_*.pt" > /dev/null; then
    echo "Skipping $out_dir: it already has tau_batch_*.pt files (delete them first to redo)."
    return 0
  fi
  echo "===== $out_dir ====="
  python scripts/convert_tau_pkl_to_pt.py --pkl_dir "$PKL_DIR" --out_dir "$out_dir" --batch_size 50 \
    --no-crop_foreground --head_crop --spacing_mm 2 --image_shape 91 109 91 \
    --reference_mask "$MASK" --mask_erode 1 "$@" || exit 1
}

convert "$DATA/tau_T1MNI_cbl_pts" --suvr_max 5.89

# Per site: cerebellum mean in native units and how much of the mask the scan covers (ID = row in the batches)
python - "$DATA" <<'EOF'
import sys
from pathlib import Path
import pandas as pd
data = Path(sys.argv[1])
scans = pd.read_csv(data / "tau_T1MNI_cbl_pts" / "tau_pt_scans.csv")
demo = pd.read_csv(data / "demo.csv")[["ID", "site"]]
d = scans.merge(demo, left_on="index", right_on="ID", how="left")
d["site"] = d["site"].fillna("not in demo.csv (AVID)")
print("\nCerebellum mean (native units) and mask coverage per site:")
print(d.groupby("site").agg(n=("index", "size"),
                            cerebellum_median=("reference_mean_native", "median"),
                            coverage_median=("reference_coverage", "median"),
                            coverage_below_0_8=("reference_coverage", lambda c: int((c < 0.8).sum()))).round(3).to_string())
EOF
