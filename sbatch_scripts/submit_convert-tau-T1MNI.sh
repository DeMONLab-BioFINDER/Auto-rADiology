#!/bin/bash
# SLURM batch job script for Berzelius HOPPER - submit from a berzelius-hopper login node.
# One-off data onboarding for the tau_T1MNI baseline: converts the shared tau_T1MNI
# pkl batches into the cached tau_batch_*.pt format run.py reads, two ways:
#
#   data/tau_T1MNI         - exactly the tau_raw recipe (CropForeground + resize to 128^3),
#                            so raw vs MNI differs only in the image space.
#   data/tau_T1MNI_iso2    - the tau_raw_iso2 recipe (2 mm, head mask, percentiles inside the
#                            head, 96x128x96) but NOT cropped: the 91x109x91 MNI grid is only
#                            padded by the same amount for every scan, so every voxel is the
#                            same brain location in every scan (needed for atlas-based
#                            occlusion / comparing maps). tau_raw_iso2 vs tau_T1MNI_iso2 then
#                            differ only in the registration to MNI.
#
# The pkls also hold the AVID scans (demo_AVID.csv), so AVID is converted too.
# Both reuse data/demo.csv: tau_T1MNI's pkl order has the same ID -> sID as tau_raw
# (checked 2026-10-21: 2279/2279 match), so with the same seed the train/test split is
# identical to the raw model's. tau_T1 is not converted: its image values are identical
# to tau_raw (only the header affine differs), so it would give the model the same input.
#
# Refuses to overwrite an output folder that already has .pt batches.
# A GPU is requested only because Berzelius jobs need one; the work is CPU.
#
# Usage:
#   sbatch submit_convert-tau-T1MNI.sh

#SBATCH -A berzelius-2026-231
#SBATCH --gpus=1
#SBATCH -t 04:00:00
#SBATCH -J convert-tau-T1MNI
#SBATCH -o ../../logs/convert-tau-T1MNI-slurm-%j.out
#SBATCH -e ../../logs/convert-tau-T1MNI-slurm-%j.err

# Load environment (Hopper module name)
module load Miniforge3/25.3.1-0
mamba activate ai-pet

# Determine project root based on current directory
if [[ "$(basename "$PWD")" == "sbatch_scripts" ]]; then
  cd .. || exit 1
elif [[ ! -d "sbatch_scripts" ]]; then
  echo "Error: Must be run from sbatch_scripts or Auto-rADiology root directory"
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
  python scripts/convert_tau_pkl_to_pt.py --pkl_dir "$PKL_DIR" --out_dir "$out_dir" --batch_size 50 "$@"
}

STATUS=0
convert "$DATA/tau_T1MNI" --image_shape 128 128 128 || STATUS=1
convert "$DATA/tau_T1MNI_iso2" --no-crop_foreground --head_crop --spacing_mm 2 --image_shape 96 128 96 --pct_inside_head || STATUS=1
exit $STATUS
