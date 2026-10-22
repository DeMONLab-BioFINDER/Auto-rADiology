#!/bin/bash
# SLURM batch job script for Berzelius HOPPER - submit from a berzelius-hopper login node.
# One-off data onboarding for the MNI baseline: converts the shared tau_T1MNI pkl batches
# (/proj/.../dataset/tau-PET/tau_T1MNI, left untouched) into the cached tau_batch_*.pt format
# run.py reads, in data/tau_T1MNI_pts:
#
#   the MNI scans as they are: 91x109x91 at 2 mm (all scans share one SPM MNI header), no crop,
#   no stretch, no padding, no resampling (the CNN ends in adaptive pooling, so any size works).
#   Only the head mask (outside the head -> 0) and the tau_raw brightness scaling (1st-99th
#   percentile -> 0-1), as in tau_raw_2mm_pts. Every voxel is the same brain location in every
#   scan, so MNI atlas masks line up (stored in RAS order: flip a mask's x if loaded from an
#   SPM file without Orientation("RAS")). The "head touches the box" warning is expected here:
#   the MNI grid is tight around the head.
#
# The pkls also hold the AVID scans (demo_AVID.csv), so AVID is converted too.
# Reuses data/demo.csv: tau_T1MNI's pkl order has the same ID -> sID as tau_raw
# (checked 2026-10-21: 2279/2279 match), so with the same seed the train/test split is
# identical to the raw model's. Train/validate with --data_type tau_T1MNI_pts.
# tau_T1 is not converted: its image values are identical to tau_raw (only the header
# affine differs), so it would give the model the same input.
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

# --spacing_mm 2 equals the MNI grid, so nothing is resampled; centre_on_brain is skipped without crop_foreground
convert "$DATA/tau_T1MNI_pts" --no-crop_foreground --head_crop --spacing_mm 2 --image_shape 91 109 91
