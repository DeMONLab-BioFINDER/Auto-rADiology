#!/bin/bash
# SLURM batch job script for Berzelius HOPPER - submit from a berzelius-hopper login node.
# Reads only (no training, no GPU work): how blurry are the tau-PET images per site, and were
# some sites (e.g. ADNI) smoothed before we got them? Matters for whether to smooth before the
# voxelwise analysis, and as a possible site cue. See scripts/check_image_sharpness.py.
# tau_T1MNI is the main comparison (same grid and anatomy for every scan); tau_raw_backup next to it.
# A GPU is requested only because Berzelius jobs need one.
#
# Usage:
#   sbatch submit_check-image-sharpness.sh [max_scans_per_site]   (default 20, 0 = all)

#SBATCH -A berzelius-2026-231
#SBATCH --gpus=1
#SBATCH -t 01:00:00
#SBATCH -J check-image-sharpness
#SBATCH -o ../../logs/check-image-sharpness-slurm-%j.out
#SBATCH -e ../../logs/check-image-sharpness-slurm-%j.err

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

SHARED=/proj/berzelius-2024-156/dataset/tau-PET
python scripts/check_image_sharpness.py \
  --input_path /proj/berzelius-2024-156/users/x_nadpi/data \
  --data_types $SHARED/tau_T1MNI,tau_raw_backup \
  --max_per_site "${1:-20}"
