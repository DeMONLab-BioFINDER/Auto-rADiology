#!/bin/bash
# SLURM batch job script for Berzelius (Hopper or Ampere login node).
# How much would resampling into the header's frame rotate each scan? Compares tau_raw_backup
# (the raw pkls) with the shared tau_T1 pkls (same voxels, header = rigid coregistration to the
# person's T1). If the T1 headers barely rotate the scans, a "T1 2 mm" training run would give
# the model almost the same input as tau_raw_2mm_pts and isn't worth doing.
# See scripts/check_t1_rotation.py. Reads only; a GPU is requested only because Berzelius jobs need one.
#
# Output: <proj_path>/results/t1_rotation/ (rotation_<folder>.csv per scan, rotation_<folder>_by_site.csv)
#
# Usage:
#   sbatch submit_check-t1-rotation.sh

#SBATCH -A berzelius-2026-231
#SBATCH --gpus=1
#SBATCH -t 02:00:00
#SBATCH -J check-t1-rotation
#SBATCH -o ../../logs/check-t1-rotation-slurm-%j.out
#SBATCH -e ../../logs/check-t1-rotation-slurm-%j.err

module load Miniforge3/25.3.1-0 2>/dev/null || module load Miniforge3/24.7.1-2-hpc1-bdist  # Hopper name, else Ampere
mamba activate ai-pet

# Determine project root based on current directory
if [[ "$(basename "$PWD")" == "sbatch_scripts" ]]; then
  cd .. || exit 1
elif [[ ! -d "sbatch_scripts" ]]; then
  echo "Error: Must be run from sbatch_scripts or Auto-rADiology root directory"
  exit 1
fi

SHARED=/proj/berzelius-2024-156/dataset/tau-PET
python scripts/check_t1_rotation.py \
  --input_path /proj/berzelius-2024-156/users/x_nadpi/data \
  --data_types tau_raw_backup,$SHARED/tau_T1
