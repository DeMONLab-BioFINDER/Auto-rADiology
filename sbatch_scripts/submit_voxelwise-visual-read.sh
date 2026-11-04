#!/bin/bash
# SLURM batch job script for Berzelius HOPPER - submit from a berzelius-hopper login node.
# Reads only (no training, no GPU work): voxelwise average Read + scan, average Read - scan and their
# difference, on the MNI model's inputs (all scans with a visual read), as slices and on the cortical
# surface. See results_plotting/voxelwise_visual_read.py. Needs nilearn in ai-pet.
# A GPU is requested only because Berzelius jobs need one.
#
# Usage:
#   sbatch submit_voxelwise-visual-read.sh [smooth_fwhm_mm]   (default 0 = unsmoothed)

#SBATCH -A berzelius-2026-231
#SBATCH --gpus=1
#SBATCH -t 01:00:00
#SBATCH -J voxelwise-visual-read
#SBATCH -o ../../logs/voxelwise-visual-read-slurm-%j.out
#SBATCH -e ../../logs/voxelwise-visual-read-slurm-%j.err

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

python results_plotting/voxelwise_visual_read.py \
  --input_path /proj/berzelius-2024-156/users/x_nadpi/data \
  --data_type tau_T1MNI_pts \
  --smooth_fwhm "${1:-0}"
