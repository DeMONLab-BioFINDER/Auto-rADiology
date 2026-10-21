#!/bin/bash
# SLURM batch job script for Berzelius HOPPER - submit from a berzelius-hopper login node.
# Reads only (no training, no GPU work): for tau_raw / tau_T1 / tau_T1MNI, prints each
# site's native image shape and voxel size, and the physical size (mm) of one voxel of
# the 128^3 model input, plus what occlusion masks of 4/8/16 voxels are in mm.
# See scripts/check_image_geometry.py. A GPU is requested only because Berzelius jobs need one.
#
# Usage:
#   sbatch submit_check-image-geometry.sh

#SBATCH -A berzelius-2026-231
#SBATCH --gpus=1
#SBATCH -t 01:00:00
#SBATCH -J check-image-geometry
#SBATCH -o ../../logs/check-image-geometry-slurm-%j.out
#SBATCH -e ../../logs/check-image-geometry-slurm-%j.err

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

python scripts/check_image_geometry.py \
  --input_path /proj/berzelius-2024-156/users/x_nadpi/data \
  --data_types tau_raw,tau_T1,tau_T1MNI
