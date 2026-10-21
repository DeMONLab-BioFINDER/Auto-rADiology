#!/bin/bash
# SLURM batch job script for Berzelius HOPPER - submit from a berzelius-hopper login node.
# Reads only (no training, no GPU work): are the tau-PET images already SUVR images or raw
# activity values, per site? And which model-input value does a native 1.0 (= SUVR 1.0 if they
# are SUVR) become after the pipeline's percentile scaling? Decides whether an "SUVR 1.0"
# occlusion fill is possible on raw. See scripts/check_intensity_units.py.
# A GPU is requested only because Berzelius jobs need one.
#
# Usage:
#   sbatch submit_check-intensity-units.sh [max_scans_per_site]   (default 20, 0 = all)

#SBATCH -A berzelius-2026-231
#SBATCH --gpus=1
#SBATCH -t 01:00:00
#SBATCH -J check-intensity-units
#SBATCH -o ../../logs/check-intensity-units-slurm-%j.out
#SBATCH -e ../../logs/check-intensity-units-slurm-%j.err

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

# tau_raw_backup = your copy of the original tau_raw pkl batches (the model's training data);
# tau_T1MNI = the shared pkl batches, for comparison.
SHARED=/proj/berzelius-2024-156/dataset/tau-PET
python scripts/check_intensity_units.py \
  --input_path /proj/berzelius-2024-156/users/x_nadpi/data \
  --data_types tau_raw_backup,$SHARED/tau_T1MNI \
  --max_per_site "${1:-20}"
