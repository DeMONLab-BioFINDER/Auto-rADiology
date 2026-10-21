#!/bin/bash
# SLURM batch job script for Berzelius HOPPER - submit from a berzelius-hopper login node.
# Does the old head mask let background through at some sites (e.g. Geneva, biggest field of
# view)? Checks the real model inputs in data/tau_raw (what the models trained on) and replays
# the old and new (iso2) recipes on the native pkls; see scripts/check_head_mask.py.
# Reads only. A GPU is requested only because Berzelius jobs need one; the work is CPU.
#
# Usage:
#   sbatch submit_check-head-mask.sh [scans_per_site=15] [line_mm=120]
#   line_mm = how far below the top of the head counts as "low" (neck) in the brightness check

#SBATCH -A berzelius-2026-231
#SBATCH --gpus=1
#SBATCH -t 01:00:00
#SBATCH -J check-head-mask
#SBATCH -o ../../logs/check-head-mask-slurm-%j.out
#SBATCH -e ../../logs/check-head-mask-slurm-%j.err

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

python scripts/check_head_mask.py --input_path /proj/berzelius-2024-156/users/x_nadpi/data \
  --cache_type tau_raw --pkl_type tau_raw_backup --max_per_site "${1:-15}" --line_mm "${2:-120}"
