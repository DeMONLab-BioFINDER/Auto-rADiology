#!/bin/bash
# SLURM batch job script for Berzelius HOPPER - submit from a berzelius-hopper login node.
# Draws the FDA flortaucipir visual-read regions (TAUVID label) on the Desikan-Killiany atlas in MNI space, with the
# label's temporal crosshairs: axial slices, a NIfTI region map and an interactive 3D view.
# See results_plotting/plot_fda_regions.py. Needs nilearn (and plotly for the 3D view) in ai-pet.
# Runs on the CPU-only partition (no GPU needed).
#
# The first run downloads the atlas (about 2 MB) into <proj_path>/data/atlases/. Compute nodes may have no
# internet: if the download fails, run the python line below once on the login node, then submit.
#
# Usage:
#   sbatch submit_fda-regions.sh

#SBATCH -A berzelius-2026-231
#SBATCH -p berzelius-hopper-cpu
#SBATCH -c 4
#SBATCH -t 00:30:00
#SBATCH -J fda-regions
#SBATCH -o ../../logs/fda-regions-slurm-%j.out
#SBATCH -e ../../logs/fda-regions-slurm-%j.err

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

python results_plotting/plot_fda_regions.py
