#!/bin/bash
# SLURM batch job script for Berzelius HOPPER - submit from a berzelius-hopper login node.
# Reads only (no training, no GPU work): voxelwise average Read + scan, average Read - scan and their
# difference, on the MNI model's inputs (all scans with a visual read), as slices and on the cortical
# surface. See results_plotting/voxelwise_visual_read.py. Needs nilearn in ai-pet.
# Without a mask: on the MNI model's inputs (0-1 per scan, what the model sees).
# With a reference mask: SUVR from the original MNI images (shared tau_T1MNI pkls), each scan divided by its
# mean in the mask. Karagianni 2026 / CenTauR use the inferior cerebellar grey matter: voi_CerebGry_tau_2mm.nii.
# Runs on the CPU-only partition (berzelius-hopper-cpu, no GPU): the script does its maths in numpy, and the
# CPU nodes usually have a much shorter queue than the GPU nodes.
#
# Usage:
#   sbatch submit_voxelwise-visual-read.sh [smooth_fwhm_mm] [reference_mask]   (default 0 = unsmoothed, no mask)
#   e.g. sbatch submit_voxelwise-visual-read.sh 0 /proj/berzelius-2024-156/users/x_nadpi/data/masks/voi_CerebGry_tau_2mm.nii

#SBATCH -A berzelius-2026-231
#SBATCH -p berzelius-hopper-cpu
#SBATCH -c 8
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

if [[ -n "$2" ]]; then
  python results_plotting/voxelwise_visual_read.py \
    --pkl_dir /proj/berzelius-2024-156/dataset/tau-PET/tau_T1MNI \
    --ref_mask "$2" \
    --smooth_fwhm "${1:-0}"
else
  python results_plotting/voxelwise_visual_read.py \
    --input_path /proj/berzelius-2024-156/users/x_nadpi/data \
    --data_type tau_T1MNI_pts \
    --smooth_fwhm "${1:-0}"
fi
