#!/bin/bash
# SLURM batch job script for Berzelius HOPPER - submit from a berzelius-hopper login node.
# One-off data onboarding: converts Nadine's native tau pkls (tau_raw_backup) into a new
# cached input, data/tau_raw_headcrop, that differs from tau_raw only in the crop:
#
#   tau_raw           - CropForeground keeps every voxel > 0. PET background noise is > 0,
#                       so the box is ~the whole field of view, and how much the head gets
#                       stretched to 128^3 depends on the scanner/site.
#   tau_raw_headcrop  - the head mask is applied on the native scan first, so the box is
#                       the head. Still stretched to 128^3 (the head box is not a cube), but
#                       about the same for every scan instead of differing by site.
#
# tau_raw is not touched. Same pkls, same order, so data/demo.csv and the train/test split
# (same seed) are the same as the raw model's. Also writes tau_pt_scans.csv (per-scan mm per
# voxel and the native values the scaling mapped to 0 and 1) and the recipe in the manifest.
#
# AVID: a model trained on this needs AVID converted the same way (--head_crop) before
# zero-shot validation.
#
# Refuses to overwrite an output folder that already has .pt batches.
# A GPU is requested only because Berzelius jobs need one; the work is CPU.
#
# Usage:
#   sbatch submit_convert-tau-headcrop.sh

#SBATCH -A berzelius-2026-231
#SBATCH --gpus=1
#SBATCH -t 04:00:00
#SBATCH -J convert-tau-headcrop
#SBATCH -o ../../logs/convert-tau-headcrop-slurm-%j.out
#SBATCH -e ../../logs/convert-tau-headcrop-slurm-%j.err

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

DATA=/proj/berzelius-2024-156/users/x_nadpi/data
PKL_DIR=$DATA/tau_raw_backup
OUT_DIR=$DATA/tau_raw_headcrop

if compgen -G "$OUT_DIR/tau_batch_*.pt" > /dev/null; then
  echo "Skipping $OUT_DIR: it already has tau_batch_*.pt files (delete them first to redo)."
  exit 0
fi
python scripts/convert_tau_pkl_to_pt.py --pkl_dir "$PKL_DIR" --out_dir "$OUT_DIR" --batch_size 50 \
  --image_shape 128 128 128 --head_crop --crop_margin 2
