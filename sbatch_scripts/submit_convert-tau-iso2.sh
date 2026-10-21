#!/bin/bash
# SLURM batch job script for Berzelius HOPPER - submit from a berzelius-hopper login node.
# One-off data onboarding: converts Nadine's native tau pkls (tau_raw_backup, which also hold
# the AVID scans) into a new cached input, data/tau_raw_iso2. Differences from tau_raw:
#
#   tau_raw       - crop keeps every voxel > 0 (PET background noise is > 0, so ~the whole
#                   field of view), then each axis is stretched to 128^3 by its own factor:
#                   1 voxel = ~1.2-3.2 mm depending on axis and site, proportions not true.
#   tau_raw_iso2  - resampled to 2 mm cubic voxels, head masked on the native scan (threshold
#                   relative to the 99th percentile, so it works for any field of view), box
#                   cropped to the head and padded (centred, not stretched) to 112x128x96 =
#                   224x256x192 mm (x widened from 96: some skulls reached 192 mm); a head taller than 192 mm loses its bottom (neck), never its
#                   top (bottom_cut in tau_pt_scans.csv). 1 voxel = 2 mm in every scan, true proportions, about half
#                   the voxels of 128^3. Scaling percentiles (1-99 -> 0-1) taken inside the head.
#
# tau_raw is not touched. Same pkls, same order, so data/demo.csv, demo_AVID.csv and the
# train/test split (same seed) are unchanged: train/validate with --data_type tau_raw_iso2.
# Writes tau_pt_scans.csv (per scan: head size in mm, whether it touches the box edge, native
# values mapped to 0 and 1) and the recipe in tau_pt_manifest.json. A warning is printed if
# any head touches the box edge (would mean the box is too small).
#
# Refuses to overwrite an output folder that already has .pt batches.
# A GPU is requested only because Berzelius jobs need one; the work is CPU.
#
# Usage:
#   sbatch submit_convert-tau-iso2.sh

#SBATCH -A berzelius-2026-231
#SBATCH --gpus=1
#SBATCH -t 04:00:00
#SBATCH -J convert-tau-iso2
#SBATCH -o ../../logs/convert-tau-iso2-slurm-%j.out
#SBATCH -e ../../logs/convert-tau-iso2-slurm-%j.err

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
OUT_DIR=$DATA/tau_raw_iso2

if compgen -G "$OUT_DIR/tau_batch_*.pt" > /dev/null; then
  echo "Skipping $OUT_DIR: it already has tau_batch_*.pt files (delete them first to redo)."
  exit 0
fi
python scripts/convert_tau_pkl_to_pt.py --pkl_dir "$PKL_DIR" --out_dir "$OUT_DIR" --batch_size 50 \
  --spacing_mm 2 --image_shape 112 128 96 --head_crop --crop_margin 2 --pct_inside_head
