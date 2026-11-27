#!/bin/bash
# SLURM batch job script for Berzelius HOPPER (CPU partition, no GPU) - submit from a login node.
#
# Focal tau (results_plotting/focal_tau_followup.py): per scan, over the whole cortex (DK outline), peak tau, share of
# raised voxels and how much of it lies inside Universal CenTauR (the read regions). Do the Model - / Read + scans have
# a small hot spot that the voxelwise group map or a regional mean would average away? Held-out test, then AVID.
# Reads the model calls from the occlusion summary.csv.
#
# Usage (from the Auto-rADiology root):
#   sbatch sbatch_scripts/submit_focal-tau.sh <final_run_folder>
# Settings: DATA (default <proj>/data, with demo.csv, demo_AVID_unseen.csv, masks/CenTauR.nii.gz, atlases/lotter2025_mni_freesurfer), EXT (AVID_unseen).

#SBATCH -A berzelius-2026-231
#SBATCH -p berzelius-hopper-cpu
#SBATCH -c 4
#SBATCH -t 01:00:00
#SBATCH -J focal-tau
#SBATCH -o ../logs/focal-tau-%j.out
#SBATCH -e ../logs/focal-tau-%j.err

module load Miniforge3/25.3.1-0 2>/dev/null || module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate ai-pet
set -eo pipefail

if [[ "$(basename "$PWD")" == "sbatch_scripts" ]]; then cd .. || exit 1; fi
[[ -d results_plotting ]] || { echo "Run from sbatch_scripts or the Auto-rADiology root"; exit 1; }

RUN_ARG="$1"
[[ -n "$RUN_ARG" ]] || { echo "Usage: sbatch submit_focal-tau.sh <final_run_folder>"; exit 1; }
if [[ "$RUN_ARG" = /* ]]; then RUN="$RUN_ARG"; else RUN="$(cd .. && pwd)/results/$RUN_ARG"; fi
[[ -d "$RUN" ]] || { echo "No such run folder: $RUN"; exit 1; }

DATA="${DATA:-$(cd .. && pwd)/data}"
EXT="${EXT:-AVID_unseen}"
ext=$(echo "${EXT%_unseen}" | tr '[:upper:]' '[:lower:]')
MASKS="$DATA/masks"
PKL_DIR="${PKL_DIR:-/proj/berzelius-2024-156/dataset/tau-PET/tau_T1MNI}"
VIS="$RUN/validation/visualization"
[[ -f "$MASKS/CenTauR.nii.gz" ]] || { echo "No $MASKS/CenTauR.nii.gz (Universal CenTauR mask)"; exit 1; }
COMMON=(--pkl_dir "$PKL_DIR" --ref_mask "$MASKS/voi_CerebGry_tau_2mm.nii" --universal_mask "$MASKS/CenTauR.nii.gz" --atlas_dir "$DATA/atlases/lotter2025_mni_freesurfer")

echo "================ held-out test ================"
python results_plotting/focal_tau_followup.py "$RUN" --set test "${COMMON[@]}" \
  --summary_csv "$VIS/occlusion_all_logit_median_brain_mask4_ov0.5/summary.csv" --demo_csv "$DATA/demo.csv"
echo "================ $EXT ================"
python results_plotting/focal_tau_followup.py "$RUN" --set avid "${COMMON[@]}" \
  --summary_csv "$VIS/occlusion_all_${EXT}_logit_median_brain_mask4_ov0.5/summary.csv" --demo_csv "$DATA/demo_${EXT}.csv" \
  --out_dir "$RUN/figures/focal_tau_$ext"
echo "Done: $RUN/figures/focal_tau_{test,$ext}"
