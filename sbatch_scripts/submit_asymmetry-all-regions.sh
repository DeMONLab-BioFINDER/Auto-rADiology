#!/bin/bash
# SLURM batch job script for Berzelius HOPPER (CPU partition, no GPU) - submit from a login node.
#
# Left-right asymmetry (results_plotting/asymmetry_visual_read.py) in every CenTauR region: meta-temporal,
# temporoparietal, Universal, mesial temporal and frontal, for the held-out test set and AVID separately.
# Writes to new folders (asymmetry_allregions_test / _avid) so the earlier two-region results stay as they are;
# the meta-temporal and temporoparietal rows are the same numbers as before. Reading the images is the slow part.
#
# Usage (from the Auto-rADiology root):
#   sbatch sbatch_scripts/submit_asymmetry-all-regions.sh <final_run_folder>
# Settings: DATA (default <proj>/data, with demo.csv, demo_AVID_unseen.csv, masks/), EXT (default AVID_unseen).

#SBATCH -A berzelius-2026-231
#SBATCH -p berzelius-hopper-cpu
#SBATCH -c 4
#SBATCH -t 02:00:00
#SBATCH -J asymmetry-all
#SBATCH -o ../logs/asymmetry-all-%j.out
#SBATCH -e ../logs/asymmetry-all-%j.err

module load Miniforge3/25.3.1-0 2>/dev/null || module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate ai-pet
set -eo pipefail

if [[ "$(basename "$PWD")" == "sbatch_scripts" ]]; then cd .. || exit 1; fi
[[ -d results_plotting ]] || { echo "Run from sbatch_scripts or the Auto-rADiology root"; exit 1; }

RUN_ARG="$1"
[[ -n "$RUN_ARG" ]] || { echo "Usage: sbatch submit_asymmetry-all-regions.sh <final_run_folder>"; exit 1; }
if [[ "$RUN_ARG" = /* ]]; then RUN="$RUN_ARG"; else RUN="$(cd .. && pwd)/results/$RUN_ARG"; fi
[[ -d "$RUN" ]] || { echo "No such run folder: $RUN"; exit 1; }

DATA="${DATA:-$(cd .. && pwd)/data}"
EXT="${EXT:-AVID_unseen}"
ext=$(echo "${EXT%_unseen}" | tr '[:upper:]' '[:lower:]')
MASKS="$DATA/masks"
PKL_DIR="${PKL_DIR:-/proj/berzelius-2024-156/dataset/tau-PET/tau_T1MNI}"
for f in Meta_CenTauR.nii.gz TP_CenTauR.nii.gz CenTauR.nii.gz Mesial_CenTauR.nii.gz Frontal_CenTauR.nii.gz; do
  [[ -f "$MASKS/$f" ]] || echo "WARNING: $MASKS/$f missing - that region will be skipped"
done
COMMON=(--pkl_dir "$PKL_DIR" --ref_mask "$MASKS/voi_CerebGry_tau_2mm.nii" --centaur_dir "$MASKS")

echo "================ asymmetry test ================"
python results_plotting/asymmetry_visual_read.py "$RUN" --set test --demo_csv "$DATA/demo.csv" "${COMMON[@]}" \
  --out_dir "$RUN/figures/asymmetry_allregions_test"
echo "================ asymmetry $EXT ================"
python results_plotting/asymmetry_visual_read.py "$RUN" --set avid --ext_dataset "$EXT" --ext_demo_csv "$DATA/demo_${EXT}.csv" \
  "${COMMON[@]}" --out_dir "$RUN/figures/asymmetry_allregions_$ext"
echo "Done: $RUN/figures/asymmetry_allregions_{test,$ext}"
