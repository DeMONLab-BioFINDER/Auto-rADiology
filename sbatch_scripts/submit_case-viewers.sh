#!/bin/bash
# SLURM batch job script for Berzelius HOPPER (CPU partition, no GPU) - submit from a login node.
#
# Case viewers (results_plotting/make_case_viewer.py) for every discordant scan of one final MNI model, test and
# AVID, then the case review page (make_case_review.py) linking to them. Everything stays on Berzelius: open
# <run>/validation/visualization/case_review.html in Firefox inside ThinLinc. Reading the original images from the
# pkls is the slow part (several minutes per set).
#
# Usage (from the Auto-rADiology root):
#   sbatch sbatch_scripts/submit_case-viewers.sh <final_run_folder>
#   ANSWERS=<earlier review csv> sbatch ...   pre-fills the review page with earlier answers (columns Set, ID + questions)

#SBATCH -A berzelius-2026-231
#SBATCH -p berzelius-hopper-cpu
#SBATCH -c 4
#SBATCH -t 02:00:00
#SBATCH -J case-viewers
#SBATCH -o ../logs/case-viewers-%j.out
#SBATCH -e ../logs/case-viewers-%j.err

module load Miniforge3/25.3.1-0 2>/dev/null || module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate ai-pet
set -eo pipefail

if [[ "$(basename "$PWD")" == "sbatch_scripts" ]]; then cd .. || exit 1; fi
[[ -d results_plotting ]] || { echo "Run from sbatch_scripts or the Auto-rADiology root"; exit 1; }

RUN_ARG="$1"
[[ -n "$RUN_ARG" ]] || { echo "Usage: sbatch submit_case-viewers.sh <final_run_folder>"; exit 1; }
if [[ "$RUN_ARG" = /* ]]; then RUN="$RUN_ARG"; else RUN="$(cd .. && pwd)/results/$RUN_ARG"; fi
[[ -d "$RUN" ]] || { echo "No such run folder: $RUN"; exit 1; }

DATA="${DATA:-/proj/berzelius-2024-156/users/x_nadpi/data}"
V="$RUN/validation/visualization"
T="$V/occlusion_all_logit_median_brain_mask4_ov0.5"
A="$V/occlusion_all_AVID_unseen_logit_median_brain_mask4_ov0.5"
CT="$RUN/figures/discordant_by_confidence/cases.csv"
CA="$RUN/validation/figures/discordant_by_confidence/cases.csv"
COMMON=(--pkl_dir "${PKL_DIR:-/proj/berzelius-2024-156/dataset/tau-PET/tau_T1MNI}" --cbl_mask "$DATA/masks/voi_WhlCbl_2mm.nii"
        --regions "$DATA/masks/fda_regions_MNI152NLin2009cAsym.nii.gz")
ids() { python -c "import pandas as pd; print(','.join(pd.read_csv('$1')['ID'].astype(str)))"; }

echo "================ test viewers ================"
python results_plotting/make_case_viewer.py --vis_dir "$T" --ids "$(ids "$CT")" "${COMMON[@]}" --out_dir "$V/viewers_discordant_test"
echo "================ AVID viewers ================"
python results_plotting/make_case_viewer.py --vis_dir "$A" --ids "$(ids "$CA")" "${COMMON[@]}" --out_dir "$V/viewers_discordant_avid"
echo "================ review page ================"
python results_plotting/make_case_review.py --cases "test=$CT" "AVID=$CA" \
  --viewers "test=$V/viewers_discordant_test" "AVID=$V/viewers_discordant_avid" \
  ${ANSWERS:+--answers "$ANSWERS"} --out "$V/case_review.html"
