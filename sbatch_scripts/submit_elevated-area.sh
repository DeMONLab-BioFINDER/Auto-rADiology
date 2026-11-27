#!/bin/bash
# SLURM batch job script for Berzelius HOPPER (CPU partition, no GPU) - submit from a login node.
#
# Follow-up on the voxelwise result (results_plotting/elevated_area_followup.py): in the area where the Model - / Read +
# scans have more tau than the Model - / Read - scans (held-out test voxelwise map), which misses carry the extra tau
# (confidence band, reader agreement, reader grade), who they are, and does the model react to it (occlusion evidence
# in the area against its tau, all scans). Held-out test, then AVID with the same (test) area. Needs the voxelwise
# output (submit_voxelwise-discordant.sh) and the merged occlusion maps.
#
# Usage (from the Auto-rADiology root):
#   sbatch sbatch_scripts/submit_elevated-area.sh <final_run_folder>
# Settings: DATA (default <proj>/data), AREA_DIR (default <run>/figures/voxelwise_discordant_test), EXT (AVID_unseen).

#SBATCH -A berzelius-2026-231
#SBATCH -p berzelius-hopper-cpu
#SBATCH -c 4
#SBATCH -t 01:00:00
#SBATCH -J elevated-area
#SBATCH -o ../logs/elevated-area-%j.out
#SBATCH -e ../logs/elevated-area-%j.err

module load Miniforge3/25.3.1-0 2>/dev/null || module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate ai-pet
set -eo pipefail

if [[ "$(basename "$PWD")" == "sbatch_scripts" ]]; then cd .. || exit 1; fi
[[ -d results_plotting ]] || { echo "Run from sbatch_scripts or the Auto-rADiology root"; exit 1; }

RUN_ARG="$1"
[[ -n "$RUN_ARG" ]] || { echo "Usage: sbatch submit_elevated-area.sh <final_run_folder>"; exit 1; }
if [[ "$RUN_ARG" = /* ]]; then RUN="$RUN_ARG"; else RUN="$(cd .. && pwd)/results/$RUN_ARG"; fi
[[ -d "$RUN" ]] || { echo "No such run folder: $RUN"; exit 1; }

DATA="${DATA:-$(cd .. && pwd)/data}"
EXT="${EXT:-AVID_unseen}"
ext=$(echo "${EXT%_unseen}" | tr '[:upper:]' '[:lower:]')
MASKS="$DATA/masks"
PKL_DIR="${PKL_DIR:-/proj/berzelius-2024-156/dataset/tau-PET/tau_T1MNI}"
AREA_DIR="${AREA_DIR:-$RUN/figures/voxelwise_discordant_test}"
VIS="$RUN/validation/visualization"
[[ -f "$AREA_DIR/model_neg_read_pos_vs_neg_logp_tfce.nii.gz" ]] || { echo "No voxelwise map in $AREA_DIR (set AREA_DIR)"; exit 1; }
COMMON=(--area_dir "$AREA_DIR" --pkl_dir "$PKL_DIR" --ref_mask "$MASKS/voi_CerebGry_tau_2mm.nii")

echo "================ held-out test ================"
python results_plotting/elevated_area_followup.py "$RUN" --set test "${COMMON[@]}" \
  --occlusion_dir "$VIS/occlusion_all_logit_median_brain_mask4_ov0.5" \
  --votes "$DATA/IDs_reads.csv" --demo_csv "$DATA/demo.csv"
echo "================ $EXT (area from the held-out test) ================"
VOTES_EXT="$DATA/AVIDIDs_reads.csv"; [[ "$EXT" == *AVID* && -f "$VOTES_EXT" ]] || VOTES_EXT=""
python results_plotting/elevated_area_followup.py "$RUN" --set avid "${COMMON[@]}" \
  --occlusion_dir "$VIS/occlusion_all_${EXT}_logit_median_brain_mask4_ov0.5" \
  ${VOTES_EXT:+--votes "$VOTES_EXT"} --demo_csv "$DATA/demo_${EXT}.csv" --out_dir "$RUN/figures/elevated_area_$ext"
echo "Done: $RUN/figures/elevated_area_{test,$ext}"
