#!/bin/bash
# SLURM batch job script for Berzelius - submit from a login node.
#
# Voxelwise maps of discordant scans (results_plotting/voxelwise_stats_visual_read.py) for one final MNI
# visual-read model, on the held-out test set and each external set, one after the other (never pooled).
# Default (the extra analyses):
#   read_neg_above_cutoff      Read - scans: meta temporal SUVR above vs below the CenTauRz cut point (Read - / SUVR +)
#   read_pos_below_cutoff      Read + scans: below vs above the cut point (Read + / SUVR -)
#   model_neg_read_pos_vs_neg  Model - scans: Read + vs Read - (what readers see in the Model - / Read + scans)
# Any analysis of the script can be chosen with ANALYSES= (e.g. read_pos_vs_neg, missed_vs_detected).
# Cortex only (DK outline); results reported per CenTauR region (masks from $DATA/masks).
# Groups with fewer than 5 scans are skipped. CPU work only (a GPU is requested because Berzelius jobs need one).
# Outputs: <run>/figures/voxelwise_discordant_<set>/ (maps, summary.csv, centaur_regions.csv)
#
# Usage (from the Auto-rADiology root):
#   sbatch sbatch_scripts/submit_voxelwise-discordant.sh <final_run_folder>
#   ANALYSES=model_neg_read_pos_vs_neg sbatch sbatch_scripts/submit_voxelwise-discordant.sh <final_run_folder>   # one map only
#   EXTS="OASIS_unseen" N_PERM=1000 sbatch sbatch_scripts/submit_voxelwise-discordant.sh <final_run_folder>   # quicker check
# Original frozen MNI attention model (test + AVID, full training pool demographics), discordant-case set:
#   DEMO=/proj/berzelius-2024-156/users/x_nadpi/data/demo.csv EXTS=AVID_unseen \
#   ANALYSES="read_pos_vs_neg missed_vs_detected model_neg_read_pos_vs_neg" \
#   sbatch sbatch_scripts/submit_voxelwise-discordant.sh <final-17-attention run folder>

#SBATCH -A berzelius-2026-231
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH -t 08:00:00
#SBATCH -J voxelwise-discordant
#SBATCH -o ../logs/voxelwise-discordant-%j.out
#SBATCH -e ../logs/voxelwise-discordant-%j.err

module load Miniforge3/25.3.1-0 2>/dev/null || module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate ai-pet
set -eo pipefail

if [[ "$(basename "$PWD")" == "sbatch_scripts" ]]; then cd .. || exit 1; fi
[[ -d results_plotting ]] || { echo "Run from sbatch_scripts or the Auto-rADiology root"; exit 1; }

RUN_ARG="$1"
[[ -n "$RUN_ARG" ]] || { echo "Usage: sbatch submit_voxelwise-discordant.sh <final_run_folder>"; exit 1; }
if [[ "$RUN_ARG" = /* ]]; then RUN="$RUN_ARG"; else RUN="$(cd .. && pwd)/results/$RUN_ARG"; fi
[[ -d "$RUN" ]] || { echo "No such run folder: $RUN"; exit 1; }

DATA="${DATA:-/proj/berzelius-2024-156/users/x_nadpi/data}"
DEMO="${DEMO:-$DATA/demo_noOASIS.csv}"          # training pool + test set (left-out cohort removed)
EXTS="${EXTS:-OASIS_unseen AVID_unseen}"
PKL_DIR="${PKL_DIR:-/proj/berzelius-2024-156/dataset/tau-PET/tau_T1MNI}"
N_PERM="${N_PERM:-5000}"
N_JOBS="${SLURM_CPUS_PER_TASK:-16}"
ANALYSES="${ANALYSES:-read_neg_above_cutoff read_pos_below_cutoff model_neg_read_pos_vs_neg}"   # e.g. ANALYSES=model_neg_read_pos_vs_neg
# WHOLE_BRAIN=1: whole brain instead of cortex; RIM_MM=8: plus a rim outside the brain (meninges / skull, off-target).
# Results then go to voxelwise_discordant_wholebrain_<set> so the cortex results are kept.
EXTRA=(); TAG=""
if [[ "${WHOLE_BRAIN:-0}" == 1 ]]; then EXTRA+=(--whole_brain --rim_mm "${RIM_MM:-0}"); TAG="_wholebrain"; fi
COMMON=(--pkl_dir "$PKL_DIR" --ref_mask "$DATA/masks/voi_CerebGry_tau_2mm.nii"
        --atlas_dir "$DATA/atlases/lotter2025_mni_freesurfer" --demo_csv "$DEMO"
        --n_perm "$N_PERM" --n_jobs "$N_JOBS" --analyses $ANALYSES "${EXTRA[@]}")
echo "Run: $RUN | test + $EXTS | $N_PERM permutations on $N_JOBS CPUs"

echo -e "\n================ test ================"
python results_plotting/voxelwise_stats_visual_read.py "$RUN" --set test "${COMMON[@]}" \
  --out_dir "$RUN/figures/voxelwise_discordant${TAG}_test"
for EXT in $EXTS; do
  label=$(echo "${EXT%_unseen}" | tr '[:upper:]' '[:lower:]')
  echo -e "\n================ $EXT ================"
  python results_plotting/voxelwise_stats_visual_read.py "$RUN" --set avid --ext_dataset "$EXT" \
    --ext_demo_csv "$DATA/demo_$EXT.csv" "${COMMON[@]}" --out_dir "$RUN/figures/voxelwise_discordant${TAG}_$label"
done
