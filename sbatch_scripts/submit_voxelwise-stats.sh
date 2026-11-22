#!/bin/bash
# SLURM batch job script for Berzelius HOPPER - submit from a berzelius-hopper login node.
#
# Voxelwise permutation statistics (results_plotting/voxelwise_stats_visual_read.py) for one final MNI
# visual-read model, on its held-out test set and on AVID, one after the other (never pooled).
# CPU work only (a GPU is requested because Berzelius jobs need one); the permutations use every CPU of the job.
# Outputs: <run>/figures/voxelwise_stats_test/ and <run>/figures/voxelwise_stats_avid/
#
# Usage (from sbatch_scripts, like the other scripts; relative run names resolve under <proj_path>/results):
#   sbatch submit_voxelwise-stats.sh <final_run_folder>
#   SETS=test N_PERM=1000 sbatch submit_voxelwise-stats.sh <final_run_folder>     # quicker check
# The AVID part needs the model's AVID result (submit_val_AVID_unseen_visual-read.sh).

#SBATCH -A berzelius-2026-231
#SBATCH --gpus=1
#SBATCH -t 06:00:00
#SBATCH -J voxelwise-stats
#SBATCH -o ../../logs/voxelwise-stats-slurm-%j.out
#SBATCH -e ../../logs/voxelwise-stats-slurm-%j.err

module load Miniforge3/25.3.1-0 2>/dev/null || module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate ai-pet

if [[ "$(basename "$PWD")" == "sbatch_scripts" ]]; then cd .. || exit 1; fi
if [[ ! -d results_plotting ]]; then echo "Run from sbatch_scripts or the Auto-rADiology root"; exit 1; fi

RUN_ARG="$1"
if [[ -z "$RUN_ARG" ]]; then echo "Usage: sbatch submit_voxelwise-stats.sh <final_run_folder>"; exit 1; fi
if [[ "$RUN_ARG" = /* ]]; then RUN="$RUN_ARG"; else RUN="$(cd .. && pwd)/results/$RUN_ARG"; fi
[[ -d "$RUN" ]] || { echo "No such run folder: $RUN"; exit 1; }

DATA=/proj/berzelius-2024-156/users/x_nadpi/data
PKL_DIR="${PKL_DIR:-/proj/berzelius-2024-156/dataset/tau-PET/tau_T1MNI}"
REF_MASK="${REF_MASK:-$DATA/masks/voi_CerebGry_tau_2mm.nii}"
ATLAS_DIR="${ATLAS_DIR:-$DATA/atlases/lotter2025_mni_freesurfer}"   # DK cortex mask + peak names
SETS="${SETS:-test avid}"
N_PERM="${N_PERM:-5000}"
N_JOBS="${SLURM_CPUS_ON_NODE:-$(nproc)}"
echo "Run: $RUN | sets: $SETS | $N_PERM permutations on $N_JOBS CPUs"

for SET in $SETS; do
  echo -e "\n================ $SET ================"
  python results_plotting/voxelwise_stats_visual_read.py "$RUN" --set "$SET" \
    --pkl_dir "$PKL_DIR" --ref_mask "$REF_MASK" --atlas_dir "$ATLAS_DIR" \
    --demo_csv "$DATA/demo.csv" --ext_demo_csv "$DATA/demo_AVID_unseen.csv" \
    --n_perm "$N_PERM" --n_jobs "$N_JOBS"
done
