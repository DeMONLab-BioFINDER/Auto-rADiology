#!/bin/bash
# SLURM batch job script for Berzelius HOPPER - submit from a berzelius-hopper login node.
#
# Perturbation experiment A (run_perturb_offtarget.py): set the striatum + thalamus signal of every scan to a typical
# and to a high level, with the per-scan scaling recomputed and with it kept fixed, and rerun the frozen model.
# Held-out test set, then AVID. Prints a check that the unchanged scans reproduce the model's saved probabilities.
# About 5 model passes per scan on one GPU: minutes.
# Outputs: <run>/figures/perturb_offtarget_test/ and _avid/ (perturb_by_group.csv, perturb_tests.csv,
#   perturb_change.png; perturb_per_scan.csv HAS IDs)
#
# Usage (from the Auto-rADiology root):
#   POOL=attention sbatch sbatch_scripts/submit_perturb-offtarget.sh <final_run_folder>
#   MAX_SCANS=20 ...   quick check on the first 20 scans of each set
#   REGION=chp ...     choroid plexus stand-in instead of striatum + thalamus (outputs perturb_chp_test/ and _avid/)

#SBATCH -A berzelius-2026-231
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH -J perturb-offtarget
#SBATCH -o ../logs/perturb-offtarget-%j.out
#SBATCH -e ../logs/perturb-offtarget-%j.err

module load Miniforge3/25.3.1-0 2>/dev/null || module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate ai-pet
set -eo pipefail

if [[ "$(basename "$PWD")" == "sbatch_scripts" ]]; then cd .. || exit 1; fi
[[ -f run_perturb_offtarget.py ]] || { echo "Run from sbatch_scripts or the Auto-rADiology root"; exit 1; }

RUN_ARG="$1"
[[ -n "$RUN_ARG" ]] || { echo "Usage: sbatch submit_perturb-offtarget.sh <final_run_folder>"; exit 1; }
if [[ "$RUN_ARG" = /* ]]; then RUN="$RUN_ARG"; else RUN="$(cd .. && pwd)/results/$RUN_ARG"; fi
[[ -d "$RUN" ]] || { echo "No such run folder: $RUN"; exit 1; }

DATA="${DATA:-/proj/berzelius-2024-156/users/x_nadpi/data}"
COMMON=(--best_model_folder "$RUN" --pool "${POOL:-attention}" --data_type "${DATA_TYPE:-tau_T1MNI_pts}"
        --input_path "$DATA" --pkl_dir "${PKL_DIR:-/proj/berzelius-2024-156/dataset/tau-PET/tau_T1MNI}"
        --ref_mask "$DATA/masks/voi_WhlCbl_2mm.nii" --atlas_dir "$DATA/atlases/lotter2025_mni_freesurfer"
        --max_scans "${MAX_SCANS:-0}" --region "${REGION:-deep}")
nvidia-smi --query-gpu=name,memory.total --format=csv

echo -e "\n================ test ================"
python run_perturb_offtarget.py --dataset Gothenburg "${COMMON[@]}"
echo -e "\n================ AVID ================"
python run_perturb_offtarget.py --dataset AVID_unseen "${COMMON[@]}"
