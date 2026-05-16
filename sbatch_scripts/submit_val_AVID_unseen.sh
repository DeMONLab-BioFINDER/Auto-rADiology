#!/bin/bash
# SLURM batch job script for Berzelius
# Zero-shot evaluation on the AVID unseen test set using a trained final model.

# Usage:
#   sbatch submit_val_AVID_unseen.sh <best_model_folder_name>

#SBATCH -A berzelius-2026-31
#SBATCH --gpus=1
#SBATCH -t 06:00:00
#SBATCH -J AVID_unseen_eval
#SBATCH -o ../../logs/AVID-unseen-eval-slurm-%j.out
#SBATCH -e ../../logs/AVID-unseen-eval-slurm-%j.err

BEST_MODEL_FOLDER="${1:?Usage: sbatch submit_val_AVID_unseen.sh <best_model_folder_name>}"

# Load environment
module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate ai-pet

# Resolve project root robustly for Slurm.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${PROJECT_ROOT}" || exit 1

python "${PROJECT_ROOT}/run_val.py" \
  --dataset AVID_unseen \
  --best_model_folder "${BEST_MODEL_FOLDER}" \
  --model CNN3D \
  --data_type tau_raw \
  --input_path /proj/berzelius-2024-156/users/x_nadpi/data/data/tau_raw \
  --targets MetaTemporal,MesialTemporal,Frontal,TemporoParietal \
  --few_shot 0