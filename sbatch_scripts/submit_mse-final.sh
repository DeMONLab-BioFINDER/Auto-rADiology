#!/bin/bash
# SLURM batch job script for Berzelius

#SBATCH -A berzelius-2026-31
#SBATCH --gpus=1
#SBATCH -t 10:00:00
#SBATCH -J mse-final
#SBATCH -o ../../logs/mse-final-slurm-%j.out
#SBATCH -e ../../logs/mse-final-slurm-%j.err

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

python "${PROJECT_ROOT}/run.py" \
  --no-tune \
  --dataset Gothenburg \
  --data_type tau_raw \
  --targets MetaTemporal,MesialTemporal,Frontal,TemporoParietal \
  --stratifycvby site,Universal \
  --reg_loss mse \
  --train_size 0.64 \
  --val_size 0.16 \
  --test_size 0.20 \
  --es_patience 15 \
  --model_name_extra mse-final