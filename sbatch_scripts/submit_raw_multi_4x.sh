#!/bin/bash
# SLURM batch job script for Berzelius

#SBATCH -A berzelius-2026-31
#SBATCH --gpus=1
#SBATCH -t 00:20:00
#SBATCH -J 4x-val-test
#SBATCH -o ../../logs/4x-val-test-slurm-%j.out
#SBATCH -e ../../logs/4x-val-test-slurm-%j.err

# Load environment
module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate ai-pet

# Resolve project root robustly for Slurm.
# Prefer SLURM_SUBMIT_DIR (directory where sbatch was invoked),
# fallback to script location for interactive/local runs.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${PROJECT_ROOT}" || exit 1

# small test on subset dataset path
python "${PROJECT_ROOT}/run.py" \
  --no-tune \
  --dataset Gothenburg \
  --data_type tau_raw \
  --targets MetaTemporal,MesialTemporal,Frontal,TemporoParietal \
  --train_repeat 4 \
  --stratifycvby site,Universal \
  --model_name_extra 4x-val-test
