#!/bin/bash
# SLURM batch job script for Berzelius (raw quick test)

#SBATCH -A berzelius-2026-31
#SBATCH --gpus=1
#SBATCH -t 12:00:00
#SBATCH -J tau-raw-test

# Load environment
module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate ai-pet

# 1-epoch smoke test on raw dataset path
python run.py \
  --no-tune \
  --dataset Gothenburg \
  --data_type tau_raw \
  --targets visual_read \
  --stratifycvby site,visual_read \
  --epochs 30 \
  --model_name_extra raw-quick-test
