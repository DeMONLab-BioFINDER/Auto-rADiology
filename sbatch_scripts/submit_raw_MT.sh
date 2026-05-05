#!/bin/bash
# SLURM batch job script for Berzelius

#SBATCH -A berzelius-2026-31
#SBATCH --gpus=1
#SBATCH -t 12:00:00
#SBATCH -J tau-raw-MT
#SBATCH -o ../logs/slurm-%j.out
#SBATCH -e ../logs/slurm-%j.err

# Load environment
module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate ai-pet

# 1-epoch smoke test on raw dataset path
python run.py \
  --no-tune \
  --dataset Gothenburg \
  --data_type tau_raw \
  --targets MetaTemporal \
  --stratifycvby site,MetaTemporal \
  --model_name_extra raw-metatemporal-test
