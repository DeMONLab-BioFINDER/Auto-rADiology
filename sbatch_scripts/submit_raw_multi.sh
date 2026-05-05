#!/bin/bash
# SLURM batch job script for Berzelius (raw multi-output regression)

#SBATCH -A berzelius-2026-31
#SBATCH --gpus=1
#SBATCH -t 12:00:00
#SBATCH -J tau-raw-multi
#SBATCH -o ../logs/final-slurm-%j.out
#SBATCH -e ../logs/final-slurm-%j.err

# Load environment
module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate ai-pet

# Multi-output regression on 4 regions
python run.py \
  --no-tune \
  --dataset Gothenburg \
  --data_type tau_raw \
  --batch_size 8 \
  --targets MetaTemporal,MesialTemporal,Frontal,TemporoParietal \
  --stratifycvby site,MetaTemporal,MesialTemporal,Frontal,TemporoParietal \
  --model_name_extra raw-multioutput-test-final
