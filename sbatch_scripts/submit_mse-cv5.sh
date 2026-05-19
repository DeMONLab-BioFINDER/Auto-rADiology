#!/bin/bash
# SLURM batch job script for Berzelius
# 5-fold CV on the training pool with an untouched 20% hold-out test split

#SBATCH -A berzelius-2026-31
#SBATCH --gpus=1
#SBATCH -t 12:00:00
#SBATCH -J mse-cv5
#SBATCH -o ../../logs/mse-cv5-slurm-%j.out
#SBATCH -e ../../logs/mse-cv5-slurm-%j.err

# Load environment
module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate ai-pet

# Determine project root based on current directory
if [[ "$(basename "$PWD")" == "sbatch_scripts" ]]; then
  cd .. || exit 1
elif [[ ! -d "sbatch_scripts" ]]; then
  echo "Error: Must be run from sbatch_scripts or Auto-rADiology root directory"
  exit 1
fi

python "./run.py" \
  --no-tune \
  --dataset Gothenburg \
  --data_type tau_raw \
  --targets MetaTemporal,MesialTemporal,Frontal,TemporoParietal \
  --stratifycvby site,Universal \
  --reg_loss mse \
  --train_size 0.80 \
  --val_size 0.00 \
  --test_size 0.20 \
  --run_kfold_cv \
  --n_splits 5 \
  --es_patience 15 \
  --es_min_delta 0.001 \
  --model_name_extra mse-cv5