#!/bin/bash
# SLURM batch job script for Berzelius - Inference on unseen AVID test set with FINAL model

#SBATCH -A berzelius-2025-231
#SBATCH --gpus=1
#SBATCH -t 06:00:00
#SBATCH -J VR_final_unseen

# Load your environment
module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate ai-pet

# Execute your code - Final model inference on unseen AVID test set
cd /proj/berzelius-2024-156/users/x_nadpi/Auto-rADiology
python run_val.py \
  --dataset AVID_unseen \
  --data_type tau_raw \
  --input_path /proj/berzelius-2024-156/users/x_nadpi/data/data/tau_raw \
  --targets MetaTemporal,MesialTemporal,Frontal,TemporoParietal \
  --few_shot 0 \
  --stratifycvby site,MetaTemporal,MesialTemporal,Frontal,TemporoParietal \
  --model_name_extra final \
  --best_model_folder '/proj/berzelius-2024-156/users/x_nadpi/results/tau_raw_Gothenburg_CNN3D_MetaTemporal,MesialTemporal,Frontal,TemporoParietal_2split80-20_stratify-site,MetaTemporal,MesialTemporal,Frontal,TemporoParietal_raw-multioutput-test-final_20260429_034428' \
  --model CNN3D
