#!/bin/bash
# SLURM batch job script for Berzelius

#SBATCH -A berzelius-2026-31
#SBATCH --gpus=1
#SBATCH -t 24:00:00
#SBATCH -J T1MNI-l1-50

# Load your environment
module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate ai-pet

# Execute your code
python run_val.py --dataset Gothenburg --data_type tau_T1MNI --best_model_folder tau_T1MNI_Gothenburg_CNN3D_visual_read_2split80-20_stratify-site,visual_read_20260311_203154 --few_shot 50 --unfreeze_layers 1 --few_shot_iterations 10
# python run_val.py --dataset IDEAS --best_model_folder IDEAS_ADNI_CNN3D_CL_2split80-20_stratify-dataset,CL_Balanced_sampling_L1-10_all-scans_20260106_033903
# IDEAS_ADNI_CNN3D_visual_read_2split80-20_stratify-dataset,visual_read_Balanced_sampling_L1-10_all-scans_20260106_034136
# --targets CL --few_shot 5 --unfreeze_layers 1 --few_shot_iterations 10 --stratifycvby visual_read
# --model Unet3D --unfreeze_layers 3 --few_shot_iterations 10