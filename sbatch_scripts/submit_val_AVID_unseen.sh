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

RUN_LOG=$(mktemp)
python "./run_val.py" \
  --dataset AVID_unseen \
  --best_model_folder "/proj/berzelius-2024-156/users/x_nadpi/results/final-tau_raw_Gothenburg_CNN3D_MetaTemporal,MesialTemporal,Frontal,TemporoParietal_2split80-20_stratify-site,Universal_mse-final-47_20260512_222544" \
  --model CNN3D \
  --data_type tau_raw \
  --input_path /proj/berzelius-2024-156/users/x_nadpi/data \
  --targets MetaTemporal,MesialTemporal,Frontal,TemporoParietal \
  --few_shot 0 \
  2>&1 | tee "$RUN_LOG"
RUN_STATUS=${PIPESTATUS[0]}

# Figures are a separate, non-fatal step: a plotting bug shouldn't mark this (expensive,
# GPU) validation job as failed. See results_plotting/plot_unseen_validation_results.py.
# For run_val.py, "Output directory created at:" is <best_model_folder>/validation - exactly
# what --final_run_dir expects, and only produced for --few_shot 0 (zero-shot) runs.
RUN_DIR=$(grep -m1 '^Output directory created at: ' "$RUN_LOG" | sed 's/^Output directory created at: //')
rm -f "$RUN_LOG"

if [[ $RUN_STATUS -ne 0 ]]; then
  echo "run_val.py exited with status $RUN_STATUS; skipping figure generation."
elif [[ -z "$RUN_DIR" ]]; then
  echo "WARNING: could not determine run output directory; skipping figure generation."
else
  echo "Generating figures for $RUN_DIR ..."
  python results_plotting/plot_unseen_validation_results.py --final_run_dir "$RUN_DIR" \
    || echo "WARNING: figure generation failed (validation results are unaffected)."
fi

exit $RUN_STATUS