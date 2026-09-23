#!/bin/bash
# SLURM batch job script for Berzelius
# Zero-shot evaluation of the visual_read classifier on the real AVID unseen
# test set, using the trained final model from submit_visual-read-final.sh.
# Mirrors submit_val_AVID_unseen.sh (the regional-SUVR AVID validation workflow).

# Usage:
#   sbatch submit_val_AVID_unseen_visual-read.sh <best_model_folder_name>

#SBATCH -A berzelius-2026-231
#SBATCH --gpus=1
#SBATCH -t 06:00:00
#SBATCH -J AVID-unseen-visual-read
#SBATCH -o ../../logs/AVID-unseen-visual-read-slurm-%j.out
#SBATCH -e ../../logs/AVID-unseen-visual-read-slurm-%j.err

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

BEST_MODEL_FOLDER="$1"
if [[ -z "$BEST_MODEL_FOLDER" ]]; then
  echo "Usage: sbatch submit_val_AVID_unseen_visual-read.sh <best_model_folder_name>"
  exit 1
fi

DATASET=AVID_unseen

RUN_LOG=$(mktemp)
python "./run_val.py" \
  --dataset "$DATASET" \
  --best_model_folder "$BEST_MODEL_FOLDER" \
  --model CNN3D \
  --data_type tau_raw \
  --input_path /proj/berzelius-2024-156/users/x_nadpi/data \
  --targets visual_read \
  --few_shot 0 \
  2>&1 | tee "$RUN_LOG"
RUN_STATUS=${PIPESTATUS[0]}

# Figures are a separate, non-fatal step: a plotting bug shouldn't mark this (expensive,
# GPU) validation job as failed. See results_plotting/plot_visual_read_results.py (not
# plot_unseen_validation_results.py - that one's regional-SUVR-only: scatter/Bland-Altman
# panels don't apply to a binary classifier).
# For run_val.py, "Output directory created at:" is <best_model_folder>/validation - exactly
# what plot_visual_read_results.py's run_dir expects, and only produced for --few_shot 0
# (zero-shot) runs.
RUN_DIR=$(grep -m1 '^Output directory created at: ' "$RUN_LOG" | sed 's/^Output directory created at: //')
rm -f "$RUN_LOG"

if [[ $RUN_STATUS -ne 0 ]]; then
  echo "run_val.py exited with status $RUN_STATUS; skipping figure generation."
elif [[ -z "$RUN_DIR" ]]; then
  echo "WARNING: could not determine run output directory; skipping figure generation."
else
  echo "Generating figures for $BEST_MODEL_FOLDER (dataset $DATASET) ..."
  python results_plotting/plot_visual_read_results.py "$BEST_MODEL_FOLDER" --dataset "$DATASET" \
    || echo "WARNING: figure generation failed (validation results are unaffected)."
fi

exit $RUN_STATUS
