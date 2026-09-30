#!/bin/bash
# SLURM batch job script for Berzelius
# Ensembles the 5 fold checkpoints from a completed submit_visual-read-cv5*.sh
# run and evaluates the averaged predictions on that run's shared held-out
# test set (<best_model_folder>/splits/test_subjects.csv) - the same test set
# submit_visual-read-final*.sh evaluates its single retrained model on, so the
# two are directly comparable. Optionally also averages --tta_passes augmented
# inference passes per fold model (test-time augmentation).
#
# --cls_loss pinned to match whichever cv5 variant produced the checkpoints
# being ensembled (softmax for the unweighted/site-balanced/class-balanced
# runs; keep in sync if you ever ensemble a bce run).

# Usage:
#   sbatch submit_visual-read-cv5-ensemble-eval.sh <best_model_folder> [tta_passes]

#SBATCH -A berzelius-2026-231
#SBATCH --gpus=1
#SBATCH -t 02:00:00
#SBATCH -J visual-read-cv5-ensemble-eval
#SBATCH -o ../../logs/visual-read-cv5-ensemble-eval-slurm-%j.out
#SBATCH -e ../../logs/visual-read-cv5-ensemble-eval-slurm-%j.err

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
  echo "Usage: sbatch submit_visual-read-cv5-ensemble-eval.sh <best_model_folder> [tta_passes]"
  exit 1
fi
TTA_PASSES="${2:-1}"

python "./run_ensemble_eval.py" \
  --dataset Gothenburg \
  --data_type tau_raw \
  --input_path /proj/berzelius-2024-156/users/x_nadpi/data \
  --targets visual_read \
  --cls_loss softmax \
  --best_model_folder "$BEST_MODEL_FOLDER" \
  --tta_passes "$TTA_PASSES"
exit $?
