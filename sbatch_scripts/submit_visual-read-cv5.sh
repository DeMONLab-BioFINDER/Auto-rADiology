#!/bin/bash
# SLURM batch job script for Berzelius
# 5-fold CV for the visual_read classifier on the real Gothenburg training pool,
# with an untouched 20% hold-out test split. Mirrors submit_mse-cv5.sh (the
# regional-SUVR CV5 workflow), swapping --targets/--stratifycvby for the
# classification head.
# --cls_loss softmax pinned explicitly (matches run.py's current default, but
# pinned so this script's own record of what ran doesn't drift if that default
# changes again - see --class_weight_cls in submit_visual-read-cv5-weighted.sh
# for the weighted comparison run).
# Fixed learning rate (--no-lr_scheduler): early stopping only records each fold's best
# epoch, and the final model (submit_visual-read-final.sh, no validation set, fixed LR)
# is then trained for the median of those - same LR in both, so the epoch count carries over.
#
# Usage (from sbatch_scripts, Hopper or Ampere login node):
#   sbatch submit_visual-read-cv5.sh [data_type]     default tau_raw_2mm_pts
#   e.g. sbatch submit_visual-read-cv5.sh tau_raw_2mm_pts
#        sbatch submit_visual-read-cv5.sh tau_T1MNI_pts
# The data type is part of the output folder name, so runs don't overwrite each other.
# The log ends with "Median best_epoch across folds: N" -> pass N to submit_visual-read-final.sh.

#SBATCH -A berzelius-2026-231
#SBATCH --gpus=1
#SBATCH -t 12:00:00
#SBATCH -J visual-read-cv5
#SBATCH -o ../../logs/visual-read-cv5-slurm-%j.out
#SBATCH -e ../../logs/visual-read-cv5-slurm-%j.err

# Load environment
module load Miniforge3/25.3.1-0 2>/dev/null || module load Miniforge3/24.7.1-2-hpc1-bdist  # Hopper name, else Ampere
mamba activate ai-pet

# Determine project root based on current directory
if [[ "$(basename "$PWD")" == "sbatch_scripts" ]]; then
  cd .. || exit 1
elif [[ ! -d "sbatch_scripts" ]]; then
  echo "Error: Must be run from sbatch_scripts or Auto-rADiology root directory"
  exit 1
fi

DATA_TYPE="${1:-tau_raw_2mm_pts}"
echo "data_type: $DATA_TYPE"

RUN_LOG=$(mktemp)
python "./run.py" \
  --no-tune \
  --dataset Gothenburg \
  --data_type "$DATA_TYPE" \
  --input_path /proj/berzelius-2024-156/users/x_nadpi/data \
  --targets visual_read \
  --stratifycvby site,visual_read \
  --train_size 0.80 \
  --val_size 0.00 \
  --test_size 0.20 \
  --run_kfold_cv \
  --n_splits 5 \
  --es_patience 15 \
  --es_min_delta 0.001 \
  --cls_loss softmax \
  --no-lr_scheduler \
  --model_name_extra visual-read-cv5-fixedlr \
  2>&1 | tee "$RUN_LOG"
RUN_STATUS=${PIPESTATUS[0]}

# Figures are a separate, non-fatal step: a plotting bug shouldn't mark this (expensive,
# GPU) training job as failed. See results_plotting/plot_cv_summary.py.
RUN_DIR=$(grep -m1 '^Output directory created at: ' "$RUN_LOG" | sed 's/^Output directory created at: //')
rm -f "$RUN_LOG"

if [[ $RUN_STATUS -ne 0 ]]; then
  echo "run.py exited with status $RUN_STATUS; skipping figure generation."
elif [[ -z "$RUN_DIR" ]]; then
  echo "WARNING: could not determine run output directory; skipping figure generation."
else
  echo "Generating figures for $RUN_DIR ..."
  python results_plotting/plot_cv_summary.py "$RUN_DIR" \
    || echo "WARNING: figure generation failed (training results are unaffected)."
fi

exit $RUN_STATUS
