#!/bin/bash
# SLURM batch job script for Berzelius
# 5-fold CV for the visual_read classifier on the real Gothenburg training pool,
# with an untouched 20% hold-out test split. Mirrors submit_mse-cv5.sh (the
# regional-SUVR CV5 workflow), swapping --targets/--stratifycvby for the
# classification head.

#SBATCH -A berzelius-2026-231
#SBATCH --gpus=1
#SBATCH -t 12:00:00
#SBATCH -J visual-read-cv5
#SBATCH -o ../../logs/visual-read-cv5-slurm-%j.out
#SBATCH -e ../../logs/visual-read-cv5-slurm-%j.err

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
python "./run.py" \
  --no-tune \
  --dataset Gothenburg \
  --data_type tau_raw \
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
  --model_name_extra visual-read-cv5 \
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
