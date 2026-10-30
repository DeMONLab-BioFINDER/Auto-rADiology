#!/bin/bash
# SLURM batch job script for Berzelius
# Final visual_read model: retrain on the full 80% training pool for a fixed
# number of epochs, then evaluate once on the untouched 20% hold-out test set.
# Mirrors submit_mse-final-47.sh (the regional-SUVR final-model workflow):
# Fixed learning rate, no validation set, last epoch kept. The epoch count comes from
# submit_visual-read-cv5.sh (also fixed LR): its "Median best_epoch across folds" for the
# same data type. --cls_loss softmax pinned so the head matches that CV run.
#
# Usage (from sbatch_scripts, Hopper or Ampere login node):
#   sbatch submit_visual-read-final.sh <data_type> <epochs> [train_seed]
#   e.g. sbatch submit_visual-read-final.sh tau_raw_2mm_pts 40
#        sbatch submit_visual-read-final.sh tau_T1MNI_pts 52
# epochs = that data type's own "Median best_epoch across folds" from submit_visual-read-cv5.sh.
# train_seed only changes weight initialisation / shuffling / augmentation; the train/test split
# stays on --seed 42, so the test set is identical and the runs can be compared scan by scan.

#SBATCH -A berzelius-2026-231
#SBATCH --gpus=1
#SBATCH -t 6:00:00
#SBATCH -J visual-read-final
#SBATCH -o ../../logs/visual-read-final-slurm-%j.out
#SBATCH -e ../../logs/visual-read-final-slurm-%j.err

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

DATASET=Gothenburg
DATA_TYPE="$1"
EPOCHS="$2"
if [[ -z "$DATA_TYPE" || -z "$EPOCHS" ]]; then
  echo "Usage: sbatch submit_visual-read-final.sh <data_type> <epochs> [train_seed]  (epochs from submit_visual-read-cv5.sh)"
  exit 1
fi
TRAIN_SEED="${3:-}"
SEED_ARGS=()
NAME_EXTRA="visual-read-final-${EPOCHS}"
if [[ -n "$TRAIN_SEED" ]]; then
  SEED_ARGS=(--train_seed "$TRAIN_SEED")
  NAME_EXTRA="${NAME_EXTRA}-seed${TRAIN_SEED}"
fi
echo "data_type: $DATA_TYPE, epochs: $EPOCHS, train_seed: ${TRAIN_SEED:-same as split seed}"

RUN_LOG=$(mktemp)
python ./run.py \
  --no-tune \
  --dataset "$DATASET" \
  --data_type "$DATA_TYPE" \
  --input_path /proj/berzelius-2024-156/users/x_nadpi/data \
  --targets visual_read \
  --stratifycvby site,visual_read \
  --train_size 0.80 \
  --val_size 0.00 \
  --test_size 0.20 \
  --epochs "$EPOCHS" \
  --cls_loss softmax \
  --model_name_extra "$NAME_EXTRA" \
  "${SEED_ARGS[@]}" \
  2>&1 | tee "$RUN_LOG"
RUN_STATUS=${PIPESTATUS[0]}

# Figures are a separate, non-fatal step: a plotting bug shouldn't mark this (expensive,
# GPU) training job as failed. See results_plotting/plot_visual_read_results.py
# (not plot_heldout_test_results.py - that one's regional-SUVR-only: scatter/Bland-Altman
# panels don't apply to a binary classifier).
RUN_DIR=$(grep -m1 '^Output directory created at: ' "$RUN_LOG" | sed 's/^Output directory created at: //')
rm -f "$RUN_LOG"

if [[ $RUN_STATUS -ne 0 ]]; then
  echo "run.py exited with status $RUN_STATUS; skipping figure generation."
elif [[ -z "$RUN_DIR" ]]; then
  echo "WARNING: could not determine run output directory; skipping figure generation."
else
  echo "Generating figures for $RUN_DIR ..."
  python results_plotting/plot_visual_read_results.py "$RUN_DIR" --dataset "$DATASET" \
    || echo "WARNING: figure generation failed (training results are unaffected)."
fi

exit $RUN_STATUS
