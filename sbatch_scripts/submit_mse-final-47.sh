#!/bin/bash
# SLURM batch job script for Berzelius
# Final retrain on the full 80% training pool for a fixed 47 epochs

#SBATCH -A berzelius-2026-31
#SBATCH --gpus=1
#SBATCH -t 6:00:00
#SBATCH -J mse-final-47
#SBATCH -o ../../logs/mse-final-47-slurm-%j.out
#SBATCH -e ../../logs/mse-final-47-slurm-%j.err

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

DATASET=Gothenburg

RUN_LOG=$(mktemp)
python ./run.py \
  --no-tune \
  --dataset "$DATASET" \
  --data_type tau_raw \
  --targets MetaTemporal,MesialTemporal,Frontal,TemporoParietal \
  --stratifycvby site,Universal \
  --reg_loss mse \
  --train_size 0.80 \
  --val_size 0.00 \
  --test_size 0.20 \
  --epochs 47 \
  --model_name_extra mse-final-47 \
  2>&1 | tee "$RUN_LOG"
RUN_STATUS=${PIPESTATUS[0]}

# Figures are a separate, non-fatal step: a plotting bug shouldn't mark this (expensive,
# GPU) training job as failed. See results_plotting/plot_heldout_test_results.py.
RUN_DIR=$(grep -m1 '^Output directory created at: ' "$RUN_LOG" | sed 's/^Output directory created at: //')
rm -f "$RUN_LOG"

if [[ $RUN_STATUS -ne 0 ]]; then
  echo "run.py exited with status $RUN_STATUS; skipping figure generation."
elif [[ -z "$RUN_DIR" ]]; then
  echo "WARNING: could not determine run output directory; skipping figure generation."
else
  echo "Generating figures for $RUN_DIR ..."
  python results_plotting/plot_heldout_test_results.py "$RUN_DIR" --dataset "$DATASET" \
    || echo "WARNING: figure generation failed (training results are unaffected)."
fi

exit $RUN_STATUS