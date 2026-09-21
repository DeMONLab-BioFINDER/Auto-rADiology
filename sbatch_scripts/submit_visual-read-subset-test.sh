#!/bin/bash
# SLURM batch job script for Berzelius
# Smoke test: short visual_read training run on a small data subset, to sanity-check
# the pipeline against real cluster data before a full run. Not one of the three
# standard workflows (cv5 / final-47 / AVID validation) — ad hoc test only.
#
# NOTE: requires the subset CSV to be named to match run.py's cache lookup convention
# (demo_{dataset}_{data_type}.csv, see src/data.py build_master_table). Rename/symlink
# before running:
#   ln -s demo_test_subset.csv /proj/berzelius-2024-156/users/x_nadpi/data/demo_test_subset_tau_raw.csv

#SBATCH -A berzelius-2026-31
#SBATCH --gpus=1
#SBATCH -t 01:00:00
#SBATCH -J visual-read-subset-test
#SBATCH -o ../../logs/visual-read-subset-test-slurm-%j.out
#SBATCH -e ../../logs/visual-read-subset-test-slurm-%j.err

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

DATASET=test_subset
DEMO_CSV=/proj/berzelius-2024-156/users/x_nadpi/data/demo_test_subset_tau_raw.csv

RUN_LOG=$(mktemp)
python "./run.py" \
  --no-tune \
  --dataset "$DATASET" \
  --data_type tau_raw \
  --input_path /proj/berzelius-2024-156/users/x_nadpi/data \
  --targets visual_read \
  --stratifycvby site,visual_read \
  --train_size 0.70 \
  --val_size 0.00 \
  --test_size 0.30 \
  --epochs 5 \
  --es_patience 3 \
  --model_name_extra visual-read-subset-test \
  2>&1 | tee "$RUN_LOG"
RUN_STATUS=${PIPESTATUS[0]}

# Figures are a separate, non-fatal step: a plotting bug shouldn't mark this (expensive,
# GPU) training job as failed. See results_plotting/plot_visual_read_results.py.
RUN_DIR=$(grep -m1 '^Output directory created at: ' "$RUN_LOG" | sed 's/^Output directory created at: //')
rm -f "$RUN_LOG"

if [[ $RUN_STATUS -ne 0 ]]; then
  echo "run.py exited with status $RUN_STATUS; skipping figure generation."
elif [[ -z "$RUN_DIR" ]]; then
  echo "WARNING: could not determine run output directory; skipping figure generation."
else
  echo "Generating figures for $RUN_DIR ..."
  python results_plotting/plot_visual_read_results.py "$RUN_DIR" --dataset "$DATASET" --demo_csv "$DEMO_CSV" \
    || echo "WARNING: figure generation failed (training results are unaffected)."
fi

exit $RUN_STATUS
