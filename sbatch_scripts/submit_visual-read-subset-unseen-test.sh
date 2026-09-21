#!/bin/bash
# SLURM batch job script for Berzelius
# Smoke test: zero-shot "unseen" validation of a visual_read model, reusing test_subset
# itself as a stand-in unseen dataset - to sanity-check the run_val.py + plotting
# pipeline for external validation on the cluster. Not a real generalization test:
# test_subset is the same data the model may have already been trained/evaluated on.
# Not one of the three standard workflows - ad hoc test only.
#
# Usage:
#   sbatch submit_visual-read-subset-unseen-test.sh <best_model_folder>
# where <best_model_folder> is the output folder from a prior visual_read training run
# (e.g. from submit_visual-read-subset-test.sh or submit_visual-read-subset-cv.sh).
#
# NOTE: requires a second symlink (distinct from the training one, so run_val.py's
# demo-csv lookup - demo_{dataset}.csv - doesn't collide with the real AVID dataset name):
#   ln -s demo_test_subset.csv /proj/berzelius-2024-156/users/x_nadpi/data/demo_AVID_test_subset.csv

#SBATCH -A berzelius-2026-31
#SBATCH --gpus=1
#SBATCH -t 01:00:00
#SBATCH -J visual-read-subset-unseen-test
#SBATCH -o ../../logs/visual-read-subset-unseen-test-slurm-%j.out
#SBATCH -e ../../logs/visual-read-subset-unseen-test-slurm-%j.err

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
  echo "Usage: sbatch submit_visual-read-subset-unseen-test.sh <best_model_folder>"
  exit 1
fi

DATASET=AVID_test_subset

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

# Figures are a separate, non-fatal step: a plotting bug shouldn't mark this validation
# job as failed. See results_plotting/plot_visual_read_results.py (not
# plot_unseen_validation_results.py - that one's regional-SUVR-only: scatter/Bland-Altman
# panels don't apply to a binary classifier).
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
