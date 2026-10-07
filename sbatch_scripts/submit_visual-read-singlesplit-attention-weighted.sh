#!/bin/bash
# SLURM batch job script for Berzelius
# Fast diagnostic run (single train/val/test split, not 5-fold CV): attention/MIL
# pooling + weighted loss, with the fixed (Right-Left-only) flip augmentation.
#
# Why: the attention + class-balanced-sampling run collapsed on AVID (sensitivity=0,
# all predicted probabilities compressed under ~0.25 - see visual_read_probability_by_class
# for that run). Class-balanced sampling oversamples (repeats) minority-class scans every
# epoch; weighted loss does not - it reweights the loss but still shows every scan once
# per epoch. Pairing attention pooling with weighted loss instead isolates whether the
# AVID collapse was an oversampling x attention-capacity interaction (repeated draws +
# attention's extra learnable parameters memorizing training-cohort-specific patterns),
# or something inherent to attention pooling itself regardless of balancing method.
#
# Single train/val/test split instead of CV5: this is a yes/no diagnostic question, not
# final model selection, so the 5x cost of full CV isn't needed here. Uses the same
# early-stopping path as a CV fold (run.py "option 3: direct train/val/test"), writing
# directly to evaluation/<dataset>/Eval_<dataset>_results.csv + _metrics.csv - both
# plot_visual_read_results.py and site_breakdown_visual_read.py already handle this path
# (verified against src/cv.py's run_fold, which is shared with the CV/final-retrain code).
# Held-out test_size kept at 0.20 to stay comparable with all other held-out results.
#
# If this looks promising, promote to a full cv5 + final retrain the same way the other
# variants were - this is a screen, not the final answer.

#SBATCH -A berzelius-2026-231
#SBATCH --gpus=1
#SBATCH -t 6:00:00
#SBATCH -J visual-read-singlesplit-attention-weighted
#SBATCH -o ../../logs/visual-read-singlesplit-attention-weighted-slurm-%j.out
#SBATCH -e ../../logs/visual-read-singlesplit-attention-weighted-slurm-%j.err

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
  --input_path /proj/berzelius-2024-156/users/x_nadpi/data \
  --targets visual_read \
  --stratifycvby site,visual_read \
  --train_size 0.65 \
  --val_size 0.15 \
  --test_size 0.20 \
  --es_patience 15 \
  --es_min_delta 0.001 \
  --cls_loss softmax \
  --class_weight_cls \
  --model_kwargs '{"pool": "attention"}' \
  --model_name_extra "visual-read-singlesplit-attention-weighted" \
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
  python results_plotting/plot_visual_read_results.py "$RUN_DIR" --dataset "$DATASET" \
    || echo "WARNING: figure generation failed (training results are unaffected)."
fi

exit $RUN_STATUS
