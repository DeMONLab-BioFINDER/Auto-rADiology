#!/bin/bash
# SLURM batch job script for Berzelius
# Fast diagnostic run (single train/val/test split, not 5-fold CV): attention/MIL
# pooling + GroupNorm (instead of the default BatchNorm), class-balanced sampling,
# with the fixed (Right-Left-only) flip augmentation.
#
# Why: the attention + class-balanced run collapsed on AVID (sensitivity=0, all
# predicted probabilities compressed under ~0.25). BatchNorm freezes its running
# mean/variance from training and reuses them unchanged at inference - a plausible
# contributor if AVID's scanner/intensity distribution differs enough that those
# frozen stats no longer describe it well, especially feeding into attention's
# learned (and intensity-magnitude-sensitive) gating. GroupNorm computes
# normalization per-sample, identically at train and test time, with no running
# statistic to go stale against a new site. This run isolates whether that's the
# fix, independent of the weighted-vs-class-balanced question tested in the
# sibling script (submit_visual-read-singlesplit-attention-weighted.sh).
#
# Deliberately NOT combined with the weighted-loss change above - one variable at a
# time. Sampling stays class-balanced here so only the norm layer differs from the
# original attention + class-balanced run that showed the AVID collapse.
#
# Single train/val/test split instead of CV5, same rationale as the sibling script -
# this is a yes/no diagnostic, not final model selection. Held-out test_size kept at
# 0.20 to stay comparable with all other held-out results.

#SBATCH -A berzelius-2026-231
#SBATCH --gpus=1
#SBATCH -t 6:00:00
#SBATCH -J visual-read-singlesplit-attention-groupnorm
#SBATCH -o ../../logs/visual-read-singlesplit-attention-groupnorm-slurm-%j.out
#SBATCH -e ../../logs/visual-read-singlesplit-attention-groupnorm-slurm-%j.err

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
  --balance_sampling_by visual_read \
  --model_kwargs '{"pool": "attention", "norm": "group"}' \
  --model_name_extra "visual-read-singlesplit-attention-groupnorm" \
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
