#!/bin/bash
# SLURM batch job script for Berzelius
# Final visual_read model trained exactly like a CV fold model: the same 20% hold-out test set as
# submit_visual-read-final.sh (same --seed, so the test scans are identical), and 20% of the training
# pool (16% of all scans, ~365 - the size of a CV validation fold) held out as a validation set.
# Training stops early on validation loss (patience 15, as in submit_visual-read-cv5.sh), the learning
# rate is halved when validation loss plateaus, and the best-validation-loss epoch is the model that
# predicts the test set. --epochs is only a ceiling.
#
# Why: with no validation set (submit_visual-read-final.sh) the final model trains at the full
# learning rate for a fixed number of epochs and stops wherever its scores happen to be, so where
# 0.5 lands changes a lot between training seeds (MNI: 2 / 10 / 40 Model + / Visual read - on test,
# same AUC). The CV fold models, trained this way, were stable at 0.5 (spec .971-.989 over 15 models).
#
# Usage (from sbatch_scripts, Hopper or Ampere login node):
#   sbatch submit_visual-read-final-val.sh [data_type] [train_seed]     defaults tau_raw, none
#   e.g. sbatch submit_visual-read-final-val.sh tau_raw_2mm_pts
#        sbatch submit_visual-read-final-val.sh tau_raw_2mm_pts 7      (second training seed)
# train_seed only changes weight initialisation / shuffling / augmentation; test and validation
# scans stay the same.
# Pooling (CNN3D pool=...) instead of the default global average pooling: POOL=attention (gated
# attention/MIL), POOL=avgmax (average and max side by side) or POOL=topk (mean of the strongest 5% of
# locations), e.g.  POOL=topk sbatch submit_visual-read-final-val.sh tau_T1MNI_pts
# The run name gets "-<POOL>".
# The validation scans' predictions are saved too (evaluation/<dataset>/Val_<dataset>_results.csv), so
# variants can be compared on validation with results_plotting/score_borderline_validation.py. The AVID and occlusion scripts need the same POOL=... to rebuild
# the model (a mismatched checkpoint now stops with an error).
# Batch size: BATCH=8 (default 4, from src/params.py) - a cheap check of BatchNorm with a larger batch;
# same learning rate. The run name gets "-bs<BATCH>". e.g.  BATCH=8 sbatch submit_visual-read-final-val.sh tau_T1MNI_pts

#SBATCH -A berzelius-2026-231
#SBATCH --gpus=1
#SBATCH -t 6:00:00
#SBATCH -J visual-read-final-val
#SBATCH -o ../../logs/visual-read-final-val-slurm-%j.out
#SBATCH -e ../../logs/visual-read-final-val-slurm-%j.err

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
DATA_TYPE="${1:-tau_raw}"
TRAIN_SEED="${2:-}"
SEED_ARGS=()
NAME_EXTRA="visual-read-final-val"
POOL="${POOL:-avg}"
MODEL_ARGS=()
if [[ "$POOL" != "avg" ]]; then
  MODEL_ARGS=(--model_kwargs "{\"pool\": \"$POOL\"}")
fi
[[ "$POOL" != "avg" ]] && NAME_EXTRA="${NAME_EXTRA}-${POOL}"
BATCH_ARGS=()
if [[ -n "${BATCH:-}" ]]; then
  BATCH_ARGS=(--batch_size "$BATCH")
  NAME_EXTRA="${NAME_EXTRA}-bs${BATCH}"
fi
if [[ -n "$TRAIN_SEED" ]]; then
  SEED_ARGS=(--train_seed "$TRAIN_SEED")
  NAME_EXTRA="${NAME_EXTRA}-seed${TRAIN_SEED}"
fi
echo "data_type: $DATA_TYPE, train_seed: ${TRAIN_SEED:-same as split seed}, pool: $POOL, batch_size: ${BATCH:-default}"

RUN_LOG=$(mktemp)
python ./run.py \
  --no-tune \
  --dataset "$DATASET" \
  --data_type "$DATA_TYPE" \
  --input_path /proj/berzelius-2024-156/users/x_nadpi/data \
  --targets visual_read \
  --stratifycvby site,visual_read \
  --train_size 0.64 \
  --val_size 0.16 \
  --test_size 0.20 \
  --epochs 200 \
  --es_patience 15 \
  --es_min_delta 0.001 \
  --cls_loss softmax \
  --model_name_extra "$NAME_EXTRA" \
  "${SEED_ARGS[@]}" \
  "${MODEL_ARGS[@]}" \
  "${BATCH_ARGS[@]}" \
  2>&1 | tee "$RUN_LOG"
RUN_STATUS=${PIPESTATUS[0]}

# Figures are a separate, non-fatal step (see submit_visual-read-final.sh).
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
