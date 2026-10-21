#!/bin/bash
# SLURM batch job script for Berzelius HOPPER (H200) - submit from a
# berzelius-hopper login node (berzelius-hopper.nsc.liu.se), not the Ampere one.
# Ampere's Miniforge module name doesn't exist on Hopper, hence the different
# module load line below.
#
# Occlusion timing only (run_vis.py --vis_benchmark): no maps are made. Loads a
# trained visual_read model, times a few batches of forward passes on one
# subject and prints the projected whole-volume occlusion time for several
# mask size / overlap settings. Runs twice (--vis_n_batch 16 and 64) to see
# whether bigger batches help on the H200.
#
# Mask size is in voxels of the 128^3 model input, not mm: the tau_raw scans
# are resized from different native sizes, so one mask covers a different
# physical size per dataset.
#
# Usage:
#   sbatch submit_vis-occlusion-benchmark.sh <best_model_folder_name> [subject_ID]
# Without subject_ID the first subject in the model's test split is used.

#SBATCH -A berzelius-2026-231
#SBATCH --gpus=1
#SBATCH -t 00:30:00
#SBATCH -J vis-occlusion-benchmark
#SBATCH -o ../../logs/vis-occlusion-benchmark-slurm-%j.out
#SBATCH -e ../../logs/vis-occlusion-benchmark-slurm-%j.err

# Load environment (Hopper module name)
module load Miniforge3/25.3.1-0
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
  echo "Usage: sbatch submit_vis-occlusion-benchmark.sh <best_model_folder_name> [subject_ID]"
  exit 1
fi

# Relative folder names resolve under <proj_path>/results, same as src/params.py does.
SUBJECT_ID="$2"
if [[ -z "$SUBJECT_ID" ]]; then
  SUBJECT_ID=$(python -c "
import os, sys, pandas as pd
from src.params import parse_arguments
sys.argv = ['x', '--best_model_folder', '$BEST_MODEL_FOLDER']
args = parse_arguments()
print(int(pd.read_csv(os.path.join(args.best_model_folder, 'splits', 'test_subjects.csv'))['ID'].iloc[0]))
" | tail -n 1)
fi
echo "Benchmarking on subject ID $SUBJECT_ID"

nvidia-smi --query-gpu=name,memory.total --format=csv

RUN_STATUS=0
for N_BATCH in 16 64; do
  echo "===== --vis_n_batch $N_BATCH ====="
  python ./run_vis.py \
    --dataset Gothenburg \
    --best_model_folder "$BEST_MODEL_FOLDER" \
    --model CNN3D \
    --data_type tau_raw \
    --input_path /proj/berzelius-2024-156/users/x_nadpi/data \
    --targets visual_read \
    --cls_loss softmax \
    --visualization_name occlusion \
    --vis_img_list "$SUBJECT_ID" \
    --vis_benchmark \
    --vis_mask_size 8 \
    --vis_overlap 0.5 \
    --vis_n_batch "$N_BATCH" \
    || RUN_STATUS=$?
done

exit $RUN_STATUS
