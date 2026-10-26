#!/bin/bash
# SLURM batch job script for Berzelius HOPPER (H200) - submit from a
# berzelius-hopper login node (berzelius-hopper.nsc.liu.se), not the Ampere one.
#
# Occlusion maps for hand-picked scans (e.g. a confident positive, a confident
# negative and a discrepant case), inside the head only (--vis_brain_bbox).
# Pick the IDs from <best_model_folder>/figures/visual_read_confidence_groups.csv
# (column ID_ind, with prob and group) or visual_read_confidently_wrong_cases.csv,
# written by results_plotting/analyze_model_confidence_visual_read.py.
# target: prob = drop in probability (default), logit = drop in the model's log-odds, needed for
# very confident scans (P ~1 or ~0), where the probability cannot move and the map is noise.
# Maps go to <best_model_folder>/validation/visualization/occlusion_cases_<target>_<fill>_mask<M>_ov<O>
# (earlier output of the same settings there is replaced).
#
# Usage:
#   sbatch submit_vis-occlusion-cases.sh <best_model_folder_name> <ID1,ID2,...> [target] [mask_size] [overlap] [fill]
# Defaults: target prob, mask 4, overlap 0.5, fill gaussian.
#   e.g. sbatch submit_vis-occlusion-cases.sh <folder> 2021,201,1922 logit
# Input images: set DATA_TYPE to the data type the model was trained on (default tau_raw), e.g.
#   DATA_TYPE=tau_T1MNI_pts sbatch submit_vis-occlusion-cases.sh <folder> <IDs> logit
# Time: ~6 min per scan.

#SBATCH -A berzelius-2026-231
#SBATCH --gpus=1
#SBATCH -t 02:00:00
#SBATCH -J vis-occlusion-cases
#SBATCH -o ../../logs/vis-occlusion-cases-slurm-%j.out
#SBATCH -e ../../logs/vis-occlusion-cases-slurm-%j.err

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
SUBJECT_IDS="$2"
if [[ -z "$BEST_MODEL_FOLDER" || -z "$SUBJECT_IDS" ]]; then
  echo "Usage: sbatch submit_vis-occlusion-cases.sh <best_model_folder_name> <ID1,ID2,...> [target] [mask_size] [overlap] [fill]"
  exit 1
fi
TARGET="${3:-prob}"
MASK_SIZE="${4:-4}"
OVERLAP="${5:-0.5}"
FILL="${6:-gaussian}"
DATA_TYPE="${DATA_TYPE:-tau_raw}"

# Relative folder names resolve under <proj_path>/results, same as src/params.py does.
VIS_DIR=$(python -c "
import os, sys
from src.params import parse_arguments
sys.argv = ['x', '--best_model_folder', '$BEST_MODEL_FOLDER']
print(os.path.join(parse_arguments().output_path, 'visualization'))
" | tail -n 1)
echo "Data type $DATA_TYPE, subjects $SUBJECT_IDS, target $TARGET, mask $MASK_SIZE, overlap $OVERLAP, fill $FILL"

nvidia-smi --query-gpu=name,memory.total --format=csv

RUN_STATUS=0
START=$SECONDS
python ./run_vis.py \
  --dataset Gothenburg \
  --best_model_folder "$BEST_MODEL_FOLDER" \
  --model CNN3D \
  --data_type "$DATA_TYPE" \
  --input_path /proj/berzelius-2024-156/users/x_nadpi/data \
  --targets visual_read \
  --cls_loss softmax \
  --visualization_name occlusion \
  --vis_img_list "$SUBJECT_IDS" \
  --vis_mask_size "$MASK_SIZE" \
  --vis_overlap "$OVERLAP" \
  --vis_n_batch 16 \
  --vis_occlusion_mode "$FILL" \
  --vis_brain_bbox \
  --vis_occlusion_target "$TARGET" \
  || RUN_STATUS=$?
ELAPSED=$((SECONDS - START))
echo "[timing] ${ELAPSED}s = $((ELAPSED / 60)) min for all subjects"

RUN_DIR="$VIS_DIR/occlusion_cases_${TARGET}_${FILL}_mask${MASK_SIZE}_ov${OVERLAP}"
if [[ -d "$VIS_DIR/occlusion" ]]; then
  rm -rf "$RUN_DIR" && mv "$VIS_DIR/occlusion" "$RUN_DIR" && echo "maps saved in $RUN_DIR"
fi

exit $RUN_STATUS
