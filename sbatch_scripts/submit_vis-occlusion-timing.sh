#!/bin/bash
# SLURM batch job script for Berzelius HOPPER (H200) - submit from a
# berzelius-hopper login node (berzelius-hopper.nsc.liu.se), not the Ampere one.
#
# Times one REAL occlusion map on one subject (not the projection that
# submit_vis-occlusion-benchmark.sh prints), so data loading, MONAI's masking
# and saving are included. Runs the projection first for comparison, then the
# full occlusion only inside the brain bounding box (--vis_brain_bbox; the whole-volume
# run was dropped after the 2026-10-21 timing: 15 min vs 6 min per scan). The maps are
# saved too, in <best_model_folder>/validation/visualization/occlusion_mask<M>_ov<O>_bbox
# (run_vis.py always writes to .../occlusion, so the output is moved there afterwards).
#
# Mask size is in voxels of the 128^3 model input: 4 voxels ~ 8x8x5 mm on a
# median raw scan.
#
# Usage:
#   sbatch submit_vis-occlusion-timing.sh <best_model_folder_name> [subject_ID] [mask_size] [overlap]
# Defaults: first subject in the model's test split, mask 4, overlap 0.5.

#SBATCH -A berzelius-2026-231
#SBATCH --gpus=1
#SBATCH -t 02:00:00
#SBATCH -J vis-occlusion-timing
#SBATCH -o ../../logs/vis-occlusion-timing-slurm-%j.out
#SBATCH -e ../../logs/vis-occlusion-timing-slurm-%j.err

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
  echo "Usage: sbatch submit_vis-occlusion-timing.sh <best_model_folder_name> [subject_ID] [mask_size] [overlap]"
  exit 1
fi
MASK_SIZE="${3:-4}"
OVERLAP="${4:-0.5}"

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
VIS_DIR=$(python -c "
import os, sys
from src.params import parse_arguments
sys.argv = ['x', '--best_model_folder', '$BEST_MODEL_FOLDER']
print(os.path.join(parse_arguments().output_path, 'visualization'))
" | tail -n 1)
echo "Subject ID $SUBJECT_ID, mask $MASK_SIZE, overlap $OVERLAP"

nvidia-smi --query-gpu=name,memory.total --format=csv

COMMON_ARGS=(
  --dataset Gothenburg
  --best_model_folder "$BEST_MODEL_FOLDER"
  --model CNN3D
  --data_type tau_raw
  --input_path /proj/berzelius-2024-156/users/x_nadpi/data
  --targets visual_read
  --cls_loss softmax
  --visualization_name occlusion
  --vis_img_list "$SUBJECT_ID"
  --vis_mask_size "$MASK_SIZE"
  --vis_overlap "$OVERLAP"
  --vis_n_batch 16
)

RUN_STATUS=0
echo "===== projection (--vis_benchmark) ====="
python ./run_vis.py "${COMMON_ARGS[@]}" --vis_benchmark || RUN_STATUS=$?

echo "===== real occlusion (brain bbox) ====="
START=$SECONDS
python ./run_vis.py "${COMMON_ARGS[@]}" --vis_brain_bbox || RUN_STATUS=$?
ELAPSED=$((SECONDS - START))
echo "[timing] mask $MASK_SIZE, overlap $OVERLAP, brain bbox: ${ELAPSED}s = $((ELAPSED / 60)) min $((ELAPSED % 60)) s for 1 scan (incl. loading model and data)"
RUN_DIR="$VIS_DIR/occlusion_mask${MASK_SIZE}_ov${OVERLAP}_bbox"
if [[ -d "$VIS_DIR/occlusion" ]]; then
  rm -rf "$RUN_DIR" && mv "$VIS_DIR/occlusion" "$RUN_DIR" && echo "maps saved in $RUN_DIR"
fi

exit $RUN_STATUS
