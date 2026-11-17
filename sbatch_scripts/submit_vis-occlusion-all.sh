#!/bin/bash
# SLURM batch job script for Berzelius HOPPER (H200) - submit from a berzelius-hopper login node.
#
# Occlusion maps for EVERY scan of a final model's held-out test set (or of the external AVID set), split
# over a SLURM job array so the chunks run in parallel. Same settings as submit_vis-occlusion-cases.sh
# (logit target, mask 4, overlap 0.5, median_brain fill, head box only), so the maps are comparable.
#
# Each array task takes every N-th scan (N = array size) and writes into its own folder
#   <best_model_folder>/validation/visualization/occlusion_all_<DATASET_>logit_median_brain_mask4_ov0.5_chunk<k>
# (parallel jobs never share a summary.csv). When all tasks are done, combine them with
#   python results_plotting/merge_occlusion_chunks.py <that folder without _chunk<k>>
# Scans already in the combined folder or in the task's chunk folder are skipped, so a task that ran out of
# time can simply be resubmitted with the same command (only the missing scans run).
#
# Usage (8 chunks; ~6 min per scan -> test set ~6 h per task, AVID ~3 h per task):
#   POOL=attention DATA_TYPE=tau_T1MNI_pts sbatch --array=0-7 submit_vis-occlusion-all.sh <best_model_folder>
#   POOL=attention DATA_TYPE=tau_T1MNI_pts DATASET=AVID_unseen sbatch --array=0-7 submit_vis-occlusion-all.sh <best_model_folder>
# Scans: test set = <best_model_folder>/splits/test_subjects.csv; AVID = the IDs in the model's AVID result
# (validation/External_validation_AVID_unseen__*_zeroshot_results.csv), so run submit_val_AVID_unseen_visual-read.sh first.

#SBATCH -A berzelius-2026-231
#SBATCH --gpus=1
#SBATCH -t 08:00:00
#SBATCH -J vis-occlusion-all
#SBATCH -o ../../logs/vis-occlusion-all-slurm-%A_%a.out
#SBATCH -e ../../logs/vis-occlusion-all-slurm-%A_%a.err

module load Miniforge3/25.3.1-0
mamba activate ai-pet

if [[ "$(basename "$PWD")" == "sbatch_scripts" ]]; then
  cd .. || exit 1
elif [[ ! -d "sbatch_scripts" ]]; then
  echo "Error: Must be run from sbatch_scripts or Auto-rADiology root directory"
  exit 1
fi

BEST_MODEL_FOLDER="$1"
if [[ -z "$BEST_MODEL_FOLDER" || -z "$SLURM_ARRAY_TASK_ID" ]]; then
  echo "Usage: sbatch --array=0-7 submit_vis-occlusion-all.sh <best_model_folder>  (needs --array)"
  exit 1
fi
TASK=$SLURM_ARRAY_TASK_ID
N_TASKS=${SLURM_ARRAY_TASK_COUNT:-1}
DATA_TYPE="${DATA_TYPE:-tau_T1MNI_pts}"
DATASET="${DATASET:-Gothenburg}"
POOL="${POOL:-avg}"
NPOOLS="${NPOOLS:-}"
TARGET=logit; MASK_SIZE=4; OVERLAP=0.5; FILL=median_brain
KW=()
[[ "$POOL" != "avg" ]] && KW+=("\"pool\": \"$POOL\"")
[[ -n "$NPOOLS" ]] && KW+=("\"n_pools\": $NPOOLS")
MODEL_ARGS=()
if (( ${#KW[@]} )); then
  MODEL_ARGS=(--model_kwargs "{$(IFS=,; echo "${KW[*]}")}")
fi

DATASET_TAG=""
[[ "$DATASET" != "Gothenburg" ]] && DATASET_TAG="${DATASET}_"
BASE_NAME="occlusion_all_${DATASET_TAG}${TARGET}_${FILL}_mask${MASK_SIZE}_ov${OVERLAP}"
OUT_NAME="${BASE_NAME}_chunk${TASK}"

# This task's scans: every N-th of the sorted list, minus scans already done
IDS=$(python - "$BEST_MODEL_FOLDER" "$DATASET" "$BASE_NAME" "$OUT_NAME" "$TASK" "$N_TASKS" <<'EOF'
import glob, os, sys
import pandas as pd
from src.params import parse_arguments
folder, dataset, base, out, task, n = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
sys.argv = ["x", "--best_model_folder", folder]
args = parse_arguments()
run = args.best_model_folder
vis = os.path.join(args.output_path, "visualization")
if dataset == "Gothenburg":
    ids = pd.read_csv(os.path.join(run, "splits", "test_subjects.csv"))["ID"]
else:
    hits = sorted(glob.glob(os.path.join(run, "validation", f"External_validation_{dataset}__*_zeroshot_results.csv")))
    if not hits:
        sys.exit(f"No {dataset} zero-shot result in {run}/validation: run the AVID evaluation first")
    ids = pd.read_csv(hits[0])["ID_ind"]
ids = sorted(int(i) for i in ids.unique())
mine = ids[task::n]
done = lambda i: any(os.path.exists(os.path.join(vis, d, f"{i}_occlusion_class.nii.gz")) for d in (base, out))
todo = [i for i in mine if not done(i)]
print(f"[ids] {dataset}: {len(ids)} scans in total; task {task}/{n}: {len(mine)} scans, {len(mine) - len(todo)} already done",
      file=sys.stderr)
print(",".join(map(str, todo)))
EOF
)
IDS=$(echo "$IDS" | tail -n 1)
if [[ -z "$IDS" ]]; then echo "Nothing to run for task $TASK."; exit 0; fi
echo "Task $TASK of $N_TASKS: dataset $DATASET, data type $DATA_TYPE, pool $POOL -> $OUT_NAME"
echo "IDs: $IDS"
nvidia-smi --query-gpu=name,memory.total --format=csv

START=$SECONDS
python ./run_vis.py \
  --dataset "$DATASET" \
  --best_model_folder "$BEST_MODEL_FOLDER" \
  --model CNN3D \
  --data_type "$DATA_TYPE" \
  --input_path /proj/berzelius-2024-156/users/x_nadpi/data \
  --targets visual_read \
  --cls_loss softmax \
  --visualization_name occlusion \
  --vis_img_list "$IDS" \
  --vis_mask_size "$MASK_SIZE" \
  --vis_overlap "$OVERLAP" \
  --vis_n_batch 16 \
  --vis_occlusion_mode "$FILL" \
  --vis_brain_bbox \
  --vis_occlusion_target "$TARGET" \
  --vis_out_name "$OUT_NAME" \
  "${MODEL_ARGS[@]}"
RUN_STATUS=$?
ELAPSED=$((SECONDS - START))
echo "[timing] ${ELAPSED}s = $((ELAPSED / 60)) min"
exit $RUN_STATUS
