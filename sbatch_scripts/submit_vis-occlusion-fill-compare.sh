#!/bin/bash
# SLURM batch job script for Berzelius HOPPER (H200) - submit from a
# berzelius-hopper login node (berzelius-hopper.nsc.liu.se), not the Ampere one.
#
# Does the occlusion map depend on what the hidden patch is filled with? Runs
# occlusion (inside the head only, --vis_brain_bbox) on the same few test
# subjects with each fill in FILLS, then prints per subject how similar the
# pos_vs_neg maps are:
#   r          = voxelwise correlation inside the head (1 = same pattern)
#   top5_dice  = overlap of each map's 5% most important voxels (1 = same spots)
# If they agree, the map doesn't hinge on the fill choice.
#
# Fills: gaussian = MONAI default (soft dimming, only the centre fully hidden),
# median_brain = the scan's own median inside the head ("ordinary uptake").
# Maps go to <best_model_folder>/validation/visualization/occlusion_fill-<fill>_mask<M>_ov<O>
#
# Usage:
#   sbatch submit_vis-occlusion-fill-compare.sh <best_model_folder_name> [n_subjects] [mask_size] [overlap]
# Defaults: 6 test subjects (half visual read positive, half negative), mask 4, overlap 0.5.
# Time: ~6 min per subject per fill with mask 4 -> ~70 min for the defaults.

#SBATCH -A berzelius-2026-231
#SBATCH --gpus=1
#SBATCH -t 03:00:00
#SBATCH -J vis-occlusion-fill-compare
#SBATCH -o ../../logs/vis-occlusion-fill-compare-slurm-%j.out
#SBATCH -e ../../logs/vis-occlusion-fill-compare-slurm-%j.err

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
  echo "Usage: sbatch submit_vis-occlusion-fill-compare.sh <best_model_folder_name> [n_subjects] [mask_size] [overlap]"
  exit 1
fi
N_SUBJECTS="${2:-6}"
MASK_SIZE="${3:-4}"
OVERLAP="${4:-0.5}"
FILLS=(gaussian median_brain)

# Test subjects (half visual read positive, half negative when the split has that column) and the output folder.
# Relative folder names resolve under <proj_path>/results, same as src/params.py does.
read -r SUBJECT_IDS VIS_DIR < <(python -c "
import os, sys, pandas as pd
from src.params import parse_arguments
sys.argv = ['x', '--best_model_folder', '$BEST_MODEL_FOLDER']
args = parse_arguments()
df = pd.read_csv(os.path.join(args.best_model_folder, 'splits', 'test_subjects.csv'))
n = $N_SUBJECTS
if 'visual_read' in df.columns:
    pos, neg = df[df['visual_read'] == 1], df[df['visual_read'] == 0]
    ids = list(pos['ID'][:(n + 1) // 2]) + list(neg['ID'][:n // 2])
else:
    ids = list(df['ID'][:n])
print(','.join(str(int(i)) for i in ids), os.path.join(args.output_path, 'visualization'))
" | tail -n 1)
echo "Subjects $SUBJECT_IDS, mask $MASK_SIZE, overlap $OVERLAP, fills ${FILLS[*]}"
echo "Maps under $VIS_DIR"

nvidia-smi --query-gpu=name,memory.total --format=csv

RUN_STATUS=0
for FILL in "${FILLS[@]}"; do
  echo "===== fill: $FILL ====="
  START=$SECONDS
  python ./run_vis.py \
    --dataset Gothenburg \
    --best_model_folder "$BEST_MODEL_FOLDER" \
    --model CNN3D \
    --data_type tau_raw \
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
    || RUN_STATUS=$?
  ELAPSED=$((SECONDS - START))
  echo "[timing] fill $FILL: ${ELAPSED}s = $((ELAPSED / 60)) min for all subjects"
  RUN_DIR="$VIS_DIR/occlusion_fill-${FILL}_mask${MASK_SIZE}_ov${OVERLAP}"
  if [[ -d "$VIS_DIR/occlusion" ]]; then
    rm -rf "$RUN_DIR" && mv "$VIS_DIR/occlusion" "$RUN_DIR" && echo "maps saved in $RUN_DIR"
  fi
done

echo "===== how similar are the maps? ====="
python - "$VIS_DIR" "$MASK_SIZE" "$OVERLAP" "$SUBJECT_IDS" "${FILLS[@]}" <<'PY' || RUN_STATUS=$?
import sys, os
import numpy as np, nibabel as nib, pandas as pd
vis_dir, mask, ov, ids, fills = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(","), sys.argv[5:]
a, b = fills[:2]
dir_of = lambda f: os.path.join(vis_dir, f"occlusion_fill-{f}_mask{mask}_ov{ov}")
summ = pd.read_csv(os.path.join(dir_of(a), "summary.csv")).set_index("ID")
rows = []
for i in ids:
    m = [nib.load(os.path.join(dir_of(f), f"{i}_occlusion_pos_vs_neg.nii.gz")).get_fdata() for f in (a, b)]
    inside = (m[0] != 0) | (m[1] != 0)  # 0 = outside the head / suppressed in both
    x, y = m[0][inside], m[1][inside]
    r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
    top = [np.abs(v) >= np.quantile(np.abs(v), 0.95) for v in (x, y)]
    dice = 2 * (top[0] & top[1]).sum() / (top[0].sum() + top[1].sum())
    s = summ.loc[int(i)] if int(i) in summ.index else {}
    rows.append({"ID": i, "visual_read": s.get("visual_read"), "P_positive": s.get("P_positive"),
                 "r": round(r, 3), "top5_dice": round(float(dice), 3),
                 f"max_abs_{a}": round(float(np.abs(x).max()), 4), f"max_abs_{b}": round(float(np.abs(y).max()), 4)})
out = pd.DataFrame(rows)
print(out.to_string(index=False))
print(f"\nmedian r = {out['r'].median():.3f}, median top5_dice = {out['top5_dice'].median():.3f}")
out.to_csv(os.path.join(vis_dir, f"occlusion_fill_compare_mask{mask}_ov{ov}.csv"), index=False)
PY

exit $RUN_STATUS
