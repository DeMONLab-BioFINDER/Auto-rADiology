#!/bin/bash
# SLURM batch job script for Berzelius HOPPER CPU nodes - submit from a berzelius-hopper login node.
# One-off data onboarding: an MRI-free "straightened" version of the 2 mm input,
# data/tau_raw_2mm_rigid_pts (or _rigidscale_pts, below). Same pkls (tau_raw_backup, AVID included) and the exact same 2 mm
# recipe as submit_convert-tau-raw-2mm.sh, with one step added in front:
#
#   each native scan is rigidly aligned (rotation + shift; "scale" adds one overall size factor; no warping, no MRI)
#   to a tau-PET template = the average of the MNI tau scans (scripts/make_pet_template.py),
#   so every head has the template's pose. Head tilt differs by site in the 2 mm input
#   (median 4-13 deg, mostly pitch; scripts/check_t1_rotation.py); this removes it.
#
# Head size, voxel size (2 mm) and brightness scaling stay as in tau_raw_2mm_pts, so a run on this
# input differs from the 2 mm run only in head pose. Same order, so demo.csv / demo_AVID_unseen.csv
# and the train/test split are unchanged: train with --data_type tau_raw_2mm_rigid_pts.
#
# tau_pt_scans.csv gets rigid_pitch_deg / rigid_roll_deg / rigid_yaw_deg / rigid_total_deg /
# rigid_shift_mm per scan; the log prints the median turn and warns about scans turned > 30 deg
# (possible failed registration - look at them with results_plotting/inspect_scans.py
# --data_type tau_raw_2mm_rigid_pts before training).
#
# Needs SimpleITK in ai-pet (once: mamba activate ai-pet && pip install SimpleITK).
# Refuses to overwrite an output folder that already has .pt batches; the template is made once.
# Time: ~1.5-3 s per scan -> ~2 h for ~2,500 scans.
#
# Three versions (argument 1):
#   rigid (default) -> data/tau_raw_2mm_rigid_pts       head turned/moved to the template's pose; true head size kept
#   scale           -> data/tau_raw_2mm_rigidscale_pts  same, plus ONE overall scale factor so every head has about
#                      the template's size (same factor in every direction: shape unchanged, no warping);
#                      rigid_scale in tau_pt_scans.csv = head size vs template, the log warns outside 0.8-1.25
#                      (FAILED Nov 2026: fitted on the whole head, skull uptake pulled the size, heads shrunk)
#   template        -> data/tau_template_pts            MRI-free stand-in for tau_T1MNI: rigid fit from a multi-start
#                      search (template head tried at every height of the scan, front/back offsets, nod angles,
#                      scored on 8 mm copies; best 3 fitted, best kept - no neck / top-of-head assumption; the
#                      single top-of-head start failed for ~1 in 4 hard pilot scans), then ONE size
#                      factor fitted only inside the MNI152 brain mask (shrunk 1 voxel, so skull/meninges uptake is
#                      left out) and limited to 0.85-1.15, then turn/shift re-fitted on the brain; repeated twice.
#                      No warping. Output on the template's own grid, 91x109x91 at 2 mm, with the same recipe as
#                      tau_T1MNI_pts, so it is directly comparable with the MNI model (same model settings).
#                      tau_pt_scans.csv: rigid_scale (> 1 = brain bigger than template), rigid_scale_at_limit,
#                      rigid_*_deg, rigid_r_template (match with the template inside the brain = quality score,
#                      needs no MRI). The log lists the 10 lowest and warns below QC_MIN_R (default 0.5, provisional:
#                      set it from the pilot) and about scans that stopped at the size limit. ~7 s per scan. Afterwards, compare
#                      with each person's MNI scan: scripts/check_template_alignment.py (runs at the end of this job).
#                      Train with --data_type tau_template_pts.
#   pilot           -> data/tau_template_pilot_pts      the "template" version on a few scans only, to check it
#                      before the ~2 h full run: per site the 3 scans turned most by the rigid run
#                      (tau_raw_2mm_rigid_pts/tau_pt_scans.csv, if there) + 5 random ones (seed 42), AVID
#                      included -> ~55 scans, minutes. Redone from scratch on every run (only a check, never
#                      train on it). Check output: <proj>/results/template_alignment_pilot/, including
#                      sheet_<site>.png = each pilot scan next to its own MNI scan (view on Berzelius /
#                      ThinLinc, don't download). PILOT_TILTED / PILOT_RANDOM change the counts.
#
# Template: rigid / scale use the 11 Nov 2026 template (all 2,497 MNI scans, AVID and test included).
# template / pilot use tau_pet_template_mni2mm_train = training scans only (no AVID, no held-out test),
# built on the first run from TRAIN_CSV = a seed-42 80/20 run's splits/train_subjects.csv (default: the
# frozen MNI attention run, FROZEN_RUN below; the split is the same for every input). Its <name>_ids.txt lists the scans averaged.
#
# Usage:
#   sbatch submit_convert-tau-raw-2mm-rigid.sh pilot      # (TRAIN_CSV defaults to the frozen MNI attention run's)      # try the template version on ~55 scans first
#   sbatch submit_convert-tau-raw-2mm-rigid.sh            # rigid
#   sbatch submit_convert-tau-raw-2mm-rigid.sh scale      # rigid + size (whole head; failed, kept for the record)
#   sbatch submit_convert-tau-raw-2mm-rigid.sh template   # rigid + size on the brain, MNI-like grid

#SBATCH -A berzelius-2026-231
#SBATCH -p berzelius-hopper-cpu
#SBATCH -c 8
#SBATCH -t 12:00:00
#SBATCH -J convert-tau-raw-2mm-rigid
#SBATCH -o ../../logs/convert-tau-raw-2mm-rigid-slurm-%j.out
#SBATCH -e ../../logs/convert-tau-raw-2mm-rigid-slurm-%j.err

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

python -c "import SimpleITK" 2>/dev/null || { echo "SimpleITK missing: mamba activate ai-pet && pip install SimpleITK"; exit 1; }
export SITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=${SLURM_CPUS_PER_TASK:-8}

DATA=/proj/berzelius-2024-156/users/x_nadpi/data
MNI_PKL_DIR=/proj/berzelius-2024-156/dataset/tau-PET/tau_T1MNI
TEMPLATE=$DATA/templates/tau_pet_template_mni2mm
TEMPLATE_ARGS_MAKE=()
# training scans for the training-only template: the frozen MNI attention model's 80% pool (seed 42; the
# split is the same for every input). Override with TRAIN_CSV=... if needed.
FROZEN_RUN=/proj/berzelius-2024-156/users/x_nadpi/results/tau_T1MNI_pts_Gothenburg_CNN3D_visual_read_2split80-20_stratify-site,visual_read_visual-read-final-17-attention_20261104_160037
TRAIN_CSV="${TRAIN_CSV:-$FROZEN_RUN/splits/train_subjects.csv}"
PKL_DIR=$DATA/tau_raw_backup
MODE="${1:-rigid}"
RECIPE_2MM=(--spacing_mm 2 --image_shape 112 128 96 --head_crop --crop_margin 2)
TEMPLATE_ARGS=(--template_grid --multistart --qc_min_r "${QC_MIN_R:-0.5}" --scale_range 0.85 1.15 --brain_mask_erode 1
               --no-crop_foreground --head_crop --spacing_mm 2 --image_shape 91 109 91)
CHECK_DIR="$(cd .. && pwd)/results/template_alignment"
case "$MODE" in
  rigid) OUT_DIR=$DATA/tau_raw_2mm_rigid_pts; SCALE_ARGS=("${RECIPE_2MM[@]}") ;;
  scale) OUT_DIR=$DATA/tau_raw_2mm_rigidscale_pts; SCALE_ARGS=("${RECIPE_2MM[@]}" --rigid_scale) ;;
  # same recipe as submit_convert-tau-T1MNI.sh after the alignment
  template) OUT_DIR=$DATA/tau_template_pts; SCALE_ARGS=("${TEMPLATE_ARGS[@]}") ;;
  pilot) OUT_DIR=$DATA/tau_template_pilot_pts; SCALE_ARGS=("${TEMPLATE_ARGS[@]}" --ids_file "$OUT_DIR/pilot_ids.txt")
         CHECK_DIR="${CHECK_DIR}_pilot" ;;
  *) echo "Usage: sbatch submit_convert-tau-raw-2mm-rigid.sh [rigid|scale|template|pilot]"; exit 1 ;;
esac

if [[ "$MODE" == "template" || "$MODE" == "pilot" ]]; then
  TEMPLATE=${TEMPLATE}_train  # training scans only
  if [[ ! -f "$TEMPLATE.nii.gz" ]]; then
    [[ -f "$TRAIN_CSV" ]] || { echo "Missing $TRAIN_CSV - set TRAIN_CSV=<seed-42 run>/splits/train_subjects.csv to build $TEMPLATE"; exit 1; }
    TEMPLATE_ARGS_MAKE=(--ids_csv "$TRAIN_CSV")
  fi
fi
if [[ ! -f "$TEMPLATE.nii.gz" ]]; then
  echo "===== template ====="
  python scripts/make_pet_template.py --pkl_dir "$MNI_PKL_DIR" --out "$TEMPLATE" "${TEMPLATE_ARGS_MAKE[@]}" || exit 1
fi
echo "Template: $TEMPLATE.nii.gz"

if [[ "$MODE" == "pilot" ]]; then
  # a check only: start clean every time, then pick the scans
  rm -f "$OUT_DIR"/tau_batch_*.pt "$OUT_DIR"/tau_pt_scans.csv "$OUT_DIR"/tau_pt_manifest.json
  mkdir -p "$OUT_DIR"
  python - "$DATA" "$OUT_DIR/pilot_ids.txt" "${PILOT_TILTED:-3}" "${PILOT_RANDOM:-5}" <<'PY' || exit 1
import sys
from pathlib import Path
import pandas as pd
data, out, n_tilt, n_rand = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
demo = pd.concat([pd.read_csv(data / f)[["ID", "site"]] for f in ("demo.csv", "demo_AVID_unseen.csv")
                  if (data / f).exists()]).drop_duplicates("ID")
rigid = data / "tau_raw_2mm_rigid_pts" / "tau_pt_scans.csv"
if rigid.exists():
    r = pd.read_csv(rigid)
    demo = demo.merge(r[["index", "rigid_total_deg"]].rename(columns={"index": "ID"}), on="ID", how="left")
else:
    print(f"No {rigid}: random scans only")
    demo["rigid_total_deg"] = float("nan")
ids = []
for site, g in demo.groupby("site"):
    tilted = g.dropna(subset=["rigid_total_deg"]).nlargest(n_tilt, "rigid_total_deg")["ID"].tolist()
    rest = g[~g["ID"].isin(tilted)]
    rand = rest.sample(min(n_rand, len(rest)), random_state=42)["ID"].tolist()
    ids += tilted + rand
    print(f"{site}: most turned {tilted}, random {rand}")
out.write_text(" ".join(str(int(i)) for i in sorted(ids)) + "\n")
print(f"{len(ids)} pilot scans -> {out}")
PY
elif compgen -G "$OUT_DIR/tau_batch_*.pt" > /dev/null; then
  echo "Skipping $OUT_DIR: it already has tau_batch_*.pt files (delete them first to redo)."
  exit 0
fi
echo "===== $OUT_DIR ====="
python scripts/convert_tau_pkl_to_pt.py --pkl_dir "$PKL_DIR" --out_dir "$OUT_DIR" --batch_size 50 \
  --rigid_template "$TEMPLATE.nii.gz" "${SCALE_ARGS[@]}" || exit 1

if [[ "$MODE" == "template" || "$MODE" == "pilot" ]]; then
  echo "===== check against the MRI-based MNI scans ====="
  SHEETS=(); [[ "$MODE" == "pilot" ]] && SHEETS=(--sheets)
  python scripts/check_template_alignment.py --template_dir "$OUT_DIR" --mni_dir "$DATA/tau_T1MNI_pts" \
    --demo_csv "$DATA/demo.csv" "$DATA/demo_AVID_unseen.csv" --out_dir "$CHECK_DIR" "${SHEETS[@]}"
fi
