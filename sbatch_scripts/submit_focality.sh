#!/bin/bash
# SLURM batch job script for Berzelius - submit from a login node.
#
# What kind of tau does the model miss? (results_plotting/focality_by_group.py) for one final MNI visual-read model,
# on the held-out test set and AVID, one after the other. Per scan, SUVR on the readers' scale (whole cerebellum),
# cortex only: peak (2 ml spot around the hottest point) and where it is, extent above 1.65 x cerebellum, striatum +
# thalamus signal, the reference difference (cerebellar grey vs whole), and the share of the model's positive occlusion
# evidence on the hot spot; per-group map of where scans are above 1.65. A few minutes of CPU work (a GPU is requested
# because Berzelius jobs need one).
# Outputs: <run>/figures/focality_test/ and <run>/validation/figures/focality_avid/
#   focality_by_group.csv, focality_tests.csv, focality_scatter.png, focality_maps.png (no IDs);
#   focality_per_scan.csv (HAS IDs)
#
# Usage (from the Auto-rADiology root):
#   sbatch sbatch_scripts/submit_focality.sh <final_run_folder>
# Occlusion maps of every scan are taken from <run>/validation/visualization/occlusion_all_*_logit_median_brain_mask4_ov0.5
# (override with OCC_TEST= / OCC_AVID=).

#SBATCH -A berzelius-2026-231
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH -t 01:00:00
#SBATCH -J focality
#SBATCH -o ../logs/focality-%j.out
#SBATCH -e ../logs/focality-%j.err

module load Miniforge3/25.3.1-0 2>/dev/null || module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate ai-pet
set -eo pipefail

if [[ "$(basename "$PWD")" == "sbatch_scripts" ]]; then cd .. || exit 1; fi
[[ -d results_plotting ]] || { echo "Run from sbatch_scripts or the Auto-rADiology root"; exit 1; }

RUN_ARG="$1"
[[ -n "$RUN_ARG" ]] || { echo "Usage: sbatch submit_focality.sh <final_run_folder>"; exit 1; }
if [[ "$RUN_ARG" = /* ]]; then RUN="$RUN_ARG"; else RUN="$(cd .. && pwd)/results/$RUN_ARG"; fi
[[ -d "$RUN" ]] || { echo "No such run folder: $RUN"; exit 1; }

DATA="${DATA:-/proj/berzelius-2024-156/users/x_nadpi/data}"
DEMO="${DEMO:-$DATA/demo.csv}"
PKL_DIR="${PKL_DIR:-/proj/berzelius-2024-156/dataset/tau-PET/tau_T1MNI}"
V="$RUN/validation/visualization"
OCC_TEST="${OCC_TEST:-$V/occlusion_all_logit_median_brain_mask4_ov0.5}"
OCC_AVID="${OCC_AVID:-$V/occlusion_all_AVID_unseen_logit_median_brain_mask4_ov0.5}"
COMMON=(--pkl_dir "$PKL_DIR" --ref_mask "$DATA/masks/voi_WhlCbl_2mm.nii" --grey_ref "$DATA/masks/voi_CerebGry_tau_2mm.nii" --centaur_dir "$DATA/masks"
        --atlas_dir "$DATA/atlases/lotter2025_mni_freesurfer")
for d in "$OCC_TEST" "$OCC_AVID"; do [[ -d "$d" ]] || echo "WARNING: no occlusion folder $d (step 2 skipped for that set)"; done

echo -e "\n================ test ================"
python results_plotting/focality_by_group.py "$RUN" --set test --demo_csv "$DEMO" "${COMMON[@]}" \
  $([[ -d "$OCC_TEST" ]] && echo --occ_dir "$OCC_TEST") --out_dir "$RUN/figures/focality_test"
echo -e "\n================ AVID ================"
python results_plotting/focality_by_group.py "$RUN" --set avid --ext_dataset AVID_unseen \
  --ext_demo_csv "$DATA/demo_AVID_unseen.csv" "${COMMON[@]}" $([[ -d "$OCC_AVID" ]] && echo --occ_dir "$OCC_AVID") \
  --out_dir "$RUN/validation/figures/focality_avid"
