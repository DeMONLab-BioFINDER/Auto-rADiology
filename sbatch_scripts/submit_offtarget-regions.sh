#!/bin/bash
# SLURM batch job script for Berzelius - submit from a login node.
#
# Off-target signal per scan (results_plotting/offtarget_by_group.py) for one final MNI visual-read model, on the
# held-out test set and AVID, one after the other: cortical tau (CenTauR meta-temporal, temporoparietal) and
# off-target signal (choroid plexus via the lateral ventricles, striatum, thalamus, brainstem, rim outside the brain)
# per model / read group. A few minutes of CPU work (a GPU is requested because Berzelius jobs need one).
# Outputs: <run>/figures/offtarget_test/ and <run>/validation/figures/offtarget_avid/
#   offtarget_by_group.csv, offtarget_tests.csv, offtarget_scatter.png (no IDs); offtarget_per_scan.csv (HAS IDs)
#
# Usage (from the Auto-rADiology root):
#   DEMO=/proj/berzelius-2024-156/users/x_nadpi/data/demo.csv sbatch sbatch_scripts/submit_offtarget-regions.sh <final_run_folder>
#   MARK_TEST=12,34,56 ...   ring these test scans on the scatter (e.g. the ones judged 'off-target only' by eye)

#SBATCH -A berzelius-2026-231
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH -t 01:00:00
#SBATCH -J offtarget-regions
#SBATCH -o ../logs/offtarget-regions-%j.out
#SBATCH -e ../logs/offtarget-regions-%j.err

module load Miniforge3/25.3.1-0 2>/dev/null || module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate ai-pet
set -eo pipefail

if [[ "$(basename "$PWD")" == "sbatch_scripts" ]]; then cd .. || exit 1; fi
[[ -d results_plotting ]] || { echo "Run from sbatch_scripts or the Auto-rADiology root"; exit 1; }

RUN_ARG="$1"
[[ -n "$RUN_ARG" ]] || { echo "Usage: sbatch submit_offtarget-regions.sh <final_run_folder>"; exit 1; }
if [[ "$RUN_ARG" = /* ]]; then RUN="$RUN_ARG"; else RUN="$(cd .. && pwd)/results/$RUN_ARG"; fi
[[ -d "$RUN" ]] || { echo "No such run folder: $RUN"; exit 1; }

DATA="${DATA:-/proj/berzelius-2024-156/users/x_nadpi/data}"
DEMO="${DEMO:-$DATA/demo.csv}"
PKL_DIR="${PKL_DIR:-/proj/berzelius-2024-156/dataset/tau-PET/tau_T1MNI}"
COMMON=(--pkl_dir "$PKL_DIR" --ref_mask "$DATA/masks/voi_CerebGry_tau_2mm.nii"
        --atlas_dir "$DATA/atlases/lotter2025_mni_freesurfer" --rim_mm "${RIM_MM:-8}")

echo -e "\n================ test ================"
python results_plotting/offtarget_by_group.py "$RUN" --set test --demo_csv "$DEMO" "${COMMON[@]}" \
  ${MARK_TEST:+--mark_ids "$MARK_TEST"} --out_dir "$RUN/figures/offtarget_test"
echo -e "\n================ AVID ================"
python results_plotting/offtarget_by_group.py "$RUN" --set avid --ext_dataset AVID_unseen \
  --ext_demo_csv "$DATA/demo_AVID_unseen.csv" "${COMMON[@]}" ${MARK_AVID:+--mark_ids "$MARK_AVID"} \
  --out_dir "$RUN/validation/figures/offtarget_avid"
