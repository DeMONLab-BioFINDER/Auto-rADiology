#!/bin/bash
# SLURM batch job script for Berzelius - submit from a login node.
#
# Per scan: SUVR and model evidence inside the clusters of one voxelwise map (results_plotting/cluster_by_scan.py).
# Default map: the precuneus clusters of 'Model - scans: confident Read + vs Read -' on the held-out test set.
# Test = descriptive (how many of the 12 carry it); AVID = independent check with the same clusters. A few minutes.
# Outputs: <run>/figures/cluster_by_scan_test/ and _avid/ (cluster_by_group.csv, cluster_by_scan.png;
#   cluster_per_scan.csv HAS IDs)
#
# Usage (from the Auto-rADiology root):
#   sbatch sbatch_scripts/submit_cluster-by-scan.sh <final_run_folder>
#   CLUSTERS=<path prefix of another voxelwise map> ...

#SBATCH -A berzelius-2026-231
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH -t 00:45:00
#SBATCH -J cluster-by-scan
#SBATCH -o ../logs/cluster-by-scan-%j.out
#SBATCH -e ../logs/cluster-by-scan-%j.err

module load Miniforge3/25.3.1-0 2>/dev/null || module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate ai-pet
set -eo pipefail

if [[ "$(basename "$PWD")" == "sbatch_scripts" ]]; then cd .. || exit 1; fi
[[ -d results_plotting ]] || { echo "Run from sbatch_scripts or the Auto-rADiology root"; exit 1; }

RUN_ARG="$1"
[[ -n "$RUN_ARG" ]] || { echo "Usage: sbatch submit_cluster-by-scan.sh <final_run_folder>"; exit 1; }
if [[ "$RUN_ARG" = /* ]]; then RUN="$RUN_ARG"; else RUN="$(cd .. && pwd)/results/$RUN_ARG"; fi
[[ -d "$RUN" ]] || { echo "No such run folder: $RUN"; exit 1; }

DATA="${DATA:-/proj/berzelius-2024-156/users/x_nadpi/data}"
CLUSTERS="${CLUSTERS:-$RUN/figures/voxelwise_discordant_wholebrain_confident_test/confident_neg_pos_vs_neg}"
[[ -f "${CLUSTERS}_logp_tfce.nii.gz" ]] || { echo "No map ${CLUSTERS}_logp_tfce.nii.gz"; exit 1; }
V="$RUN/validation/visualization"
COMMON=(--clusters "$CLUSTERS" --pkl_dir "${PKL_DIR:-/proj/berzelius-2024-156/dataset/tau-PET/tau_T1MNI}"
        --ref_mask "$DATA/masks/voi_WhlCbl_2mm.nii" --atlas_dir "$DATA/atlases/lotter2025_mni_freesurfer")

echo -e "\n================ test ================"
python results_plotting/cluster_by_scan.py "$RUN" --set test --demo_csv "${DEMO:-$DATA/demo.csv}" "${COMMON[@]}" \
  --occ_dir "$V/occlusion_all_logit_median_brain_mask4_ov0.5" --out_dir "$RUN/figures/cluster_by_scan_test"
echo -e "\n================ AVID ================"
python results_plotting/cluster_by_scan.py "$RUN" --set avid --ext_dataset AVID_unseen \
  --ext_demo_csv "$DATA/demo_AVID_unseen.csv" "${COMMON[@]}" \
  --occ_dir "$V/occlusion_all_AVID_unseen_logit_median_brain_mask4_ov0.5" --out_dir "$RUN/figures/cluster_by_scan_avid"
