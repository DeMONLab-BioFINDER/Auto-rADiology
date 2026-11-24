#!/bin/bash
# SLURM batch job script for Berzelius (or run with bash on a login node: CPU only, ~15-30 min).
#
# Discordant-case follow-up for one final MNI visual-read model that already went through
# run_visual-read-analysis.sh and occlusion (submit_vis-occlusion-all.sh + merge_occlusion_chunks.py).
# Runs only the new / changed steps, on the held-out test set and AVID separately (never pooled):
#   1. reader votes (adds 6_grade_at_same_suvr.csv: reader tau score at the same meta-temporal SUVR)
#   2. occlusion group maps (CenTauR shares + the share of evidence outside the cortex)
#   3. occlusion_vs_tau.py (evidence share per CenTauR region next to the Read + vs Read - SUVR difference)
#   4. discordant_summary_table.py (one table: plain + same-tau-level results, collated)
#   5. discordant_by_confidence.py (every disagreeing scan by confidence band, everything joined)
# The voxelwise maps are a separate job: submit_voxelwise-discordant.sh (see its usage for this model).
#
# Usage (from the Auto-rADiology root):
#   sbatch sbatch_scripts/run_discordant-followup.sh <final run folder>
#   bash   sbatch_scripts/run_discordant-followup.sh <final run folder>
# Settings: DATA (default <proj>/data, with demo.csv, demo_AVID_unseen.csv, IDs_reads.csv, AVIDIDs_reads.csv, masks/),
#           EXT (default AVID_unseen).

#SBATCH -A berzelius-2026-231
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH -J discordant-followup
#SBATCH -o ../logs/discordant-followup-%j.out
#SBATCH -e ../logs/discordant-followup-%j.err

module load Miniforge3/25.3.1-0 2>/dev/null || module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate ai-pet 2>/dev/null || true
set -o pipefail

if [[ "$(basename "$PWD")" == "sbatch_scripts" ]]; then cd .. || exit 1; fi
[[ -d results_plotting ]] || { echo "Run from sbatch_scripts or the Auto-rADiology root"; exit 1; }
RUN_ARG="$1"
[[ -n "$RUN_ARG" ]] || { echo "Usage: sbatch run_discordant-followup.sh <final run folder>"; exit 1; }
if [[ "$RUN_ARG" = /* ]]; then RUN="$RUN_ARG"; else RUN="$(cd .. && pwd)/results/$RUN_ARG"; fi
[[ -d "$RUN" ]] || { echo "No such run folder: $RUN"; exit 1; }

PROJ_RESULTS="${PROJ_RESULTS:-$(cd .. && pwd)/results}"
DATA="${DATA:-$(cd .. && pwd)/data}"
EXT="${EXT:-AVID_unseen}"
ext=$(echo "${EXT%_unseen}" | tr '[:upper:]' '[:lower:]')
MASKS="$DATA/masks"
RUN_NAME=$(basename "$RUN")
VIS="$RUN/validation/visualization"
LOG="$RUN/discordant_followup_log_$(date +%Y%m%d_%H%M).txt"
FAILED=()
step() {  # step "<name>" cmd args...
  local name="$1"; shift
  echo -e "\n================ $name ================" | tee -a "$LOG"
  if "$@" 2>&1 | tee -a "$LOG"; [[ ${PIPESTATUS[0]} -eq 0 ]]; then :; else FAILED+=("$name"); fi
}
echo "Run: $RUN | test + $EXT" | tee "$LOG"

# set: label | demo csv | votes csv | predictions dataset name | confidence/figures folder | asymmetry folder | occlusion folder
EXT_FIG="$RUN/validation/figures_$ext"; [[ -d "$EXT_FIG" ]] || EXT_FIG="$RUN/validation/figures"
SETS=(
  "test|$DATA/demo.csv|$DATA/IDs_reads.csv|Gothenburg|$RUN/figures|$RUN/figures/asymmetry_test|$VIS/occlusion_all_logit_median_brain_mask4_ov0.5"
  "$ext|$DATA/demo_${EXT}.csv|$DATA/AVIDIDs_reads.csv|$EXT|$EXT_FIG|$RUN/figures/asymmetry_$ext|$VIS/occlusion_all_${EXT}_logit_median_brain_mask4_ov0.5"
)
[[ "$EXT" == *AVID* ]] || SETS[1]="${SETS[1]/AVIDIDs_reads.csv/IDs_reads.csv}"  # a left-out cohort is in the pool file

for S in "${SETS[@]}"; do
  IFS="|" read -r label demo votes dataset fig asym occ <<< "$S"
  name="Held-out test"; [[ "$label" != test ]] && name="$EXT"
  votes_out="$PROJ_RESULTS/reader_votes/${RUN_NAME}_${dataset}"

  [[ -f "$votes" ]] && step "1 reader votes $label" python results_plotting/analyze_reader_votes.py "$RUN" \
    --votes "$votes" --dataset "$dataset" --demo_csv "$demo" --out_dir "$PROJ_RESULTS/reader_votes"
  if [[ -f "$occ/summary.csv" ]]; then
    step "2 occlusion group maps $label" python results_plotting/occlusion_group_maps.py "$occ" --centaur_dir "$MASKS" \
      --atlas_dir "$DATA/atlases/lotter2025_mni_freesurfer" --label "$name"
    step "3 evidence vs tau $label" python results_plotting/occlusion_vs_tau.py "$occ/group_maps" --demo_csv "$demo" --label "$name"
  else
    echo "No merged occlusion folder $occ - steps 2-3 skipped for $label" | tee -a "$LOG"; FAILED+=("occlusion $label")
  fi
  step "4 summary table $label" python results_plotting/discordant_summary_table.py --figures "$fig" \
    --asymmetry_tests "$asym/asymmetry_tests.csv" --votes_dir "$votes_out" --label "$name"
  OCC_ARG=(); [[ -f "$occ/group_maps/centaur_shares_per_scan.csv" ]] && OCC_ARG=(--occlusion "$occ/group_maps/centaur_shares_per_scan.csv")
  step "5 by confidence $label" python results_plotting/discordant_by_confidence.py \
    --confidence "$fig/visual_read_confidence_groups.csv" --demo_csv "$demo" --votes "$votes" \
    --asymmetry "$asym/asymmetry_per_scan.csv" "${OCC_ARG[@]}" --label "$name" --out_dir "$fig/discordant_by_confidence"
done

echo -e "\n================ done ================" | tee -a "$LOG"
if (( ${#FAILED[@]} )); then echo "FAILED / skipped: ${FAILED[*]}" | tee -a "$LOG"; else echo "All steps ran." | tee -a "$LOG"; fi
echo "Log: $LOG"
