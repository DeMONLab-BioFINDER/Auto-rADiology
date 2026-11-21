#!/bin/bash
# End-to-end analysis of ONE final visual-read model: every results_plotting analysis on its held-out test
# set and, separately, on each external set it was evaluated on zero-shot (AVID, or a cohort the run left out
# such as OASIS). CPU only, ~15-30 min; can run on a login node (from the Auto-rADiology root)
#   bash sbatch_scripts/run_visual-read-analysis.sh <final_run_folder>
# or as a job (from the root as well)
#   sbatch sbatch_scripts/run_visual-read-analysis.sh <final_run_folder>
# Each step is independent: a failing step is reported at the end and the rest still run.
#
# Settings (put them in front of the command, e.g. EXTS="OASIS_unseen AVID_unseen" bash ...):
#   EXTS      external sets to analyse (default: every External_validation_*_zeroshot_results.csv in the run)
#   DEMO      demographics of the training pool + test set (default data/demo.csv or ../data/demo.csv; when the
#             run left a cohort out, pass the file without it, so the dataset description doesn't include it)
#   DATA      folder with demo_<external set>.csv, the reader-vote files and masks/ (default ../data)
#   PKL_DIR   original MNI images, for the asymmetry analysis (default the shared tau_T1MNI folder)
#
# Outputs: <run>/figures/... (test set); per external set <ext>: <run>/validation/figures_<ext>/ and
# <run>/figures/<analysis>_<ext>/; <proj_path>/results/dataset_description[_<ext>], excluded_regions,
# reader_votes. Log of everything: <run>/analysis_log_<date>.txt
# Not here (long jobs of their own): occlusion maps (submit_vis-occlusion-all.sh, then merge_occlusion_chunks.py
# and occlusion_group_maps.py) and voxelwise statistics (submit_voxelwise-stats.sh).
# run_visual-read-pipeline.sh runs all of it, from training onwards.

#SBATCH -A berzelius-2026-231
#SBATCH --gpus=1
#SBATCH -t 02:00:00
#SBATCH -J visual-read-analysis
#SBATCH -o ../logs/visual-read-analysis-slurm-%j.out
#SBATCH -e ../logs/visual-read-analysis-slurm-%j.err

if [[ -n "$SLURM_JOB_ID" ]]; then
  module load Miniforge3/25.3.1-0 2>/dev/null || module load Miniforge3/24.7.1-2-hpc1-bdist
  mamba activate ai-pet
fi
if [[ "$(basename "$PWD")" == "sbatch_scripts" ]]; then cd .. || exit 1; fi
if [[ ! -d results_plotting ]]; then echo "Run from the Auto-rADiology root (or sbatch_scripts)"; exit 1; fi

RUN_ARG="$1"
if [[ -z "$RUN_ARG" ]]; then echo "Usage: bash sbatch_scripts/run_visual-read-analysis.sh <final_run_folder>"; exit 1; fi
# relative folder names resolve under <proj_path>/results, like the sbatch scripts
if [[ "$RUN_ARG" = /* ]]; then RUN="$RUN_ARG"; else RUN="$(cd .. && pwd)/results/$RUN_ARG"; fi
[[ -d "$RUN" ]] || { echo "No such run folder: $RUN"; exit 1; }
PROJ_RESULTS="${PROJ_RESULTS:-$(cd .. && pwd)/results}"
DATA="${DATA:-$(cd .. && pwd)/data}"
first_existing() { for f in "$@"; do [[ -f "$f" ]] && { echo "$f"; return; }; done; echo "$1"; }
DEMO="${DEMO:-$(first_existing data/demo.csv "$DATA/demo.csv")}"
PKL_DIR="${PKL_DIR:-/proj/berzelius-2024-156/dataset/tau-PET/tau_T1MNI}"
MASKS="$DATA/masks"
[[ -f "$DEMO" ]] || echo "WARNING: $DEMO not found (pass DEMO=...)"
LABEL=$(basename "$RUN" | grep -o "final-[^_]*" || echo model)
LOG="$RUN/analysis_log_$(date +%Y%m%d_%H%M).txt"
FAILED=()

step() {  # step "<name>" cmd args...
  local name="$1"; shift
  echo -e "\n================ $name ================" | tee -a "$LOG"
  if "$@" 2>&1 | tee -a "$LOG"; [[ ${PIPESTATUS[0]} -eq 0 ]]; then :; else FAILED+=("$name"); fi
}

# external sets: given, or every zero-shot result the run has
if [[ -z "${EXTS:-}" ]]; then
  EXTS=$(ls "$RUN"/validation/External_validation_*__*_zeroshot_results.csv 2>/dev/null \
         | sed -E 's#.*/External_validation_(.*)__.*#\1#' | sort -u | tr '\n' ' ')
fi
echo "Run: $RUN" | tee "$LOG"
echo "Demographics: $DEMO (test); external sets: ${EXTS:-none}" | tee -a "$LOG"

# ---- held-out test set
T=(--dataset Gothenburg --demo_csv "$DEMO")
step "A test: performance summary" python results_plotting/performance_summary_visual_read.py "$RUN" --sets test --demo_csv "$DEMO"
step "F dataset description" python results_plotting/describe_dataset.py --demo_csv "$DEMO" --run_dir "$RUN"
step "A test: per site" python results_plotting/site_breakdown_visual_read.py "$RUN" "${T[@]}"
step "A test: vs SUVR threshold" python results_plotting/compare_model_vs_suvr_threshold.py "$RUN" "${T[@]}"
step "B test: confidence and calibration" python results_plotting/analyze_model_confidence_visual_read.py "$RUN" "${T[@]}"
step "C test: discordant cases" python results_plotting/list_discordant_cases.py "$RUN" "${T[@]}"
step "C test: discordant demographics" python results_plotting/plot_discordant_demographics.py "$RUN" "${T[@]}"
step "C test: discordant findings figure" python results_plotting/plot_discordant_findings.py "$RUN" "${T[@]}"
step "C test: subgroups" python results_plotting/analyze_subgroup_performance_visual_read.py "$RUN" "${T[@]}"
step "C test: same-SUVR check" python results_plotting/analyze_errors_at_same_suvr.py "$RUN" "${T[@]}"
step "C test: regressions" python results_plotting/regress_model_output_visual_read.py "$RUN" --sets test --demo_csv "$DEMO"
step "region rule test" python results_plotting/check_excluded_regions.py "$RUN" --labels "$LABEL" --sets test --demo_csv "$DEMO" \
  --out_dir "$PROJ_RESULTS/excluded_regions"
if [[ -f "$DATA/IDs_reads.csv" ]]; then
  step "reader votes test" python results_plotting/analyze_reader_votes.py "$RUN" --votes "$DATA/IDs_reads.csv" --demo_csv "$DEMO"
fi
step "asymmetry test" python results_plotting/asymmetry_visual_read.py "$RUN" --set test --pkl_dir "$PKL_DIR" \
  --ref_mask "$MASKS/voi_CerebGry_tau_2mm.nii" --centaur_dir "$MASKS" --demo_csv "$DEMO"
step "Figure 1 test" python results_plotting/ohbm_figure1.py "$RUN" --demo_csv "$DEMO"
step "Figure 1 test, Universal" python results_plotting/ohbm_figure1.py "$RUN" --demo_csv "$DEMO" --region Universal

# ---- each external set (zero-shot), never pooled with the test set
for EXT in $EXTS; do
  EXT_DEMO="$DATA/demo_${EXT}.csv"
  ext=$(echo "${EXT%_unseen}" | tr '[:upper:]' '[:lower:]')
  if [[ ! -f "$EXT_DEMO" ]]; then
    echo "No $EXT_DEMO: $EXT skipped" | tee -a "$LOG"; FAILED+=("$EXT (no demographics)"); continue
  fi
  # the per-dataset scripts write to <run>/validation/figures: start empty, and move it to figures_<ext> after
  [[ -d "$RUN/validation/figures" ]] && mv "$RUN/validation/figures" "$RUN/validation/figures_previous_$(date +%s)"
  A=(--dataset "$EXT" --demo_csv "$EXT_DEMO")
  X=(--ext_dataset "$EXT" --ext_demo_csv "$EXT_DEMO")
  step "A $EXT: performance summary (next to test)" python results_plotting/performance_summary_visual_read.py "$RUN" \
    --sets test avid --demo_csv "$DEMO" "${X[@]}" --out_dir "$RUN/figures/performance_summary_$ext"
  step "F $EXT dataset description" python results_plotting/describe_dataset.py --demo_csv "$EXT_DEMO" \
    --out_dir "$PROJ_RESULTS/dataset_description_$ext"
  step "A $EXT: vs SUVR threshold" python results_plotting/compare_model_vs_suvr_threshold.py "$RUN" "${A[@]}" --train_demo_csv "$DEMO"
  step "B $EXT: confidence and calibration" python results_plotting/analyze_model_confidence_visual_read.py "$RUN" "${A[@]}"
  step "C $EXT: discordant cases" python results_plotting/list_discordant_cases.py "$RUN" "${A[@]}"
  step "C $EXT: discordant demographics" python results_plotting/plot_discordant_demographics.py "$RUN" "${A[@]}"
  step "C $EXT: discordant findings figure" python results_plotting/plot_discordant_findings.py "$RUN" "${A[@]}"
  step "C $EXT: subgroups" python results_plotting/analyze_subgroup_performance_visual_read.py "$RUN" "${A[@]}"
  step "C $EXT: same-SUVR check" python results_plotting/analyze_errors_at_same_suvr.py "$RUN" "${A[@]}"
  step "C $EXT: regressions" python results_plotting/regress_model_output_visual_read.py "$RUN" --sets avid "${X[@]}" \
    --out_dir "$RUN/figures/regressions_$ext"
  step "region rule $EXT" python results_plotting/check_excluded_regions.py "$RUN" --labels "$LABEL" --sets avid "${X[@]}" \
    --out_dir "$PROJ_RESULTS/excluded_regions/$ext"
  # reader votes: AVID has its own file; a left-out cohort is in the training-pool file
  VOTES="$DATA/IDs_reads.csv"; [[ "$EXT" == *AVID* ]] && VOTES="$DATA/AVIDIDs_reads.csv"
  if [[ -f "$VOTES" ]]; then
    step "reader votes $EXT" python results_plotting/analyze_reader_votes.py "$RUN" --votes "$VOTES" --dataset "$EXT" \
      --demo_csv "$EXT_DEMO"
  fi
  step "asymmetry $EXT" python results_plotting/asymmetry_visual_read.py "$RUN" --set avid "${X[@]}" --pkl_dir "$PKL_DIR" \
    --ref_mask "$MASKS/voi_CerebGry_tau_2mm.nii" --centaur_dir "$MASKS" --out_dir "$RUN/figures/asymmetry_$ext"
  step "Figure 1 $EXT" python results_plotting/ohbm_figure1.py "$RUN" --sets avid "${X[@]}"
  step "Figure 1 $EXT, Universal" python results_plotting/ohbm_figure1.py "$RUN" --sets avid "${X[@]}" --region Universal
  if [[ -d "$RUN/validation/figures" ]]; then
    rm -rf "$RUN/validation/figures_$ext"
    mv "$RUN/validation/figures" "$RUN/validation/figures_$ext"
  fi
done

echo -e "\n================ done ================" | tee -a "$LOG"
if (( ${#FAILED[@]} )); then
  echo "FAILED steps (see the log): ${FAILED[*]}" | tee -a "$LOG"
else
  echo "All steps ran." | tee -a "$LOG"
fi
echo "Log: $LOG"
