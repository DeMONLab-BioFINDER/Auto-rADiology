#!/bin/bash
# End-to-end analysis of ONE final visual-read model: every results_plotting analysis on its held-out test
# set and, separately, on its zero-shot AVID result. CPU only, ~10-20 min; can run on a login node
#   bash sbatch_scripts/run_visual-read-analysis.sh <final_run_folder>
# or as a job
#   sbatch sbatch_scripts/run_visual-read-analysis.sh <final_run_folder>
# Each step is independent: a failing step is reported at the end and the rest still run.
# Needs data/demo.csv (training pool + test set) and data/demo_AVID_unseen.csv (AVID), and the model's AVID
# evaluation (submit_val_AVID_unseen_visual-read.sh) for the AVID part (skipped if it is missing).
#
# Outputs land where each script puts them: <run>/figures/... (test set), <run>/validation/figures/... (AVID),
# <proj_path>/results/dataset_description[_AVID], <proj_path>/results/excluded_regions. Log of everything:
#   <run>/analysis_log_<date>.txt
#
# Parts of the OHBM plan covered: A (performance, kappa, per cohort, SUVR baseline), B (confidence, calibration),
# C (discordant cases, subgroups, same-SUVR check), region rule, F (dataset description).
# Not here yet: C multivariable regressions, D occlusion group maps, E voxelwise statistics.

#SBATCH -A berzelius-2026-231
#SBATCH --gpus=1
#SBATCH -t 01:00:00
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
DEMO="${DEMO:-data/demo.csv}"
DEMO_AVID="${DEMO_AVID:-data/demo_AVID_unseen.csv}"
LABEL=$(basename "$RUN" | grep -o "final-[^_]*" || echo model)
LOG="$RUN/analysis_log_$(date +%Y%m%d_%H%M).txt"
FAILED=()

step() {  # step "<name>" cmd args...
  local name="$1"; shift
  echo -e "\n================ $name ================" | tee -a "$LOG"
  if "$@" 2>&1 | tee -a "$LOG"; [[ ${PIPESTATUS[0]} -eq 0 ]]; then :; else FAILED+=("$name"); fi
}

echo "Run: $RUN" | tee "$LOG"
HAS_AVID=$(ls "$RUN"/validation/External_validation_AVID_unseen__*_zeroshot_results.csv 2>/dev/null | head -1)
[[ -n "$HAS_AVID" ]] || echo "No AVID result in $RUN/validation: AVID steps skipped" | tee -a "$LOG"
SETS=(test); [[ -n "$HAS_AVID" ]] && SETS+=(avid)

# ---- A. headline performance (both sets, side by side but never pooled)
step "A performance summary (${SETS[*]})" python results_plotting/performance_summary_visual_read.py "$RUN" \
  --sets "${SETS[@]}" --demo_csv "$DEMO" --ext_demo_csv "$DEMO_AVID"

# ---- held-out test set
step "F dataset description" python results_plotting/describe_dataset.py --demo_csv "$DEMO" --run_dir "$RUN"
step "A test: per site" python results_plotting/site_breakdown_visual_read.py "$RUN" --dataset Gothenburg --demo_csv "$DEMO"
step "A test: vs SUVR threshold" python results_plotting/compare_model_vs_suvr_threshold.py "$RUN" --dataset Gothenburg --demo_csv "$DEMO"
step "B test: confidence and calibration" python results_plotting/analyze_model_confidence_visual_read.py "$RUN" --dataset Gothenburg --demo_csv "$DEMO"
step "C test: discordant cases" python results_plotting/list_discordant_cases.py "$RUN" --dataset Gothenburg --demo_csv "$DEMO"
step "C test: discordant demographics" python results_plotting/plot_discordant_demographics.py "$RUN" --dataset Gothenburg --demo_csv "$DEMO"
step "C test: discordant findings figure" python results_plotting/plot_discordant_findings.py "$RUN" --dataset Gothenburg --demo_csv "$DEMO"
step "C test: subgroups" python results_plotting/analyze_subgroup_performance_visual_read.py "$RUN" --dataset Gothenburg --demo_csv "$DEMO"
step "C test: same-SUVR check" python results_plotting/analyze_errors_at_same_suvr.py "$RUN" --dataset Gothenburg --demo_csv "$DEMO"
step "region rule test" python results_plotting/check_excluded_regions.py "$RUN" --labels "$LABEL" --sets test --demo_csv "$DEMO" --out_dir "$PROJ_RESULTS/excluded_regions"

# ---- AVID (external, zero-shot)
if [[ -n "$HAS_AVID" ]]; then
  A=(--dataset AVID_unseen --demo_csv "$DEMO_AVID")
  step "F AVID dataset description" python results_plotting/describe_dataset.py --demo_csv "$DEMO_AVID" \
    --out_dir "$PROJ_RESULTS/dataset_description_AVID"
  step "A AVID: vs SUVR threshold" python results_plotting/compare_model_vs_suvr_threshold.py "$RUN" "${A[@]}" --train_demo_csv "$DEMO"
  step "B AVID: confidence and calibration" python results_plotting/analyze_model_confidence_visual_read.py "$RUN" "${A[@]}"
  step "C AVID: discordant cases" python results_plotting/list_discordant_cases.py "$RUN" "${A[@]}"
  step "C AVID: discordant demographics" python results_plotting/plot_discordant_demographics.py "$RUN" "${A[@]}"
  step "C AVID: discordant findings figure" python results_plotting/plot_discordant_findings.py "$RUN" "${A[@]}"
  step "C AVID: subgroups" python results_plotting/analyze_subgroup_performance_visual_read.py "$RUN" "${A[@]}"
  step "C AVID: same-SUVR check" python results_plotting/analyze_errors_at_same_suvr.py "$RUN" "${A[@]}"
  step "region rule AVID" python results_plotting/check_excluded_regions.py "$RUN" --labels "$LABEL" --sets avid --out_dir "$PROJ_RESULTS/excluded_regions" \
    --ext_demo_csv "$DEMO_AVID"
fi

echo -e "\n================ done ================" | tee -a "$LOG"
if (( ${#FAILED[@]} )); then
  echo "FAILED steps (see the log): ${FAILED[*]}" | tee -a "$LOG"
else
  echo "All steps ran." | tee -a "$LOG"
fi
echo "Log: $LOG"
