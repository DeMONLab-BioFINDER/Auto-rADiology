# Auto-rADiology

## Purpose

This repository contains the code used to develop and evaluate a 3D convolutional neural network for predicting regional tau-PET SUVR values from PET images. The analyses reported in this project comprise model development on the training cohort, final train/test evaluation, and external validation on the AVID unseen dataset.

The main final outputs are:

- results from 5-fold cross-validation on the training pool
- final train/test model
- zero-shot external validation on the AVID unseen dataset
- summary plots for these results

The code used for data preprocessing, model training, and evaluation is available at: https://github.com/DeMONLab-BioFINDER/Auto-rADiology/tree/regional-suvr-nadine

## Project Overview

Input:

- tau-PET images
- demographic / label tables

Prediction targets used in the final project:

- `MetaTemporal`
- `MesialTemporal`
- `Frontal`
- `TemporoParietal`

Main model:

- `CNN3D`

Main training script:

- [run.py](run.py)

Main external validation script:

- [run_val.py](run_val.py)

Main plotting scripts:

- [results_plotting/plot_heldout_test_results.py](results_plotting/plot_heldout_test_results.py)
- [results_plotting/plot_unseen_validation_results.py](results_plotting/plot_unseen_validation_results.py)
- [results_plotting/plot_visual_read_results.py](results_plotting/plot_visual_read_results.py)

## Repository Structure

Main scripts:

- [run.py](run.py): training, train/validation/test split, and k-fold CV
- [run_val.py](run_val.py): external validation / zero-shot evaluation

Core modules in [src/](src/):

- [src/params.py](src/params.py): command-line arguments and output folder naming
- [src/data.py](src/data.py): data loading, demographics loading, image discovery
- [src/train.py](src/train.py): model training and inference
- [src/cv.py](src/cv.py): cross-validation logic
- [src/models.py](src/models.py): model definitions
- [src/hypertune.py](src/hypertune.py): hyperparameter tuning with optuna
- [src/validation.py](src/validation.py): external validation logic
- [src/utils.py](src/utils.py): utility functions

Plotting and utilities:

- [results_plotting/](results_plotting/): plotting scripts for results visualization
- [sbatch_scripts/](sbatch_scripts/): SLURM scripts used on Berzelius
- [environment_Berzelius.yml](environment_Berzelius.yml): conda environment file

Data utilities (onboarding new raw data into the cached tensor format the pipeline expects):

- [scripts/convert_tau_pkl_to_pt.py](scripts/convert_tau_pkl_to_pt.py): convert intermediate pickle batches to PyTorch `.pt` batches
- [scripts/transform.py](scripts/transform.py): preprocess raw BIDS PET images into per-scan `.pt` tensors, anonymizing subject IDs in the process (original ID → shuffled ID mapping saved alongside the output)

## Software Environment

The project was run using NSC's Berzelius high performance computing environment. The environment used on Berzelius is defined in:

- [environment_Berzelius.yml](environment_Berzelius.yml)

Create environment:

```bash
conda env create -f environment_Berzelius.yml
conda activate ai-pet
```

Key packages:

- Python 3.10
- PyTorch 2.4
- MONAI
- pandas
- scikit-learn
- matplotlib
- nibabel
- optuna

## Data Requirements

The code expects image data plus demographics/label tables.

Important note:

- the repository does **not** contain the original medical image data
- to reproduce results, the original data must be available in the expected folder format
- on Berzelius, the cached `tau_batch_*.pt` files were generated with [scripts/convert_tau_pkl_to_pt.py](scripts/convert_tau_pkl_to_pt.py) so that the MONAI preprocessing step could be completed on CPU before training
- this preprocessing step reduced the risk of GPU starvation caused by slow data loading, which can otherwise lead Berzelius to terminate a job while the GPU waits for input
- the resulting `.pt` batches are already preprocessed; when `tau_batch_*.pt` files are present, the loader path in [src/data.py](src/data.py) bypasses MONAI transforms
- if only raw images are available, the code can still apply MONAI preprocessing on the fly through the raw-image loading path rather than the cached `.pt` path

By default, if `--input_path` is not given, the code uses:

```text
<project_parent>/data
```

Demographic table loading is handled in [src/data.py](src/data.py).

Expected metadata columns include:

- `ID`
- `site`
- `visual_read`
- `CL`
- `age`
- `gender`

For this final project, the target columns also need to exist:

- `MetaTemporal`
- `MesialTemporal`
- `Frontal`
- `TemporoParietal`

## Primary Workflows

The final project includes three main training/evaluation workflows:

### 1. 5-fold CV

Estimate performance on the 80% training pool using 5-fold cross-validation, keeping the 20% hold-out test split untouched.

Run via:

```bash
sbatch sbatch_scripts/submit_mse-cv5.sh
```

Or directly:

```bash
python run.py \
  --no-tune \
  --dataset Gothenburg \
  --data_type tau_raw \
  --targets MetaTemporal,MesialTemporal,Frontal,TemporoParietal \
  --stratifycvby site,Universal \
  --reg_loss mse \
  --train_size 0.80 \
  --val_size 0.00 \
  --test_size 0.20 \
  --run_kfold_cv \
  --n_splits 5 \
  --es_patience 15 \
  --es_min_delta 0.001 \
  --model_name_extra mse-cv5
```

Results are saved to a timestamped folder in `results/`.

### 2. Final Train/Test Model

Train final model on the 80% training pool and evaluate once on the 20% hold-out test set, using the median epoch for early stopping that we found in the previous step (47).

Run via:

```bash
sbatch sbatch_scripts/submit_mse-final-47.sh
```

Or directly:

```bash
python run.py \
  --no-tune \
  --dataset Gothenburg \
  --data_type tau_raw \
  --targets MetaTemporal,MesialTemporal,Frontal,TemporoParietal \
  --stratifycvby site,Universal \
  --reg_loss mse \
  --train_size 0.80 \
  --val_size 0.00 \
  --test_size 0.20 \
  --epochs 47 \
  --model_name_extra mse-final-47
```

Results are saved to a timestamped folder in `results/`.

### 3. AVID Unseen External Validation

Evaluate the final Gothenburg-trained model on an unseen external dataset (zero-shot).

Run via:

```bash
sbatch sbatch_scripts/submit_val_AVID_unseen.sh
```

Or directly:

```bash
python run_val.py \
  --dataset AVID_unseen \
  --best_model_folder <best_model_folder_name> \
  --model CNN3D \
  --data_type tau_raw \
  --targets MetaTemporal,MesialTemporal,Frontal,TemporoParietal \
  --few_shot 0
```

Replace `<best_model_folder_name>` with the folder name from the final Gothenburg run.

Training and evaluation runs automatically create timestamped folders in a `results/` directory (created at runtime).

Typical saved outputs include:

- `RUN_INFO.txt`: dataset/targets/mode/split summary for the run
- `splits/`: the overall train/val/test partition (subject-level CSVs) and a leakage audit
- `final_model/`: checkpoints, per-epoch metrics/plots, and test predictions for the one trained model (direct train/test, train/val/test, or a hypertune outer retrain)
- `evaluation/<dataset>/`: final held-out test predictions and summary metrics
- `metrics.csv` and `oof_predictions.csv` (`--run_kfold_cv` runs only): per-fold summary metrics, and out-of-fold per-subject predictions tagged by fold; each `kfold-N/` folder has its own checkpoints/metrics/predictions

## Plotting Workflow

After generating result folders from the training/validation runs, use the plotting scripts in [results_plotting/](results_plotting/):

```bash
# Regional SUVR model: held-out test results (test-set scatter panel + agreement + subgroup MAE)
python results_plotting/plot_heldout_test_results.py \
  <path_to_result_folder>

# Regional SUVR model: AVID unseen external validation results
python results_plotting/plot_unseen_validation_results.py \
  --final_run_dir <path_to_external_validation_result_folder>

# Regional SUVR model: ground-truth SUVR distributions (discovery + AVID, Universal + per-region)
python results_plotting/plot_suvr_distributions.py

# Regional SUVR model: per-fold metric spread from a --run_kfold_cv run (e.g. submit_mse-cv5.sh)
python results_plotting/plot_cv_summary.py \
  <path_to_result_folder>

# Visual-read model: class balance + agreement with the expert clinical read
python results_plotting/plot_visual_read_results.py \
  <path_to_result_folder>
```

All figures land in `<path_to_result_folder>/figures/` (AVID validation figures land in `<path_to_result_folder>/validation/figures/`, since `--final_run_dir` there is the validation subfolder). Shared plotting helpers (styling, formatting, figure builders, stats) live in [results_plotting/plot_utils.py](results_plotting/plot_utils.py) and are imported by all five scripts above.

`submit_mse-final-47.sh`, `submit_mse-cv5.sh`, and `submit_visual-read-subset-test.sh` already call the matching plotting script automatically right after training, as a separate non-fatal step (a plotting bug won't mark the training job itself as failed) — you only need to run these by hand for AVID validation, ground-truth distributions, or to regenerate a run's figures later.

## Generated Figures

Regional SUVR model, from `plot_heldout_test_results.py` (held-out test set):

- `suvr_true_vs_predicted_test.png` — 4-panel per-region scatter, annotated with Pearson r, MAE, RMSE, R²
- `suvr_true_vs_predicted_test_stats.csv` — the same per-region stats as a table
- `suvr_bland_altman_test.png` — per-region mean-vs-difference agreement plot with bias and 95% limits of agreement
- `suvr_mae_subgroup_test.png` — per-subject MAE boxplots by diagnosis, site, sex, APOE, amyloid status, and age group (whichever are present in the demographics table)

From `plot_unseen_validation_results.py` (AVID external validation) — the same pattern: `suvr_true_vs_predicted_avid_unseen.png`, `suvr_true_vs_predicted_avid_unseen_stats.csv`, `suvr_bland_altman_avid_unseen.png`.

From `plot_suvr_distributions.py` (dataset-level, not tied to one run — saved to `<proj_path>/results/suvr_distributions/` by default, i.e. alongside the run folders, never inside the git repo):

- `suvr_distribution_universal_demo.png`, `suvr_distribution_universal_avid.png` — Universal SUVR distribution, discovery vs. AVID
- `suvr_distribution_regions_demo.png`, `suvr_distribution_regions_avid.png` — same, broken out per region

From `plot_cv_summary.py` (a `--run_kfold_cv` run, e.g. `submit_mse-cv5.sh` — reads the fold-level `metrics.csv` that `src/cv.py`'s `kfold_cv()` writes; there's no single held-out test split in CV mode, so this is fold-to-fold spread rather than a scatter panel):

- `suvr_cv_fold_metrics.png` — MAE/RMSE/R² per fold (regression targets)
- `visual_read_cv_fold_metrics.png` — AUC/accuracy per fold (only if CV was run with a `visual_read` target)
- `cv_summary_stats.csv` — mean ± std per metric across folds

Visual-read model, from `plot_visual_read_results.py`:

- `visual_read_class_balance.png` — positive/negative counts, overall and by site
- `visual_read_roc_confusion_panel.png` — ROC curve (AUC) and confusion matrix at the Youden-optimal threshold, i.e. agreement with the expert clinical read
- `visual_read_performance_stats.csv` — accuracy, sensitivity, specificity, balanced accuracy, F1, MCC, AUC at both the 0.5 and optimal thresholds

## Reproducibility Notes

- Use the same train/test split settings as the example scripts in [sbatch_scripts/](sbatch_scripts/).
- Result folder naming follows the convention defined in [src/params.py](src/params.py).
- External validation requires the trained model folder from a previous run to be available.
- The project was developed for Berzelius SLURM runs (use `sbatch sbatch_scripts/<script>.sh`), but commands can also be run manually if the environment and data are available.

## Summary

This repository contains all code used to train, evaluate, and plot the final models in the project.  
The main scripts needed for reproduction are:

- [run.py](run.py)
- [run_val.py](run_val.py)
- [results_plotting/plot_heldout_test_results.py](results_plotting/plot_heldout_test_results.py)
- [results_plotting/plot_unseen_validation_results.py](results_plotting/plot_unseen_validation_results.py)

These scripts, together with the original data and the conda environment file, are sufficient to recreate the current final results.
