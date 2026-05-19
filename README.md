# Auto-rADiology

## Purpose

This repository contains the code used for my project on predicting regional tau-PET SUVR values from PET images using a 3D CNN.  
The main final outputs are:

- results from 5-fold cross-validation on the training pool
- final train/test model
- zero-shot external validation on the AVID unseen dataset
- journal-style summary plots for these results


## What This Project Does

Input:

- preprocessed tau-PET images
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

- [results_plotting/plot_multi_regression_results.py](results_plotting/plot_multi_regression_results.py)
- [results_plotting/plot_cv_summary.py](results_plotting/plot_cv_summary.py)

## Repository Structure

Main scripts:

- [run.py](run.py): training, train/validation/test split, and k-fold CV
- [run_val.py](run_val.py): external validation / zero-shot evaluation
- [run_vis.py](run_vis.py): visualization script

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

Data utilities:

- [convert_tau_pkl_to_pt.py](convert_tau_pkl_to_pt.py): convert pickle to PyTorch format
- [transform.py](transform.py): data transformation utilities

Notebooks:

- [plot.ipynb](plot.ipynb): interactive plotting notebook
- [visualize.ipynb](visualize.ipynb): visualization notebook

## Software Environment

The project environment used on Berzelius is defined in:

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

## Final Runs Used in This Project

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

- `metrics.csv`: overall performance metrics
- `results.csv`: per-sample predictions and ground truth
- `evaluation/<dataset>/` folder with per-dataset results
- `splits/`: train/validation/test split indices
- Model checkpoints and per-epoch training metrics
- Per-region performance summaries

## How To Recreate The Plots

After generating result folders from the training/validation runs, use the plotting scripts in [results_plotting/](results_plotting/):

```bash
# Plot a single regression run
python results_plotting/plot_multi_regression_results.py \
  --run_dir <path_to_result_folder>

# Plot CV summary
python results_plotting/plot_cv_summary.py \
  --run_dir <path_to_cv_result_folder>

# Plot other visualizations
python results_plotting/plot_metatemporal_results.py \
  --run_dir <path_to_result_folder>

python results_plotting/plot_unseen_validation_scatter.py \
  --run_dir <path_to_external_validation_result_folder>

python results_plotting/plot_visual_read_results.py \
  --run_dir <path_to_result_folder>
## Final Figures Produced

The simplified plotting code generates:

- `true_vs_predicted_panel.png`
- `training_validation_loss.png` when per-epoch training curves exist
- `metrics_summary.png`
- `per_region_metrics.csv`
- `residual_panel.png`
- `error_histograms.png`
- `mae_boxstrip_by_site_panel.png` when metadata are available
- `mae_boxstrip_by_gender_panel.png` when metadata are available
- `mae_boxstrip_by_apoe_panel.png` when metadata are available

CV plotting generates:

- `cv_metrics_by_fold.png`
- `cv_best_epoch_histogram.png`
- `cv_mean_training_validation_loss.png`
- `cv_metrics_table.csv`

## Notes For Reproducibility

- Set the random seed using `--seed` argument in the scripts. Default is `42`.
- Keep the same target order when specifying `--targets`:
  - `MetaTemporal`
  - `MesialTemporal`
  - `Frontal`
  - `TemporoParietal`
- Use the same train/test split settings as the example scripts in [sbatch_scripts/](sbatch_scripts/).
- Result folder naming follows the convention defined in [src/params.py](src/params.py).
- External validation requires the trained model folder from a previous run to be available.
- The project was developed for Berzelius SLURM runs (use `sbatch sbatch_scripts/<script>.sh`), but commands can also be run manually if the environment and data are available.

To reproduce the project:

1. Create the conda environment from [environment_Berzelius.yml](environment_Berzelius.yml):
   ```bash
   conda env create -f environment_Berzelius.yml
   conda activate ai-pet
   ```
2. Prepare tau-PET image data and demographics tables in the expected folder structure.
3. Run the Gothenburg 5-fold CV training (see "Final Runs" section).
4. Run the final Gothenburg train/test training.
5. Run the AVID unseen external validation using the final Gothenburg model folder.
6. Run the plotting scripts to generate visualizations.
7. All results and figures are saved in the auto-generated `results/` folder.

## Short Summary

This repository contains all code used to train, evaluate, and plot the final models in the project.  
The main scripts needed for reproduction are:

- [run.py](run.py)
- [run_val.py](run_val.py)
- [results_plotting/plot_multi_regression_results.py](results_plotting/plot_multi_regression_results.py)
- [results_plotting/plot_cv_summary.py](results_plotting/plot_cv_summary.py)

These scripts, together with the original data and the conda environment file, are sufficient to recreate the current final results.