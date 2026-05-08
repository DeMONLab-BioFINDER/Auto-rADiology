from src.warnings import ignore_warnings
ignore_warnings()

import os
import pandas as pd

from src.params import parse_arguments
from src.utils import set_seed, train_val_test_split, hold_out_set, make_splits, save_train_test_subjects, clone_args
from src.data import build_master_table
from src.cv import get_stratify_labels, run_fold, cv_median_best_epoch, kfold_cv
from src.hypertune import create_study_from_args, run_optuna, objective, print_best, get_best_args


TRAIN_SIZE = 0.64
VAL_SIZE = 0.16
TEST_SIZE = 0.20


def main(args):
    # 1) data
    df = build_master_table(args.input_path, args.data_suffix, args.targets, args.dataset, args.data_type)
    df_clean, stratify_labels = get_stratify_labels(df, args.stratifycvby, args.seed)

    use_validation_split = VAL_SIZE > 0

    if use_validation_split:
        tr_idx, va_idx, te_idx = train_val_test_split(
            df_clean,
            stratify_labels,
            subject_col=args.samesubject_col,
            train_size=TRAIN_SIZE,
            val_size=VAL_SIZE,
            test_size=TEST_SIZE,
            seed=args.seed,
        )
        df_train = df_clean.iloc[tr_idx].reset_index(drop=True)
        df_test = df_clean.iloc[te_idx].reset_index(drop=True)
        df_val = df_clean.iloc[va_idx].reset_index(drop=True)
    else:
        tr_idx, te_idx = hold_out_set(
            df_clean,
            stratify_labels,
            subject_col=args.samesubject_col,
            test_size=TEST_SIZE,
            seed=args.seed,
        )
        df_train = df_clean.iloc[tr_idx].reset_index(drop=True)
        df_test = df_clean.iloc[te_idx].reset_index(drop=True)
        df_val = df_train.copy()
    
    # Save the splits
    split_dir = os.path.join(args.output_path, 'splits')
    os.makedirs(split_dir, exist_ok=True)
    save_train_test_subjects(df_train, df_test, split_dir, 'Hold-out')
    if use_validation_split:
        df_val.to_csv(os.path.join(split_dir, 'Hold-out_validation-set.csv'), index=False)

    if 'dataset' in df_train.columns: 
        print('train:', df_train['dataset'].value_counts(), 
              '\nval:', df_val['dataset'].value_counts() if use_validation_split else 'not used',
              '\ntest:', df_test['dataset'].value_counts())
    _, stratify_labels_train = get_stratify_labels(df_train, args.stratifycvby, args.seed)

    # 2) tuning, cv, or direct training
    if args.tune: # option 1: hyperparameter tuning with nested CV
        # --- Hyperparameter Tuning ---
        print('Running Hyperparameter tuning...')
        splits = make_splits(df_train, stratify_labels_train, args.n_splits, args.seed)
        # to save the splits
        study  = create_study_from_args(args)
        study  = run_optuna(study, objective, args, df_train, splits, args.model)
        print_best(study)

        # Retrain with best params (full epochs) on df_train and evaluate ONCE on df_test
        best_args = get_best_args(args, study, out_subdir="best_params")
        E_final = cv_median_best_epoch(df_train, stratify_labels_train, best_args)
        print(f"Final retrain epochs (median best_epoch across folds): {E_final}")
        best_args_fixed = clone_args(best_args, epochs=E_final)
        print("\nRetraining with best params on full training set…")
        print(best_args_fixed)

        print("\nRetraining on FULL TRAIN pool with fixed epochs (no early stop), then one-shot TEST eval…")
        metrics_te, df_result_te = run_fold(df_train, df_val, df_test, best_args_fixed, fold_name="nestedcv-outer-test")

        print(f"\nHypertune OUTER TEST: AUC={metrics_te.get('auc'):.3f} "
              f"ACC={metrics_te.get('acc'):.3f} MAE={metrics_te.get('mae'):.2f} "
              f"RMSE={metrics_te.get('rmse'):.2f} R2={metrics_te.get('r2'):.3f}")

    # elif args.n_splits > 1: # option 2: k-fold CV without hyperparameter tuning
    #     print(f'Running {args.n_splits}-fold cross-validation on training set…')
    #     kfold_cv(df_train, stratify_labels_train, args)
    #     metrics_te = None
    #     df_result_te = None

    else: # option 3: direct train/val/test (single split)
        if use_validation_split:
            print('Direct training with validation and test split…')
            metrics_te, df_result_te = run_fold(df_train, df_val, df_test, args, fold_name="train-val-test")
            print(f"\nTest set: AUC={metrics_te.get('auc'):.3f} "
                  f"ACC={metrics_te.get('acc'):.3f} MAE={metrics_te.get('mae'):.2f} "
                  f"RMSE={metrics_te.get('rmse'):.2f} R2={metrics_te.get('r2'):.3f}")
        else:
            print('Direct training with train/test split only…')
            metrics_te, df_result_te = run_fold(df_train, df_train, df_test, args, fold_name="train-test-split")
            print(f"\nTrain/test split: AUC={metrics_te.get('auc'):.3f} "
                  f"ACC={metrics_te.get('acc'):.3f} MAE={metrics_te.get('mae'):.2f} "
                  f"RMSE={metrics_te.get('rmse'):.2f} R2={metrics_te.get('r2'):.3f}")

    # Save Results (skip for CV since kfold_cv handles its own saving)
    if metrics_te is not None and df_result_te is not None:
        evaluation_folder = os.path.join(args.output_path, 'evaluation', args.dataset)
        os.makedirs(evaluation_folder, exist_ok=True)
        df_result_te.to_csv(os.path.join(evaluation_folder, f'Eval_{args.dataset}_results.csv'), index=False)
        pd.DataFrame([metrics_te]).to_csv(os.path.join(evaluation_folder, f'Eval_{args.dataset}_metrics.csv'), index=False)
    
    print('DONE!')


if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    #args.device = get_device(force_cpu=True)
    print("Using device:", args.device)

    set_seed(args.seed)

    main(args)
