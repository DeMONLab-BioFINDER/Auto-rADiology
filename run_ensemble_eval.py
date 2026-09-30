from src.warnings import ignore_warnings
ignore_warnings()

import os
import glob
import numpy as np
import pandas as pd
import torch

from src.params import parse_arguments
from src.utils import get_device, set_seed
from src.data import get_train_val_loaders
from src.model_factory import build_model_from_args, classifier_out_dim
from src.checkpoints import load_best_checkpoint
from src.train import validate_one_epoch, compute_metrics

import torch.multiprocessing as mp
os.environ["NIBABEL_KEEP_FILE_OPEN"] = "0"
mp.set_sharing_strategy("file_system")


def find_fold_checkpoints(best_model_folder: str) -> list[str]:
    """
    Discover kfold-N/checkpoints/kfold-N_best.pt checkpoints from a completed
    --run_kfold_cv run, sorted by fold number.
    """
    pattern = os.path.join(best_model_folder, "kfold-*", "checkpoints", "kfold-*_best.pt")
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(
            f"No kfold-*/checkpoints/*_best.pt found under {best_model_folder}. "
            "run_ensemble_eval.py ensembles the fold models from a completed --run_kfold_cv "
            "run, not a single final-model run (see submit_visual-read-cv5*.sh)."
        )
    paths.sort(key=lambda p: int(os.path.basename(os.path.dirname(os.path.dirname(p))).split("-")[1]))
    return paths


def run_pass(model, loader, device, decision_threshold) -> pd.DataFrame:
    _, _, df_result = validate_one_epoch(model, loader, device, decision_threshold=decision_threshold, desc="Eval")
    return df_result.sort_values("ID_ind").reset_index(drop=True)


def main(args):
    if not args.best_model_folder:
        raise ValueError("--best_model_folder is required: path to a completed --run_kfold_cv output directory.")

    device = args.device
    test_csv = os.path.join(args.best_model_folder, "splits", "test_subjects.csv")
    df_test = pd.read_csv(test_csv)
    print(f"Held-out test set: {len(df_test)} subjects ({test_csv})")

    n_unique = int(df_test["visual_read"].dropna().nunique())
    out_dim = classifier_out_dim(n_unique, args.cls_loss)

    ckpt_paths = find_fold_checkpoints(args.best_model_folder)
    print(f"Found {len(ckpt_paths)} fold checkpoints:")
    for p in ckpt_paths:
        print(f"  {p}")

    tta_passes = max(1, args.tta_passes)
    eval_augment = tta_passes > 1
    if eval_augment:
        print(f"Test-time augmentation: averaging {tta_passes} augmented passes per model.")

    subject_ids, y_true = None, None
    fold_names, fold_probs = [], []

    for ckpt_path in ckpt_paths:
        fold_name = os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))
        model = build_model_from_args(args, device=device, n_classes=out_dim)
        model = load_best_checkpoint(model, ckpt_path, device)

        _, loader = get_train_val_loaders(df_test, df_test, args, repeat_train=False, eval_augment=eval_augment)

        pass_probs = []
        for pass_i in range(tta_passes):
            if eval_augment:
                # Reseed torch's global RNG so the random-flip augmentation in
                # PETDataset.__getitem__ actually differs pass-to-pass - see --tta_passes
                # help text for the caveat with --num_workers > 0.
                torch.manual_seed(args.seed + 1000 * pass_i)
            df_result = run_pass(model, loader, device, args.decision_threshold)
            pass_probs.append(df_result["prob"].to_numpy())
            if subject_ids is None:
                subject_ids = df_result["ID_ind"].to_numpy()
                y_true = df_result["y"].to_numpy()

        fold_prob = np.mean(pass_probs, axis=0)
        fold_metrics = compute_metrics([y_true], [(fold_prob > args.decision_threshold).astype(int)],
                                        [fold_prob], True, [], [], False)
        print(f"[{fold_name}] on shared test set: AUC={fold_metrics['auc']:.3f} ACC={fold_metrics['acc']:.3f}")

        fold_names.append(fold_name)
        fold_probs.append(fold_prob)

    ens_prob = np.mean(fold_probs, axis=0)
    ens_pred = (ens_prob > args.decision_threshold).astype(int)
    ens_metrics = compute_metrics([y_true], [ens_pred], [ens_prob], True, [], [], False)

    print("\n================================================================================")
    print(f"Ensemble of {len(ckpt_paths)} folds on shared held-out test set (decision_threshold={args.decision_threshold})")
    print("================================================================================")
    print(f"AUC={ens_metrics['auc']:.3f}  ACC={ens_metrics['acc']:.3f}  "
          f"eval_metric={ens_metrics['eval_metric']:.3f}")
    print(f"(diagnostic, computed at this test set's own optimal threshold, not deployed): "
          f"acc_opt={ens_metrics.get('acc_opt', float('nan')):.3f}  best_thr={ens_metrics.get('best_thr', float('nan')):.3f}")

    out_dir = os.path.join(args.best_model_folder, "evaluation", "kfold_ensemble")
    os.makedirs(out_dir, exist_ok=True)

    df_out = pd.DataFrame({"ID": subject_ids, "y": y_true, "ensemble_prob": ens_prob, "ensemble_pred": ens_pred})
    for name, prob in zip(fold_names, fold_probs):
        df_out[f"{name}_prob"] = prob
    df_out.to_csv(os.path.join(out_dir, "ensemble_predictions.csv"), index=False)

    pd.DataFrame([{"fold": "ensemble", **ens_metrics}]).to_csv(
        os.path.join(out_dir, "ensemble_metrics.csv"), index=False)

    print(f"\nSaved per-subject predictions and metrics to {out_dir}")
    print("DONE!")


if __name__ == "__main__":
    args = parse_arguments()
    args.device = get_device()
    print("Using device:", args.device)
    print(args)

    set_seed(args.seed)

    main(args)
