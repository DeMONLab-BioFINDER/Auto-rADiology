"""Predict one split of a trained visual_read run with one of its models, without retraining.

Final model (default): writes <best_model_folder>/evaluation/<dataset>/<Prefix>_<dataset>_results.csv
(+ _metrics.csv) for the scans in <best_model_folder>/splits/<split>_subjects.csv, in the same format as
the test results (ID_ind, y, pred, prob). Runs trained since the borderline-weighting change save their
validation predictions themselves (Val_...); this fills them in for older runs, e.g. as a baseline for
results_plotting/score_borderline_validation.py. Not the test split: that would overwrite the run's own
Eval_ results.

CV fold model (--fold N, a run made with --run_kfold_cv): the fold's best checkpoint
(kfold-N/checkpoints/kfold-N_best.pt) predicts the run's held-out test scans (or another split), written to
<best_model_folder>/fold_predictions/kfold-N/evaluation/<dataset>/Eval_<dataset>_results.csv - laid out like
a run folder, so compare_discordant_between_runs.py / inspect_scans.py take fold_predictions/kfold-N as a
run. Each fold model learned from a different 4/5 of the training pool and stopped on its own validation
fold, so comparing them scan by scan on the same test scans asks whether different training data gives a
different strategy (not just a different seed).

Usage (repo root; pass the same --model_kwargs / --cls_loss / --data_type the run was trained with):
    python run_predict_split.py --best_model_folder <run_dir> --data_type tau_T1MNI_pts \
        --input_path /proj/berzelius-2024-156/users/x_nadpi/data --targets visual_read --cls_loss softmax \
        --model_kwargs '{"pool": "attention"}'   [--predict_split val]
    python run_predict_split.py --best_model_folder <cv_run_dir> --fold 3 ...   (test split by default)
"""
from src.warnings import ignore_warnings
ignore_warnings()

import argparse
import os
import sys

import pandas as pd
import torch

from src.params import parse_arguments
from src.utils import get_device, set_seed
from src.model_factory import build_model_from_args, classifier_out_dim
from src.checkpoints import load_state_dict_checked
from src.data import get_train_val_loaders
from src.train import inference

PREFIX = {"val": "Val", "train": "Train", "test": "Eval"}


def main(args, split, fold):
    split_csv = os.path.join(args.best_model_folder, "splits", f"{split}_subjects.csv")
    if not os.path.exists(split_csv):
        raise SystemExit(f"{split_csv} not found (a run without a {split} split?)")
    df = pd.read_csv(split_csv)
    _, dl = get_train_val_loaders(df, df, args, repeat_train=False)

    n_classes = classifier_out_dim(int(df["visual_read"].dropna().nunique()), args.cls_loss)
    model = build_model_from_args(args, device=args.device, n_classes=n_classes)
    if fold is None:
        ckpt_dir = os.path.join(args.best_model_folder, "final_model", "checkpoints")
        ckpt_last, ckpt_best = (os.path.join(ckpt_dir, f"final_model_{k}.pt") for k in ("last", "best"))
        ckpt = ckpt_last if os.path.exists(ckpt_last) else ckpt_best
        out_dir = os.path.join(args.best_model_folder, "evaluation", args.dataset)
    else:
        ckpt = os.path.join(args.best_model_folder, f"kfold-{fold}", "checkpoints", f"kfold-{fold}_best.pt")
        out_dir = os.path.join(args.best_model_folder, "fold_predictions", f"kfold-{fold}", "evaluation", args.dataset)
    if not os.path.exists(ckpt):
        raise SystemExit(f"checkpoint {ckpt} not found")
    sd = torch.load(ckpt, map_location=args.device, weights_only=True)
    load_state_dict_checked(model, sd.get("model", sd) if isinstance(sd, dict) else sd, ckpt)
    model.eval()

    metrics, df_result = inference(model, dl, args.device, decision_threshold=args.decision_threshold)
    os.makedirs(out_dir, exist_ok=True)
    prefix = PREFIX[split]
    df_result.to_csv(os.path.join(out_dir, f"{prefix}_{args.dataset}_results.csv"), index=False)
    pd.DataFrame([metrics]).to_csv(os.path.join(out_dir, f"{prefix}_{args.dataset}_metrics.csv"), index=False)
    print(f"{'fold ' + str(fold) + ' ' if fold else ''}{split}: {len(df_result)} scans, AUC={metrics.get('auc'):.3f} "
          f"-> {out_dir}/{prefix}_{args.dataset}_results.csv")


if __name__ == "__main__":
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--predict_split", default=None, choices=["val", "train", "test"])
    pre.add_argument("--fold", type=int, default=None, help="CV fold model to use instead of the final model.")
    extra, rest = pre.parse_known_args()
    split = extra.predict_split or ("test" if extra.fold else "val")
    if split == "test" and extra.fold is None:
        raise SystemExit("--predict_split test only with --fold (the final model's test results already exist)")
    sys.argv = [sys.argv[0]] + rest
    args = parse_arguments()
    args.device = get_device()
    set_seed(args.seed)
    main(args, split, extra.fold)
