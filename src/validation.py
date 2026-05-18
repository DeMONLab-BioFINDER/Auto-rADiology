#src/validation.py
from src.warnings import ignore_warnings
ignore_warnings()

import os
import copy
import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit

from src.data import build_master_table, get_transforms, get_loader
from src.train import inference
from src.cv import train_model
from src.utils import build_model_from_args, load_best_checkpoint, add_quantile_bins

import torch.multiprocessing as mp
os.environ["NIBABEL_KEEP_FILE_OPEN"] = "0"
mp.set_sharing_strategy("file_system")


def run_few_shots(args, df, tfm, data_file, base_model, targets_list):
    finetune_path = os.path.join(args.output_path, f"fewshot-{args.few_shot}")
    os.makedirs(finetune_path, exist_ok=True)

    all_metrics, all_fs_ids = [], [] 
    df_results = pd.DataFrame([])
    for it in range(args.few_shot_iterations):
        print(f"\n=== Few-shot iteration {it+1}/{args.few_shot_iterations} ===")

        seed_it = args.seed + it

        # ----- stratification -----
        # Only works for visual_read and CL
        if "visual_read" in targets_list:
            strat_col = "visual_read"
            df_use = df.dropna(subset=["visual_read"])
        else:
            df_use = df.dropna(subset=["CL"]).copy()
            df_use = add_quantile_bins(df_use, "CL")
            strat_col = "CL_qbin"

        # ----- split -----
        print('few shot stratified by', strat_col, 'for', df_use.shape, 'scans')
        df_fs, df_eval = stratified_few_shot_split(df_use, n_shot=args.few_shot, stratify_col=strat_col, seed=seed_it)
        all_fs_ids.append({"iteration": it, "ids": df_fs["ID"].tolist()})
        df_ids = pd.DataFrame(all_fs_ids)

        # ----- FEW-SHOT -----
        metrics, df_result = few_shots(base_model, df_fs, df_eval, tfm, data_file, args, it) # results on test set
        print("metrics:", metrics)

        # ----- COLLECT -----
        metrics["iteration"] = it
        all_metrics.append(metrics)
        df_result["iteration"] = it
        df_results = pd.concat([df_results, df_result], ignore_index=True)

    df_metrics = pd.DataFrame(all_metrics)

    return df_metrics, df_results, df_ids


def few_shots(base_model, df_fs, df_eval, tfm, data_file, args, it):
    # loaders
    dl_fs_tr = get_loader(df_fs, tfm, data_file, args, batch_size=max(1, args.batch_size // 2), augment=True, shuffle=True)
    dl_fs_va = get_loader(df_fs, tfm, data_file, args, batch_size=max(1, args.batch_size // 2), augment=False, shuffle=False)
    dl_eval  = get_loader(df_eval, tfm, data_file, args, batch_size=max(1, args.batch_size // 2), augment=False, shuffle=False)

    model = copy.deepcopy(base_model) # clone model (important!)
    # freeze backbone
    model = freeze_all_but_last_k(model, args.unfreeze_layers)

    # finetune
    model, save_path = finetune(model, dl_fs_tr, dl_fs_va, it, args, fold_name=f"fewshot-{args.few_shot}_iter-{args.few_shot_iterations}-{it}")

    # evaluate
    metrics, df_result = inference(model, dl_eval, args.device, cls_threshold=args.cls_threshold)
    pickle.dump([metrics, df_result], open(f"{save_path}_inference_testset.pkl", "wb"))
    return metrics, df_result


def finetune(model, dl_tr, dl_va, it, args, fold_name="few-shots"):
    '''
    Finetune the given model using the few-shot training and validation on the same set of dataloaders.
    '''
    
    finetune_path = os.path.join(args.output_path, f"fewshot-{args.few_shot}") ### ! Same as in run_few_shots()
    save_path = f"{finetune_path}/iter-{args.few_shot_iterations}-{it}"
    path_list = {"train_eval_csv": f"{save_path}_training_metrics_per_epoch.csv",
                 "train_loss_csv": f"{save_path}_trainning_loss_allfinetunesubjects_per_epoch.csv",
                 "ckpt": f"{save_path}_finetuned_best.pt"}

    model, _ = train_model(model, dl_tr, dl_va, args=args, fold_name=fold_name, path_list=path_list)

    if os.path.exists(path_list['ckpt']):
        model = load_best_checkpoint(model, ckpt_path=path_list['ckpt'], device=args.device)
    else:
        print("! No finetune checkpoint found, using last-epoch weights")

    return model, save_path


def load_validation_data(args):
    """
    Load the validation dataset based on the specified dataset type in args.

    Returns:
        tuple: A tuple containing the transformations and the validation dataloader. 
    Notes
    -----
    - when data_suffix exists, NIfTI files are loaded and smoothed if voxel sizes are provided.
    - when not, torch tensors are loaded directly from the specified input path. 
    !!! remains cleaning !!!
    """
    data_file = None
    tfm = get_transforms(target_shape=tuple(args.image_shape), add_spacing=args.spacing, pixdim=args.pixdim,
                                     intensity_norm=args.intensity_norm, pct_lo=args.intensity_pct[0], pct_hi=args.intensity_pct[1])
    if args.data_suffix: # Berkeley server, load NIfTI files
        print(f'Validate on {args.dataset} - {args.data_suffix} test set...')
        test_set = os.path.join(args.proj_path, "data", f'{args.dataset}_found_scans_{args.data_suffix}_{args.targets}.csv')
        if os.path.exists(test_set):
            print('loading validation dataframe')
            df = pd.read_csv(test_set, index_col=0)
        else:
            print('finding scans from folder')
            df = build_master_table(args.input_path, args.data_suffix, args.targets, args.dataset, args.data_type)
            df.to_csv(os.path.join(args.proj_path, "data", f'{args.dataset}_found_scans_{args.data_suffix}_{args.targets}.csv'))
    else: # should be torch tensor images, without any preprocessing
        print(f'Validate on {args.dataset} ...')
        test_set = os.path.join(args.proj_path, "data", f'demo_{args.dataset}_{args.data_type}_validation.csv')
        print(test_set)
        df = pd.read_csv(test_set, index_col=0)
        data_file = Path(args.input_path) / 'data' / args.data_type
    #elif 'IDEAS' in args.dataset: # Berzelius, load torch tensors
    #    print('Validate on IDEAS test set...')
    #    test_set = os.path.join(args.best_model_folder,'Hold-out_testing-set.csv')
    #    print(test_set)
    #    df = pd.read_csv(test_set, index_col=0)
    #    data_file = Path(args.input_path) / 'data' / args.data_type
    #    tfm = None
    #elif 'A4' in args.dataset: # Berzelius, load torch tensors
    #    print('Validate on A4 dataset...')
    #    test_set = os.path.join(args.proj_path, "data", f'demo_{args.dataset}.csv')
    #    print(test_set)
    #    df = pd.read_csv(test_set, index_col=0)
    #    data_file = Path(args.input_path) / 'data' / args.data_type
    #    tfm = None
    
    # remove nan
    print(df)
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    df = df.dropna(subset=targets)
    print(f'Validation set size: {len(df)} images')

    dl_va = get_loader(df, tfm, data_file, args, batch_size=max(1, args.batch_size // 2), augment=False, shuffle=False, train_test='test')

    return tfm, data_file, dl_va, df


def load_preatrained_model(args, df) -> torch.nn.Module:
    """
    Load a pretrained model checkpoint into the given model architecture.
    Parameters
    ----------
    args : argparse.Namespace in params.py
    df : DataFrame containing the dataset information (used to determine n_classes)

    Returns:
        torch.nn.Module: The model with loaded weights.
    """
    targets_list = [t.strip() for t in args.targets.split(",") if t.strip()]
    #n_classes = int(df["visual_read"].dropna().nunique()) if 'visual_read' in targets_list else None
    if 'visual_read' in targets_list:
        n_classes = 1 if args.cls_loss in {"bce", "weighted_bce"} else 2
    else:
        n_classes = None

    model = build_model_from_args(args, device=args.device, n_classes=n_classes)

    ckpts = [os.path.join(args.best_model_folder, "nestedcv-outer-test/checkpoints/nestedcv-outer-test_best.pt"),
             os.path.join(args.best_model_folder, "train-test-split/checkpoints/train-test-split_best.pt")]
    ckpt_found = None
    for ckpt in ckpts:
        if os.path.exists(ckpt):
            ckpt_found = ckpt
            break
    if ckpt_found is None: raise FileNotFoundError(f"No pretrained checkpoint found. Tried: {ckpts}")

    print(f"Loading pretrained model: {ckpt_found}")
    sd = torch.load(ckpt_found, map_location=args.device, weights_only=True)
    state_dict = sd.get("model", sd) if isinstance(sd, dict) else sd
    model.load_state_dict(state_dict, strict=False)

    return model, targets_list


def freeze_all_but_last_k(model: nn.Module, k: int):
    """
    Freeze all parameters except the last K *parameterized layers*.
    """

    # 1. Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # 2. Collect modules that own parameters (in forward order)
    param_layers = []
    for module in model.modules():
        # Only count modules that *directly* own parameters
        if any(p.requires_grad is False for p in module.parameters(recurse=False)) \
           and len(list(module.parameters(recurse=False))) > 0:
            param_layers.append(module)

    if len(param_layers) == 0: raise ValueError("No parameterized layers found in model.")

    # 3. Clamp k
    k = min(k, len(param_layers))

    # 4. Unfreeze last K parameterized layers
    for layer in param_layers[-k:]:
        for p in layer.parameters(recurse=False):
            p.requires_grad = True

    # 5. Logging
    print(f"✓ Unfroze last {k} parameterized layers:")
    for layer in param_layers[-k:]:
        print(f"  - {layer.__class__.__name__}")

    return model


def stratified_few_shot_split(df: pd.DataFrame, n_shot: int, stratify_col: str, seed: int):
    """
    Returns:
      df_fs  : few-shot dataframe
      df_eval: disjoint evaluation dataframe
    """
    if n_shot >= len(df):
        raise ValueError("few-shot size must be < dataset size")

    y = df[stratify_col]

    vc = df[stratify_col].value_counts()
    if n_shot < len(vc):
        raise ValueError(f"few_shot={n_shot} is smaller than number of strata={len(vc)}. Increase few_shot or reduce stratification granularity.")
    if (vc < 2).any():
        raise ValueError(f"Some strata have <2 samples, cannot make disjoint few-shot/eval split:\n{vc}")

    splitter = StratifiedShuffleSplit(n_splits=1, train_size=n_shot, random_state=seed,)

    fs_idx, eval_idx = next(splitter.split(df, y))
    df_fs = df.iloc[fs_idx].reset_index(drop=True)
    df_eval = df.iloc[eval_idx].reset_index(drop=True)

    return df_fs, df_eval


def bootstrap_ci(values, n_boot=2000, ci=95, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)

    boots = [
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ]

    lo = np.percentile(boots, (100 - ci) / 2)
    hi = np.percentile(boots, 100 - (100 - ci) / 2)

    return float(np.mean(values)), float(lo), float(hi)


def summarize_results(zero_metrics, few_metrics):
    print("\n========== SUMMARY (mean ± CI) ==========")

    keys = zero_metrics[0].keys()

    for k in keys:
        z_vals = [m[k] for m in zero_metrics if np.isfinite(m[k])]
        f_vals = [m[k] for m in few_metrics if np.isfinite(m[k])]

        if not z_vals or not f_vals:
            continue

        z_mean, z_lo, z_hi = bootstrap_ci(z_vals)
        f_mean, f_lo, f_hi = bootstrap_ci(f_vals)

        print(
            f"{k:>10} | "
            f"Zero: {z_mean:.3f} [{z_lo:.3f}, {z_hi:.3f}] | "
            f"Few:  {f_mean:.3f} [{f_lo:.3f}, {f_hi:.3f}]"
        )
    print("=================DONE====================\n")


def save_predictions(ycls, preds, probs, yreg, cents, any_cls, any_reg, out_csv):
    if any_cls:
        df = pd.DataFrame({"y":    np.concatenate(ycls),
                           "pred": np.concatenate(preds),
                           "prob": np.concatenate(probs)})
    elif any_reg:
        df = pd.DataFrame({"y":    np.concatenate(yreg),
                           "pred": np.concatenate(cents)})
    else:
        raise ValueError("No valid predictions to save")

    df.to_csv(out_csv, index=False)
    return df
