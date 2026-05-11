# abpet/utils.py
import os, csv, json, time, math
import torch, random, inspect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Mapping, Tuple
from types import SimpleNamespace
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from src.models import *


def set_seed(seed: int = 42, deterministic: bool = False, set_pythonhashseed: bool = True):
    if set_pythonhashseed:
        os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    # Seeds CPU and (per PyTorch docs) CUDA RNG; no separate call for MPS exists.
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True

    # Global determinism (may error if an op lacks a deterministic variant)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            print(f"[warn] deterministic_algorithms not fully supported: {e}")

    #  MONAI convenience:
    from monai.utils import set_determinism as monai_set_det
    monai_set_det(seed=seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def save_checkpoint(model, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model.state_dict(), out_path)

def get_device(prefer_cuda=True, force_cpu=False):
    """
    Returns a torch.device among: cuda, mps, cpu.
    prefer_cuda: if True, choose CUDA over MPS when both are present (e.g., eGPU on Mac).
    """
    has_cuda = torch.cuda.is_available()
    has_mps = getattr(torch.backends, "mps", None) is not None \
              and torch.backends.mps.is_built() and torch.backends.mps.is_available()
    if prefer_cuda and has_cuda and not force_cpu: return torch.device("cuda")
    if not prefer_cuda and has_mps: return torch.device("mps")
    if not force_cpu and has_cuda: return torch.device("cuda")
    if not force_cpu and has_mps: return torch.device("mps")

    torch.backends.cudnn.benchmark = True  # 3D convs benefit
    return torch.device("cpu")

def load_best_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    """
    Load model weights from checkpoint path into the given model.
    """
    if os.path.exists(ckpt_path):
        try:
            sd = torch.load(ckpt_path, map_location=device, weights_only=True)
        except TypeError:
            sd = torch.load(ckpt_path, map_location=device)
        print(f"Loading checkpoint: {ckpt_path}")
        state_dict = sd.get("model", sd) if isinstance(sd, dict) else sd
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"[warn] checkpoint not found: {ckpt_path}")
    return model


def add_quantile_bins(df, col, n_bins=5):
    df = df.copy()
    df[f"{col}_qbin"] = pd.qcut(df[col], q=n_bins, duplicates="drop")
    return df


def is_continuous_numeric(s, min_unique_ratio=0.05):
    """
    Heuristic check for continuous numeric variable.
    """
    if not pd.api.types.is_numeric_dtype(s): return False

    s = s.dropna()
    if len(s) == 0: return False

    unique_ratio = s.nunique() / len(s)
    return unique_ratio >= min_unique_ratio


def collapse_dx_to_other(df: pd.DataFrame, col: str = "dx", main_groups=None, other_label: str = "Other"):
    """
    Collapse diagnosis labels outside the main groups into a single catch-all class.

    This is intended for stratification and should run before any split that uses dx.
    """
    if main_groups is None:
        main_groups = ["CU", "MCI", "AlzCS dem"]

    if col not in df.columns:
        return df

    dx = df[col].astype("string")
    keep = {g.strip().lower() for g in main_groups}
    mask_other = dx.notna() & ~dx.str.strip().str.lower().isin(keep)

    if mask_other.any():
        df = df.copy()
        df.loc[mask_other, col] = other_label

    return df


def compute_smooth_sigma_vox(voxel_sizes_mm: tuple[float, float, float], fwhm_current_mm: float, fwhm_target_mm: float) -> tuple[float, float, float] | None:
    """
    Returns per-axis sigma in *voxels* for MONAI.GaussianSmooth to top-up smoothing
    from fwhm_current_mm to fwhm_target_mm. If no extra smoothing needed, returns (0,0,0).
    """
    if fwhm_target_mm <= fwhm_current_mm:
        return (0.0, 0.0, 0.0)

    fwhm_extra_mm = np.sqrt(fwhm_target_mm**2 - fwhm_current_mm**2)  # quadrature
    sigma_mm = fwhm_extra_mm / 2.354820045  # mm → σ
    vx, vy, vz = voxel_sizes_mm
    return (sigma_mm / vx, sigma_mm / vy, sigma_mm / vz)


def append_metrics_csv(csv_path: str, data: dict, mode: str = "row"):
    """
    Append metrics to a CSV file using pandas.

    Parameters
    ----------
    csv_path : str
        Path to CSV file.
    data : dict
        Metrics to append.
    mode : {"row", "column"}
        - "row": append one experiment per row (recommended)
        - "column": append one experiment per column
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if mode == "row":
        df_new = pd.DataFrame([data])  # one row
        if os.path.exists(csv_path):
            df_old = pd.read_csv(csv_path)
            df = pd.concat([df_old, df_new], ignore_index=True)
    elif mode == "column":
        df_new = pd.DataFrame.from_dict(data, orient="index", columns=["value"])
        if os.path.exists(csv_path):
            df_old = pd.read_csv(csv_path, index_col=0)
            df = pd.concat([df_old, df_new], axis=1)
    else:
        raise ValueError("mode must be 'row' or 'column'")
    if not os.path.exists(csv_path): df = df_new
    df.to_csv(csv_path)


def combine_metrics_for_minimize(m: dict) -> float:
    """
    Turn your fold metrics into a scalar to *minimize*.
    Adjust weights if you prefer.
    """
    auc  = m.get("auc")
    mae  = m.get("mae")
    rmse = m.get("rmse")
    r2   = m.get("r2")

    parts = []
    if auc  is not None and auc  == auc: parts.append(1.0 - float(auc))  # 1 - AUC
    if mae  is not None and mae  == mae: parts.append(float(mae))
    if rmse is not None and rmse == rmse: parts.append(float(rmse))
    if r2   is not None and r2   == r2:   parts.append(1.0 - float(r2))  # 1 - R2

    return sum(parts) if parts else 1e9  # big penalty if missing


def clone_args(args, **overrides):
    """Create a shallow, mutable copy of args with some fields overridden."""
    d = vars(args).copy()
    d.update(overrides)
    return SimpleNamespace(**d)


def make_splits(df, labels, n_splits, seed):
    """Freeze splits once so objective() is deterministic across trials."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(df, labels))


def _subject_level_split_inputs(df, labels, subject_col):
    if subject_col is None:
        subject_col = "ID"
    if subject_col not in df.columns:
        raise ValueError(f"Subject column '{subject_col}' not found in dataframe.")

    df = df.copy()
    if df[subject_col].isna().any():
        raise ValueError(f"Subject column '{subject_col}' contains NaN values.")

    print(f'[split] keep only first scan per subject, based on column "{subject_col}"')
    subj_df = df.drop_duplicates(subset=subject_col, keep="first")
    if len(subj_df) != subj_df[subject_col].nunique():
        raise ValueError(f"Subject column '{subject_col}' must uniquely identify subject-level rows after de-duplication.")

    subj_labels = labels.loc[subj_df.index]
    if not (subj_df.index == subj_labels.index).all():
        raise ValueError("Subject-level dataframe and stratification labels are not aligned.")

    return df, subject_col, subj_df, subj_labels


def _check_stratification_counts(labels, split_name):
    counts = labels.astype(str).value_counts()
    if counts.empty:
        raise ValueError(f"{split_name} split failed: no stratification labels available.")
    sparse = counts[counts < 2]
    if not sparse.empty:
        examples = sparse.head(10).to_dict()
        raise ValueError(
            f"{split_name} split failed: each stratum needs at least 2 subjects for stratified splitting. "
            f"Sparse strata examples: {examples}. Reduce stratification columns/bins or use more data."
        )


def _stratified_subject_shuffle_split(subj_df, subj_labels, test_size, seed, split_name):
    _check_stratification_counts(subj_labels, split_name)
    try:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(subj_df, subj_labels))
    except ValueError as e:
        raise ValueError(
            f"{split_name} split failed with subject-level stratification. "
            "Reduce the number of stratification columns/bins, use a larger dataset, "
            "or adjust split fractions so each split can contain all strata. "
            f"Original error: {e}"
        ) from e

    return train_idx, test_idx


def _scan_indices_from_subjects(df, subject_col, subject_ids):
    return df[df[subject_col].isin(subject_ids)].index.to_numpy()


def hold_out_set(df, labels, subject_col, test_size: float = 0.2, seed: int = 42,) -> Tuple[np.ndarray, np.ndarray]:
    """
    Subject-level hold-out split with stratified shuffle splitting.

    - Splits by subject (no leakage)
    - Stratifies using subject-level labels
    - Returns scan-level indices
    """
    df, subject_col, subj_df, subj_labels = _subject_level_split_inputs(df, labels, subject_col)
    train_s, test_s = _stratified_subject_shuffle_split(
        subj_df,
        subj_labels,
        test_size=test_size,
        seed=seed,
        split_name="Hold-out train/test",
    )

    train_ids = subj_df.iloc[train_s][subject_col]
    test_ids  = subj_df.iloc[test_s][subject_col]

    train_idx = _scan_indices_from_subjects(df, subject_col, train_ids)
    test_idx  = _scan_indices_from_subjects(df, subject_col, test_ids)

    print("hold out training vs testing:", len(train_idx), len(test_idx),
          f"(subjects: {len(train_ids)} / {len(test_ids)})")

    return train_idx, test_idx


def train_val_test_split(
    df,
    labels,
    subject_col,
    train_size=0.75,
    val_size=0.10,
    test_size=0.15,
    seed=42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Three-way subject-level stratified split: train / validation / test.
    
    - Splits by subject (no leakage)
    - Stratifies using subject-level labels
    - Returns scan-level indices for train, val, and test
    
    Args:
        train_size: fraction for training (default 0.75)
        val_size: fraction for validation (default 0.10)
        test_size: fraction for testing (default 0.15)
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Sizes must sum to 1.0"
    
    df, subject_col, subj_df, subj_labels = _subject_level_split_inputs(df, labels, subject_col)

    # ---- 2. first split: (train+val) vs test ----
    trainval_s, test_s = _stratified_subject_shuffle_split(
        subj_df,
        subj_labels,
        test_size=test_size,
        seed=seed,
        split_name="Hold-out trainval/test",
    )
    
    # ---- 3. second split: train vs val (within the remaining fraction) ----
    temp_subj_df = subj_df.iloc[trainval_s].reset_index(drop=True)
    temp_subj_labels = subj_labels.iloc[trainval_s].reset_index(drop=True)
    
    remaining = train_size + val_size
    if val_size <= 0:
        train_s_temp = np.arange(len(temp_subj_df))
        val_s_temp = np.array([], dtype=int)
    else:
        val_ratio_within_temp = val_size / remaining
        train_s_temp, val_s_temp = _stratified_subject_shuffle_split(
            temp_subj_df,
            temp_subj_labels,
            test_size=val_ratio_within_temp,
            seed=seed + 1,
            split_name="Hold-out train/validation",
        )
    
    # Map back to original indices
    train_s = trainval_s[train_s_temp]
    val_s = trainval_s[val_s_temp]

    # ---- 4. map back to scan-level indices ----
    train_ids = subj_df.iloc[train_s][subject_col]
    val_ids   = subj_df.iloc[val_s][subject_col]
    test_ids  = subj_df.iloc[test_s][subject_col]

    train_idx = _scan_indices_from_subjects(df, subject_col, train_ids)
    val_idx   = _scan_indices_from_subjects(df, subject_col, val_ids)
    test_idx  = _scan_indices_from_subjects(df, subject_col, test_ids)

    print(f"[split] train/val/test (scans): {len(train_idx)} / {len(val_idx)} / {len(test_idx)} "
          f"(subjects: {len(train_ids)} / {len(val_ids)} / {len(test_ids)})")

    return train_idx, val_idx, test_idx


def save_split_audit(df, split_indices: Mapping[str, np.ndarray], labels, output_path, subject_col=None):
    """
    Save and print a compact audit for subject leakage and stratification balance.
    """
    subject_col = subject_col or "ID"
    if subject_col not in df.columns:
        raise ValueError(f"Subject column '{subject_col}' not found in dataframe.")

    os.makedirs(output_path, exist_ok=True)

    subject_sets = {}
    rows = []
    for split_name, idx in split_indices.items():
        split_df = df.iloc[idx]
        subjects = set(split_df[subject_col].astype(str))
        subject_sets[split_name] = subjects
        audit_df = split_df[[subject_col]].copy()
        audit_df["_stratum"] = labels.iloc[idx].astype(str).to_numpy()

        rows.append({
            "split": split_name,
            "stratum": "__ALL__",
            "n_scans": len(split_df),
            "n_subjects": len(subjects),
        })
        for stratum, split_stratum_df in audit_df.groupby("_stratum", sort=True):
            rows.append({
                "split": split_name,
                "stratum": stratum,
                "n_scans": len(split_stratum_df),
                "n_subjects": split_stratum_df[subject_col].nunique(),
            })

    names = list(split_indices)
    overlap_rows = []
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = subject_sets[left] & subject_sets[right]
            overlap_rows.append({"left": left, "right": right, "n_overlap_subjects": len(overlap)})
            if overlap:
                examples = sorted(overlap)[:10]
                raise ValueError(
                    f"Subject leakage between {left} and {right}: {len(overlap)} overlapping subjects. "
                    f"Examples: {examples}"
                )

    pd.DataFrame(rows).to_csv(os.path.join(output_path, "split_audit_strata.csv"), index=False)
    pd.DataFrame(overlap_rows).to_csv(os.path.join(output_path, "split_audit_overlap.csv"), index=False)

    size_msg = ", ".join(
        f"{name}: {len(df.iloc[idx])} scans / {len(subject_sets[name])} subjects"
        for name, idx in split_indices.items()
    )
    print(f"[split-audit] {size_msg}")
    print(f"[split-audit] no subject overlap across: {', '.join(names)}")


def _resolve_model_class(name: str):
    """Return a model class by name from models.py; raise a helpful error if missing."""
    try:
        return globals()[name]
    except KeyError as e:
        raise ValueError(
            f"Unknown model '{name}'. Ensure it's defined in models.py "
            f"and the name matches exactly (case-sensitive)."
        ) from e

def save_train_test_subjects(df_train, df_test, output_path, savename):
    df_train.to_csv(os.path.join(output_path, f'{savename}_training-set.csv'), index=False)
    df_test.to_csv(os.path.join(output_path, f'{savename}_testing-set.csv'), index=False)


def build_model_from_args(args, device=None, n_classes: int | None = None):
    """
    Dynamically instantiate a model by name from models.py using args.model.
    - Filters kwargs to what the class __init__ accepts.
    - Merges defaults from args with optional --model_kwargs (JSON or dict).
    - If args.resume is set, loads weights with strict=False.
    """
    ModelCls = _resolve_model_class(args.model)

    # Defaults from args (common across your models)
    defaults = {"in_channels": getattr(args, "in_channels", 1),
                "widths": tuple(getattr(args, "widths", (32, 64, 128, 256))),
                "dropout": getattr(args, "dropout", 0.3)}
    
    extra_dim = 0
    # scalar CL input
    if getattr(args, "input_cl", None) is not None:
        extra_dim += 1
        print('add extra input CL to the last FC layer')
    # global image-derived features
    if getattr(args, "extra_global_feats", None):
        # expect comma-separated string: "p95,std,frac_hi"
        feats = [f for f in args.extra_global_feats.split(",") if f.strip()]
        extra_dim += len(feats)
        print(f'add extra global input {args.extra_global_feats} to the last FC layer')
    defaults["extra_dim"] = extra_dim

    # JSON-only model kwargs
    extra = {}
    if hasattr(args, "model_kwargs") and args.model_kwargs:
        try:
            extra = json.loads(args.model_kwargs)
        except json.JSONDecodeError as e:
            raise ValueError(f"--model_kwargs must be valid JSON: {e}")
        if "widths" in extra and isinstance(extra["widths"], list):
            extra["widths"] = tuple(extra["widths"])

    # keep only params accepted by __init__
    allowed = set(inspect.signature(ModelCls.__init__).parameters) - {"self", "*args", "**kwargs"}
    params = {k: v for k, v in {**defaults, **extra}.items() if k in allowed}

    # auto-wire class count if provided by caller
    if n_classes is not None:
        if "num_classes" in allowed and "num_classes" not in params:
            params["num_classes"] = n_classes
        elif "out_channels" in allowed and "out_channels" not in params:
            params["out_channels"] = n_classes

    # Instantiate:  Build + move
    model = ModelCls(**params)
    if device is not None:
        model = model.to(device)

    # resume weights (Optional)
    if getattr(args, "resume", ""):
        state = torch.load(args.resume, map_location=device or "cpu")
        model.load_state_dict(state, strict=False)

    print(model)
    return model


def plot_metrics_from_csv(csv_path: str, out_png: str, val_out_png: str | None = None):
    """
    Plot per-epoch metrics from the training log CSV.
    - `out_png`: validation metric overview
    - `val_out_png`: optional training-vs-validation loss curve
    """
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    if df.empty or "epoch" not in df.columns: return

    all_metric_defs = [
        ("eval_metric", "Evaluation metric"),
        ("auc", "AUC"),
        ("acc", "Accuracy"),
        ("mae", "MAE"),
        ("rmse", "RMSE"),
        ("r2", "R2"),
    ]
    metric_defs = [(k, lbl) for k, lbl in all_metric_defs if k in df.columns]

    if metric_defs:
        plt.figure(figsize=(10, 6), dpi=300)
        ax1 = plt.gca()

        x = df["epoch"]
        _plot_lines(ax1, x, df, metric_defs, 'Value')

        ax1.set_xlabel("Epoch")
        plt.title("Validation metrics per epoch")
        plt.tight_layout()
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=300)
        plt.close()

    if val_out_png and {"train_loss", "val_loss"}.issubset(df.columns):
        plt.figure(figsize=(10, 6), dpi=300)
        ax = plt.gca()
        x = df["epoch"]
        _plot_lines(
            ax,
            x,
            df,
            [("train_loss", "Training loss"), ("val_loss", "Validation loss")],
            "Loss",
        )
        ax.set_xlabel("Epoch")
        plt.title("Training and validation loss per epoch")
        plt.tight_layout()
        os.makedirs(os.path.dirname(val_out_png), exist_ok=True)
        plt.savefig(val_out_png, dpi=300)
        plt.close()

def _plot_lines(ax, x, df, metrics, ylabel):
    for k, lbl in metrics:
        ax.plot(x, df[k], label=lbl)
    ax.set_ylabel(ylabel)
    if metrics:
        ax.legend(loc="best")

def random_assign_nan_labels(df: pd.DataFrame, labels, seed: int):
    """
    Randomly assigns NaN rows to existing label combinations
    (used ONLY for stratification).
    Modifies df in place.
    """
    y = df[labels]
    nan_mask = y.isna().any(axis=1) # row-wise

    if not nan_mask.any():
        return df

    rng = np.random.default_rng(seed)

    vc = (y[~nan_mask].astype(str) # avoid mixed dtype issues
          .agg("|".join, axis=1).value_counts())

    classes = vc.index.to_numpy()
    probs = (vc / vc.sum()).to_numpy()

    assigned = rng.choice(classes, nan_mask.sum(), p=probs)
    assigned_df = (pd.Series(assigned).str.split("|", expand=True).set_axis(labels, axis=1))

    df.loc[nan_mask, labels] = assigned_df.values

    print(f"Stratified CV split, Assigned {nan_mask.sum()} NaN rows:",
          assigned_df.value_counts())
    
    return df
