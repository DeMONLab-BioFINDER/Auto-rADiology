# src/splits.py
import os
import pandas as pd
import numpy as np

from typing import Mapping, Tuple
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


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
