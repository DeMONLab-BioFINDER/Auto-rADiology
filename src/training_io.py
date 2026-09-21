# src/training_io.py
import os
import pandas as pd
import matplotlib.pyplot as plt


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
    if mode == "row":
        df.to_csv(csv_path, index=False)
    else:
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


def save_train_test_subjects(df_train, df_test, output_path):
    df_train.to_csv(os.path.join(output_path, 'train_subjects.csv'), index=False)
    df_test.to_csv(os.path.join(output_path, 'test_subjects.csv'), index=False)


def plot_metrics_from_csv(csv_path: str, out_dir: str, class_present: bool, reg_present: bool):
    """
    Build per-epoch plots from the training log CSV (epoch, train_loss, val_loss,
    eval_metric, auc, acc, mae, rmse, r2 - whichever columns are present).

    Always writes `epoch_train_loss.png` (the one curve that's meaningful even when
    there's no validation split, e.g. a fixed-epoch direct train/test run).

    Only when a real validation split exists for this fold (`val_loss` has at least
    one non-NaN value) does it also write:
    - `epoch_loss_train_vs_val.png` (train vs val loss)
    - `epoch_classification_metrics.png` (auc, acc) if `class_present`
    - `epoch_regression_metrics.png` (mae, rmse, r2) if `reg_present`
    Skipping these when there's no validation avoids plots with an empty/NaN line
    still showing up in the legend.
    """
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    if df.empty or "epoch" not in df.columns: return

    os.makedirs(out_dir, exist_ok=True)
    x = df["epoch"]

    if "train_loss" in df.columns:
        _save_line_plot(x, df, [("train_loss", "Training loss")], "Loss",
                         "Training loss per epoch", os.path.join(out_dir, "epoch_train_loss.png"))

    has_validation = "val_loss" in df.columns and df["val_loss"].notna().any()
    if not has_validation:
        return

    _save_line_plot(x, df, [("train_loss", "Training loss"), ("val_loss", "Validation loss")], "Loss",
                     "Training vs validation loss per epoch", os.path.join(out_dir, "epoch_loss_train_vs_val.png"))

    if class_present:
        class_defs = [(k, lbl) for k, lbl in [("auc", "AUC"), ("acc", "Accuracy")] if k in df.columns]
        if class_defs:
            _save_line_plot(x, df, class_defs, "Value", "Classification metrics per epoch (validation)",
                             os.path.join(out_dir, "epoch_classification_metrics.png"))

    if reg_present:
        reg_defs = [(k, lbl) for k, lbl in [("mae", "MAE"), ("rmse", "RMSE"), ("r2", "R2")] if k in df.columns]
        if reg_defs:
            _save_line_plot(x, df, reg_defs, "Value", "Regression metrics per epoch (validation)",
                             os.path.join(out_dir, "epoch_regression_metrics.png"))


def _save_line_plot(x, df, metrics, ylabel, title, out_png):
    plt.figure(figsize=(10, 6), dpi=300)
    ax = plt.gca()
    for k, lbl in metrics:
        ax.plot(x, df[k], label=lbl)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Epoch")
    ax.legend(loc="best")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def write_run_info(output_path, args):
    """Human-readable summary of what this run is, written once at the top of output_path."""
    targets_list = [t.strip() for t in args.targets.split(",") if t.strip()]
    class_present = "visual_read" in targets_list
    reg_targets = [t for t in targets_list if t != "visual_read"]

    if args.tune:
        mode = "hyperparameter tuning (Optuna, nested CV)"
    elif args.run_kfold_cv:
        mode = f"{args.n_splits}-fold cross-validation"
    elif args.val_size > 0:
        mode = "direct train/val/test split"
    else:
        mode = "direct train/test split (fixed epochs, no validation)"

    lines = [
        f"Run:         {args.output_name}",
        f"Started:     {args.output_date_time}",
        f"Dataset:     {args.dataset} ({args.data_type})",
        f"Targets:     {args.targets}"
        + (f" [classification: visual_read]" if class_present else "")
        + (f" [regression: {', '.join(reg_targets)}]" if reg_targets else ""),
        f"Model:       {args.model}",
        f"Mode:        {mode}",
        f"Split:       train={args.train_size}, val={args.val_size}, test={args.test_size}",
        f"Stratify by: {args.stratifycvby}",
        "",
        "Folder guide:",
        "  splits/        - the overall dataset train/val/test partition (subject-level CSVs)",
        "  final_model/   - the one trained model for this run: checkpoints/, per-epoch metrics/plots, test predictions",
        "  kfold-N/       - (CV runs only) per-fold checkpoints/metrics/predictions",
        "  evaluation/    - final held-out test predictions and summary metrics",
    ]
    with open(os.path.join(output_path, "RUN_INFO.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
