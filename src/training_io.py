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


def save_train_test_subjects(df_train, df_test, output_path, savename):
    df_train.to_csv(os.path.join(output_path, f'{savename}_training-set.csv'), index=False)
    df_test.to_csv(os.path.join(output_path, f'{savename}_testing-set.csv'), index=False)


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
