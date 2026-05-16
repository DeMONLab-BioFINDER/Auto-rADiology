#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


RUN_DIR = Path(
    "../results/tau_raw_Gothenburg_CNN3D_visual_read_2split80-20_stratify-site,visual_read_raw-quick-test_20260408_141238"
)


def save_figure(fig: plt.Figure, out_path: Path, **kwargs) -> bool:
    if out_path.exists():
        print(f"[skip] {out_path} exists")
        return False
    fig.savefig(out_path, **kwargs)
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Create plots for visual_read runs.")
    parser.add_argument(
        "--run_dir",
        default=str(RUN_DIR),
        help="Path to the run directory (defaults to the hardcoded example).",
    )
    return parser.parse_args()


def load_results(run_dir: Path):
    metrics_path = run_dir / "validation" / "GothenburgTest_Gothenburg_metrics.csv"
    preds_path = run_dir / "validation" / "Gothenburg" / "Test_Gothenburg_results.csv"
    train_curve_path = run_dir / "train-test-split" / "trainning_metrics_per_epoch.csv"

    df_metrics = pd.read_csv(metrics_path)
    df_preds = pd.read_csv(preds_path)
    df_curve = pd.read_csv(train_curve_path)

    # The training csv has many repeated unnamed columns from append mode.
    df_curve = df_curve.loc[:, ~df_curve.columns.str.contains(r"^Unnamed")]

    return df_metrics, df_preds, df_curve


def make_training_curve(df_curve: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_curve["epoch"], df_curve["train_loss"], label="train loss", linewidth=2)
    ax.plot(df_curve["epoch"], df_curve["auc"], label="AUC", linewidth=2)
    ax.plot(df_curve["epoch"], df_curve["acc"], label="accuracy", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_title("Training-Set Metrics Per Epoch")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, out_dir / "training_curves.png", dpi=300)
    plt.close(fig)


def make_probability_histogram(df_preds: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        df_preds.loc[df_preds["y"] == 0, "prob"],
        bins=20,
        alpha=0.6,
        label="true class 0",
    )
    ax.hist(
        df_preds.loc[df_preds["y"] == 1, "prob"],
        bins=20,
        alpha=0.6,
        label="true class 1",
    )
    ax.set_xlabel("Predicted probability for class 1")
    ax.set_ylabel("Count")
    ax.set_title("Predicted Probability Distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, out_dir / "probability_histogram.png", dpi=300)
    plt.close(fig)


def make_confusion_matrix(df_preds: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        df_preds["y"],
        df_preds["pred"],
        display_labels=["visual_read=0", "visual_read=1"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    save_figure(fig, out_dir / "confusion_matrix.png", dpi=300)
    plt.close(fig)


def make_roc_curve(df_preds: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(df_preds["y"], df_preds["prob"], ax=ax)
    ax.set_title("ROC Curve")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, out_dir / "roc_curve.png", dpi=300)
    plt.close(fig)


def print_summary(df_metrics: pd.DataFrame, df_preds: pd.DataFrame):
    tn = ((df_preds["y"] == 0) & (df_preds["pred"] == 0)).sum()
    fp = ((df_preds["y"] == 0) & (df_preds["pred"] == 1)).sum()
    fn = ((df_preds["y"] == 1) & (df_preds["pred"] == 0)).sum()
    tp = ((df_preds["y"] == 1) & (df_preds["pred"] == 1)).sum()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    print("Test metrics from saved csv:")
    print(df_metrics.to_string(index=False))
    print()
    print(f"Test set size: {len(df_preds)}")
    print("Class counts:")
    print(df_preds["y"].value_counts().sort_index().to_string())
    print()
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Sensitivity: {sensitivity:.3f}")
    print(f"Specificity: {specificity:.3f}")
    print(
        "\nNote: in this --no-tune run, the per-epoch training csv reflects"
        " performance on the training set, because early stopping monitored train data."
    )


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    out_dir = run_dir / "figures"
    out_dir.mkdir(exist_ok=True)

    df_metrics, df_preds, df_curve = load_results(run_dir)

    make_training_curve(df_curve, out_dir)
    make_probability_histogram(df_preds, out_dir)
    make_confusion_matrix(df_preds, out_dir)
    make_roc_curve(df_preds, out_dir)
    print_summary(df_metrics, df_preds)

    print(f"\nSaved figures to: {out_dir}")


if __name__ == "__main__":
    main()
