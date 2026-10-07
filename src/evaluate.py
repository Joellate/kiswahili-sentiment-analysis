"""Shared evaluation so every experiment is scored the same way.

Primary metric: macro-F1. The classes are imbalanced (~59% neutral, ~30%
positive, ~11% negative), so accuracy would reward always predicting
"neutral"; macro-F1 weights the rare negative class equally. This is also the
metric used in the AfriSenti benchmark / SemEval-2023 Task 12.
"""
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from data import LABELS

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
EXPERIMENTS_CSV = RESULTS_DIR / "experiments.csv"


def compute_metrics(y_true, y_pred) -> dict:
    """Accuracy, macro precision/recall/F1, weighted F1 and per-class F1."""
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(LABELS)), average="macro", zero_division=0
    )
    per_class = f1_score(y_true, y_pred, labels=range(len(LABELS)), average=None, zero_division=0)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": p,
        "macro_recall": r,
        "macro_f1": f,
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    metrics.update({f"f1_{name}": score for name, score in zip(LABELS, per_class)})
    return {k: round(float(v), 4) for k, v in metrics.items()}


def aggregate_seeds(runs: list[dict]) -> dict:
    """Mean of each metric over several random seeds, plus the std of macro-F1.

    With only ~450 validation / ~750 test tweets, a single run can swing by a few
    F1 points, so neural models are always reported as mean +- std over seeds.
    """
    df = pd.DataFrame(runs)
    out = df.mean().round(4).to_dict()
    out["macro_f1_std"] = round(float(df["macro_f1"].std(ddof=0)), 4)
    out["n_seeds"] = len(runs)
    return out


def report(y_true, y_pred) -> str:
    return classification_report(
        y_true, y_pred, labels=range(len(LABELS)), target_names=LABELS, digits=3, zero_division=0
    )


def plot_confusion(y_true, y_pred, title: str, save_as: str | None = None):
    """Row-normalised confusion matrix (each row sums to 1 = recall per class)."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(LABELS)), normalize="true")
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                xticklabels=LABELS, yticklabels=LABELS, ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    if save_as:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / save_as, dpi=150)
    return fig


def log_experiment(name: str, split: str, metrics: dict, config: dict | None = None):
    """Append one row to results/experiments.csv (replacing a previous run of the same name+split)."""
    RESULTS_DIR.mkdir(exist_ok=True)
    row = {"experiment": name, "split": split, **metrics,
           "config": json.dumps(config or {}), "timestamp": datetime.now().isoformat(timespec="seconds")}
    if EXPERIMENTS_CSV.exists():
        df = pd.read_csv(EXPERIMENTS_CSV)
        df = df[~((df.experiment == name) & (df.split == split))]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(EXPERIMENTS_CSV, index=False)
    return df
