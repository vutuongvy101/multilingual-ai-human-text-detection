"""Evaluation metrics and utilities."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: list,
    y_pred: list,
    y_prob: Optional[list] = None,
    average: str = "weighted"
) -> Dict[str, float]:
    """Compute comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional, for AUC)
        average: Averaging method for multiclass metrics

    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average),
        "recall": recall_score(y_true, y_pred, average=average),
        "f1": f1_score(y_true, y_pred, average=average),
    }

    # Add AUC if probabilities are provided
    if y_prob is not None:
        try:
            # For binary classification, use the positive class probabilities
            if len(np.unique(y_true)) == 2:
                y_prob_pos = np.array(y_prob)[:, 1] if len(y_prob[0]) > 1 else y_prob
                metrics["auc"] = roc_auc_score(y_true, y_prob_pos)
            else:
                # For multiclass, use one-vs-rest AUC
                metrics["auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr")
        except Exception as e:
            print(f"Could not compute AUC: {e}")

    return metrics


def compute_per_class_metrics(
    y_true: list,
    y_pred: list,
    class_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """Compute per-class precision, recall, and F1 scores.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names for each class

    Returns:
        DataFrame with per-class metrics
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )

    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(precision))]

    df = pd.DataFrame({
        "class": class_names,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })

    return df


def compute_per_language_metrics(
    dataset: pd.DataFrame,
    predictions: list,
    probabilities: Optional[list] = None
) -> pd.DataFrame:
    """Compute metrics broken down by language.

    Args:
        dataset: Dataset with 'lang' column
        predictions: Predicted labels
        probabilities: Predicted probabilities (optional)

    Returns:
        DataFrame with per-language metrics
    """
    results = []

    for lang in dataset["lang"].unique():
        mask = dataset["lang"] == lang
        y_true_lang = dataset.loc[mask, "label"].tolist()
        y_pred_lang = [predictions[i] for i in mask[mask].index]
        y_prob_lang = [probabilities[i] for i in mask[mask].index] if probabilities else None

        metrics = compute_classification_metrics(
            y_true_lang, y_pred_lang, y_prob_lang
        )
        metrics["language"] = lang
        metrics["count"] = len(y_true_lang)

        results.append(metrics)

    return pd.DataFrame(results)


def generate_confusion_matrix(
    y_true: list,
    y_pred: list,
    class_names: Optional[List[str]] = None,
    normalize: Optional[str] = None
) -> pd.DataFrame:
    """Generate confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names for each class
        normalize: Normalization method ('true', 'pred', 'all', or None)

    Returns:
        Confusion matrix as DataFrame
    """
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)

    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]

    return pd.DataFrame(cm, index=class_names, columns=class_names)