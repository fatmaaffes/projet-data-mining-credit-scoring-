"""Module 4 - evaluation metrics and curve helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def ks_statistic(y_true, y_prob) -> float:
    """Kolmogorov-Smirnov statistic for binary scoring."""

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def metrics_at_threshold(
    y_true,
    y_prob,
    threshold: float = 0.50,
) -> dict[str, float]:
    """Compute business-oriented classification metrics at a decision threshold."""

    y_pred = (np.asarray(y_prob) >= threshold).astype(int)

    roc_auc = roc_auc_score(y_true, y_prob)
    precision_curve, recall_curve_values, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_curve_values, precision_curve)

    return {
        "AUC-ROC": float(roc_auc),
        "Gini": float(2 * roc_auc - 1),
        "KS": ks_statistic(y_true, y_prob),
        "Précision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Rappel": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "PR-AUC": float(pr_auc),
    }


def confusion_at_threshold(y_true, y_prob, threshold: float = 0.50) -> np.ndarray:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def roc_points(y_true, y_prob) -> pd.DataFrame:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return pd.DataFrame(
        {
            "fpr": fpr,
            "tpr": tpr,
            "threshold": thresholds,
        }
    )


def precision_recall_points(y_true, y_prob) -> pd.DataFrame:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # thresholds has one element fewer than precision/recall.
    threshold_values = np.append(thresholds, np.nan)

    return pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "threshold": threshold_values,
        }
    )


def comparison_table(results: dict) -> pd.DataFrame:
    rows = []

    for name, result in results.items():
        row = {"Modèle": name, **result.metrics}
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    table = pd.DataFrame(rows)
    metric_columns = [
        "AUC-ROC",
        "Gini",
        "KS",
        "Précision",
        "Rappel",
        "F1",
        "PR-AUC",
    ]

    for column in metric_columns:
        if column in table.columns:
            table[column] = table[column].astype(float).round(4)

    return table.sort_values("AUC-ROC", ascending=False).reset_index(drop=True)
