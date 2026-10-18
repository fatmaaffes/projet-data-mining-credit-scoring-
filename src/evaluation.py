"""Module 4 - evaluation metrics, curves and threshold helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
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
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline


def ks_statistic(y_true, y_prob) -> float:
    """Kolmogorov-Smirnov statistic for binary scoring."""

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def metrics_at_threshold(
    y_true,
    y_prob,
    threshold: float = 0.50,
) -> dict[str, float]:
    """Compute scoring metrics, including threshold-dependent metrics."""

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
    """Return confusion matrix ordered as [[TN, FP], [FN, TP]]."""

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


def cross_validated_probabilities(
    estimator,
    X_train,
    y_train,
    *,
    cv_folds: int = 5,
    random_state: int = 42,
    preprocessor=None,
) -> np.ndarray:
    """Generate leakage-safe out-of-fold probabilities from training data only.

    If a preprocessor is provided, it is refitted inside every CV fold rather
    than being learned once from the whole training split.
    """

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )

    if preprocessor is None:
        cv_estimator = clone(estimator)
    else:
        cv_estimator = Pipeline(
            [
                ("preprocessor", clone(preprocessor)),
                ("model", clone(estimator)),
            ]
        )

    probabilities = cross_val_predict(
        cv_estimator,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )
    return np.asarray(probabilities)[:, 1]


def threshold_analysis(
    y_true,
    y_prob,
    *,
    minimum: float = 0.05,
    maximum: float = 0.95,
    step: float = 0.01,
) -> pd.DataFrame:
    """Evaluate the precision/recall trade-off over a grid of thresholds."""

    thresholds = np.round(
        np.arange(minimum, maximum + step / 2, step),
        2,
    )
    rows = []

    for threshold in thresholds:
        metrics = metrics_at_threshold(y_true, y_prob, float(threshold))
        matrix = confusion_at_threshold(y_true, y_prob, float(threshold))
        tn, fp, fn, tp = matrix.ravel()

        rows.append(
            {
                "Seuil": float(threshold),
                "Précision": metrics["Précision"],
                "Rappel": metrics["Rappel"],
                "F1": metrics["F1"],
                "VN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "VP": int(tp),
            }
        )

    return pd.DataFrame(rows)


def recommend_threshold(
    y_true,
    y_prob,
    *,
    objective: str = "balance",
    fn_cost: float = 1.0,
    fp_cost: float = 1.0,
) -> dict:
    """Recommend a threshold from validation predictions.

    objective:
      - "balance": maximise F1.
      - "cost": minimise fn_cost * FN + fp_cost * FP.
    """

    table = threshold_analysis(y_true, y_prob)

    if objective == "balance":
        best_index = table["F1"].idxmax()
        criterion = "F1 maximal sur validation"
        table["Coût"] = np.nan
    elif objective == "cost":
        table["Coût"] = fn_cost * table["FN"] + fp_cost * table["FP"]
        min_cost = table["Coût"].min()
        candidates = table[table["Coût"] == min_cost]
        best_index = candidates["F1"].idxmax()
        criterion = (
            f"coût minimal sur validation "
            f"(FN×{fn_cost:g} + FP×{fp_cost:g})"
        )
    else:
        raise ValueError("objective must be 'balance' or 'cost'")

    best = table.loc[best_index].to_dict()
    best["Critère"] = criterion
    return best


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
