"""Module 5 - global and local model explanations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def native_importance(model, feature_names: list[str]) -> pd.DataFrame | None:
    """Return model-native importance when the estimator exposes one."""

    values = None
    source = None

    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        if coef.ndim == 2:
            coef = coef[0]
        values = np.abs(coef)
        source = "coefficient_absolu"

    elif hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
        source = "importance_modele"

    if values is None:
        return None

    result = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": values,
            "source": source,
        }
    )
    return result.sort_values("importance", ascending=False).reset_index(drop=True)


def permutation_importance_df(
    model,
    X_test,
    y_test,
    feature_names: list[str],
    *,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Permutation importance measured with ROC-AUC."""

    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
    )

    table = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": result.importances_mean,
            "std": result.importances_std,
        }
    )
    return table.sort_values("importance", ascending=False).reset_index(drop=True)


def compute_shap_values(
    model,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    feature_names: list[str],
):
    """Compute model-agnostic SHAP values for the positive-class probability.

    A small background sample is used so the operation remains practical in a
    Streamlit application.
    """

    import shap

    background = X_background.sample(
        n=min(30, len(X_background)),
        random_state=42,
    )

    def positive_probability(values):
        frame = pd.DataFrame(values, columns=feature_names)
        return model.predict_proba(frame)[:, 1]

    explainer = shap.Explainer(
        positive_probability,
        background,
        feature_names=feature_names,
        algorithm="permutation",
    )

    values = explainer(X_explain)
    return explainer, values


def lime_explanation(
    model,
    X_train: pd.DataFrame,
    row: pd.Series,
    feature_names: list[str],
    *,
    num_features: int = 10,
) -> pd.DataFrame:
    """Explain one transformed row with LIME."""

    from lime.lime_tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(
        training_data=np.asarray(X_train, dtype=float),
        feature_names=feature_names,
        class_names=["Bon payeur", "Défaut"],
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )

    explanation = explainer.explain_instance(
        np.asarray(row, dtype=float),
        model.predict_proba,
        labels=(1,),
        num_features=min(num_features, len(feature_names)),
    )

    return pd.DataFrame(
        explanation.as_list(),
        columns=["condition", "contribution"],
    )


def original_feature_name(
    transformed_name: str,
    original_features: list[str],
) -> str:
    """Map a transformed/OHE feature back to its original variable."""

    for feature in sorted(original_features, key=len, reverse=True):
        if transformed_name == feature or transformed_name.startswith(feature + "_"):
            return feature
    return transformed_name


def grouped_shap_importance(
    shap_values,
    feature_names: list[str],
    original_features: list[str],
) -> pd.DataFrame:
    """Aggregate mean absolute SHAP values at original-variable level."""

    values = np.asarray(shap_values.values)
    if values.ndim == 1:
        values = values.reshape(1, -1)

    importance = np.abs(values).mean(axis=0)
    table = pd.DataFrame(
        {
            "feature_transformee": feature_names,
            "variable": [
                original_feature_name(name, original_features)
                for name in feature_names
            ],
            "importance": importance,
        }
    )

    return (
        table.groupby("variable", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def local_shap_contributions(
    shap_row,
    feature_names: list[str],
    original_features: list[str],
) -> pd.DataFrame:
    """Aggregate local SHAP contributions at original-variable level."""

    values = np.asarray(shap_row.values).reshape(-1)
    table = pd.DataFrame(
        {
            "feature_transformee": feature_names,
            "variable": [
                original_feature_name(name, original_features)
                for name in feature_names
            ],
            "contribution": values,
        }
    )

    grouped = (
        table.groupby("variable", as_index=False)["contribution"]
        .sum()
    )
    grouped["importance_absolue"] = grouped["contribution"].abs()
    grouped["sens"] = np.where(
        grouped["contribution"] > 0,
        "Augmente le risque",
        np.where(
            grouped["contribution"] < 0,
            "Réduit le risque",
            "Effet quasi nul",
        ),
    )

    return (
        grouped.sort_values("importance_absolue", ascending=False)
        .reset_index(drop=True)
    )


def lime_contributions_by_feature(
    lime_df: pd.DataFrame,
    feature_names: list[str],
    original_features: list[str],
) -> pd.DataFrame:
    """Aggregate LIME conditions back to original variables."""

    transformed_sorted = sorted(feature_names, key=len, reverse=True)

    rows = []
    for item in lime_df.itertuples(index=False):
        condition = str(item.condition)
        contribution = float(item.contribution)

        transformed = next(
            (name for name in transformed_sorted if name in condition),
            condition,
        )
        variable = original_feature_name(transformed, original_features)
        rows.append(
            {
                "variable": variable,
                "contribution": contribution,
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return pd.DataFrame(
            columns=["variable", "contribution", "importance_absolue", "sens"]
        )

    grouped = (
        table.groupby("variable", as_index=False)["contribution"]
        .sum()
    )
    grouped["importance_absolue"] = grouped["contribution"].abs()
    grouped["sens"] = np.where(
        grouped["contribution"] > 0,
        "Augmente le risque",
        np.where(
            grouped["contribution"] < 0,
            "Réduit le risque",
            "Effet quasi nul",
        ),
    )

    return (
        grouped.sort_values("importance_absolue", ascending=False)
        .reset_index(drop=True)
    )


def plain_language_summary(
    feature_names: list[str],
    shap_row,
    *,
    top_n: int = 3,
    original_features: list[str] | None = None,
) -> str:
    """Create a short deterministic explanation from local SHAP contributions."""

    if original_features:
        table = local_shap_contributions(
            shap_row,
            feature_names,
            original_features,
        ).head(top_n)

        positive = table.loc[
            table["contribution"] > 0,
            "variable",
        ].tolist()
        negative = table.loc[
            table["contribution"] < 0,
            "variable",
        ].tolist()
    else:
        values = np.asarray(shap_row.values).reshape(-1)
        order = np.argsort(np.abs(values))[::-1][:top_n]

        positive = []
        negative = []

        for index in order:
            feature = feature_names[index]
            value = float(values[index])
            if value > 0:
                positive.append(feature)
            elif value < 0:
                negative.append(feature)

    parts = []

    if positive:
        parts.append(
            "Le risque est augmenté principalement par : "
            + ", ".join(positive)
            + "."
        )

    if negative:
        parts.append(
            "Le risque est réduit principalement par : "
            + ", ".join(negative)
            + "."
        )

    if not parts:
        return "Aucune contribution dominante n'a été identifiée pour cet individu."

    return " ".join(parts)
