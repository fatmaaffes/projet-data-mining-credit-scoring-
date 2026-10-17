"""Data loading and exploratory helpers for the German Credit project.

Module 1 only:
- automatic download from UCI
- readable feature names
- binary target mapping (0 = good, 1 = bad/default)
- descriptive statistics
- missing-value summary
- class proportions
- simple EDA helpers
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from ucimlrepo import fetch_ucirepo


COLUMN_NAMES = {
    "Attribute1": "checking_status",
    "Attribute2": "duration",
    "Attribute3": "credit_history",
    "Attribute4": "purpose",
    "Attribute5": "credit_amount",
    "Attribute6": "savings_status",
    "Attribute7": "employment",
    "Attribute8": "installment_rate",
    "Attribute9": "personal_status_sex",
    "Attribute10": "other_debtors",
    "Attribute11": "residence_since",
    "Attribute12": "property",
    "Attribute13": "age",
    "Attribute14": "other_installment_plans",
    "Attribute15": "housing",
    "Attribute16": "existing_credits",
    "Attribute17": "job",
    "Attribute18": "dependents",
    "Attribute19": "telephone",
    "Attribute20": "foreign_worker",
}

TARGET_LABELS = {
    0: "Bon payeur",
    1: "Mauvais payeur / défaut",
}


@dataclass(frozen=True)
class CreditData:
    """Container returned by the UCI loader."""

    X: pd.DataFrame
    y: pd.Series
    df: pd.DataFrame
    variables: pd.DataFrame


def load_german_credit() -> CreditData:
    """Download German Credit (UCI id=144) and return clean analysis tables."""

    dataset = fetch_ucirepo(id=144)

    X = dataset.data.features.copy()
    y_raw = dataset.data.targets.copy()

    X = X.rename(columns=COLUMN_NAMES)

    # UCI original target: 1 = Good, 2 = Bad.
    y = y_raw["class"].map({1: 0, 2: 1}).rename("default")

    if y.isna().any():
        raise ValueError(
            "La cible UCI n'a pas les valeurs attendues 1/2. "
            "Vérifier le chargement du dataset."
        )

    df = X.copy()
    df["default"] = y

    return CreditData(
        X=X,
        y=y,
        df=df,
        variables=dataset.variables.copy(),
    )


def build_data_dictionary(variables: pd.DataFrame) -> pd.DataFrame:
    """Build a readable feature dictionary from UCI metadata."""

    dictionary = variables.loc[variables["role"].eq("Feature")].copy()
    dictionary["variable"] = dictionary["name"].map(COLUMN_NAMES)

    columns = [
        "variable",
        "type",
        "description",
        "units",
        "missing_values",
    ]
    return dictionary[columns].reset_index(drop=True)


def split_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return numeric and non-numeric feature names."""

    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    return numeric, categorical


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing-value counts and percentages."""

    count = df.isna().sum()
    percentage = (count / len(df) * 100).round(2)

    return pd.DataFrame(
        {
            "valeurs_manquantes": count,
            "pourcentage": percentage,
        }
    )


def class_distribution(y: pd.Series) -> pd.DataFrame:
    """Return target counts and proportions."""

    count = y.value_counts().sort_index()
    percentage = (y.value_counts(normalize=True).sort_index() * 100).round(2)

    return pd.DataFrame(
        {
            "classe": count.index,
            "libelle": [TARGET_LABELS.get(int(v), str(v)) for v in count.index],
            "effectif": count.values,
            "pourcentage": percentage.values,
        }
    )


def numeric_describe(X: pd.DataFrame) -> pd.DataFrame:
    """Return standard descriptive statistics for numeric features."""

    numeric, _ = split_feature_types(X)
    return X[numeric].describe().T


def categorical_distribution(
    df: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """Return counts and percentages for one categorical feature."""

    count = df[column].value_counts(dropna=False)
    percentage = (
        df[column]
        .value_counts(dropna=False, normalize=True)
        .mul(100)
        .round(2)
    )

    return pd.DataFrame(
        {
            "modalite": count.index.astype(str),
            "effectif": count.values,
            "pourcentage": percentage.values,
        }
    )


def default_rate_by_category(
    df: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """Return positive-class (default=1) rate for each category."""

    result = (
        df.groupby(column, dropna=False)["default"]
        .agg(effectif="size", taux_defaut="mean")
        .reset_index()
    )
    result["taux_defaut"] = (result["taux_defaut"] * 100).round(2)
    return result.sort_values("taux_defaut", ascending=False)


def correlation_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix for numeric features only."""

    numeric, _ = split_feature_types(X)
    return X[numeric].corr()
