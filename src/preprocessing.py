"""Module 2 - preprocessing for supervised credit scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class PreparedData:
    X_train_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    preprocessor: ColumnTransformer
    feature_names: list[str]
    numeric_features: list[str]
    categorical_features: list[str]


def _clean_feature_names(names) -> list[str]:
    cleaned = []
    for name in names:
        text = str(name)
        if "__" in text:
            text = text.split("__", 1)[1]
        cleaned.append(text)
    return cleaned


def prepare_data(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.20,
    random_state: int = 42,
    scale_numeric: bool = True,
    numeric_imputation: str = "median",
    categorical_imputation: str = "most_frequent",
) -> PreparedData:
    """Split first, then fit preprocessing on training data only.

    This avoids data leakage. Numerical variables are imputed and optionally
    standardized. Categorical variables are imputed and one-hot encoded.
    """

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = [
        column for column in X.columns if column not in numeric_features
    ]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    numeric_steps = [
        ("imputer", SimpleImputer(strategy=numeric_imputation)),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_steps = [
        ("imputer", SimpleImputer(strategy=categorical_imputation)),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        ),
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric_features),
            ("cat", Pipeline(categorical_steps), categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    X_train_array = preprocessor.fit_transform(X_train_raw)
    X_test_array = preprocessor.transform(X_test_raw)

    feature_names = _clean_feature_names(preprocessor.get_feature_names_out())

    X_train = pd.DataFrame(
        X_train_array,
        columns=feature_names,
        index=X_train_raw.index,
    )
    X_test = pd.DataFrame(
        X_test_array,
        columns=feature_names,
        index=X_test_raw.index,
    )

    return PreparedData(
        X_train_raw=X_train_raw,
        X_test_raw=X_test_raw,
        y_train=y_train,
        y_test=y_test,
        X_train=X_train,
        X_test=X_test,
        preprocessor=preprocessor,
        feature_names=feature_names,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
