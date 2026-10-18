"""Module 3 - model training, tuning and comparison."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.evaluation import metrics_at_threshold
from src.models import build_model, search_space


@dataclass
class ModelResult:
    name: str
    estimator: object
    y_prob: np.ndarray
    metrics: dict[str, float]
    best_params: dict = field(default_factory=dict)
    tuned: bool = False


def train_one_model(
    name: str,
    prepared,
    *,
    params: dict | None = None,
    tune: bool = False,
    cv_folds: int = 5,
    n_iter: int = 8,
) -> ModelResult:
    """Train one classifier, optionally with leakage-safe randomized search.

    During hyperparameter tuning, preprocessing is refitted inside every
    cross-validation fold. The held-out test set remains untouched.
    """

    params = params or {}
    estimator = build_model(name, **params)
    best_params = dict(params)

    if tune:
        cv = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=42,
        )

        tuning_pipeline = Pipeline(
            [
                ("preprocessor", clone(prepared.preprocessor)),
                ("model", estimator),
            ]
        )
        prefixed_search_space = {
            f"model__{key}": values
            for key, values in search_space(name).items()
        }

        search = RandomizedSearchCV(
            estimator=tuning_pipeline,
            param_distributions=prefixed_search_space,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=cv,
            random_state=42,
            n_jobs=-1,
            refit=True,
        )
        search.fit(prepared.X_train_raw, prepared.y_train)

        # The refitted model has seen preprocessing learned only from the full
        # training split. Its transformed feature order matches prepared.X_train.
        estimator = search.best_estimator_.named_steps["model"]
        best_params = {
            key.removeprefix("model__"): value
            for key, value in search.best_params_.items()
        }
    else:
        estimator.fit(prepared.X_train, prepared.y_train)

    y_prob = estimator.predict_proba(prepared.X_test)[:, 1]

    return ModelResult(
        name=name,
        estimator=estimator,
        y_prob=y_prob,
        metrics=metrics_at_threshold(
            prepared.y_test,
            y_prob,
            threshold=0.50,
        ),
        best_params=best_params,
        tuned=tune,
    )


def train_models(
    names: list[str],
    prepared,
    *,
    model_params: dict[str, dict] | None = None,
    tune: bool = False,
    cv_folds: int = 5,
    n_iter: int = 8,
) -> dict[str, ModelResult]:
    results = {}
    model_params = model_params or {}

    for name in names:
        results[name] = train_one_model(
            name,
            prepared,
            params=model_params.get(name, {}),
            tune=tune,
            cv_folds=cv_folds,
            n_iter=n_iter,
        )

    return results
