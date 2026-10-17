"""Module 3 - model training, tuning and comparison."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

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
    """Train one classifier, optionally with randomized hyperparameter search."""

    params = params or {}
    estimator = build_model(name, **params)
    best_params = dict(params)

    if tune:
        cv = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=42,
        )

        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=search_space(name),
            n_iter=n_iter,
            scoring="roc_auc",
            cv=cv,
            random_state=42,
            n_jobs=-1,
            refit=True,
        )
        search.fit(prepared.X_train, prepared.y_train)
        estimator = search.best_estimator_
        best_params = search.best_params_
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
