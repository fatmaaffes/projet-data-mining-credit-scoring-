"""XGBoost model definition."""

from xgboost import XGBClassifier


DEFAULT_PARAMS = {
    "n_estimators": 250,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
}

SEARCH_SPACE = {
    "n_estimators": [150, 250, 400, 600],
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
}


def build_model(**params):
    config = {**DEFAULT_PARAMS, **params}
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        **config,
    )
