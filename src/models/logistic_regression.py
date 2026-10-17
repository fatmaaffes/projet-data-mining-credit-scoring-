"""Logistic Regression model definition."""

from sklearn.linear_model import LogisticRegression


DEFAULT_PARAMS = {
    "C": 1.0,
    "penalty": "l2",
    "class_weight": None,
}

SEARCH_SPACE = {
    "C": [0.01, 0.1, 0.5, 1.0, 2.0, 10.0],
    "penalty": ["l1", "l2"],
    "class_weight": [None, "balanced"],
}


def build_model(**params):
    config = {**DEFAULT_PARAMS, **params}
    return LogisticRegression(
        solver="liblinear",
        max_iter=2000,
        random_state=42,
        **config,
    )
