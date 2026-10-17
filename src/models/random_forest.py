"""Random Forest model definition."""

from sklearn.ensemble import RandomForestClassifier


DEFAULT_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 2,
    "class_weight": None,
}

SEARCH_SPACE = {
    "n_estimators": [150, 250, 400, 600],
    "max_depth": [4, 6, 8, 12, None],
    "min_samples_split": [2, 5, 10],
    "class_weight": [None, "balanced"],
}


def build_model(**params):
    config = {**DEFAULT_PARAMS, **params}
    return RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        **config,
    )
