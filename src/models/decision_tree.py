"""Decision Tree model definition."""

from sklearn.tree import DecisionTreeClassifier


DEFAULT_PARAMS = {
    "max_depth": 5,
    "min_samples_split": 2,
    "criterion": "gini",
    "class_weight": None,
}

SEARCH_SPACE = {
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_split": [2, 5, 10, 20],
    "criterion": ["gini", "entropy"],
    "class_weight": [None, "balanced"],
}


def build_model(**params):
    config = {**DEFAULT_PARAMS, **params}
    return DecisionTreeClassifier(
        random_state=42,
        **config,
    )
