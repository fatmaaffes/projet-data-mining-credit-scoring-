"""k-Nearest Neighbors model definition."""

from sklearn.neighbors import KNeighborsClassifier


DEFAULT_PARAMS = {
    "n_neighbors": 7,
    "weights": "uniform",
    "p": 2,
}

SEARCH_SPACE = {
    "n_neighbors": [3, 5, 7, 9, 11, 15, 21],
    "weights": ["uniform", "distance"],
    "p": [1, 2],
}


def build_model(**params):
    config = {**DEFAULT_PARAMS, **params}
    return KNeighborsClassifier(**config)
