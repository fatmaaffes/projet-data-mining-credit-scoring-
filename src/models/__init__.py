"""Model registry for Module 3."""

from . import decision_tree, knn, logistic_regression, random_forest, xgboost_model


MODEL_MODULES = {
    "Régression logistique": logistic_regression,
    "Arbre de décision": decision_tree,
    "kNN": knn,
    "Random Forest": random_forest,
    "XGBoost": xgboost_model,
}


def build_model(name: str, **params):
    if name not in MODEL_MODULES:
        raise KeyError(f"Modèle inconnu: {name}")
    return MODEL_MODULES[name].build_model(**params)


def search_space(name: str):
    if name not in MODEL_MODULES:
        raise KeyError(f"Modèle inconnu: {name}")
    return MODEL_MODULES[name].SEARCH_SPACE
