# One-pager — Credit Scoring explicable

## Problème métier
Prédire le risque de défaut d'un demandeur de crédit afin d'aider à la décision, tout en fournissant une justification interprétable.

## Données
Statlog German Credit Data (UCI, id=144) : 1 000 observations, 20 variables explicatives, cible binaire.
Dans l'application : 0 = bon payeur, 1 = mauvais payeur / défaut.

## Préparation
Séparation train/test stratifiée, imputation configurable, one-hot encoding des catégories et normalisation configurable des variables numériques. Le préprocesseur est appris uniquement sur le train afin d'éviter la fuite de données.

## Modèles comparés
Régression logistique, arbre de décision, kNN, Random Forest et XGBoost. Les hyperparamètres sont configurables dans l'interface et une optimisation RandomizedSearchCV est disponible.

## Résultats
À compléter après l'exécution finale du Module 3 :
- meilleur modèle : ...
- AUC-ROC : ...
- Gini : ...
- KS : ...
- précision : ...
- rappel : ...
- F1 : ...
- PR-AUC : ...

## Explicabilité
L'application fournit une importance native quand elle existe, une importance par permutation, une explication SHAP globale, une dépendance SHAP, une explication SHAP locale et LIME pour un individu.

## Recommandations et limites
Le seuil de décision doit être choisi selon le coût métier des faux négatifs et faux positifs. Les variables sensibles telles que l'âge, le sexe/statut personnel ou le statut de travailleur étranger doivent faire l'objet d'une analyse d'équité. Corrélation et importance ne signifient pas causalité.
