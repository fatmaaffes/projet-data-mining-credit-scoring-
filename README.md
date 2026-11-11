# Credit Scoring explicable

Projet Challenge 1 — Application Web Interactive de Classification Explicable.

Cas métier : scoring de crédit avec Statlog German Credit Data (UCI, id=144).

Cible :
- default = 0 : bon payeur
- default = 1 : mauvais payeur / défaut

## Modules

1. Données & EDA
   - téléchargement automatique UCI
   - aperçu, dictionnaire, valeurs manquantes, proportion des classes
   - statistiques descriptives
   - distributions, boxplots, taux de défaut par modalité
   - corrélations

2. Préparation
   - choix des variables X / cible y
   - imputation configurable
   - one-hot encoding des variables catégorielles
   - normalisation configurable
   - séparation train/test stratifiée et paramétrable

3. Modélisation
   - Régression logistique
   - Arbre de décision
   - kNN
   - Random Forest
   - XGBoost
   - hyperparamètres configurables
   - optimisation optionnelle par RandomizedSearchCV

4. Évaluation
   - AUC-ROC, précision, rappel, F1, PR-AUC
   - Gini et KS pour le scoring crédit
   - ROC superposées
   - courbes précision-rappel
   - matrice de confusion
   - seuil de décision interactif

5. Explainable AI
   - importance native
   - permutation importance
   - SHAP global et local
   - dépendance SHAP sur la variable principale
   - LIME local
   - synthèse automatique
   - rappel sur variables sensibles et équité

6. FastAPI
   - GET /health
   - POST /predict
   - POST /explain
   - documentation Swagger automatique sur /docs

## Installation

PowerShell :

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

## Lancer Streamlit

Depuis la racine :

streamlit run app/app.py

## Lancer FastAPI

Dans un deuxième terminal :

.\.venv\Scripts\Activate.ps1
uvicorn api.main:app --reload

Swagger :
http://127.0.0.1:8000/docs

## Architecture

app/app.py : interface Streamlit
api/main.py : API FastAPI
src/data.py : chargement et EDA
src/preprocessing.py : préparation ML
src/training.py : entraînement et tuning
src/evaluation.py : métriques crédit et courbes
src/explainability.py : SHAP, LIME, importances
src/models/ : un fichier par modèle

## Démarche

Le notebook reste le laboratoire d'exploration et d'apprentissage.
Le dossier src contient la logique réutilisable.
Streamlit orchestre les six modules.
FastAPI expose la prédiction et l'interprétation locale.

## Déploiement

L'application Streamlit peut être déployée sur Streamlit Community Cloud.
L'API FastAPI doit être déployée séparément puis son URL peut être saisie dans le Module 6 de l'application.

## Recherche reproductible (audit académique)

La commande ci-dessous exécute une validation croisée stratifiée répétée (5 folds × 10 répétitions),
calcule les IC bootstrap à 95 % pour l'AUC de test, sélectionne un seuil minimisant
le coût 5×FN + 1×FP sur les seules prédictions hors pli du train et produit des
CSV/figures vectorielles. Elle réalise aussi un audit exploratoire des erreurs
par `personal_status_sex` (combinaison sexe/statut marital) et par tranche d'âge,
avec une étude d'ablation. Les métriques du test ne servent **jamais** à choisir le seuil.

```bash
python -m pip install -r requirements.txt pytest
python -m pytest -q tests/test_evaluation_robust.py
python run_experiments.py --output results/research
# optionnel et plus coûteux : cohérence SHAP-LIME sur 50 observations
python run_experiments.py --output results/research --with-explanations
```

Si l'accès réseau au dataset UCI est impossible, télécharger préalablement le fichier
original `german.data`, puis utiliser `--data-path /chemin/vers/german.data`.
La matrice de coût 5:1 provient de la documentation UCI ; le seuil théorique 1/6
suppose des probabilités bien calibrées (ce qui n'est pas automatiquement garanti).

La CV répétée est calculée sur le TRAIN pour préserver l'indépendance de l'évaluation finale.
`Average Precision` (AP) et l'aire trapézoïdale sous la courbe précision–rappel
(`PR-AUC`) sont deux statistiques différentes et doivent être nommées séparément.

L'audit par groupe est **descriptif**, pas une certification de non-discrimination.
Le jeu public est ancien, certaines modalités sont de petits effectifs et
`personal_status_sex` mélange sexe et statut matrimonial. L'usage en décision bancaire
réelle exige une vérification juridique et une validation externe.

Les analyses sont également exécutables via **GitHub Actions** ; les CSV et PNG/SVG
sont disponibles dans l'artefact `german-credit-experiments` du workflow.