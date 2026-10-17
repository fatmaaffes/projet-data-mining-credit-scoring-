# Credit Scoring explicable

Projet de Data Mining — application web interactive de classification explicable.

## Cas métier

Scoring de crédit à partir du jeu Statlog German Credit Data (UCI, id=144).

Cible utilisée dans le projet :

- default = 0 : bon payeur
- default = 1 : mauvais payeur / défaut

## État actuel

### Module 1 — Données & EDA

Implémenté :

- téléchargement automatique via ucimlrepo
- renommage lisible des 20 variables
- mapping de la cible
- aperçu des données
- dictionnaire des variables
- contrôle des valeurs manquantes
- proportion des classes
- statistiques descriptives
- histogrammes et boxplots
- analyse des variables numériques par rapport à la cible
- fréquences des variables catégorielles
- taux de défaut par modalité
- matrice de corrélation de Pearson

Les modules 2 à 6 seront ajoutés progressivement.

## Structure

app/app.py : interface Streamlit
api/main.py : API FastAPI
src/data.py : chargement et logique du Module 1
src/preprocessing.py : préparation des données
src/evaluation.py : évaluation
src/explainability.py : XAI
src/models/ : modèles de classification

## Installation

Sous PowerShell :

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

## Lancer l'application

Depuis la racine du projet :

streamlit run app/app.py

## Principe pédagogique

Le notebook sert de laboratoire pour comprendre les données et expérimenter.
Le code dans src/ contient la logique réutilisable.
L'application Streamlit expose progressivement les six modules demandés par le projet.
