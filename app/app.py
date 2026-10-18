"""Streamlit application covering the six modules of the project."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import (
    TARGET_LABELS,
    build_data_dictionary,
    categorical_distribution,
    class_distribution,
    correlation_matrix,
    default_rate_by_category,
    load_german_credit,
    missing_summary,
    numeric_describe,
    split_feature_types,
)
from src.evaluation import (
    comparison_table,
    confusion_at_threshold,
    cross_validated_probabilities,
    metrics_at_threshold,
    precision_recall_points,
    recommend_threshold,
    roc_points,
)
from src.explainability import (
    compute_shap_values,
    lime_explanation,
    native_importance,
    permutation_importance_df,
    plain_language_summary,
)
from src.preprocessing import prepare_data
from src.training import train_models


st.set_page_config(
    page_title="Credit Scoring XAI",
    page_icon="💳",
    layout="wide",
)

st.markdown(
    """
<style>
div[data-testid="stMetric"] {
    border: 1px solid rgba(120, 130, 150, 0.22);
    border-radius: 12px;
    padding: 0.65rem 0.8rem;
}
[data-testid="stSidebar"] {
    border-right: 1px solid rgba(120, 130, 150, 0.16);
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_data():
    return load_german_credit()


def module_intro(goal: str, topics: str, importance: str) -> None:
    """Shared pedagogical header used at the beginning of each module."""

    st.markdown("### 🎯 But du module")
    st.info(goal)
    st.markdown("### 🧩 Ce qu’on va traiter")
    st.markdown(topics)
    st.markdown("### 🔗 Pourquoi c’est important pour notre Credit Scoring")
    st.success(importance)


def interpretation_box(text: str, *, title: str = "📌 Interprétation") -> None:
    st.markdown(f"**{title}**")
    st.info(text)


def metric_guide() -> None:
    guide = pd.DataFrame(
        [
            ["AUC-ROC", "Capacité globale à classer un mauvais payeur au-dessus d’un bon.", "Non"],
            ["Gini", "Indicateur bancaire dérivé de l’AUC : 2 × AUC − 1.", "Non"],
            ["KS", "Séparation maximale entre les scores des bons et mauvais payeurs.", "Non"],
            ["Précision", "Parmi les clients prédits en défaut, part réellement en défaut.", "Oui"],
            ["Rappel", "Parmi tous les vrais défauts, part détectée par le modèle.", "Oui"],
            ["F1", "Compromis entre précision et rappel.", "Oui"],
            ["PR-AUC", "Qualité du compromis précision-rappel sur l’ensemble des seuils.", "Non"],
        ],
        columns=["Métrique", "Comment la lire ?", "Dépend du seuil choisi ?"],
    )
    st.dataframe(guide, use_container_width=True, hide_index=True)


@st.cache_resource(show_spinner=False)
def get_prepared(
    selected_features: tuple[str, ...],
    test_size: float,
    scale_numeric: bool,
    numeric_imputation: str,
    categorical_imputation: str,
):
    data = get_data()
    return prepare_data(
        data.X[list(selected_features)],
        data.y,
        test_size=test_size,
        random_state=42,
        scale_numeric=scale_numeric,
        numeric_imputation=numeric_imputation,
        categorical_imputation=categorical_imputation,
    )


data = get_data()
X = data.X
y = data.y
df = data.df
numeric_features, categorical_features = split_feature_types(X)

if "prep_config" not in st.session_state:
    st.session_state.prep_config = {
        "selected_features": X.columns.tolist(),
        "test_size": 0.20,
        "scale_numeric": True,
        "numeric_imputation": "median",
        "categorical_imputation": "most_frequent",
    }

st.title("💳 Credit Scoring explicable")
st.caption("German Credit — UCI Statlog · Application interactive complète")

st.success(
    f"Dataset chargé : {len(df)} clients · {X.shape[1]} variables explicatives · "
    "cible binaire default"
)

module = st.sidebar.radio(
    "Navigation",
    [
        "1 · Données & EDA",
        "1.1 · Conclusion EDA",
        "2 · Préparation",
        "2.1 · Conclusion Préparation",
        "3 · Modélisation",
        "3.1 · Conclusion Modélisation",
        "4 · Évaluation",
        "4.1 · Conclusion Évaluation",
        "5 · Explainable AI",
        "6 · FastAPI",
    ],
)

st.sidebar.caption("Classe positive : default = 1 (mauvais payeur / défaut)")


if module == "1 · Données & EDA":
    st.header("Module 1 — Données & EDA")
    module_intro(
        "Comprendre la qualité, la structure et les relations présentes dans les données avant toute modélisation.",
        "Aperçu du dataset, statistiques descriptives, distributions numériques et catégorielles, cible et corrélations.",
        "Un modèle de crédit fiable commence par des données comprises : on vérifie ici les déséquilibres, les valeurs extrêmes, les variables potentiellement discriminantes et les risques de redondance.",
    )

    (
        tab_overview,
        tab_stats,
        tab_numeric,
        tab_categorical,
        tab_target,
        tab_corr,
    ) = st.tabs(
        [
            "Aperçu",
            "Statistiques",
            "EDA numérique",
            "EDA catégorielle",
            "Variable cible",
            "Corrélations",
        ]
    )

    with tab_overview:
        st.subheader("Aperçu des données")
        st.dataframe(df.head(10), use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Observations", len(df))
        col2.metric("Features", X.shape[1])
        col3.metric("Numériques", len(numeric_features))
        col4.metric("Catégorielles / binaires", len(categorical_features))

        st.subheader("Dictionnaire des variables")
        st.dataframe(
            build_data_dictionary(data.variables),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Valeurs manquantes")
        missing = missing_summary(df)
        st.dataframe(missing, use_container_width=True)

        total_missing = int(missing["valeurs_manquantes"].sum())
        if total_missing == 0:
            st.info("Aucune valeur manquante dans ce dataset.")
            interpretation_box(
                "Le dataset contient 1 000 clients, 20 variables explicatives et aucune valeur manquante. "
                "La qualité structurelle est donc suffisante pour poursuivre sans correction de valeurs absentes."
            )
        else:
            st.warning(f"{total_missing} valeurs manquantes détectées.")

    with tab_stats:
        st.subheader("Statistiques descriptives — variables numériques")
        st.dataframe(numeric_describe(X), use_container_width=True)

        st.markdown(
            """
**Lecture rapide de describe()**

- count : nombre de valeurs présentes
- mean : moyenne
- std : dispersion autour de la moyenne
- min / max : valeurs extrêmes
- 25% : premier quartile (Q1)
- 50% : médiane
- 75% : troisième quartile (Q3)
"""
        )
        interpretation_box(
            "Les variables numériques n'ont pas toutes la même échelle ni la même dispersion. "
            "Par exemple, credit_amount peut prendre des valeurs beaucoup plus élevées que age ou installment_rate. "
            "Les valeurs extrêmes observées ne sont pas automatiquement des erreurs : elles peuvent correspondre à de vrais profils de crédit et seront conservées sauf preuve contraire."
        )

    with tab_numeric:
        st.subheader("Analyse d'une variable numérique")

        feature = st.selectbox(
            "Variable numérique",
            numeric_features,
            index=numeric_features.index("duration")
            if "duration" in numeric_features
            else 0,
            key="numeric_feature",
        )

        nbins = st.slider(
            "Nombre de classes de l'histogramme",
            5,
            50,
            20,
            key="nbins",
        )

        histogram = px.histogram(
            df,
            x=feature,
            nbins=nbins,
            marginal="box",
            title=f"Distribution de {feature}",
        )
        histogram.update_layout(
            xaxis_title=feature,
            yaxis_title="Nombre de clients",
        )
        st.plotly_chart(histogram, use_container_width=True)

        st.subheader("Comparaison avec la cible default")
        comparison_df = df.copy()
        comparison_df["classe"] = comparison_df["default"].map(TARGET_LABELS)

        box = px.box(
            comparison_df,
            x="classe",
            y=feature,
            points="outliers",
            title=f"{feature} selon la classe de paiement",
        )
        box.update_layout(
            xaxis_title="Classe",
            yaxis_title=feature,
            height=520,
        )
        st.plotly_chart(box, use_container_width=True)

        grouped_numeric = (
            df.groupby("default")[feature]
            .describe()
            .rename(index=TARGET_LABELS)
        )
        st.dataframe(grouped_numeric, use_container_width=True)

        series = df[feature].dropna()
        skewness = float(series.skew()) if len(series) > 2 else 0.0
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
        med_good = float(df.loc[df["default"] == 0, feature].median())
        med_bad = float(df.loc[df["default"] == 1, feature].median())
        shape_text = (
            "asymétrique vers les grandes valeurs"
            if skewness > 0.5
            else "asymétrique vers les petites valeurs"
            if skewness < -0.5
            else "relativement équilibrée"
        )
        interpretation_box(
            f"La distribution de **{feature}** est {shape_text} (skewness ≈ {skewness:.2f}) et "
            f"{outliers} observations sont signalées comme valeurs extrêmes par la règle IQR. "
            f"La médiane est {med_good:.2f} chez les bons payeurs contre {med_bad:.2f} chez les défauts. "
            "Une différence entre les deux groupes suggère un pouvoir discriminant potentiel, sans prouver une causalité."
        )

    with tab_categorical:
        st.subheader("Analyse d'une variable catégorielle")

        feature = st.selectbox(
            "Variable catégorielle",
            categorical_features,
            index=(
                categorical_features.index("checking_status")
                if "checking_status" in categorical_features
                else 0
            ),
            key="categorical_feature",
        )

        distribution = categorical_distribution(df, feature)
        st.dataframe(distribution, use_container_width=True, hide_index=True)

        counts_chart = px.bar(
            distribution,
            x="modalite",
            y="effectif",
            title=f"Répartition de {feature}",
        )
        counts_chart.update_layout(
            xaxis_title=feature,
            yaxis_title="Nombre de clients",
        )
        st.plotly_chart(counts_chart, use_container_width=True)

        st.subheader("Taux de défaut par modalité")
        default_rate = default_rate_by_category(df, feature)
        st.dataframe(default_rate, use_container_width=True, hide_index=True)

        rate_chart = px.bar(
            default_rate,
            x=feature,
            y="taux_defaut",
            title=f"Taux de défaut selon {feature}",
        )
        rate_chart.update_layout(
            xaxis_title=feature,
            yaxis_title="Taux de défaut (%)",
        )
        st.plotly_chart(rate_chart, use_container_width=True)

        highest = default_rate.iloc[0]
        lowest = default_rate.iloc[-1]
        interpretation_box(
            f"Pour **{feature}**, le taux de défaut varie de {lowest['taux_defaut']:.2f}% "
            f"({lowest[feature]}) à {highest['taux_defaut']:.2f}% ({highest[feature]}). "
            "Plus l'écart entre modalités est marqué, plus cette variable peut aider le modèle à distinguer les profils de risque. "
            "Cette association reste descriptive et ne doit pas être interprétée comme une relation causale."
        )

    with tab_target:
        st.subheader("Répartition de la cible")

        target_distribution = class_distribution(y)
        st.dataframe(
            target_distribution,
            use_container_width=True,
            hide_index=True,
        )

        target_chart = px.bar(
            target_distribution,
            x="libelle",
            y="effectif",
            text="pourcentage",
            title="Répartition des classes",
        )
        target_chart.update_traces(texttemplate="%{text:.1f}%")
        target_chart.update_layout(
            xaxis_title="Classe",
            yaxis_title="Nombre de clients",
        )
        st.plotly_chart(target_chart, use_container_width=True)

        st.caption(
            "Dans ce projet, la classe positive est default = 1 "
            "(mauvais payeur / défaut)."
        )
        interpretation_box(
            "La cible contient environ 70 % de bons payeurs et 30 % de mauvais payeurs. "
            "Le déséquilibre est modéré : il justifie une séparation stratifiée et l'utilisation de métriques autres que l'accuracy, notamment rappel, PR-AUC, Gini et KS."
        )

    with tab_corr:
        st.subheader("Matrice de corrélation — variables numériques")
        corr = correlation_matrix(X).round(2)

        heatmap = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                zmin=-1,
                zmax=1,
                text=corr.values,
                texttemplate="%{text:.2f}",
                hovertemplate="%{y} ↔ %{x}<br>corrélation = %{z:.2f}<extra></extra>",
            )
        )
        heatmap.update_layout(
            title="Corrélation de Pearson",
            xaxis_title="Variables",
            yaxis_title="Variables",
            height=650,
        )
        st.plotly_chart(heatmap, use_container_width=True)

        st.info(
            "Une corrélation proche de +1 indique une forte relation linéaire positive, "
            "proche de -1 une forte relation linéaire négative, et proche de 0 peu de "
            "relation linéaire. Corrélation ne signifie pas causalité."
        )
        if {"duration", "credit_amount"}.issubset(corr.columns):
            duration_credit_corr = float(corr.loc["duration", "credit_amount"])
            interpretation_box(
                f"La plupart des corrélations numériques sont faibles : il n'y a pas de forte redondance linéaire généralisée. "
                f"La relation la plus notable métier est **duration ↔ credit_amount = {duration_credit_corr:.2f}** : "
                "les montants de crédit plus élevés tendent à être associés à des durées plus longues. "
                "La relation reste non causale et n'est pas assez extrême pour justifier de supprimer automatiquement l'une des deux variables."
            )


elif module == "1.1 · Conclusion EDA":
    st.header("1.1 — Conclusion EDA")
    st.markdown("### 📌 Ce que nous avons observé")
    st.write(
        "German Credit contient 1 000 clients, 20 variables explicatives et une cible binaire. "
        "Aucune valeur manquante n'est présente. La cible est répartie approximativement à 70 % de bons payeurs et 30 % de défauts."
    )
    st.markdown("### 🧠 Ce que cela signifie")
    st.write(
        "Les variables numériques et catégorielles apportent des informations complémentaires. "
        "Certaines modalités, notamment checking_status, présentent des taux de défaut nettement différents. "
        "La majorité des corrélations numériques restent faibles ; duration et credit_amount présentent toutefois une corrélation positive notable d'environ 0,62."
    )
    st.markdown("### ⚠️ Points à garder en tête")
    st.write(
        "Les valeurs extrêmes ne sont pas automatiquement des erreurs, et une association statistique n'implique pas une causalité. "
        "Aucune redondance linéaire majeure ne justifie à ce stade une suppression automatique de variables."
    )
    st.markdown("### ✅ Décision / transition")
    st.success(
        "Le dataset est exploitable pour la modélisation. L'étape suivante consiste à séparer train/test, encoder les variables catégorielles et mettre les variables numériques sur une échelle adaptée, sans fuite de données."
    )


elif module == "2 · Préparation":
    st.header("Module 2 — Préparation")
    module_intro(
        "Transformer les données brutes en une représentation exploitable par les algorithmes, tout en protégeant le jeu de test.",
        "Sélection des variables, séparation train/test, valeurs manquantes, encodage, normalisation et vérification des données préparées.",
        "La préparation doit être apprise uniquement sur le train : le test représente des clients jamais vus et ne doit pas influencer les transformations.",
    )

    config = st.session_state.prep_config

    (
        tab_select,
        tab_split,
        tab_missing,
        tab_encoding,
        tab_scaling,
        tab_ready,
    ) = st.tabs(
        [
            "1 · Sélection des variables",
            "2 · Train / Test",
            "3 · Valeurs manquantes",
            "4 · Encodage",
            "5 · Normalisation",
            "6 · Données préparées",
        ]
    )

    with tab_select:
        st.subheader("Sélection des variables explicatives")
        st.write(
            "On définit ici X, c'est-à-dire les informations utilisées pour prédire la cible default. "
            "La cible elle-même n'est jamais incluse dans les variables explicatives."
        )
        selected_features = st.multiselect(
            "Variables explicatives à conserver",
            X.columns.tolist(),
            default=config["selected_features"],
            key="prep_features",
        )
        st.caption(
            f"{len(selected_features)} variable(s) sélectionnée(s) sur {X.shape[1]}."
        )

    with tab_split:
        st.subheader("Séparation Train / Test")
        test_size = st.slider(
            "Part du jeu de test",
            min_value=0.10,
            max_value=0.40,
            value=float(config["test_size"]),
            step=0.05,
            key="prep_test_size",
        )
        st.info(
            "La séparation est **stratifiée** : elle conserve approximativement la proportion 70/30 de la cible dans le train et le test. "
            "Le modèle apprend sur le train ; le test reste réservé à l'évaluation finale."
        )

    with tab_missing:
        st.subheader("Traitement des valeurs manquantes")
        current_missing = int(missing_summary(df)["valeurs_manquantes"].sum())
        st.metric("Valeurs manquantes détectées", current_missing)

        col_num, col_cat = st.columns(2)
        with col_num:
            numeric_imputation = st.selectbox(
                "Imputation numérique",
                ["median", "mean"],
                index=0 if config["numeric_imputation"] == "median" else 1,
                key="prep_num_imputation",
            )
            st.caption(
                "median : robuste aux valeurs extrêmes ; mean : moyenne des valeurs disponibles."
            )
        with col_cat:
            categorical_imputation = st.selectbox(
                "Imputation catégorielle",
                ["most_frequent"],
                index=0,
                key="prep_cat_imputation",
            )
            st.caption(
                "most_frequent : remplace une catégorie manquante par la modalité la plus fréquente."
            )

        if current_missing == 0:
            interpretation_box(
                "Dans German Credit, aucune valeur n'est manquante : **aucune imputation n'est réellement effectuée**. "
                "Les imputers restent dans le pipeline pour rendre la préparation robuste si une future source contient des valeurs absentes."
            )

    with tab_encoding:
        st.subheader("Encodage des variables catégorielles")
        selected_categorical = [
            feature for feature in selected_features if feature in categorical_features
        ]
        st.write(
            f"{len(selected_categorical)} variable(s) catégorielle(s) sélectionnée(s). "
            "Elles sont transformées par **One-Hot Encoding** en colonnes binaires 0/1."
        )
        st.info(
            "Exemple : une variable à 4 modalités devient 4 colonnes indicatrices. "
            "C'est pourquoi le nombre de colonnes après préparation peut être supérieur aux 20 variables originales."
        )

    with tab_scaling:
        st.subheader("Normalisation / standardisation")
        scale_numeric = st.checkbox(
            "Standardiser les variables numériques",
            value=bool(config["scale_numeric"]),
            key="prep_scale",
        )
        st.caption(
            "StandardScaler transforme chaque variable numérique selon z = (x − moyenne) / écart-type. "
            "C'est particulièrement utile pour la régression logistique et kNN ; les arbres en ont moins besoin."
        )
        if scale_numeric:
            st.success(
                "Standardisation activée : les variables numériques seront ramenées sur des échelles comparables."
            )
        else:
            st.warning(
                "Standardisation désactivée : attention surtout aux modèles basés sur les distances comme kNN."
            )

    with tab_ready:
        st.subheader("Aperçu des données préparées")
        if not selected_features:
            st.warning("Sélectionnez au moins une variable explicative.")
        else:
            preview_prepared = get_prepared(
                tuple(selected_features),
                test_size,
                scale_numeric,
                numeric_imputation,
                categorical_imputation,
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Train", len(preview_prepared.X_train))
            c2.metric("Test", len(preview_prepared.X_test))
            c3.metric("Features brutes", len(selected_features))
            c4.metric(
                "Features après encodage",
                len(preview_prepared.feature_names),
            )

            st.caption(
                f"Affichage de **10 lignes d'exemple** parmi les {len(preview_prepared.X_train)} lignes du train complet."
            )
            st.dataframe(
                preview_prepared.X_train.head(10),
                use_container_width=True,
            )

            interpretation_box(
                f"La préparation transforme {len(selected_features)} variables brutes en "
                f"{len(preview_prepared.feature_names)} colonnes numériques utilisables par les modèles. "
                "Le préprocesseur est ajusté uniquement sur le train puis appliqué au test, ce qui protège l'évaluation contre la fuite de données."
            )

    if not selected_features:
        st.error("Sélectionnez au moins une variable explicative.")
    elif st.button("Appliquer et enregistrer la préparation", type="primary"):
        st.session_state.prep_config = {
            "selected_features": selected_features,
            "test_size": test_size,
            "scale_numeric": scale_numeric,
            "numeric_imputation": numeric_imputation,
            "categorical_imputation": categorical_imputation,
        }
        st.session_state.pop("model_results", None)
        st.session_state.pop("prepared_data", None)
        st.session_state.pop("threshold_recommendations", None)
        st.success(
            "Configuration enregistrée. Les anciens modèles ont été invalidés afin qu'ils soient réentraînés avec cette préparation."
        )


elif module == "2.1 · Conclusion Préparation":
    st.header("2.1 — Conclusion Préparation")
    current = st.session_state.prep_config
    prepared = get_prepared(
        tuple(current["selected_features"]),
        current["test_size"],
        current["scale_numeric"],
        current["numeric_imputation"],
        current["categorical_imputation"],
    )

    st.markdown("### 📌 Ce que nous avons appliqué")
    st.write(
        f"Les données ont été séparées de façon stratifiée en {len(prepared.X_train)} observations de train et "
        f"{len(prepared.X_test)} observations de test. Les variables catégorielles sont encodées en One-Hot et "
        + ("les variables numériques sont standardisées." if current["scale_numeric"] else "la standardisation numérique est désactivée.")
    )

    st.markdown("### 🧠 Ce que cela signifie")
    st.write(
        f"Les {len(current['selected_features'])} variables brutes deviennent {len(prepared.feature_names)} colonnes numériques. "
        "Cette représentation peut être fournie aux différents algorithmes sans perdre la séparation entre train et test."
    )

    st.markdown("### ⚠️ Points à garder en tête")
    st.write(
        "German Credit ne contient aucune valeur manquante : les étapes d'imputation n'ont donc remplacé aucune valeur dans cette exécution. "
        "Elles sont conservées uniquement pour la robustesse du pipeline. Le préprocesseur apprend ses statistiques et catégories sur le train uniquement."
    )

    st.markdown("### ✅ Décision / transition")
    st.success(
        "Les données sont prêtes pour la classification supervisée. On peut maintenant entraîner et comparer les cinq modèles du Module 3."
    )


elif module == "3 · Modélisation":
    st.header("Module 3 — Modélisation et comparaison")

    config = st.session_state.prep_config
    prepared = get_prepared(
        tuple(config["selected_features"]),
        config["test_size"],
        config["scale_numeric"],
        config["numeric_imputation"],
        config["categorical_imputation"],
    )

    model_names = [
        "Régression logistique",
        "Arbre de décision",
        "kNN",
        "Random Forest",
        "XGBoost",
    ]

    st.markdown(
        """
### Comprendre les modèles avant de les entraîner

Chaque algorithme est présenté selon la même logique : **principe → apprentissage → hyperparamètres → avantages / limites → rôle dans notre projet**.
"""
    )

    with st.expander("Régression logistique — comprendre le modèle", expanded=True):
        st.markdown(
            """
**Principe.** Le modèle combine linéairement les variables puis transforme le score obtenu en **probabilité de défaut** grâce à une fonction sigmoïde.

**Comment il apprend.** Il ajuste automatiquement ses coefficients afin de réduire la log-loss sur les données d'entraînement.

**Hyperparamètres dans notre application.** C contrôle la force de régularisation, penalty choisit L1 ou L2, et class_weight peut donner davantage de poids à une classe.

**Avantages / limites.** Modèle simple, stable et particulièrement intéressant en scoring car ses coefficients sont interprétables. En revanche, sa structure reste essentiellement linéaire et peut manquer certaines relations complexes.

**Dans notre projet.** C'est notre modèle de référence : il fournit directement une probabilité de défaut et constitue un bon compromis entre performance et explicabilité.
"""
        )

    with st.expander("Arbre de décision — comprendre le modèle"):
        st.markdown(
            """
**Principe.** L'arbre découpe progressivement les clients avec des règles simples afin d'obtenir des groupes aussi homogènes que possible vis-à-vis de default.

**Comment il apprend.** À chaque nœud, il cherche une coupure qui sépare au mieux les classes, puis répète l'opération jusqu'aux feuilles.

**Hyperparamètres dans notre application.** max_depth, min_samples_split, criterion et class_weight.

**Avantages / limites.** Très lisible et facile à expliquer, mais un arbre trop profond peut surapprendre et devenir instable.

**Dans notre projet.** Il sert de modèle non linéaire simple et interprétable, puis de base conceptuelle pour comprendre Random Forest et XGBoost.
"""
        )

    with st.expander("kNN — comprendre le modèle"):
        st.markdown(
            """
**Principe.** Pour un nouveau client, kNN cherche les k clients historiques les plus proches et attribue la classe majoritaire.

**Comment il apprend.** Il n'estime pas de coefficients : il mémorise essentiellement les observations d'entraînement et effectue le calcul au moment de la prédiction.

**Hyperparamètres dans notre application.** n_neighbors (k), weights et p pour la distance.

**Avantages / limites.** Simple et pédagogique, mais sensible au choix de k et à l'échelle des variables ; la prédiction devient aussi plus coûteuse lorsque le dataset grandit.

**Dans notre projet.** Il permet de comparer une approche fondée sur la proximité des profils clients avec des modèles paramétriques ou à base d'arbres. La standardisation est particulièrement importante pour lui.
"""
        )

    with st.expander("Random Forest — comprendre le modèle"):
        st.markdown(
            """
**Principe.** Random Forest entraîne beaucoup d'arbres sur des échantillons et des sous-ensembles de variables différents, puis agrège leurs prédictions.

**Comment il apprend.** Chaque arbre est construit séparément ; la randomisation réduit leur corrélation et le vote final stabilise la prédiction.

**Hyperparamètres dans notre application.** n_estimators, max_depth, min_samples_split et class_weight.

**Avantages / limites.** Robuste au surapprentissage d'un arbre isolé et efficace sur les données tabulaires, mais moins directement interprétable qu'un arbre unique ou une régression logistique.

**Dans notre projet.** Il représente une méthode ensembliste de type bagging et permet de vérifier si l'agrégation de nombreux arbres améliore le scoring.
"""
        )

    with st.expander("XGBoost — comprendre le modèle"):
        st.markdown(
            """
**Principe.** XGBoost construit des arbres de manière séquentielle : chaque nouvel arbre cherche à corriger les erreurs laissées par les précédents.

**Comment il apprend.** Les nouveaux arbres sont ajoutés progressivement selon un objectif d'optimisation ; le learning_rate contrôle l'importance de chaque correction.

**Hyperparamètres dans notre application.** n_estimators, max_depth, learning_rate, subsample et colsample_bytree.

**Avantages / limites.** Très performant sur les données tabulaires et capable de modéliser des relations complexes, mais plus coûteux à régler et moins transparent qu'un modèle linéaire.

**Dans notre projet.** Il représente notre modèle de boosting avancé et sert à mesurer le compromis entre performance prédictive et interprétabilité.
"""
        )

    selected_models = st.multiselect(
        "Modèles à entraîner",
        model_names,
        default=model_names,
    )

    tune = st.checkbox(
        "Optimiser automatiquement les hyperparamètres",
        value=False,
        help="Utilise RandomizedSearchCV avec validation croisée et AUC-ROC.",
    )

    if tune:
        n_iter = st.slider("Nombre de configurations testées par modèle", 3, 15, 8)
        cv_folds = st.slider("Nombre de folds", 3, 5, 5)
    else:
        n_iter = 8
        cv_folds = 5

    model_params = {}

    for name in selected_models:
        with st.expander(f"Hyperparamètres — {name}"):
            if name == "Régression logistique":
                model_params[name] = {
                    "C": st.number_input(
                        "C",
                        min_value=0.001,
                        max_value=100.0,
                        value=1.0,
                        step=0.1,
                        key="lr_c",
                    ),
                    "penalty": st.selectbox(
                        "penalty",
                        ["l2", "l1"],
                        key="lr_penalty",
                    ),
                    "class_weight": st.selectbox(
                        "class_weight",
                        [None, "balanced"],
                        key="lr_weight",
                    ),
                }

            elif name == "Arbre de décision":
                depth = st.slider(
                    "max_depth (0 = illimité)",
                    0,
                    20,
                    5,
                    key="dt_depth",
                )
                model_params[name] = {
                    "max_depth": None if depth == 0 else depth,
                    "min_samples_split": st.slider(
                        "min_samples_split",
                        2,
                        20,
                        2,
                        key="dt_split",
                    ),
                    "criterion": st.selectbox(
                        "criterion",
                        ["gini", "entropy"],
                        key="dt_criterion",
                    ),
                    "class_weight": st.selectbox(
                        "class_weight",
                        [None, "balanced"],
                        key="dt_weight",
                    ),
                }

            elif name == "kNN":
                model_params[name] = {
                    "n_neighbors": st.slider(
                        "n_neighbors",
                        3,
                        31,
                        7,
                        step=2,
                        key="knn_k",
                    ),
                    "weights": st.selectbox(
                        "weights",
                        ["uniform", "distance"],
                        key="knn_weights",
                    ),
                    "p": st.selectbox(
                        "distance p",
                        [1, 2],
                        index=1,
                        key="knn_p",
                    ),
                }

            elif name == "Random Forest":
                depth = st.slider(
                    "max_depth (0 = illimité)",
                    0,
                    20,
                    0,
                    key="rf_depth",
                )
                model_params[name] = {
                    "n_estimators": st.slider(
                        "n_estimators",
                        100,
                        700,
                        300,
                        step=50,
                        key="rf_n",
                    ),
                    "max_depth": None if depth == 0 else depth,
                    "min_samples_split": st.slider(
                        "min_samples_split",
                        2,
                        20,
                        2,
                        key="rf_split",
                    ),
                    "class_weight": st.selectbox(
                        "class_weight",
                        [None, "balanced"],
                        key="rf_weight",
                    ),
                }

            elif name == "XGBoost":
                model_params[name] = {
                    "n_estimators": st.slider(
                        "n_estimators",
                        100,
                        700,
                        250,
                        step=50,
                        key="xgb_n",
                    ),
                    "max_depth": st.slider(
                        "max_depth",
                        2,
                        10,
                        3,
                        key="xgb_depth",
                    ),
                    "learning_rate": st.select_slider(
                        "learning_rate",
                        options=[0.01, 0.03, 0.05, 0.1, 0.2],
                        value=0.05,
                        key="xgb_lr",
                    ),
                    "subsample": st.slider(
                        "subsample",
                        0.5,
                        1.0,
                        0.9,
                        0.05,
                        key="xgb_sub",
                    ),
                    "colsample_bytree": st.slider(
                        "colsample_bytree",
                        0.5,
                        1.0,
                        0.9,
                        0.05,
                        key="xgb_col",
                    ),
                }

    if not selected_models:
        st.warning("Sélectionnez au moins un modèle.")
    elif st.button("Entraîner les modèles", type="primary"):
        with st.spinner("Entraînement et comparaison des modèles..."):
            results = train_models(
                selected_models,
                prepared,
                model_params=model_params,
                tune=tune,
                cv_folds=cv_folds,
                n_iter=n_iter,
            )

        st.session_state.model_results = results
        st.session_state.prepared_data = prepared
        st.success("Entraînement terminé.")

    if "model_results" in st.session_state:
        results = st.session_state.model_results
        table = comparison_table(results)
        st.subheader("Comparaison au seuil 0,50")
        st.dataframe(table, use_container_width=True, hide_index=True)

        for name, result in results.items():
            with st.expander(f"Paramètres retenus — {name}"):
                st.json(result.best_params)


elif module == "4 · Évaluation":
    st.header("Module 4 — Évaluation")

    if "model_results" not in st.session_state:
        st.warning("Entraînez d'abord au moins un modèle dans le Module 3.")
    else:
        results = st.session_state.model_results
        prepared = st.session_state.prepared_data

        st.subheader("Tableau comparatif")
        st.dataframe(
            comparison_table(results),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Courbes ROC superposées")
        roc_frames = []
        for name, result in results.items():
            frame = roc_points(prepared.y_test, result.y_prob)
            frame["modèle"] = name
            roc_frames.append(frame)

        roc_df = pd.concat(roc_frames, ignore_index=True)
        roc_fig = px.line(
            roc_df,
            x="fpr",
            y="tpr",
            color="modèle",
            title="Courbes ROC",
        )
        roc_fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=1,
            y1=1,
            line={"dash": "dash"},
        )
        roc_fig.update_layout(
            xaxis_title="Taux de faux positifs (FPR)",
            yaxis_title="Taux de vrais positifs (TPR / rappel)",
        )
        st.plotly_chart(roc_fig, use_container_width=True)

        st.subheader("Courbes précision-rappel")
        pr_frames = []
        for name, result in results.items():
            frame = precision_recall_points(prepared.y_test, result.y_prob)
            frame["modèle"] = name
            pr_frames.append(frame)

        pr_df = pd.concat(pr_frames, ignore_index=True)
        pr_fig = px.line(
            pr_df,
            x="recall",
            y="precision",
            color="modèle",
            title="Courbes précision-rappel",
        )
        pr_fig.update_layout(
            xaxis_title="Rappel",
            yaxis_title="Précision",
        )
        st.plotly_chart(pr_fig, use_container_width=True)

        st.subheader("Analyse du seuil de décision")
        selected_model = st.selectbox(
            "Modèle",
            list(results.keys()),
            key="eval_model",
        )
        threshold = st.slider(
            "Seuil de décision",
            0.05,
            0.95,
            0.50,
            0.01,
            key="eval_threshold",
        )

        result = results[selected_model]
        metrics = metrics_at_threshold(
            prepared.y_test,
            result.y_prob,
            threshold,
        )

        metric_cols = st.columns(7)
        for column, (name, value) in zip(metric_cols, metrics.items()):
            column.metric(name, f"{value:.3f}")

        matrix = confusion_at_threshold(
            prepared.y_test,
            result.y_prob,
            threshold,
        )

        confusion_fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=["Prédit bon (0)", "Prédit défaut (1)"],
                y=["Réel bon (0)", "Réel défaut (1)"],
                text=matrix,
                texttemplate="%{text}",
                hovertemplate="%{y}<br>%{x}<br>n=%{z}<extra></extra>",
            )
        )
        confusion_fig.update_layout(
            title=f"Matrice de confusion — {selected_model} — seuil {threshold:.2f}",
            xaxis_title="Prédiction",
            yaxis_title="Classe réelle",
            height=500,
        )
        st.plotly_chart(confusion_fig, use_container_width=True)

        st.info(
            "Baisser le seuil détecte généralement davantage de mauvais payeurs "
            "(rappel plus élevé), mais augmente aussi les faux positifs. "
            "Le seuil est donc une décision métier."
        )


elif module == "5 · Explainable AI":
    st.header("Module 5 — Explainable AI")

    if "model_results" not in st.session_state:
        st.warning("Entraînez d'abord au moins un modèle dans le Module 3.")
    else:
        results = st.session_state.model_results
        prepared = st.session_state.prepared_data

        selected_model = st.selectbox(
            "Modèle à expliquer",
            list(results.keys()),
            key="xai_model",
        )
        result = results[selected_model]
        model = result.estimator

        st.subheader("Explication globale")

        native = native_importance(model, prepared.feature_names)
        if native is None:
            st.info(
                "Ce modèle n'expose pas directement une importance native des variables."
            )
        else:
            top_native = native.head(15)
            native_fig = px.bar(
                top_native.sort_values("importance"),
                x="importance",
                y="feature",
                orientation="h",
                title="Importance native du modèle",
            )
            st.plotly_chart(native_fig, use_container_width=True)

        with st.spinner("Calcul de l'importance par permutation..."):
            perm = permutation_importance_df(
                model,
                prepared.X_test,
                prepared.y_test,
                prepared.feature_names,
                n_repeats=5,
            )

        perm_top = perm.head(15)
        perm_fig = px.bar(
            perm_top.sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            error_x="std",
            title="Importance par permutation",
        )
        st.plotly_chart(perm_fig, use_container_width=True)

        if st.button("Calculer SHAP global", type="primary"):
            import shap

            sample = prepared.X_test.sample(
                n=min(25, len(prepared.X_test)),
                random_state=42,
            )

            with st.spinner("Calcul SHAP en cours..."):
                _, shap_values = compute_shap_values(
                    model,
                    prepared.X_train,
                    sample,
                    prepared.feature_names,
                )

            st.session_state.global_shap = shap_values
            st.session_state.global_shap_model = selected_model

        if (
            "global_shap" in st.session_state
            and st.session_state.get("global_shap_model") == selected_model
        ):
            import shap

            shap_values = st.session_state.global_shap

            st.markdown("**SHAP beeswarm — importance et sens des contributions**")
            plt.figure()
            shap.plots.beeswarm(
                shap_values,
                max_display=15,
                show=False,
            )
            st.pyplot(plt.gcf(), clear_figure=True)

            mean_abs = abs(shap_values.values).mean(axis=0)
            top_index = int(mean_abs.argmax())
            top_feature = prepared.feature_names[top_index]

            st.markdown(f"**Dépendance SHAP — {top_feature}**")
            plt.figure()
            shap.plots.scatter(
                shap_values[:, top_index],
                show=False,
            )
            st.pyplot(plt.gcf(), clear_figure=True)

        st.divider()
        st.subheader("Explication locale d'un individu")

        row_position = st.number_input(
            "Position de l'individu dans le jeu de test",
            min_value=0,
            max_value=len(prepared.X_test) - 1,
            value=0,
            step=1,
        )

        local_row = prepared.X_test.iloc[[int(row_position)]]
        probability = float(model.predict_proba(local_row)[0, 1])

        threshold = st.slider(
            "Seuil pour la décision locale",
            0.05,
            0.95,
            0.50,
            0.01,
            key="xai_threshold",
        )

        decision = 1 if probability >= threshold else 0

        c1, c2 = st.columns(2)
        c1.metric("Probabilité de défaut", f"{probability:.1%}")
        c2.metric(
            "Décision",
            "Défaut / risque élevé" if decision == 1 else "Bon payeur",
        )

        if st.button("Générer SHAP + LIME pour cet individu"):
            import shap

            with st.spinner("Génération des explications locales..."):
                _, local_shap = compute_shap_values(
                    model,
                    prepared.X_train,
                    local_row,
                    prepared.feature_names,
                )

                lime_df = lime_explanation(
                    model,
                    prepared.X_train,
                    local_row.iloc[0],
                    prepared.feature_names,
                    num_features=10,
                )

            st.markdown("**SHAP waterfall**")
            plt.figure()
            shap.plots.waterfall(
                local_shap[0],
                max_display=12,
                show=False,
            )
            st.pyplot(plt.gcf(), clear_figure=True)

            st.markdown("**LIME — même individu**")
            st.dataframe(lime_df, use_container_width=True, hide_index=True)

            summary = plain_language_summary(
                prepared.feature_names,
                local_shap[0],
                top_n=3,
            )
            st.success(summary)

        st.warning(
            "Équité : certaines variables peuvent être sensibles, notamment l'âge, "
            "le sexe / statut personnel ou le statut de travailleur étranger. "
            "Une bonne performance ne suffit pas : leur usage doit être discuté "
            "dans l'analyse critique."
        )


elif module == "6 · FastAPI":
    st.header("Module 6 — API d'interprétation")

    st.code(
        "uvicorn api.main:app --reload",
        language="powershell",
    )

    st.markdown(
        """
L'API fournit :

- GET /health
- POST /predict
- POST /explain
- documentation Swagger : /docs
"""
    )

    api_url = st.text_input(
        "URL de l'API",
        value="http://127.0.0.1:8000",
    ).rstrip("/")

    if st.button("Tester /health"):
        try:
            response = requests.get(f"{api_url}/health", timeout=15)
            response.raise_for_status()
            st.success(response.json())
        except Exception as exc:
            st.error(f"API inaccessible : {exc}")

    st.subheader("Tester un client réel du dataset")

    raw_position = st.number_input(
        "Ligne du dataset",
        min_value=0,
        max_value=len(X) - 1,
        value=0,
        step=1,
        key="api_row",
    )

    raw_row = X.iloc[int(raw_position)]
    st.dataframe(
        raw_row.rename("valeur").to_frame(),
        use_container_width=True,
    )

    api_threshold = st.slider(
        "Seuil API",
        0.05,
        0.95,
        0.50,
        0.01,
        key="api_threshold",
    )

    features_payload = json.loads(raw_row.to_json())
    payload = {
        "features": features_payload,
        "threshold": api_threshold,
    }

    col_predict, col_explain = st.columns(2)

    with col_predict:
        if st.button("Appeler /predict", use_container_width=True):
            try:
                response = requests.post(
                    f"{api_url}/predict",
                    json=payload,
                    timeout=60,
                )
                response.raise_for_status()
                st.json(response.json())
            except Exception as exc:
                st.error(f"Erreur API : {exc}")

    with col_explain:
        if st.button("Appeler /explain", use_container_width=True):
            try:
                response = requests.post(
                    f"{api_url}/explain",
                    json=payload,
                    timeout=180,
                )
                response.raise_for_status()
                result_api = response.json()
                st.json(result_api)

                if "top_contributions" in result_api:
                    contributions = pd.DataFrame(
                        result_api["top_contributions"]
                    )
                    st.dataframe(
                        contributions,
                        use_container_width=True,
                        hide_index=True,
                    )
            except Exception as exc:
                st.error(f"Erreur API : {exc}")
