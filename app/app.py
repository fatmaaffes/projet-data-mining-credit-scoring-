"""Streamlit application - Module 1: Données & EDA."""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
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


st.set_page_config(
    page_title="Credit Scoring XAI",
    page_icon="💳",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def get_data():
    """Cache UCI loading so Streamlit reruns do not redownload every time."""

    return load_german_credit()


st.title("💳 Credit Scoring explicable")
st.caption("German Credit — UCI Statlog · Module 1 : Données & EDA")

with st.spinner("Chargement automatique du jeu German Credit depuis UCI..."):
    data = get_data()

X = data.X
y = data.y
df = data.df
numeric_features, categorical_features = split_feature_types(X)

st.success(
    f"Dataset chargé : {len(df)} clients · {X.shape[1]} variables explicatives · "
    "cible binaire default"
)

tab_overview, tab_stats, tab_numeric, tab_categorical, tab_target, tab_corr = st.tabs(
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
    else:
        st.warning(f"{total_missing} valeurs manquantes détectées.")

with tab_stats:
    st.subheader("Statistiques descriptives — variables numériques")
    st.dataframe(
        numeric_describe(X),
        use_container_width=True,
    )

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

with tab_numeric:
    st.subheader("Analyse univariée d'une variable numérique")

    feature = st.selectbox(
        "Variable numérique",
        numeric_features,
        index=numeric_features.index("duration") if "duration" in numeric_features else 0,
        key="numeric_feature",
    )

    nbins = st.slider("Nombre de classes de l'histogramme", 5, 50, 20)

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
    )
    st.plotly_chart(box, use_container_width=True)

    st.dataframe(
        df.groupby("default")[feature]
        .describe()
        .rename(index=TARGET_LABELS),
        use_container_width=True,
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
    )
    st.plotly_chart(heatmap, use_container_width=True)

    st.info(
        "Une corrélation proche de +1 indique une forte relation linéaire positive, "
        "proche de -1 une forte relation linéaire négative, et proche de 0 peu de "
        "relation linéaire. Corrélation ne signifie pas causalité."
    )
