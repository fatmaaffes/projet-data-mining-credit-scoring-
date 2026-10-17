"""Module 6 - FastAPI service for prediction and local interpretation."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_german_credit
from src.explainability import compute_shap_values, plain_language_summary
from src.models import build_model
from src.preprocessing import prepare_data


app = FastAPI(
    title="Credit Scoring XAI API",
    version="1.0.0",
    description=(
        "API de prédiction et d'interprétation locale pour le projet German Credit."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    features: dict[str, Any]
    threshold: float = Field(default=0.50, ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    model: str
    probability_default: float
    threshold: float
    predicted_class: int
    label: str


class ExplanationResponse(PredictionResponse):
    top_contributions: list[dict[str, Any]]
    summary: str


@lru_cache(maxsize=1)
def get_bundle():
    """Train a lightweight baseline once and reuse it for API requests."""

    data = load_german_credit()
    prepared = prepare_data(
        data.X,
        data.y,
        test_size=0.20,
        random_state=42,
        scale_numeric=True,
    )

    model = build_model(
        "Régression logistique",
        C=1.0,
        penalty="l2",
        class_weight=None,
    )
    model.fit(prepared.X_train, prepared.y_train)

    return data, prepared, model


def _transform_features(features: dict[str, Any]):
    data, prepared, model = get_bundle()

    missing = [column for column in data.X.columns if column not in features]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Variables manquantes: {missing}",
        )

    ordered = {column: features[column] for column in data.X.columns}
    raw_row = pd.DataFrame([ordered])
    transformed_array = prepared.preprocessor.transform(raw_row)

    transformed = pd.DataFrame(
        transformed_array,
        columns=prepared.feature_names,
    )

    return prepared, model, transformed


def _prediction_payload(model, transformed, threshold: float):
    probability = float(model.predict_proba(transformed)[0, 1])
    predicted_class = int(probability >= threshold)

    return {
        "model": "Régression logistique",
        "probability_default": probability,
        "threshold": threshold,
        "predicted_class": predicted_class,
        "label": "Mauvais payeur / défaut" if predicted_class == 1 else "Bon payeur",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Credit Scoring XAI API",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    prepared, model, transformed = _transform_features(request.features)
    return _prediction_payload(model, transformed, request.threshold)


@app.post("/explain", response_model=ExplanationResponse)
def explain(request: PredictionRequest):
    prepared, model, transformed = _transform_features(request.features)

    _, shap_values = compute_shap_values(
        model,
        prepared.X_train,
        transformed,
        prepared.feature_names,
    )

    local = shap_values[0]
    values = local.values

    order = sorted(
        range(len(values)),
        key=lambda index: abs(float(values[index])),
        reverse=True,
    )[:10]

    top = [
        {
            "feature": prepared.feature_names[index],
            "contribution": float(values[index]),
            "direction": (
                "augmente le risque"
                if float(values[index]) > 0
                else "réduit le risque"
            ),
        }
        for index in order
    ]

    payload = _prediction_payload(model, transformed, request.threshold)
    payload["top_contributions"] = top
    payload["summary"] = plain_language_summary(
        prepared.feature_names,
        local,
        top_n=3,
    )
    return payload
