"""Router : endpoints de prediction."""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query

from schemas.candle import CandleInput, BatchInput
from schemas.responses import PredictionResponse, BatchPredictionResponse
from services.prediction import (
    ensure_model,
    candle_to_features,
    candles_to_features,
    make_prediction,
    make_batch_prediction,
)

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("", response_model=PredictionResponse)
def predict(candle: CandleInput, model_version: Optional[str] = Query(None)):
    """Prediction BUY/SELL sur une bougie M15."""
    try:
        ensure_model(model_version)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Version '{model_version}' introuvable"
        )

    features = candle_to_features(candle)
    return make_prediction(features)


@router.post("/batch", response_model=BatchPredictionResponse)
def predict_batch(batch: BatchInput, model_version: Optional[str] = Query(None)):
    """Predictions sur un batch de bougies M15 (max 100)."""
    try:
        ensure_model(model_version)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Version '{model_version}' introuvable"
        )

    features = candles_to_features(batch.candles)
    predictions = make_batch_prediction(features)

    return BatchPredictionResponse(predictions=predictions, count=len(predictions))
