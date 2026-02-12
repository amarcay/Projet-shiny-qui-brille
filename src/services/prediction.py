"""Service de prediction : logique metier."""

import numpy as np
from typing import Optional

from core.config import DEFAULT_VERSION
from core.registry import registry
from schemas.candle import CandleInput
from schemas.responses import PredictionResponse


def ensure_model(version: Optional[str] = None):
    """Charge le modele si necessaire (lazy loading)."""
    target = version or DEFAULT_VERSION
    if registry.model is None or (version and version != registry.version):
        registry.load(target)


def candle_to_features(candle: CandleInput) -> np.ndarray:
    """Convertit une bougie en array de features."""
    return np.array([[
        candle.return_1, candle.return_4, candle.ema_diff, candle.rsi_14,
        candle.rolling_std_20, candle.range_15m, candle.body, candle.upper_wick,
        candle.lower_wick, candle.distance_to_ema200, candle.slope_ema50,
        candle.atr_14, candle.rolling_std_100, candle.volatility_ratio,
        candle.adx_14, candle.macd, candle.macd_signal,
    ]])


def candles_to_features(candles: list[CandleInput]) -> np.ndarray:
    """Convertit un batch de bougies en array de features."""
    return np.array([
        [c.return_1, c.return_4, c.ema_diff, c.rsi_14,
         c.rolling_std_20, c.range_15m, c.body, c.upper_wick,
         c.lower_wick, c.distance_to_ema200, c.slope_ema50,
         c.atr_14, c.rolling_std_100, c.volatility_ratio,
         c.adx_14, c.macd, c.macd_signal]
        for c in candles
    ])


def make_prediction(features: np.ndarray) -> PredictionResponse:
    """Effectue une prediction sur un vecteur de features."""
    pred = int(registry.predict(features)[0])

    confidence = None
    proba = registry.predict_proba(features)
    if proba is not None:
        confidence = float(proba[0][pred])

    signal = "BUY" if pred == 1 else "SELL"

    return PredictionResponse(
        signal=signal,
        prediction=pred,
        confidence=confidence,
        model_version=registry.version,
        model_type=registry.model_type,
    )


def make_batch_prediction(features: np.ndarray) -> list[PredictionResponse]:
    """Effectue des predictions sur un batch de features."""
    preds = registry.predict(features)
    probas = registry.predict_proba(features)

    results = []
    for i, pred in enumerate(preds):
        pred_int = int(pred)
        confidence = float(probas[i][pred_int]) if probas is not None else None
        signal = "BUY" if pred_int == 1 else "SELL"
        results.append(PredictionResponse(
            signal=signal,
            prediction=pred_int,
            confidence=confidence,
            model_version=registry.version,
            model_type=registry.model_type,
        ))

    return results
