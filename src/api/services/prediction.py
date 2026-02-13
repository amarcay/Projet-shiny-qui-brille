"""Service de prediction : logique metier."""

import numpy as np
from typing import Optional

from core.config import DEFAULT_VERSION
from core.registry import registry
from schemas.candle import CandleInput
from schemas.responses import PredictionResponse


# Features de base (17) communes a tous les modeles
_BASE_FEATURES = [
    "return_1", "return_4", "ema_diff", "rsi_14",
    "rolling_std_20", "range_15m", "body", "upper_wick",
    "lower_wick", "distance_to_ema200", "slope_ema50",
    "atr_14", "rolling_std_100", "volatility_ratio",
    "adx_14", "macd", "macd_signal",
]

# Features supplementaires pour v2/v3 (DQN)
_EXTRA_FEATURES = ["ema_20", "ema_50"]

# Mapping action DQN -> signal
_DQN_ACTION_MAP = {0: "HOLD", 1: "BUY", 2: "SELL"}


def ensure_model(version: Optional[str] = None):
    """Charge le modele si necessaire (lazy loading)."""
    target = version or DEFAULT_VERSION
    if registry.model is None or (version and version != registry.version):
        registry.load(target)


def _get_feature_names() -> list[str]:
    """Retourne la liste des features attendues par le modele charge."""
    if registry.feature_cols:
        return registry.feature_cols
    return _BASE_FEATURES


def candle_to_features(candle: CandleInput) -> np.ndarray:
    """Convertit une bougie en array de features."""
    feature_names = _get_feature_names()
    values = []
    for f in feature_names:
        val = getattr(candle, f, None)
        values.append(val if val is not None else 0.0)
    return np.array([values])


def candles_to_features(candles: list[CandleInput]) -> np.ndarray:
    """Convertit un batch de bougies en array de features."""
    feature_names = _get_feature_names()
    rows = []
    for c in candles:
        values = []
        for f in feature_names:
            val = getattr(c, f, None)
            values.append(val if val is not None else 0.0)
        rows.append(values)
    return np.array(rows)


def make_prediction(features: np.ndarray) -> PredictionResponse:
    """Effectue une prediction sur un vecteur de features."""
    pred = int(registry.predict(features)[0])

    confidence = None
    proba = registry.predict_proba(features)
    if proba is not None:
        confidence = float(proba[0][pred])

    # Signal selon le type de modele
    if registry.model_type == "stable-baselines3":
        signal = _DQN_ACTION_MAP.get(pred, "HOLD")
    else:
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

        if registry.model_type == "stable-baselines3":
            signal = _DQN_ACTION_MAP.get(pred_int, "HOLD")
        else:
            signal = "BUY" if pred_int == 1 else "SELL"

        results.append(PredictionResponse(
            signal=signal,
            prediction=pred_int,
            confidence=confidence,
            model_version=registry.version,
            model_type=registry.model_type,
        ))

    return results
