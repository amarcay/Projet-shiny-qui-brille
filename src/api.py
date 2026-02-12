"""
Phase 10 – API de prediction GBP/USD M15
FastAPI exposant le meilleur modele avec versioning.

Endpoints :
  GET  /health              → statut de l'API
  GET  /model/info          → infos sur le modele charge (version, type, features)
  POST /predict             → prediction BUY/SELL/HOLD a partir de features M15
  POST /predict/batch       → predictions sur un batch de bougies

Regles :
  - L'API expose uniquement le meilleur modele
  - L'utilisateur ne peut pas relancer l'entrainement
  - Versioning modele obligatoire
  - L'API charge automatiquement la version validee
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_VERSION = os.environ.get("MODEL_VERSION", "v1")


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
class ModelRegistry:
    """Gere le chargement et le versioning des modeles."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.version = None
        self.model_type = None
        self.loaded_at = None

    def list_versions(self) -> list[str]:
        """Liste les versions disponibles."""
        versions = []
        if self.model_dir.exists():
            for d in sorted(self.model_dir.iterdir()):
                if d.is_dir() and (d / "gradient_boosting.joblib").exists():
                    versions.append(d.name)
        return versions

    def load(self, version: str = DEFAULT_VERSION):
        """Charge un modele par version."""
        version_dir = self.model_dir / version

        if not version_dir.exists():
            raise FileNotFoundError(f"Version '{version}' introuvable dans {self.model_dir}")

        # Charger le modele ML (Gradient Boosting)
        model_path = version_dir / "gradient_boosting.joblib"
        scaler_path = version_dir / "scaler.joblib"
        features_path = version_dir / "feature_cols.joblib"

        if not model_path.exists():
            raise FileNotFoundError(f"Modele introuvable: {model_path}")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        self.feature_cols = joblib.load(features_path) if features_path.exists() else None
        self.version = version
        self.model_type = type(self.model).__name__
        self.loaded_at = datetime.now().isoformat()

        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Prediction brute (0 ou 1)."""
        if self.model is None:
            raise RuntimeError("Aucun modele charge")
        return self.model.predict(features)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Probabilites de prediction."""
        if self.model is None:
            raise RuntimeError("Aucun modele charge")
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(features)
        return None


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class CandleInput(BaseModel):
    """Une bougie M15 avec ses features techniques."""
    return_1: float = Field(..., description="Rendement sur 1 periode")
    return_4: float = Field(..., description="Rendement sur 4 periodes")
    ema_diff: float = Field(..., description="EMA20 - EMA50")
    rsi_14: float = Field(..., description="RSI 14 periodes")
    rolling_std_20: float = Field(..., description="Volatilite court terme (std 20)")
    range_15m: float = Field(..., description="High - Low de la bougie")
    body: float = Field(..., description="Taille du corps |close - open|")
    upper_wick: float = Field(..., description="Meche haute")
    lower_wick: float = Field(..., description="Meche basse")
    distance_to_ema200: float = Field(..., description="Distance relative a EMA200")
    slope_ema50: float = Field(..., description="Pente de l'EMA50")
    atr_14: float = Field(..., description="ATR 14 periodes")
    rolling_std_100: float = Field(..., description="Volatilite long terme (std 100)")
    volatility_ratio: float = Field(..., description="Ratio volatilite court/long")
    adx_14: float = Field(..., description="ADX 14 periodes")
    macd: float = Field(..., description="MACD")
    macd_signal: float = Field(..., description="Signal MACD")


class BatchInput(BaseModel):
    """Batch de bougies M15."""
    candles: list[CandleInput] = Field(..., min_length=1, max_length=100)


class PredictionResponse(BaseModel):
    """Reponse de prediction."""
    signal: str = Field(..., description="BUY, SELL ou HOLD")
    prediction: int = Field(..., description="1=hausse, 0=baisse")
    confidence: Optional[float] = Field(None, description="Probabilite de la prediction")
    model_version: str
    model_type: str


class BatchPredictionResponse(BaseModel):
    """Reponse de prediction batch."""
    predictions: list[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str]
    timestamp: str


class ModelInfoResponse(BaseModel):
    version: str
    model_type: str
    features: list[str]
    available_versions: list[str]
    loaded_at: str


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="GBP/USD M15 Trading API",
    description="API de prediction de trading GBP/USD sur bougies M15. "
                "Expose le meilleur modele avec versioning.",
    version="1.0.0",
)

registry = ModelRegistry(MODEL_DIR)

# Chargement au demarrage
try:
    registry.load(DEFAULT_VERSION)
    print(f"Modele charge: {registry.model_type} (version {registry.version})")
except FileNotFoundError as e:
    print(f"[WARN] Impossible de charger le modele: {e}")


def _ensure_model(version: Optional[str] = None):
    """Charge le modele si necessaire (lazy loading)."""
    target = version or DEFAULT_VERSION
    if registry.model is None or (version and version != registry.version):
        registry.load(target)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health():
    """Statut de l'API."""
    return HealthResponse(
        status="ok" if registry.model is not None else "no_model",
        model_loaded=registry.model is not None,
        model_version=registry.version,
        timestamp=datetime.now().isoformat(),
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    """Informations sur le modele charge."""
    if registry.model is None:
        raise HTTPException(status_code=503, detail="Aucun modele charge")

    return ModelInfoResponse(
        version=registry.version,
        model_type=registry.model_type,
        features=registry.feature_cols or [],
        available_versions=registry.list_versions(),
        loaded_at=registry.loaded_at,
    )


@app.post("/model/load")
def load_model(version: str = Query(..., description="Version du modele a charger (ex: v1, v2)")):
    """Charge une version specifique du modele."""
    try:
        registry.load(version)
        return {"message": f"Modele {version} charge", "model_type": registry.model_type}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
def predict(candle: CandleInput, model_version: Optional[str] = Query(None)):
    """Prediction sur une bougie M15."""
    try:
        _ensure_model(model_version)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Version '{model_version}' introuvable")

    features = np.array([[
        candle.return_1, candle.return_4, candle.ema_diff, candle.rsi_14,
        candle.rolling_std_20, candle.range_15m, candle.body, candle.upper_wick,
        candle.lower_wick, candle.distance_to_ema200, candle.slope_ema50,
        candle.atr_14, candle.rolling_std_100, candle.volatility_ratio,
        candle.adx_14, candle.macd, candle.macd_signal,
    ]])

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


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(batch: BatchInput, model_version: Optional[str] = Query(None)):
    """Predictions sur un batch de bougies M15."""
    try:
        _ensure_model(model_version)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Version '{model_version}' introuvable")

    features = np.array([
        [c.return_1, c.return_4, c.ema_diff, c.rsi_14,
         c.rolling_std_20, c.range_15m, c.body, c.upper_wick,
         c.lower_wick, c.distance_to_ema200, c.slope_ema50,
         c.atr_14, c.rolling_std_100, c.volatility_ratio,
         c.adx_14, c.macd, c.macd_signal]
        for c in batch.candles
    ])

    preds = registry.predict(features)

    probas = registry.predict_proba(features)

    predictions = []
    for i, pred in enumerate(preds):
        pred_int = int(pred)
        confidence = float(probas[i][pred_int]) if probas is not None else None
        signal = "BUY" if pred_int == 1 else "SELL"
        predictions.append(PredictionResponse(
            signal=signal,
            prediction=pred_int,
            confidence=confidence,
            model_version=registry.version,
            model_type=registry.model_type,
        ))

    return BatchPredictionResponse(predictions=predictions, count=len(predictions))


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
