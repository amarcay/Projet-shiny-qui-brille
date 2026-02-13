"""Schemas Pydantic pour les reponses de l'API."""

from typing import Optional
from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Reponse de prediction."""

    signal: str = Field(..., description="BUY, SELL ou HOLD")
    prediction: int = Field(..., description="1=hausse, 0=baisse")
    confidence: Optional[float] = Field(
        None, description="Probabilite de la prediction"
    )
    model_version: str
    model_type: str


class BatchPredictionResponse(BaseModel):
    """Reponse de prediction batch."""

    predictions: list[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    """Reponse du health check."""

    status: str
    model_loaded: bool
    model_version: Optional[str]
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Informations sur le modele charge."""

    version: str
    model_type: str
    model_name: Optional[str] = None
    features: list[str]
    available_versions: list[str]
    loaded_at: str
