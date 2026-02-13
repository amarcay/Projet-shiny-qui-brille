"""Schemas Pydantic pour les bougies M15."""

from typing import Optional

from pydantic import BaseModel, Field


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
    ema_20: Optional[float] = Field(None, description="EMA 20 periodes (v2/v3)")
    ema_50: Optional[float] = Field(None, description="EMA 50 periodes (v2/v3)")


class BatchInput(BaseModel):
    """Batch de bougies M15."""

    candles: list[CandleInput] = Field(..., min_length=1, max_length=100)
