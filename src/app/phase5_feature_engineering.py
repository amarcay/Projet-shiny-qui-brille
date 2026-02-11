"""
Phase 5 – Feature Engineering V2
Toutes les features sont calculees uniquement a partir du passe.

Bloc court terme:
  return_1, return_4, ema_20, ema_50, ema_diff,
  rsi_14, rolling_std_20, range_15m, body, upper_wick, lower_wick

Bloc Contexte & Regime:
  Tendance long terme : ema_200, distance_to_ema200, slope_ema50
  Regime de volatilite : atr_14, rolling_std_100, volatility_ratio
  Force directionnelle : adx_14, macd, macd_signal
"""

import pandas as pd
import numpy as np
from pathlib import Path

INPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "features"


def load_m15_clean() -> pd.DataFrame:
    """Charge les donnees M15 nettoyees (sortie Phase 3)."""
    path = INPUT_DIR / "gbpusd_m15_clean.csv"
    df = pd.read_csv(path, parse_dates=["timestamp_15m"])
    print(f"  Charge: {len(df)} bougies M15")
    return df


# ---------------------------------------------------------------------------
# Bloc court terme
# ---------------------------------------------------------------------------

def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """return_1 et return_4 : rendements sur 1 et 4 periodes."""
    df["return_1"] = df["close_15m"].pct_change(1)
    df["return_4"] = df["close_15m"].pct_change(4)
    return df


def add_ema_short(df: pd.DataFrame) -> pd.DataFrame:
    """ema_20, ema_50 et ema_diff."""
    df["ema_20"] = df["close_15m"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close_15m"].ewm(span=50, adjust=False).mean()
    df["ema_diff"] = df["ema_20"] - df["ema_50"]
    return df


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calcule le RSI classique."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def add_rsi(df: pd.DataFrame) -> pd.DataFrame:
    """rsi_14."""
    df["rsi_14"] = compute_rsi(df["close_15m"], 14)
    return df


def add_rolling_std(df: pd.DataFrame) -> pd.DataFrame:
    """rolling_std_20 : volatilite court terme."""
    df["rolling_std_20"] = df["close_15m"].pct_change().rolling(20).std()
    return df


def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """range_15m, body, upper_wick, lower_wick."""
    df["range_15m"] = df["high_15m"] - df["low_15m"]
    df["body"] = (df["close_15m"] - df["open_15m"]).abs()
    df["upper_wick"] = df["high_15m"] - df[["open_15m", "close_15m"]].max(axis=1)
    df["lower_wick"] = df[["open_15m", "close_15m"]].min(axis=1) - df["low_15m"]
    return df


# ---------------------------------------------------------------------------
# Bloc Contexte & Regime
# ---------------------------------------------------------------------------

def add_ema_long(df: pd.DataFrame) -> pd.DataFrame:
    """ema_200, distance_to_ema200, slope_ema50."""
    df["ema_200"] = df["close_15m"].ewm(span=200, adjust=False).mean()
    df["distance_to_ema200"] = (df["close_15m"] - df["ema_200"]) / df["ema_200"]
    df["slope_ema50"] = df["ema_50"].diff() / df["ema_50"].shift(1)
    return df


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calcule l'Average True Range."""
    high = df["high_15m"]
    low = df["low_15m"]
    close_prev = df["close_15m"].shift(1)
    tr = pd.concat([
        high - low,
        (high - close_prev).abs(),
        (low - close_prev).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def add_volatility_regime(df: pd.DataFrame) -> pd.DataFrame:
    """atr_14, rolling_std_100, volatility_ratio."""
    df["atr_14"] = compute_atr(df, 14)
    df["rolling_std_100"] = df["close_15m"].pct_change().rolling(100).std()
    df["volatility_ratio"] = df["rolling_std_20"] / df["rolling_std_100"]
    return df


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calcule l'ADX (Average Directional Index)."""
    high = df["high_15m"]
    low = df["low_15m"]
    close = df["close_15m"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr = compute_atr(df, period)

    plus_di = 100 * plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return adx


def add_adx(df: pd.DataFrame) -> pd.DataFrame:
    """adx_14."""
    df["adx_14"] = compute_adx(df, 14)
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """macd et macd_signal (12/26/9 standard)."""
    ema_12 = df["close_15m"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close_15m"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    return df


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applique toutes les features dans l'ordre."""
    # Bloc court terme
    df = add_returns(df)
    df = add_ema_short(df)
    df = add_rsi(df)
    df = add_rolling_std(df)
    df = add_candle_features(df)

    # Bloc contexte & regime
    df = add_ema_long(df)
    df = add_volatility_regime(df)
    df = add_adx(df)
    df = add_macd(df)

    return df


def main():
    print("=" * 60)
    print("PHASE 5 – Feature Engineering V2 GBP/USD")
    print("=" * 60)

    # 1. Chargement
    print("\n[1] Chargement M15 nettoye...")
    df = load_m15_clean()

    # 2. Construction features
    print("\n[2] Construction des features...")
    df = build_features(df)

    feature_cols = [
        "return_1", "return_4",
        "ema_20", "ema_50", "ema_diff",
        "rsi_14", "rolling_std_20",
        "range_15m", "body", "upper_wick", "lower_wick",
        "ema_200", "distance_to_ema200", "slope_ema50",
        "atr_14", "rolling_std_100", "volatility_ratio",
        "adx_14", "macd", "macd_signal",
    ]
    print(f"  Features creees: {len(feature_cols)}")
    for f in feature_cols:
        print(f"    - {f}")

    # 3. NaN dus au warm-up
    n_before = len(df)
    nan_counts = df[feature_cols].isna().sum()
    print(f"\n[3] NaN par feature (warm-up):")
    for f in feature_cols:
        if nan_counts[f] > 0:
            print(f"    {f}: {nan_counts[f]}")

    warmup = df[feature_cols].isna().any(axis=1).sum()
    print(f"\n  Lignes avec au moins un NaN: {warmup}")
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    print(f"  Lignes apres suppression warm-up: {len(df)} (supprime: {n_before - len(df)})")

    # 4. Resume par annee
    print("\n[4] Resume par annee:")
    df["year"] = df["timestamp_15m"].dt.year
    for year, group in df.groupby("year"):
        print(f"  {year}: {len(group)} bougies")
    df = df.drop(columns=["year"])

    # 5. Statistiques features
    print("\n[5] Statistiques des features:")
    print(df[feature_cols].describe().round(6).to_string())

    # 6. Sauvegarde
    print("\n[6] Sauvegarde...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "gbpusd_m15_features.csv"
    df.to_csv(output_path, index=False)
    print(f"  Sauvegarde: {output_path}")
    print(f"  Shape finale: {df.shape}")
    print(f"  Colonnes: {list(df.columns)}")

    print("\n" + "=" * 60)
    print("Phase 5 terminee avec succes")
    print("=" * 60)


if __name__ == "__main__":
    main()
