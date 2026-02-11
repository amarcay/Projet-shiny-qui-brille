"""
Phase 3 – Nettoyage M15
- Suppression bougies incomplètes
- Contrôle prix négatifs
- Détection gaps anormaux
"""

import pandas as pd
import numpy as np
from pathlib import Path

INPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def load_m15_raw() -> pd.DataFrame:
    """Charge les données M15 brutes (sortie de Phase 2)."""
    path = INPUT_DIR / "gbpusd_m15_raw.csv"
    df = pd.read_csv(path, parse_dates=["timestamp_15m"])
    print(f"  Chargé: {len(df)} bougies M15 brutes")
    return df


def remove_incomplete_candles(df: pd.DataFrame, min_candles: int = 15) -> pd.DataFrame:
    """Supprime les bougies M15 avec moins de min_candles bougies M1."""
    n_before = len(df)
    incomplete = df[df["n_candles_m1"] < min_candles]
    df = df[df["n_candles_m1"] >= min_candles].reset_index(drop=True)
    n_removed = n_before - len(df)

    print(f"  Bougies incomplètes supprimées: {n_removed}")
    if n_removed > 0:
        print(f"  Distribution des bougies supprimées (n_candles_m1):")
        print(incomplete["n_candles_m1"].value_counts().sort_index().to_string(header=False))

    return df


def check_negative_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Contrôle et supprime les bougies avec prix négatifs ou nuls."""
    price_cols = ["open_15m", "high_15m", "low_15m", "close_15m"]
    n_before = len(df)

    mask_negative = pd.Series(False, index=df.index)
    for col in price_cols:
        neg = df[col] <= 0
        if neg.any():
            print(f"  [ERREUR] {neg.sum()} prix négatifs/nuls dans '{col}'")
        mask_negative = mask_negative | neg

    if mask_negative.any():
        df = df[~mask_negative].reset_index(drop=True)
        print(f"  → Supprimé {mask_negative.sum()} bougies avec prix négatifs/nuls")
    else:
        print("  Aucun prix négatif ou nul détecté")

    return df


def detect_abnormal_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Détecte les gaps anormaux entre bougies M15 consécutives."""
    expected_gap = pd.Timedelta(minutes=15)
    diffs = df["timestamp_15m"].diff().dropna()

    n_normal = (diffs == expected_gap).sum()
    n_gaps = (diffs > expected_gap).sum()
    n_total = len(diffs)

    print(f"  Intervalles réguliers (15 min): {n_normal}/{n_total} ({n_normal / n_total * 100:.1f}%)")
    print(f"  Gaps (> 15 min): {n_gaps}")

    if n_gaps > 0:
        gap_sizes = diffs[diffs > expected_gap]
        gap_hours = gap_sizes.dt.total_seconds() / 3600

        print(f"  Gap min: {gap_sizes.min()}, max: {gap_sizes.max()}, médian: {gap_sizes.median()}")
        print(f"  Gaps > 24h (weekends/jours fériés): {(gap_hours > 24).sum()}")
        print(f"  Gaps 1h-24h (sessions fermées): {((gap_hours > 1) & (gap_hours <= 24)).sum()}")
        print(f"  Gaps 15min-1h (trous intrajournaliers): {(gap_hours <= 1).sum()}")

    # Détection gaps de prix anormaux (close→open)
    price_gaps = (df["open_15m"].shift(-1) - df["close_15m"]).abs()
    price_gaps = price_gaps.dropna()
    mean_gap = price_gaps.mean()
    std_gap = price_gaps.std()
    threshold = mean_gap + 5 * std_gap

    abnormal_price_gaps = price_gaps[price_gaps > threshold]
    if len(abnormal_price_gaps) > 0:
        print(f"\n  Gaps de prix anormaux (> 5σ): {len(abnormal_price_gaps)}")
        print(f"  Seuil: {threshold:.6f}")
        print(f"  Max gap de prix: {price_gaps.max():.6f}")
    else:
        print(f"\n  Aucun gap de prix anormal détecté (seuil 5σ = {threshold:.6f})")

    return df


def check_ohlc_coherence(df: pd.DataFrame) -> pd.DataFrame:
    """Vérifie la cohérence OHLC des bougies M15."""
    issues_total = 0

    # High < Low
    mask_hl = df["high_15m"] < df["low_15m"]
    if mask_hl.any():
        print(f"  [ERREUR] {mask_hl.sum()} bougies avec high < low → supprimées")
        df = df[~mask_hl].reset_index(drop=True)
        issues_total += mask_hl.sum()

    # Open hors range [low, high]
    mask_open = (df["open_15m"] < df["low_15m"]) | (df["open_15m"] > df["high_15m"])
    if mask_open.any():
        print(f"  [WARN] {mask_open.sum()} bougies avec open hors [low, high]")
        issues_total += mask_open.sum()

    # Close hors range [low, high]
    mask_close = (df["close_15m"] < df["low_15m"]) | (df["close_15m"] > df["high_15m"])
    if mask_close.any():
        print(f"  [WARN] {mask_close.sum()} bougies avec close hors [low, high]")
        issues_total += mask_close.sum()

    if issues_total == 0:
        print("  Cohérence OHLC OK")

    return df


def generate_quality_report(df: pd.DataFrame) -> None:
    """Génère un rapport de qualité des données M15 nettoyées."""
    print("\n  --- Rapport qualité M15 nettoyé ---")
    print(f"  Nombre total de bougies: {len(df)}")
    print(f"  Plage: {df['timestamp_15m'].min()} → {df['timestamp_15m'].max()}")

    df_tmp = df.copy()
    df_tmp["year"] = df_tmp["timestamp_15m"].dt.year
    for year, group in df_tmp.groupby("year"):
        print(f"  {year}: {len(group)} bougies M15")

    print(f"\n  Statistiques prix (close_15m):")
    print(f"    Min:  {df['close_15m'].min():.5f}")
    print(f"    Max:  {df['close_15m'].max():.5f}")
    print(f"    Mean: {df['close_15m'].mean():.5f}")
    print(f"    Std:  {df['close_15m'].std():.5f}")

    returns = df["close_15m"].pct_change().dropna()
    print(f"\n  Statistiques rendements M15:")
    print(f"    Mean:     {returns.mean():.8f}")
    print(f"    Std:      {returns.std():.6f}")
    print(f"    Skewness: {returns.skew():.4f}")
    print(f"    Kurtosis: {returns.kurtosis():.4f}")


def main():
    print("=" * 60)
    print("PHASE 3 – Nettoyage M15 GBP/USD")
    print("=" * 60)

    # 1. Chargement
    print("\n[1] Chargement M15 brut...")
    df = load_m15_raw()

    # 2. Suppression bougies incomplètes
    print("\n[2] Suppression bougies incomplètes...")
    df = remove_incomplete_candles(df)
    print(f"  Restant: {len(df)} bougies")

    # 3. Contrôle prix négatifs
    print("\n[3] Contrôle prix négatifs...")
    df = check_negative_prices(df)

    # 4. Cohérence OHLC
    print("\n[4] Vérification cohérence OHLC...")
    df = check_ohlc_coherence(df)

    # 5. Détection gaps anormaux
    print("\n[5] Détection gaps anormaux...")
    df = detect_abnormal_gaps(df)

    # 6. Rapport qualité
    print("\n[6] Rapport qualité final...")
    generate_quality_report(df)

    # 7. Suppression colonne n_candles_m1 (plus nécessaire après nettoyage)
    df = df.drop(columns=["n_candles_m1"])

    # 8. Sauvegarde
    print("\n[7] Sauvegarde...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "gbpusd_m15_clean.csv"
    df.to_csv(output_path, index=False)
    print(f"  Sauvegardé: {output_path}")
    print(f"  Shape finale: {df.shape}")
    print(f"  Colonnes: {list(df.columns)}")

    print("\n" + "=" * 60)
    print("Phase 3 terminée avec succès")
    print("=" * 60)


if __name__ == "__main__":
    main()
