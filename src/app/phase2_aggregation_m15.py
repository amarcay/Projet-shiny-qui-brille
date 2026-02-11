"""
Phase 2 – Agrégation M1 → M15
- open_15m  : open de la 1ère minute
- high_15m  : max(high) sur 15 minutes
- low_15m   : min(low) sur 15 minutes
- close_15m : close de la dernière minute
"""

import pandas as pd
from pathlib import Path

INPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def load_m1() -> pd.DataFrame:
    """Charge les données M1 nettoyées."""
    path = INPUT_DIR / "gbpusd_m1.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"])
    print(f"  Chargé: {len(df)} bougies M1")
    return df


def aggregate_m1_to_m15(df: pd.DataFrame) -> pd.DataFrame:
    """Agrège les bougies M1 en bougies M15 via resample."""
    df = df.set_index("timestamp")

    df_m15 = df.resample("15min").agg(
        open_15m=("open", "first"),
        high_15m=("high", "max"),
        low_15m=("low", "min"),
        close_15m=("close", "last"),
        volume_15m=("volume", "sum"),
    )

    # Supprimer les fenêtres sans données (NaN = aucune bougie M1 dans la fenêtre)
    df_m15 = df_m15.dropna(subset=["open_15m"]).reset_index()
    df_m15 = df_m15.rename(columns={"timestamp": "timestamp_15m"})

    return df_m15


def add_candle_count(df_m1: pd.DataFrame, df_m15: pd.DataFrame) -> pd.DataFrame:
    """Ajoute le nombre de bougies M1 par fenêtre M15 (pour contrôle qualité)."""
    df_m1 = df_m1.set_index("timestamp")
    counts = df_m1.resample("15min").size().reset_index(name="n_candles_m1")
    counts = counts.rename(columns={"timestamp": "timestamp_15m"})

    df_m15 = df_m15.merge(counts, on="timestamp_15m", how="left")
    df_m15["n_candles_m1"] = df_m15["n_candles_m1"].fillna(0).astype(int)

    return df_m15


def main():
    print("=" * 60)
    print("PHASE 2 – Agrégation M1 → M15 GBP/USD")
    print("=" * 60)

    # 1. Chargement M1
    print("\n[1] Chargement des données M1...")
    df_m1 = load_m1()
    print(f"  Plage: {df_m1['timestamp'].min()} → {df_m1['timestamp'].max()}")

    # 2. Agrégation
    print("\n[2] Agrégation M1 → M15...")
    df_m15 = aggregate_m1_to_m15(df_m1)
    print(f"  Bougies M15 générées: {len(df_m15)}")

    # 3. Ajout compteur M1 par fenêtre
    print("\n[3] Ajout du compteur de bougies M1 par fenêtre...")
    df_m15 = add_candle_count(df_m1, df_m15)

    # 4. Statistiques
    print("\n[4] Statistiques de qualité:")
    print(f"  Bougies M15 complètes (15 M1): {(df_m15['n_candles_m1'] == 15).sum()}")
    print(f"  Bougies M15 incomplètes (<15 M1): {(df_m15['n_candles_m1'] < 15).sum()}")
    print(f"  Distribution n_candles_m1:")
    print(df_m15["n_candles_m1"].value_counts().sort_index().to_string(header=False))

    # 5. Résumé par année
    print("\n[5] Résumé par année:")
    df_m15["year"] = df_m15["timestamp_15m"].dt.year
    for year, group in df_m15.groupby("year"):
        print(f"  {year}: {len(group)} bougies M15, "
              f"{group['timestamp_15m'].min()} → {group['timestamp_15m'].max()}")
    df_m15 = df_m15.drop(columns=["year"])

    # 6. Sauvegarde
    print("\n[6] Sauvegarde...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "gbpusd_m15_raw.csv"
    df_m15.to_csv(output_path, index=False)
    print(f"  Sauvegardé: {output_path}")
    print(f"  Shape finale: {df_m15.shape}")
    print(f"  Colonnes: {list(df_m15.columns)}")

    print("\n" + "=" * 60)
    print("Phase 2 terminée avec succès")
    print("=" * 60)


if __name__ == "__main__":
    main()
