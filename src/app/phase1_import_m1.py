"""
Phase 1 – Importation M1
- Fusion date + time → timestamp
- Vérification régularité 1 minute
- Tri chronologique
- Détection incohérences
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

COLUMN_NAMES = ["date", "time", "open", "high", "low", "close", "volume"]

FILES = [
    RAW_DIR / "HISTDATA_COM_MT_GBPUSD_M12022" / "DAT_MT_GBPUSD_M1_2022.csv",
    RAW_DIR / "HISTDATA_COM_MT_GBPUSD_M12023" / "DAT_MT_GBPUSD_M1_2023.csv",
    RAW_DIR / "HISTDATA_COM_MT_GBPUSD_M12024" / "DAT_MT_GBPUSD_M1_2024.csv",
]


def load_raw_files() -> pd.DataFrame:
    """Charge et concatène les fichiers CSV bruts."""
    frames = []
    for f in FILES:
        df = pd.read_csv(f, header=None, names=COLUMN_NAMES)
        print(f"  Chargé {f.name}: {len(df)} lignes")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def fuse_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Fusionne les colonnes date + time en un timestamp unique."""
    df["timestamp"] = pd.to_datetime(
        df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M"
    )
    df = df.drop(columns=["date", "time"])
    # Réordonner: timestamp en premier
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    return df[cols]


def sort_chronological(df: pd.DataFrame) -> pd.DataFrame:
    """Tri chronologique et suppression des doublons de timestamp."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    n_duplicates = df["timestamp"].duplicated().sum()
    if n_duplicates > 0:
        print(f"  [WARN] {n_duplicates} timestamps dupliqués détectés → supprimés (garde le premier)")
        df = df.drop_duplicates(subset="timestamp", keep="first").reset_index(drop=True)
    return df


def check_regularity(df: pd.DataFrame) -> None:
    """Vérifie la régularité 1 minute et reporte les gaps."""
    diffs = df["timestamp"].diff().dropna()
    one_min = pd.Timedelta(minutes=1)

    n_regular = (diffs == one_min).sum()
    n_gaps = (diffs > one_min).sum()
    n_total = len(diffs)

    print(f"  Intervalles réguliers (1 min): {n_regular}/{n_total} ({n_regular/n_total*100:.1f}%)")
    print(f"  Gaps (> 1 min): {n_gaps}")

    # Distribution des tailles de gaps
    if n_gaps > 0:
        gap_sizes = diffs[diffs > one_min]
        print(f"  Gap min: {gap_sizes.min()}, max: {gap_sizes.max()}, médian: {gap_sizes.median()}")


def detect_incoherences(df: pd.DataFrame) -> pd.DataFrame:
    """Détecte et signale les incohérences OHLC."""
    issues = []

    # Prix négatifs ou nuls
    for col in ["open", "high", "low", "close"]:
        mask = df[col] <= 0
        if mask.any():
            issues.append(f"  [ERREUR] {mask.sum()} prix négatifs/nuls dans '{col}'")

    # High < Low
    mask_hl = df["high"] < df["low"]
    if mask_hl.any():
        issues.append(f"  [ERREUR] {mask_hl.sum()} lignes avec high < low")

    # Open/Close hors range [low, high]
    mask_open = (df["open"] < df["low"]) | (df["open"] > df["high"])
    if mask_open.any():
        issues.append(f"  [WARN] {mask_open.sum()} lignes avec open hors [low, high]")

    mask_close = (df["close"] < df["low"]) | (df["close"] > df["high"])
    if mask_close.any():
        issues.append(f"  [WARN] {mask_close.sum()} lignes avec close hors [low, high]")

    # Volume négatif
    mask_vol = df["volume"] < 0
    if mask_vol.any():
        issues.append(f"  [WARN] {mask_vol.sum()} volumes négatifs")

    if issues:
        for issue in issues:
            print(issue)
    else:
        print("  Aucune incohérence détectée")

    # Marquer les lignes incohérentes (high < low) pour référence
    incoherent_mask = mask_hl
    if incoherent_mask.any():
        print(f"  → Suppression de {incoherent_mask.sum()} lignes avec high < low")
        df = df[~incoherent_mask].reset_index(drop=True)

    return df


def main():
    print("=" * 60)
    print("PHASE 1 – Importation M1 GBP/USD")
    print("=" * 60)

    # 1. Chargement
    print("\n[1] Chargement des fichiers bruts...")
    df = load_raw_files()
    print(f"  Total brut: {len(df)} lignes")

    # 2. Fusion timestamp
    print("\n[2] Fusion date + time → timestamp...")
    df = fuse_timestamp(df)
    print(f"  Plage: {df['timestamp'].min()} → {df['timestamp'].max()}")

    # 3. Tri chronologique
    print("\n[3] Tri chronologique + dédoublonnage...")
    df = sort_chronological(df)
    print(f"  Lignes après dédoublonnage: {len(df)}")

    # 4. Vérification régularité
    print("\n[4] Vérification régularité 1 minute...")
    check_regularity(df)

    # 5. Détection incohérences
    print("\n[5] Détection incohérences OHLCV...")
    df = detect_incoherences(df)

    # 6. Résumé par année
    print("\n[6] Résumé par année:")
    df["year"] = df["timestamp"].dt.year
    for year, group in df.groupby("year"):
        print(f"  {year}: {len(group)} bougies, "
              f"{group['timestamp'].min()} → {group['timestamp'].max()}")
    df = df.drop(columns=["year"])

    # 7. Sauvegarde
    print("\n[7] Sauvegarde...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "gbpusd_m1.csv"
    df.to_csv(output_path, index=False)
    print(f"  Sauvegardé: {output_path}")
    print(f"  Shape finale: {df.shape}")

    print("\n" + "=" * 60)
    print("Phase 1 terminée avec succès")
    print("=" * 60)


if __name__ == "__main__":
    main()
