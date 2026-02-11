"""
Phase 4 – Analyse exploratoire GBP/USD M15 (clean)
- Distribution des rendements
- Volatilité dans le temps
- Analyse horaire
- Autocorrélation (ACF)
- Test ADF (stationnarité)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "gbpusd_m15_clean.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "reports" / "eda"

sns.set_theme(style="whitegrid")


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp_15m"])
    df = df.set_index("timestamp_15m")
    df["return_1"] = df["close_15m"].pct_change()
    df["log_return"] = np.log(df["close_15m"] / df["close_15m"].shift(1))
    return df


# ──────────────────────────────────────────────
# 1. Distribution des rendements
# ──────────────────────────────────────────────
def plot_return_distribution(df: pd.DataFrame):
    returns = df["log_return"].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogramme
    axes[0].hist(returns, bins=200, density=True, alpha=0.7, color="steelblue", edgecolor="none")
    axes[0].set_title("Distribution des log-rendements M15")
    axes[0].set_xlabel("Log-rendement")
    axes[0].set_ylabel("Densité")
    axes[0].axvline(returns.mean(), color="red", linestyle="--", label=f"Moyenne: {returns.mean():.6f}")
    axes[0].legend()

    # QQ-plot
    from scipy import stats
    stats.probplot(returns.values, dist="norm", plot=axes[1])
    axes[1].set_title("QQ-Plot vs Normale")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_distribution_rendements.png", dpi=150)
    plt.close(fig)

    # Stats descriptives
    print("  Distribution des log-rendements:")
    print(f"    Moyenne:    {returns.mean():.8f}")
    print(f"    Écart-type: {returns.std():.8f}")
    print(f"    Skewness:   {returns.skew():.4f}")
    print(f"    Kurtosis:   {returns.kurtosis():.4f}")
    print(f"    Min:        {returns.min():.6f}")
    print(f"    Max:        {returns.max():.6f}")


# ──────────────────────────────────────────────
# 2. Volatilité dans le temps
# ──────────────────────────────────────────────
def plot_volatility(df: pd.DataFrame):
    # Volatilité rolling journalière (96 bougies M15 = 1 jour de 24h)
    df["rolling_std_1d"] = df["log_return"].rolling(96).std()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Prix close
    axes[0].plot(df.index, df["close_15m"], linewidth=0.3, color="steelblue")
    axes[0].set_title("GBP/USD – Prix Close M15")
    axes[0].set_ylabel("Prix")

    # Volatilité rolling
    axes[1].plot(df.index, df["rolling_std_1d"], linewidth=0.5, color="darkorange")
    axes[1].set_title("Volatilité rolling 1 jour (std log-rendements, fenêtre 96 bougies)")
    axes[1].set_ylabel("Volatilité")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_volatilite_temps.png", dpi=150)
    plt.close(fig)

    # Volatilité par année
    print("  Volatilité annualisée par année:")
    for year in [2022, 2023, 2024]:
        yearly = df[df.index.year == year]["log_return"].dropna()
        # M15 → 96 bougies/jour × 252 jours
        vol_annualized = yearly.std() * np.sqrt(252 * 96)
        print(f"    {year}: {vol_annualized:.4f} ({vol_annualized*100:.2f}%)")

    df.drop(columns=["rolling_std_1d"], inplace=True)


# ──────────────────────────────────────────────
# 3. Analyse horaire
# ──────────────────────────────────────────────
def plot_hourly_analysis(df: pd.DataFrame):
    df_hourly = df.copy()
    df_hourly["hour"] = df_hourly.index.hour

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Nombre de bougies par heure
    counts = df_hourly.groupby("hour").size()
    axes[0].bar(counts.index, counts.values, color="steelblue", alpha=0.8)
    axes[0].set_title("Nombre de bougies M15 par heure")
    axes[0].set_xlabel("Heure (UTC)")
    axes[0].set_ylabel("Nombre de bougies")

    # Volatilité moyenne par heure
    hourly_vol = df_hourly.groupby("hour")["log_return"].std()
    axes[1].bar(hourly_vol.index, hourly_vol.values, color="darkorange", alpha=0.8)
    axes[1].set_title("Volatilité moyenne par heure")
    axes[1].set_xlabel("Heure (UTC)")
    axes[1].set_ylabel("Std des log-rendements")

    # Rendement moyen par heure
    hourly_mean = df_hourly.groupby("hour")["log_return"].mean()
    colors = ["green" if x >= 0 else "red" for x in hourly_mean.values]
    axes[2].bar(hourly_mean.index, hourly_mean.values, color=colors, alpha=0.8)
    axes[2].set_title("Rendement moyen par heure")
    axes[2].set_xlabel("Heure (UTC)")
    axes[2].set_ylabel("Log-rendement moyen")
    axes[2].axhline(0, color="black", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_analyse_horaire.png", dpi=150)
    plt.close(fig)

    print("  Top 3 heures les plus volatiles:")
    top_vol = hourly_vol.sort_values(ascending=False).head(3)
    for h, v in top_vol.items():
        print(f"    {h}h UTC: std = {v:.8f}")


# ──────────────────────────────────────────────
# 4. Autocorrélation
# ──────────────────────────────────────────────
def plot_autocorrelation(df: pd.DataFrame):
    returns = df["log_return"].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ACF des rendements
    plot_acf(returns.values, lags=60, ax=axes[0], title="ACF – Log-rendements M15")
    axes[0].set_xlabel("Lag (bougies M15)")

    # ACF des rendements au carré (clustering de volatilité)
    plot_acf(returns.values ** 2, lags=60, ax=axes[1], title="ACF – Log-rendements² (volatilité)")
    axes[1].set_xlabel("Lag (bougies M15)")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_autocorrelation.png", dpi=150)
    plt.close(fig)

    print("  ACF rendements lag 1: {:.6f}".format(
        returns.autocorr(lag=1)
    ))
    print("  ACF rendements² lag 1: {:.6f} (clustering de volatilité)".format(
        (returns**2).autocorr(lag=1)
    ))


# ──────────────────────────────────────────────
# 5. Test ADF (stationnarité)
# ──────────────────────────────────────────────
def test_adf(df: pd.DataFrame):
    print("  Test Augmented Dickey-Fuller:")
    print()

    # Test sur le prix close
    close_sample = df["close_15m"].dropna().values
    result_close = adfuller(close_sample, maxlag=50, autolag="AIC")
    print("  [Prix close M15]")
    print(f"    Stat ADF:   {result_close[0]:.6f}")
    print(f"    p-value:    {result_close[1]:.6f}")
    print(f"    Lags:       {result_close[2]}")
    for key, val in result_close[4].items():
        print(f"    Seuil {key}: {val:.6f}")
    if result_close[1] < 0.05:
        print("    → Série stationnaire (p < 0.05)")
    else:
        print("    → Série NON stationnaire (p >= 0.05)")

    print()

    # Test sur les rendements
    returns = df["log_return"].dropna().values
    result_ret = adfuller(returns, maxlag=50, autolag="AIC")
    print("  [Log-rendements M15]")
    print(f"    Stat ADF:   {result_ret[0]:.6f}")
    print(f"    p-value:    {result_ret[1]:.6f}")
    print(f"    Lags:       {result_ret[2]}")
    for key, val in result_ret[4].items():
        print(f"    Seuil {key}: {val:.6f}")
    if result_ret[1] < 0.05:
        print("    → Série stationnaire (p < 0.05)")
    else:
        print("    → Série NON stationnaire (p >= 0.05)")


def main():
    print("=" * 60)
    print("PHASE 4 – Analyse exploratoire GBP/USD M15")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nChargement des données...")
    df = load_data()
    print(f"  {len(df)} bougies M15 chargées")

    print("\n[1] Distribution des rendements...")
    plot_return_distribution(df)

    print("\n[2] Volatilité dans le temps...")
    plot_volatility(df)

    print("\n[3] Analyse horaire...")
    plot_hourly_analysis(df)

    print("\n[4] Autocorrélation...")
    plot_autocorrelation(df)

    print("\n[5] Test ADF (stationnarité)...")
    test_adf(df)

    print(f"\n  Graphiques sauvegardés dans: {OUTPUT_DIR}")
    print("\n" + "=" * 60)
    print("Phase 4 terminée avec succès")
    print("=" * 60)


if __name__ == "__main__":
    main()
