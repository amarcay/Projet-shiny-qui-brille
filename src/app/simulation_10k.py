"""
Simulation de trading GBP/USD avec 10 000€ de capital sur 2024.
Utilise le meilleur modèle du registry (v2 – DQN).

Hypothèses réalistes :
  - Capital initial : 10 000€
  - Levier : 1:30 (standard ESMA retail forex)
  - Taille position : 1 mini-lot (10 000 unités GBP/USD)
  - Spread : 1 pip (0.0001)
  - Pas de slippage (M15 suffisamment agrégé)
  - Valeur du pip (mini-lot) : ~1€ (pour GBP/USD ≈ 1.27)
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from stable_baselines3 import DQN

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "features" / "gbpusd_m15_features.csv"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "reports" / "simulation"

INITIAL_CAPITAL = 10_000.0
LOT_SIZE = 10_000          # mini-lot = 10 000 unités
SPREAD_PIPS = 1.0
PIP_VALUE = 0.0001
SPREAD_COST = SPREAD_PIPS * PIP_VALUE  # 0.0001

RL_FEATURE_COLS = [
    "return_1", "return_4", "ema_diff", "rsi_14", "rolling_std_20",
    "range_15m", "body", "upper_wick", "lower_wick",
    "distance_to_ema200", "slope_ema50",
    "atr_14", "rolling_std_100", "volatility_ratio",
    "adx_14", "macd", "macd_signal",
    "ema_20", "ema_50",
]


def main():
    print("=" * 70)
    print("SIMULATION – 10 000€ sur GBP/USD M15 (2025 & 2026)")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Chargement données 2025 & 2026 ──
    print("\n[1] Chargement des données 2025 & 2026...")
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp_15m"])
    df_2025_2026 = df[(df["timestamp_15m"].dt.year == 2025) | (df["timestamp_15m"].dt.year == 2026)].copy().reset_index(drop=True)
    print(f"  {len(df_2025_2026)} bougies M15")

    # ── Chargement modèle v2 ──
    print("\n[2] Chargement du modèle v2 (DQN)...")
    model = DQN.load(str(MODELS_DIR / "v2" / "dqn_gbpusd_m15"))
    norm_stats = joblib.load(MODELS_DIR / "v2" / "norm_stats.joblib")
    feature_mean = norm_stats["mean"]
    feature_std = norm_stats["std"]

    # Normalisation des features
    features_raw = df_2025_2026[RL_FEATURE_COLS].values
    std_safe = feature_std.values.copy()
    std_safe[std_safe == 0] = 1.0
    features_norm = (features_raw - feature_mean.values) / std_safe

    # ── Simulation ──
    print("\n[3] Simulation en cours...")
    capital = INITIAL_CAPITAL
    position = 0  # -1, 0, 1
    steps_in_position = 0
    entry_price = 0.0

    # Historique
    history = []

    for i in range(len(df_2025_2026) - 1):
        row = df_2025_2026.iloc[i]
        next_row = df_2025_2026.iloc[i + 1]
        timestamp = row["timestamp_15m"]
        close = row["close_15m"]
        next_close = next_row["close_15m"]

        # Observation (21 dims: 19 features + position + steps_in_position/100)
        obs = np.append(features_norm[i], [position, steps_in_position / 100.0]).astype(np.float32)
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        # Mapper action
        new_position = position
        if action == 1:
            new_position = 1
        elif action == 2:
            new_position = -1

        # PnL du step
        pnl_eur = 0.0
        trade_cost = 0.0

        # Si changement de position → coût de spread
        if new_position != position:
            # Coût = spread × lot_size en termes de prix
            trade_cost = SPREAD_COST * LOT_SIZE  # ≈ 1€ par trade (mini-lot)
            capital -= trade_cost
            entry_price = close
            steps_in_position = 0
        else:
            steps_in_position += 1

        # PnL de la position courante (sur la bougie suivante)
        if position != 0:
            price_change = next_close - close
            pnl_eur = position * price_change * LOT_SIZE
            capital += pnl_eur

        position = new_position

        history.append({
            "timestamp": timestamp,
            "close": close,
            "action": ["HOLD", "BUY", "SELL"][action],
            "position": position,
            "pnl_eur": pnl_eur,
            "trade_cost": trade_cost,
            "capital": capital,
        })

    hist_df = pd.DataFrame(history)
    hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"])

    # ── Métriques ──
    print("\n[4] Résultats de la simulation:")
    final_capital = hist_df["capital"].iloc[-1]
    total_return = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    total_pnl = final_capital - INITIAL_CAPITAL

    peak = hist_df["capital"].cummax()
    drawdown = hist_df["capital"] - peak
    max_dd_eur = drawdown.min()
    max_dd_pct = (max_dd_eur / peak[drawdown.idxmin()]) * 100

    daily_pnl = hist_df.set_index("timestamp")["pnl_eur"].resample("D").sum().dropna()
    daily_pnl = daily_pnl[daily_pnl != 0]
    sharpe_daily = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252) if daily_pnl.std() > 0 else 0

    n_trades = (hist_df["trade_cost"] > 0).sum()
    total_costs = hist_df["trade_cost"].sum()

    gains = hist_df["pnl_eur"][hist_df["pnl_eur"] > 0].sum()
    losses = abs(hist_df["pnl_eur"][hist_df["pnl_eur"] < 0].sum())
    pf = gains / losses if losses > 0 else np.inf

    winning_steps = (hist_df["pnl_eur"] > 0).sum()
    losing_steps = (hist_df["pnl_eur"] < 0).sum()
    win_rate = winning_steps / (winning_steps + losing_steps) * 100 if (winning_steps + losing_steps) > 0 else 0

    pos_counts = hist_df["position"].value_counts()

    print(f"  ┌─────────────────────────────────────────┐")
    print(f"  │  Capital initial:    {INITIAL_CAPITAL:>12,.2f}€       │")
    print(f"  │  Capital final:      {final_capital:>12,.2f}€       │")
    print(f"  │  PnL total:          {total_pnl:>+12,.2f}€       │")
    print(f"  │  Rendement:          {total_return:>+11,.2f}%        │")
    print(f"  │  Max drawdown:       {max_dd_eur:>12,.2f}€       │")
    print(f"  │  Max drawdown:       {max_dd_pct:>11,.2f}%        │")
    print(f"  │  Sharpe (daily):     {sharpe_daily:>12,.3f}         │")
    print(f"  │  Profit factor:      {pf:>12,.3f}         │")
    print(f"  │  Win rate:           {win_rate:>11,.1f}%        │")
    print(f"  │  Nb trades:          {n_trades:>12,}         │")
    print(f"  │  Coûts totaux:       {total_costs:>12,.2f}€       │")
    print(f"  └─────────────────────────────────────────┘")

    print(f"\n  Répartition des positions:")
    for pos_val, label in [(-1, "Short"), (0, "Flat"), (1, "Long")]:
        count = pos_counts.get(pos_val, 0)
        pct = count / len(hist_df) * 100
        print(f"    {label:6s}: {count:>6} ({pct:.1f}%)")

    # ── Résumé mensuel ──
    print(f"\n  Résumé mensuel:")
    hist_df["month"] = hist_df["timestamp"].dt.to_period("M")
    monthly = hist_df.groupby("month").agg(
        pnl=("pnl_eur", "sum"),
        costs=("trade_cost", "sum"),
        trades=("trade_cost", lambda x: (x > 0).sum()),
    )
    monthly["capital_fin"] = INITIAL_CAPITAL + hist_df.groupby("month")["pnl_eur"].apply(
        lambda x: hist_df.loc[x.index, "capital"].iloc[-1]
    ).values - hist_df.groupby("month")["trade_cost"].sum().cumsum().values + hist_df.groupby("month")["trade_cost"].sum().values

    print(f"    {'Mois':>10s}  {'PnL':>10s}  {'Coûts':>8s}  {'Trades':>7s}")
    print(f"    {'─'*10}  {'─'*10}  {'─'*8}  {'─'*7}")
    for month, row in monthly.iterrows():
        print(f"    {str(month):>10s}  {row['pnl']:>+10.2f}€  {row['costs']:>8.2f}€  {int(row['trades']):>7}")

    # ── Graphiques ──
    print(f"\n[5] Génération des graphiques...")

    fig, axes = plt.subplots(3, 1, figsize=(16, 14), gridspec_kw={"height_ratios": [3, 1, 1]})

    # 1. Évolution du capital
    ax = axes[0]
    ax.plot(hist_df["timestamp"], hist_df["capital"], linewidth=0.8, color="steelblue")
    ax.axhline(INITIAL_CAPITAL, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.fill_between(hist_df["timestamp"], INITIAL_CAPITAL, hist_df["capital"],
                     where=hist_df["capital"] >= INITIAL_CAPITAL, alpha=0.15, color="green")
    ax.fill_between(hist_df["timestamp"], INITIAL_CAPITAL, hist_df["capital"],
                     where=hist_df["capital"] < INITIAL_CAPITAL, alpha=0.15, color="red")
    ax.set_title(f"Simulation 10 000€ – GBP/USD M15 (2025 & 2026) – DQN v2\n"
                 f"Capital final: {final_capital:,.2f}€ ({total_return:+.2f}%) | "
                 f"Max DD: {max_dd_pct:.2f}% | Sharpe: {sharpe_daily:.3f}", fontsize=12)
    ax.set_ylabel("Capital (€)")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}€"))

    # 2. Prix GBP/USD
    ax = axes[1]
    ax.plot(df_2025_2026["timestamp_15m"].iloc[:-1], df_2025_2026["close_15m"].iloc[:-1],
            linewidth=0.4, color="gray")
    ax.set_ylabel("GBP/USD")
    ax.set_title("Prix GBP/USD (close M15)")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # 3. Drawdown
    ax = axes[2]
    ax.fill_between(hist_df["timestamp"], 0, drawdown, color="red", alpha=0.4)
    ax.set_ylabel("Drawdown (€)")
    ax.set_title(f"Drawdown (max: {max_dd_eur:,.2f}€ / {max_dd_pct:.2f}%)")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}€"))

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "simulation_10k_2025_2026.png", dpi=150)
    plt.close(fig)

    # Sauvegarde CSV
    hist_df.to_csv(OUTPUT_DIR / "simulation_10k_history.csv", index=False)

    print(f"\n  Fichiers sauvegardés:")
    print(f"    {OUTPUT_DIR / 'simulation_10k_2025_2026.png'}")
    print(f"    {OUTPUT_DIR / 'simulation_10k_history.csv'}")

    print("\n" + "=" * 70)
    print("Simulation terminée")
    print("=" * 70)


if __name__ == "__main__":
    main()
