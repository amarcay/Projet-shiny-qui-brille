"""
Phase 6 – Baseline obligatoire (T06)
Stratégies de référence avec backtest simple sur GBP/USD M15.

Stratégies :
  1. Buy & Hold
  2. Random (BUY/SELL/HOLD uniformes)
  3. Règles fixes (EMA crossover + RSI + ADX)

Split temporel :
  - 2022 : Entraînement (calibrage règles)
  - 2023 : Validation
  - 2024 : Test final

Métriques :
  - Profit cumulé
  - Maximum drawdown
  - Sharpe simplifié
  - Profit factor
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "features" / "gbpusd_m15_features.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "reports" / "baseline"

TRANSACTION_COST = 0.0001  # ~1 pip spread
SEED = 42


# ──────────────────────────────────────────────
# Backtest engine
# ──────────────────────────────────────────────
def backtest(df: pd.DataFrame, signals: pd.Series) -> pd.DataFrame:
    """
    Backtest simple sur signaux {1: BUY, -1: SELL, 0: HOLD}.
    Rendement = position * return_next - cost_si_changement.
    """
    df = df.copy()
    df["signal"] = signals.values
    df["position"] = df["signal"].replace(0, np.nan).ffill().fillna(0)
    df["return_next"] = df["close_15m"].pct_change().shift(-1)

    # Coût de transaction uniquement quand la position change
    df["trade"] = df["position"].diff().abs().fillna(0)
    df["cost"] = df["trade"] * TRANSACTION_COST

    df["pnl"] = df["position"] * df["return_next"] - df["cost"]
    df["pnl"] = df["pnl"].fillna(0)
    df["cumulative_pnl"] = df["pnl"].cumsum()

    return df


def compute_metrics(df: pd.DataFrame, label: str) -> dict:
    """Calcule les métriques de performance."""
    pnl = df["pnl"].dropna()
    cum = df["cumulative_pnl"]

    # Profit cumulé
    total_profit = cum.iloc[-1] if len(cum) > 0 else 0.0

    # Maximum drawdown
    running_max = cum.cummax()
    drawdown = cum - running_max
    max_dd = drawdown.min()

    # Sharpe simplifié (annualisé M15 : 96 bougies/jour × 252 jours)
    if pnl.std() > 0:
        sharpe = (pnl.mean() / pnl.std()) * np.sqrt(96 * 252)
    else:
        sharpe = 0.0

    # Profit factor
    gains = pnl[pnl > 0].sum()
    losses = abs(pnl[pnl < 0].sum())
    profit_factor = gains / losses if losses > 0 else np.inf

    # Nombre de trades
    n_trades = int(df["trade"].gt(0).sum())

    return {
        "Stratégie": label,
        "Profit cumulé": total_profit,
        "Max drawdown": max_dd,
        "Sharpe": sharpe,
        "Profit factor": profit_factor,
        "Nb trades": n_trades,
    }


# ──────────────────────────────────────────────
# Stratégies
# ──────────────────────────────────────────────
def strategy_buy_and_hold(df: pd.DataFrame) -> pd.Series:
    """Toujours long."""
    return pd.Series(1, index=df.index)


def strategy_random(df: pd.DataFrame) -> pd.Series:
    """Signaux aléatoires BUY/SELL/HOLD uniformes."""
    rng = np.random.RandomState(SEED)
    return pd.Series(rng.choice([1, -1, 0], size=len(df)), index=df.index)


def strategy_rules(df: pd.DataFrame) -> pd.Series:
    """
    Règles fixes basées sur EMA crossover + RSI + ADX.
    - BUY  : ema_diff > 0, RSI < 70, ADX > 20
    - SELL : ema_diff < 0, RSI > 30, ADX > 20
    - HOLD : sinon
    """
    signals = pd.Series(0, index=df.index)

    buy_mask = (df["ema_diff"] > 0) & (df["rsi_14"] < 70) & (df["adx_14"] > 20)
    sell_mask = (df["ema_diff"] < 0) & (df["rsi_14"] > 30) & (df["adx_14"] > 20)

    signals[buy_mask] = 1
    signals[sell_mask] = -1

    return signals


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("=" * 70)
    print("PHASE 6 – Baseline obligatoire (backtest GBP/USD M15)")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Chargement
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp_15m"])
    df["year"] = df["timestamp_15m"].dt.year
    print(f"\n  Données chargées: {len(df)} bougies M15")

    # Split temporel
    splits = {
        "2022 (Train)": df[df["year"] == 2022].copy(),
        "2023 (Valid)": df[df["year"] == 2023].copy(),
        "2024 (Test)": df[df["year"] == 2024].copy(),
    }
    for name, subset in splits.items():
        print(f"  {name}: {len(subset)} bougies")

    strategies = {
        "Buy & Hold": strategy_buy_and_hold,
        "Random": strategy_random,
        "Règles (EMA+RSI+ADX)": strategy_rules,
    }

    all_metrics = []

    # Backtest par split et par stratégie
    for split_name, split_df in splits.items():
        print(f"\n{'─' * 70}")
        print(f"  {split_name}")
        print(f"{'─' * 70}")

        fig, ax = plt.subplots(figsize=(12, 5))

        for strat_name, strat_fn in strategies.items():
            signals = strat_fn(split_df)
            result = backtest(split_df, signals)
            metrics = compute_metrics(result, strat_name)
            metrics["Split"] = split_name
            all_metrics.append(metrics)

            ax.plot(
                split_df["timestamp_15m"].values,
                result["cumulative_pnl"].values,
                label=strat_name,
                linewidth=1.2,
            )

            print(f"  {strat_name:30s} | Profit: {metrics['Profit cumulé']:+.6f} | "
                  f"MaxDD: {metrics['Max drawdown']:.6f} | Sharpe: {metrics['Sharpe']:.3f} | "
                  f"PF: {metrics['Profit factor']:.3f} | Trades: {metrics['Nb trades']}")

        ax.set_title(f"PnL cumulé – {split_name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("PnL cumulé")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        safe_name = split_name.replace(" ", "_").replace("(", "").replace(")", "")
        fig.savefig(OUTPUT_DIR / f"baseline_{safe_name}.png", dpi=150)
        plt.close(fig)

    # Tableau récapitulatif
    print(f"\n{'=' * 70}")
    print("  TABLEAU RÉCAPITULATIF")
    print(f"{'=' * 70}")
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df = metrics_df[["Split", "Stratégie", "Profit cumulé", "Max drawdown", "Sharpe", "Profit factor", "Nb trades"]]
    print(metrics_df.to_string(index=False))

    # Sauvegarde CSV
    metrics_df.to_csv(OUTPUT_DIR / "baseline_metrics.csv", index=False)

    # Graphique comparatif global (toutes années combinées)
    fig, ax = plt.subplots(figsize=(14, 6))
    for strat_name, strat_fn in strategies.items():
        signals = strat_fn(df)
        result = backtest(df, signals)
        ax.plot(
            df["timestamp_15m"].values,
            result["cumulative_pnl"].values,
            label=strat_name,
            linewidth=1.2,
        )
    for year in [2023, 2024]:
        ax.axvline(pd.Timestamp(f"{year}-01-01"), color="gray", linestyle="--", alpha=0.5)
    ax.set_title("PnL cumulé – Toutes périodes (2022-2024)")
    ax.set_xlabel("Date")
    ax.set_ylabel("PnL cumulé")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "baseline_global.png", dpi=150)
    plt.close(fig)

    print(f"\n  Graphiques sauvegardés dans: {OUTPUT_DIR}")
    print("\n" + "=" * 70)
    print("Phase 6 terminée avec succès")
    print("=" * 70)


if __name__ == "__main__":
    main()
