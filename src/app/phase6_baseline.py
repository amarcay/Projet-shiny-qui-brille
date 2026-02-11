"""
Phase 6 – Baseline obligatoire + Backtest simple
Avant ML ou RL, on etablit 3 baselines :
  1. Buy & Hold
  2. Strategie aleatoire
  3. Strategie regles fixes (EMA crossover + RSI)

Split temporel strict :
  - 2022 : Entrainement (calibration regles)
  - 2023 : Validation
  - 2024 : Test final
"""

import pandas as pd
import numpy as np
from pathlib import Path

INPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "features"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "evaluation"

TRANSACTION_COST = 0.0001  # 1 pip spread
SEED = 42


def load_features() -> pd.DataFrame:
    """Charge les donnees M15 avec features."""
    path = INPUT_DIR / "gbpusd_m15_features.csv"
    df = pd.read_csv(path, parse_dates=["timestamp_15m"])
    print(f"  Charge: {len(df)} bougies")
    return df


def split_by_year(df: pd.DataFrame) -> dict:
    """Split temporel strict par annee."""
    df["year"] = df["timestamp_15m"].dt.year
    splits = {
        "train": df[df["year"] == 2022].copy(),
        "val": df[df["year"] == 2023].copy(),
        "test": df[df["year"] == 2024].copy(),
    }
    df.drop(columns=["year"], inplace=True)
    for s in splits.values():
        s.drop(columns=["year"], inplace=True)
    return splits


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def backtest(df: pd.DataFrame, signals: pd.Series, cost: float = TRANSACTION_COST) -> pd.DataFrame:
    """
    Backtest simple sur signaux {1=BUY, -1=SELL, 0=HOLD}.
    Rendement = signal_t * return_{t+1} - cost si changement de position.
    """
    returns = df["close_15m"].pct_change().shift(-1)
    position_changes = signals.diff().abs().fillna(0)
    costs = position_changes * cost

    pnl = signals * returns - costs
    pnl = pnl.fillna(0)

    result = pd.DataFrame({
        "timestamp_15m": df["timestamp_15m"],
        "signal": signals,
        "return": returns,
        "pnl": pnl,
        "cumulative_pnl": pnl.cumsum(),
    })
    return result


def compute_metrics(result: pd.DataFrame, name: str) -> dict:
    """Calcule les metriques d'evaluation."""
    pnl = result["pnl"]
    cum = result["cumulative_pnl"]

    total_pnl = cum.iloc[-1]
    n_trades = (result["signal"].diff().abs() > 0).sum()

    # Maximum drawdown
    peak = cum.cummax()
    drawdown = cum - peak
    max_dd = drawdown.min()

    # Sharpe simplifie (annualise sur base 15min)
    periods_per_year = 4 * 24 * 252  # 4 par heure * 24h * 252 jours
    mean_pnl = pnl.mean()
    std_pnl = pnl.std()
    sharpe = (mean_pnl / std_pnl) * np.sqrt(periods_per_year) if std_pnl > 0 else 0

    # Profit factor
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = pnl[pnl < 0].sum()
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else np.inf

    return {
        "strategy": name,
        "total_pnl": total_pnl,
        "n_trades": int(n_trades),
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "profit_factor": profit_factor,
    }


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def strategy_buy_and_hold(df: pd.DataFrame) -> pd.Series:
    """Buy & Hold : toujours long."""
    return pd.Series(1, index=df.index)


def strategy_random(df: pd.DataFrame, seed: int = SEED) -> pd.Series:
    """Strategie aleatoire : BUY/SELL/HOLD equiprobable."""
    rng = np.random.RandomState(seed)
    signals = rng.choice([1, -1, 0], size=len(df))
    return pd.Series(signals, index=df.index)


def strategy_rules(df: pd.DataFrame) -> pd.Series:
    """
    Strategie regles fixes : EMA crossover + RSI.
    - BUY  si ema_diff > 0 et rsi_14 < 70
    - SELL si ema_diff < 0 et rsi_14 > 30
    - HOLD sinon
    """
    signals = pd.Series(0, index=df.index)

    buy_mask = (df["ema_diff"] > 0) & (df["rsi_14"] < 70)
    sell_mask = (df["ema_diff"] < 0) & (df["rsi_14"] > 30)

    signals[buy_mask] = 1
    signals[sell_mask] = -1

    return signals


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("PHASE 6 – Baseline + Backtest GBP/USD")
    print("=" * 60)

    # 1. Chargement
    print("\n[1] Chargement des features...")
    df = load_features()

    # 2. Split temporel
    print("\n[2] Split temporel strict...")
    splits = split_by_year(df)
    for name, data in splits.items():
        print(f"  {name}: {len(data)} bougies ({data['timestamp_15m'].dt.year.iloc[0]})")

    # 3. Strategies
    strategies = {
        "Buy & Hold": strategy_buy_and_hold,
        "Random": strategy_random,
        "Regles (EMA+RSI)": strategy_rules,
    }

    # 4. Backtest sur chaque split
    all_metrics = []

    for split_name, split_df in splits.items():
        print(f"\n{'='*60}")
        print(f"  Backtest sur {split_name.upper()} ({split_df['timestamp_15m'].dt.year.iloc[0]})")
        print(f"{'='*60}")

        for strat_name, strat_fn in strategies.items():
            signals = strat_fn(split_df)
            result = backtest(split_df, signals)
            metrics = compute_metrics(result, strat_name)
            metrics["split"] = split_name
            all_metrics.append(metrics)

            print(f"\n  {strat_name}:")
            print(f"    PnL cumule:     {metrics['total_pnl']:+.6f}")
            print(f"    Max drawdown:   {metrics['max_drawdown']:.6f}")
            print(f"    Sharpe:         {metrics['sharpe']:.4f}")
            print(f"    Profit factor:  {metrics['profit_factor']:.4f}")
            print(f"    Nb changements: {metrics['n_trades']}")

    # 5. Tableau recapitulatif
    print(f"\n{'='*60}")
    print("  TABLEAU RECAPITULATIF")
    print(f"{'='*60}")
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics = df_metrics[["split", "strategy", "total_pnl", "max_drawdown", "sharpe", "profit_factor", "n_trades"]]
    print(df_metrics.to_string(index=False))

    # 6. Sauvegarde
    print(f"\n[6] Sauvegarde...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "baseline_metrics.csv"
    df_metrics.to_csv(output_path, index=False)
    print(f"  Sauvegarde: {output_path}")

    print("\n" + "=" * 60)
    print("Phase 6 terminee avec succes")
    print("=" * 60)


if __name__ == "__main__":
    main()
