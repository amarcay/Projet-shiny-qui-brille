"""
Phase 9 – Évaluation finale (T09)
Comparaison obligatoire de toutes les stratégies :
  - Random
  - Règles fixes (EMA+RSI+ADX)
  - ML (meilleur modèle Phase 7)
  - RL (DQN Phase 8)

Métriques :
  - Profit cumulé
  - Maximum drawdown
  - Sharpe simplifié
  - Profit factor

Un modèle est valide uniquement s'il est robuste sur 2024.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import DQN
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "features" / "gbpusd_m15_features.csv"
MODEL_RL_PATH = Path(__file__).resolve().parents[2] / "models" / "v1" / "dqn_gbpusd_m15"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "reports" / "evaluation"

TRANSACTION_COST = 0.0001
SEED = 42

FEATURE_COLS = [
    "return_1", "return_4", "ema_diff", "rsi_14", "rolling_std_20",
    "range_15m", "body", "upper_wick", "lower_wick",
    "distance_to_ema200", "slope_ema50",
    "atr_14", "rolling_std_100", "volatility_ratio",
    "adx_14", "macd", "macd_signal",
]

# Features RL (inclut ema_20/ema_50 en plus)
RL_FEATURE_COLS = FEATURE_COLS + ["ema_20", "ema_50"]


# ══════════════════════════════════════════════
# Moteur de backtest unifié
# ══════════════════════════════════════════════
def backtest(df: pd.DataFrame, signals: np.ndarray) -> dict:
    """
    Backtest unifié. signals: array de {-1, 0, 1}.
    Retourne métriques + courbes.
    """
    returns = df["close_15m"].pct_change().shift(-1).fillna(0).values
    positions = pd.Series(signals).replace(0, np.nan).ffill().fillna(0).values

    trade_mask = np.abs(np.diff(positions, prepend=0))
    costs = trade_mask * TRANSACTION_COST

    pnl = positions * returns - costs
    cumulative = np.cumsum(pnl)

    # Métriques
    total_profit = cumulative[-1]
    running_max = np.maximum.accumulate(cumulative)
    max_dd = np.min(cumulative - running_max)

    sharpe = (pnl.mean() / pnl.std()) * np.sqrt(96 * 252) if pnl.std() > 0 else 0.0

    gains = pnl[pnl > 0].sum()
    losses = abs(pnl[pnl < 0].sum())
    pf = gains / losses if losses > 0 else np.inf

    n_trades = int((trade_mask > 0).sum())

    return {
        "total_profit": total_profit,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "profit_factor": pf,
        "n_trades": n_trades,
        "cumulative": cumulative,
        "pnl": pnl,
    }


# ══════════════════════════════════════════════
# Stratégies
# ══════════════════════════════════════════════
def strategy_random(n: int) -> np.ndarray:
    rng = np.random.RandomState(SEED)
    return rng.choice([1, -1, 0], size=n)


def strategy_buy_hold(n: int) -> np.ndarray:
    return np.ones(n)


def strategy_rules(df: pd.DataFrame) -> np.ndarray:
    signals = np.zeros(len(df))
    buy = (df["ema_diff"] > 0) & (df["rsi_14"] < 70) & (df["adx_14"] > 20)
    sell = (df["ema_diff"] < 0) & (df["rsi_14"] > 30) & (df["adx_14"] > 20)
    signals[buy.values] = 1
    signals[sell.values] = -1
    return signals


def strategy_ml(df_train: pd.DataFrame, df_eval: pd.DataFrame) -> np.ndarray:
    """Entraîne Gradient Boosting et génère les signaux."""
    # Target
    y_train = (df_train["close_15m"].shift(-1) > df_train["close_15m"]).astype(int).values[:-1]
    X_train = df_train[FEATURE_COLS].values[:-1]
    X_eval = df_eval[FEATURE_COLS].values

    model = HistGradientBoostingClassifier(
        max_iter=300, max_depth=6, min_samples_leaf=50,
        learning_rate=0.05, random_state=SEED,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_eval)
    # 1 → BUY, 0 → SELL
    signals = np.where(preds == 1, 1, -1)
    return signals


def strategy_rl(df_eval: pd.DataFrame, feature_mean: pd.Series, feature_std: pd.Series) -> np.ndarray:
    """Charge le modèle DQN et génère les signaux."""
    model = DQN.load(str(MODEL_RL_PATH))

    features_raw = df_eval[RL_FEATURE_COLS].values
    std_safe = feature_std.values.copy()
    std_safe[std_safe == 0] = 1.0
    features_norm = (features_raw - feature_mean.values) / std_safe

    signals = np.zeros(len(df_eval))
    position = 0

    for i in range(len(df_eval)):
        obs = np.append(features_norm[i], position).astype(np.float32)
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        if action == 1:
            position = 1
        elif action == 2:
            position = -1
        # action 0 → HOLD (garde la position)
        signals[i] = position

    return signals


# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════
def main():
    print("=" * 70)
    print("PHASE 9 – Évaluation finale (comparaison toutes stratégies)")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Chargement
    print("\n[1] Chargement des données...")
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp_15m"])
    df["year"] = df["timestamp_15m"].dt.year

    df_train = df[(df["year"] == 2022) | (df["year"] == 2023)].copy()
    df_valid = df[df["year"] == 2024].copy()
    df_test = df[(df["year"] == 2025) | (df["year"] == 2026)].copy()

    splits = {
        "2022 & 2023 (Train)": df_train,
        "2024 (Valid)": df_valid,
        "2025 & 2026 (Test)": df_test,
    }

    for name, s in splits.items():
        print(f"  {name}: {len(s)} bougies")

    # Normalisation RL (sur train uniquement)
    rl_feature_mean = df_train[RL_FEATURE_COLS].mean()
    rl_feature_std = df_train[RL_FEATURE_COLS].std()

    # ── Génération des signaux par stratégie et par split ──
    print("\n[2] Génération des signaux...")

    all_results = []

    for split_name, split_df in splits.items():
        print(f"\n  {'─' * 60}")
        print(f"  {split_name}")
        print(f"  {'─' * 60}")

        strategies = {}

        # Random
        strategies["Random"] = strategy_random(len(split_df))

        # Buy & Hold
        strategies["Buy & Hold"] = strategy_buy_hold(len(split_df))

        # Règles fixes
        strategies["Règles (EMA+RSI+ADX)"] = strategy_rules(split_df)

        # ML (Gradient Boosting)
        strategies["ML (Gradient Boosting)"] = strategy_ml(df_train, split_df)

        # RL (DQN)
        strategies["RL (DQN)"] = strategy_rl(split_df, rl_feature_mean, rl_feature_std)

        # Backtest chaque stratégie
        split_results = {}
        for strat_name, signals in strategies.items():
            m = backtest(split_df, signals)
            split_results[strat_name] = m

            all_results.append({
                "Split": split_name,
                "Stratégie": strat_name,
                "Profit cumulé": m["total_profit"],
                "Max drawdown": m["max_drawdown"],
                "Sharpe": m["sharpe"],
                "Profit factor": m["profit_factor"],
                "Nb trades": m["n_trades"],
            })

            print(f"    {strat_name:30s} | Profit: {m['total_profit']:+.6f} | "
                  f"MaxDD: {m['max_drawdown']:.6f} | Sharpe: {m['sharpe']:+.3f} | "
                  f"PF: {m['profit_factor']:.3f} | Trades: {m['n_trades']}")

        # ── Graphique PnL cumulé pour ce split ──
        fig, ax = plt.subplots(figsize=(14, 6))
        for strat_name, m in split_results.items():
            ax.plot(
                split_df["timestamp_15m"].values,
                m["cumulative"],
                label=strat_name,
                linewidth=1.0,
            )
        ax.set_title(f"PnL cumulé – {split_name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("PnL cumulé")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.5)
        fig.tight_layout()
        safe = split_name.replace(" ", "_").replace("(", "").replace(")", "")
        fig.savefig(OUTPUT_DIR / f"eval_{safe}.png", dpi=150)
        plt.close(fig)

    # ── Tableau récapitulatif complet ──
    print(f"\n{'=' * 70}")
    print("  TABLEAU RÉCAPITULATIF – TOUTES STRATÉGIES / TOUS SPLITS")
    print(f"{'=' * 70}")
    results_df = pd.DataFrame(all_results)
    print(results_df.to_string(index=False))

    # ── Focus 2025 & 2026 (Test) : robustesse ──
    print(f"\n{'=' * 70}")
    print("  FOCUS 2025 & 2026 (TEST) – ROBUSTESSE")
    print(f"{'=' * 70}")
    test_df = results_df[results_df["Split"] == "2025 & 2026 (Test)"].copy()
    test_df = test_df.sort_values("Sharpe", ascending=False)
    print(test_df[["Stratégie", "Profit cumulé", "Max drawdown", "Sharpe", "Profit factor"]].to_string(index=False))

    # Verdict
    best = test_df.iloc[0]
    print(f"\n  Meilleure stratégie sur 2025 & 2026 (par Sharpe): {best['Stratégie']}")
    print(f"    Profit: {best['Profit cumulé']:+.6f}  |  Sharpe: {best['Sharpe']:+.3f}  |  MaxDD: {best['Max drawdown']:.6f}")

    # Validité : Sharpe > 0 et Profit > 0
    valid_models = test_df[(test_df["Sharpe"] > 0) & (test_df["Profit cumulé"] > 0)]
    if len(valid_models) > 0:
        print(f"\n  Modèles VALIDES sur 2025 & 2026 (Sharpe > 0 & Profit > 0):")
        for _, row in valid_models.iterrows():
            print(f"    ✓ {row['Stratégie']}")
    else:
        print(f"\n  ⚠ Aucun modèle n'est robuste sur 2025 & 2026 (tous Sharpe ≤ 0 ou Profit ≤ 0)")
        print("    → Pistes d'amélioration : plus de features, tuning hyperparamètres,")
        print("      reward shaping RL, walk-forward, ensemble methods")

    # ── Graphique global comparatif ──
    print("\n[3] Graphique global...")

    # Recalculer sur données complètes (2022-2024 enchaînées)
    fig, ax = plt.subplots(figsize=(16, 7))

    strat_full = {
        "Random": strategy_random(len(df)),
        "Buy & Hold": strategy_buy_hold(len(df)),
        "Règles (EMA+RSI+ADX)": strategy_rules(df),
        "ML (Gradient Boosting)": strategy_ml(df_train, df),
        "RL (DQN)": strategy_rl(df, rl_feature_mean, rl_feature_std),
    }

    for strat_name, signals in strat_full.items():
        m = backtest(df, signals)
        ax.plot(df["timestamp_15m"].values, m["cumulative"], label=strat_name, linewidth=1.0)

    for year in [2025, 2026]:
        ax.axvline(pd.Timestamp(f"{year}-01-01"), color="gray", linestyle="--", alpha=0.6, linewidth=0.8)
    ax.text(pd.Timestamp("2022-06-01"), ax.get_ylim()[1] * 0.95, "TRAIN", fontsize=10, color="gray")
    ax.text(pd.Timestamp("2025-06-01"), ax.get_ylim()[1] * 0.95, "VALID", fontsize=10, color="gray")
    ax.text(pd.Timestamp("2026-06-01"), ax.get_ylim()[1] * 0.95, "TEST", fontsize=10, color="gray")

    ax.set_title("Évaluation finale – PnL cumulé toutes stratégies (2022-2024)", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("PnL cumulé")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "eval_global.png", dpi=150)
    plt.close(fig)

    # ── Heatmap résumé ──
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot = results_df.pivot(index="Stratégie", columns="Split", values="Sharpe")
    pivot = pivot[["2022 & 2023 (Train)", "2024 (Valid)", "2025 & 2026 (Test)"]]
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if abs(pivot.values[i, j]) > 10 else "black")
    ax.set_title("Sharpe ratio par stratégie et période")
    fig.colorbar(im, ax=ax, label="Sharpe")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "eval_heatmap_sharpe.png", dpi=150)
    plt.close(fig)

    # Sauvegarde CSV
    results_df.to_csv(OUTPUT_DIR / "evaluation_metrics.csv", index=False)

    print(f"\n  Graphiques et métriques sauvegardés dans: {OUTPUT_DIR}")
    print("\n" + "=" * 70)
    print("Phase 9 terminée avec succès")
    print("=" * 70)


if __name__ == "__main__":
    main()
