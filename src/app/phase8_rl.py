"""
Phase 8 – Reinforcement Learning (T08)
Système de trading GBP/USD M15 avec DQN (Stable-Baselines3).

═══════════════════════════════════════════════════════════════
CONCEPTION (obligatoire avant codage)
═══════════════════════════════════════════════════════════════

1. Problème métier
   - Objectif : maximiser le profit cumulé sur GBP/USD M15
   - Contraintes : coûts de transaction (1 pip), drawdown limité
   - Horizon : épisode = 1 année de trading (~21k-24k steps)

2. Données
   - Source : data/features/gbpusd_m15_features.csv (19 features + OHLCV)
   - Qualité : nettoyé en Phase 3, features sans look-ahead (Phase 5)
   - Split : 2022 & 2023 train / 2024 valid / 2025 & 2026 test

3. State (observation)
   - 19 features techniques normalisées (z-score sur fenêtre train)
   - Position courante encodée : {-1, 0, 1}
   - Steps dans la position courante / 100 (normalise)
   - Dimension : 21

4. Action (discret)
   - 0 = HOLD (ne rien changer)
   - 1 = BUY  (position longue)
   - 2 = SELL (position courte)

5. Reward
   - PnL réalisé à chaque step : position × return_next - coût_si_trade
   - Pénalité drawdown : -λ × max(0, drawdown - seuil)

6. Environnement
   - Simulateur custom Gymnasium
   - Coût de transaction : 0.0001 (1 pip spread)
   - Pas de slippage (données M15 suffisamment agrégées)

7. Choix algorithme : DQN
   - Justification : actions discrètes (3), espace d'état continu modéré (20D),
     bon compromis sample efficiency / stabilité pour un premier modèle.
     MlpPolicy adaptée à la dimension du state.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "features" / "gbpusd_m15_features.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "reports" / "rl"
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

TRANSACTION_COST = 0.0001
DRAWDOWN_PENALTY = 0.5
DRAWDOWN_THRESHOLD = 0.02
REWARD_SCALE = 10000
POSITION_CHANGE_BONUS = 0.05
INACTIVITY_PENALTY = 0.01
INACTIVITY_THRESHOLD = 50
SEED = 42

FEATURE_COLS = [
    "return_1", "return_4", "ema_diff", "rsi_14", "rolling_std_20",
    "range_15m", "body", "upper_wick", "lower_wick",
    "distance_to_ema200", "slope_ema50",
    "atr_14", "rolling_std_100", "volatility_ratio",
    "adx_14", "macd", "macd_signal",
    "ema_20", "ema_50",
]


# ══════════════════════════════════════════════
# Environnement Gymnasium
# ══════════════════════════════════════════════
class TradingEnv(gym.Env):
    """Environnement de trading GBP/USD M15."""

    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame, feature_mean: pd.Series, feature_std: pd.Series):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.feature_mean = feature_mean
        self.feature_std = feature_std

        # Pré-calcul des returns (close_t+1 / close_t - 1)
        self.returns = self.df["close_15m"].pct_change().shift(-1).fillna(0).values
        self.closes = self.df["close_15m"].values

        # Normalisation des features
        features_raw = self.df[FEATURE_COLS].values
        std_safe = self.feature_std.values.copy()
        std_safe[std_safe == 0] = 1.0
        self.features_norm = (features_raw - self.feature_mean.values) / std_safe

        self.n_steps = len(self.df)

        # Spaces : 19 features + 1 position encoding + 1 steps_in_position = 21
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(FEATURE_COLS) + 2,), dtype=np.float32
        )
        # Actions : 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)

        self.current_step = 0
        self.position = 0  # -1, 0, 1
        self.steps_in_position = 0
        self.cumulative_pnl = 0.0
        self.peak_pnl = 0.0

    def _get_obs(self):
        obs = np.append(self.features_norm[self.current_step],
                        [self.position, self.steps_in_position / 100.0])
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0
        self.steps_in_position = 0
        self.cumulative_pnl = 0.0
        self.peak_pnl = 0.0
        return self._get_obs(), {}

    def step(self, action):
        # Mapper action → position cible
        action_map = {0: self.position, 1: 1, 2: -1}  # HOLD garde la position
        new_position = action_map[action]

        # Coût de transaction si changement de position
        cost = TRANSACTION_COST * abs(new_position - self.position)

        # Tracker changement de position
        position_changed = new_position != self.position
        if position_changed:
            self.steps_in_position = 0
        else:
            self.steps_in_position += 1

        self.position = new_position

        # PnL du step
        step_return = self.returns[self.current_step]
        pnl = self.position * step_return - cost

        # Tracking drawdown
        self.cumulative_pnl += pnl
        self.peak_pnl = max(self.peak_pnl, self.cumulative_pnl)
        drawdown = self.peak_pnl - self.cumulative_pnl

        # Reward shaping : PnL scalé + bonus/pénalités
        scaled_pnl = pnl * REWARD_SCALE
        reward = scaled_pnl

        # Bonus pour changement de position (évite politique dégénérée)
        if position_changed:
            reward += POSITION_CHANGE_BONUS

        # Pénalité d'inactivité (même position trop longtemps)
        if self.steps_in_position > INACTIVITY_THRESHOLD:
            reward -= INACTIVITY_PENALTY

        # Pénalité drawdown (sur drawdown brut, non scalé)
        if drawdown > DRAWDOWN_THRESHOLD:
            reward -= DRAWDOWN_PENALTY * (drawdown - DRAWDOWN_THRESHOLD) * REWARD_SCALE

        self.current_step += 1
        terminated = self.current_step >= self.n_steps - 1
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {
            "pnl": pnl,
            "cumulative_pnl": self.cumulative_pnl,
            "drawdown": drawdown,
            "position": self.position,
            "steps_in_position": self.steps_in_position,
        }


# ══════════════════════════════════════════════
# Callback pour logging
# ══════════════════════════════════════════════
class PnLCallback(BaseCallback):
    """Log le PnL cumulé à la fin de chaque épisode."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_pnls = []

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_pnls.append(info.get("cumulative_pnl", 0))
        return True


# ══════════════════════════════════════════════
# Backtest (réutilisable)
# ══════════════════════════════════════════════
def run_backtest(model, env: TradingEnv) -> dict:
    """Exécute un backtest complet et retourne les métriques."""
    obs, _ = env.reset()
    pnls = []
    positions = []
    cumulative = []
    trades = 0
    prev_pos = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        pnls.append(info["pnl"])
        positions.append(info["position"])
        cumulative.append(info["cumulative_pnl"])
        if info["position"] != prev_pos:
            trades += 1
        prev_pos = info["position"]
        if terminated or truncated:
            break

    pnls = np.array(pnls)
    cumulative = np.array(cumulative)

    # Métriques
    total_profit = cumulative[-1]
    running_max = np.maximum.accumulate(cumulative)
    max_dd = np.min(cumulative - running_max)

    if pnls.std() > 0:
        sharpe = (pnls.mean() / pnls.std()) * np.sqrt(96 * 252)
    else:
        sharpe = 0.0

    gains = pnls[pnls > 0].sum()
    losses = abs(pnls[pnls < 0].sum())
    pf = gains / losses if losses > 0 else np.inf

    # Distribution des positions
    pos_arr = np.array(positions)
    pct_long = (pos_arr == 1).mean() * 100
    pct_short = (pos_arr == -1).mean() * 100
    pct_flat = (pos_arr == 0).mean() * 100

    return {
        "total_profit": total_profit,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "profit_factor": pf,
        "n_trades": trades,
        "pct_long": pct_long,
        "pct_short": pct_short,
        "pct_flat": pct_flat,
        "cumulative": cumulative,
        "positions": pos_arr,
        "timestamps": env.df["timestamp_15m"].values[:len(cumulative)],
    }


def print_metrics(label: str, m: dict):
    print(f"  {label}")
    print(f"    Profit cumulé:  {m['total_profit']:+.6f}")
    print(f"    Max drawdown:   {m['max_drawdown']:.6f}")
    print(f"    Sharpe:         {m['sharpe']:.3f}")
    print(f"    Profit factor:  {m['profit_factor']:.3f}")
    print(f"    Nb trades:      {m['n_trades']}")
    print(f"    Positions:      Long {m['pct_long']:.1f}% | Short {m['pct_short']:.1f}% | Flat {m['pct_flat']:.1f}%")


# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════
def main():
    print("=" * 70)
    print("PHASE 8 – Reinforcement Learning (DQN) GBP/USD M15")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Chargement des données ──
    print("\n[1] Chargement des données...")
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp_15m"])
    df["year"] = df["timestamp_15m"].dt.year

    df_train = df[(df["year"] == 2022) | (df["year"] == 2023)].copy()
    df_valid = df[df["year"] == 2024].copy()
    df_test = df[(df["year"] == 2025) | (df["year"] == 2026)].copy()

    print(f"  Train (2022 & 2023): {len(df_train)} steps")
    print(f"  Valid (2024): {len(df_valid)} steps")
    print(f"  Test  (2025 & 2026): {len(df_test)} steps")

    # ── Normalisation (calculée uniquement sur train) ──
    print("\n[2] Normalisation des features (z-score sur 2022)...")
    feature_mean = df_train[FEATURE_COLS].mean()
    feature_std = df_train[FEATURE_COLS].std()

    # ── Création des environnements ──
    print("\n[3] Création des environnements...")
    env_train = TradingEnv(df_train, feature_mean, feature_std)
    env_valid = TradingEnv(df_valid, feature_mean, feature_std)
    env_test = TradingEnv(df_test, feature_mean, feature_std)

    # ── Paramètres DQN ──
    print("\n[4] Configuration DQN...")
    dqn_params = {
        "policy": "MlpPolicy",
        "env": env_train,
        "learning_rate": 1e-4,
        "buffer_size": 50_000,
        "learning_starts": 1_000,
        "batch_size": 64,
        "gamma": 0.99,
        "exploration_fraction": 0.3,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "target_update_interval": 500,
        "train_freq": 4,
        "policy_kwargs": {"net_arch": [128, 128]},
        "seed": SEED,
        "verbose": 0,
    }

    print("  Paramètres:")
    for k, v in dqn_params.items():
        if k not in ("policy", "env"):
            print(f"    {k}: {v}")

    # ── Entraînement ──
    print("\n[5] Entraînement DQN...")
    model = DQN(**dqn_params)

    total_timesteps = len(df_train) * 5  # ~5 épisodes
    print(f"  Total timesteps: {total_timesteps}")

    callback = PnLCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)

    # Sauvegarde modèle
    model_path = MODEL_DIR / "v1" / "dqn_gbpusd_m15"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    print(f"  Modèle sauvegardé: {model_path}")

    # ── Évaluation ──
    print("\n[6] Évaluation...")
    results = {}
    for name, env in [("2022 & 2023 (Train)", env_train), ("2024 (Valid)", env_valid), ("2025 & 2026 (Test)", env_test)]:
        results[name] = run_backtest(model, env)
        print_metrics(name, results[name])
        print()

    # ── Graphiques ──
    print("[7] Génération des graphiques...")

    # PnL cumulé par période
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, m) in zip(axes, results.items()):
        ax.plot(m["timestamps"], m["cumulative"], linewidth=0.8, color="steelblue")
        ax.set_title(f"PnL cumulé DQN – {name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("PnL cumulé")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "rl_pnl_by_period.png", dpi=150)
    plt.close(fig)

    # PnL global
    fig, ax = plt.subplots(figsize=(14, 6))
    for name, m in results.items():
        ax.plot(m["timestamps"], m["cumulative"], label=name, linewidth=0.8)
    ax.set_title("PnL cumulé DQN – Toutes périodes")
    ax.set_xlabel("Date")
    ax.set_ylabel("PnL cumulé")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "rl_pnl_global.png", dpi=150)
    plt.close(fig)

    # Distribution des positions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, m) in zip(axes, results.items()):
        labels = ["Short", "Flat", "Long"]
        values = [m["pct_short"], m["pct_flat"], m["pct_long"]]
        colors = ["#e74c3c", "#95a5a6", "#2ecc71"]
        ax.bar(labels, values, color=colors)
        ax.set_title(f"Positions – {name}")
        ax.set_ylabel("%")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "rl_positions.png", dpi=150)
    plt.close(fig)

    # ── Tableau récapitulatif ──
    print("\n" + "=" * 70)
    print("  TABLEAU RÉCAPITULATIF DQN")
    print("=" * 70)
    rows = []
    for name, m in results.items():
        rows.append({
            "Split": name,
            "Profit cumulé": m["total_profit"],
            "Max DD": m["max_drawdown"],
            "Sharpe": m["sharpe"],
            "Profit Factor": m["profit_factor"],
            "Trades": m["n_trades"],
            "% Long": m["pct_long"],
            "% Short": m["pct_short"],
        })
    metrics_df = pd.DataFrame(rows)
    print(metrics_df.to_string(index=False))
    metrics_df.to_csv(OUTPUT_DIR / "rl_metrics.csv", index=False)

    print(f"\n  Graphiques sauvegardés dans: {OUTPUT_DIR}")
    print(f"  Modèle sauvegardé dans: {model_path}")

    print("\n" + "=" * 70)
    print("Phase 8 terminée avec succès")
    print("=" * 70)


if __name__ == "__main__":
    main()
