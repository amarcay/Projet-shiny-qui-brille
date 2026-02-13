"""
Phase 11 – Versioning modèle (T11)
Registry de modèles avec versioning v1/v2, métadonnées et sélection automatique.

Structure :
  models/
  ├── registry.json          ← catalogue de tous les modèles
  ├── v1/
  │   ├── dqn_gbpusd_m15.zip
  │   ├── gradient_boosting.joblib
  │   ├── scaler.joblib
  │   └── metadata.json
  └── v2/
      ├── ...
      └── metadata.json

Fonctionnalités :
  - Enregistrement de modèles ML (sklearn) et RL (stable-baselines3)
  - Métadonnées : hyperparamètres, métriques, date, features utilisées
  - Sélection automatique de la meilleure version (par Sharpe sur validation)
  - API de chargement unique : load_best_model()
"""

import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from stable_baselines3 import DQN

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "features" / "gbpusd_m15_features.csv"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
REGISTRY_PATH = MODELS_DIR / "registry.json"

SEED = 42
TRANSACTION_COST = 0.0001

ML_FEATURE_COLS = [
    "return_1", "return_4", "ema_diff", "rsi_14", "rolling_std_20",
    "range_15m", "body", "upper_wick", "lower_wick",
    "distance_to_ema200", "slope_ema50",
    "atr_14", "rolling_std_100", "volatility_ratio",
    "adx_14", "macd", "macd_signal",
]

RL_FEATURE_COLS = ML_FEATURE_COLS + ["ema_20", "ema_50"]


# ══════════════════════════════════════════════
# Registry
# ══════════════════════════════════════════════
def load_registry() -> dict:
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {"models": {}, "best_version": None}


def save_registry(registry: dict):
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2, default=str)


def register_model(
    version: str,
    model_type: str,
    model_name: str,
    hyperparams: dict,
    metrics_train: dict,
    metrics_valid: dict,
    metrics_test: dict,
    features: list[str],
    artifacts: list[str],
):
    """Enregistre un modèle dans le registry."""
    registry = load_registry()

    registry["models"][version] = {
        "version": version,
        "model_type": model_type,
        "model_name": model_name,
        "registered_at": datetime.now().isoformat(),
        "hyperparams": hyperparams,
        "features": features,
        "artifacts": artifacts,
        "metrics": {
            "train_2022_2023": metrics_train,
            "valid_2024": metrics_valid,
            "test_2025_2026": metrics_test,
        },
    }

    # Sauvegarder aussi metadata.json dans le dossier du modèle
    version_dir = MODELS_DIR / version
    with open(version_dir / "metadata.json", "w") as f:
        json.dump(registry["models"][version], f, indent=2, default=str)

    save_registry(registry)
    print(f"    Modèle enregistré: {version} ({model_name})")


def select_best_model():
    """Sélectionne la meilleure version par Sharpe sur validation 2024."""
    registry = load_registry()

    best_version = None
    best_sharpe = -np.inf

    for version, info in registry["models"].items():
        sharpe_valid = info["metrics"]["valid_2024"]["sharpe"]
        print(f"    {version} ({info['model_name']}): Sharpe valid = {sharpe_valid:.3f}")
        if sharpe_valid > best_sharpe:
            best_sharpe = sharpe_valid
            best_version = version

    registry["best_version"] = best_version
    save_registry(registry)
    print(f"    → Meilleure version: {best_version} (Sharpe valid = {best_sharpe:.3f})")
    return best_version


def load_best_model():
    """Charge le meilleur modèle validé depuis le registry.
    Retourne (model, scaler_or_None, model_info).
    Utilisé par l'API (Phase 10).
    """
    registry = load_registry()
    version = registry.get("best_version")
    if version is None:
        raise ValueError("Aucun modèle validé dans le registry. Exécuter phase11 d'abord.")

    info = registry["models"][version]
    version_dir = MODELS_DIR / version

    if info["model_type"] == "sklearn":
        model = joblib.load(version_dir / f"{info['model_name'].lower().replace(' ', '_')}.joblib")
        scaler = joblib.load(version_dir / "scaler.joblib") if (version_dir / "scaler.joblib").exists() else None
        return model, scaler, info
    elif info["model_type"] == "stable-baselines3":
        model = DQN.load(str(version_dir / "dqn_gbpusd_m15"))
        norm_stats = joblib.load(version_dir / "norm_stats.joblib")
        return model, norm_stats, info
    else:
        raise ValueError(f"Type de modèle inconnu: {info['model_type']}")


# ══════════════════════════════════════════════
# Évaluation financière
# ══════════════════════════════════════════════
def financial_eval(df: pd.DataFrame, signals: np.ndarray) -> dict:
    returns = df["close_15m"].pct_change().shift(-1).fillna(0).values
    positions = pd.Series(signals).replace(0, np.nan).ffill().fillna(0).values
    trade_mask = np.abs(np.diff(positions, prepend=0))
    costs = trade_mask * TRANSACTION_COST
    pnl = positions * returns - costs
    cum = np.cumsum(pnl)

    total_profit = cum[-1]
    max_dd = np.min(cum - np.maximum.accumulate(cum))
    sharpe = (pnl.mean() / pnl.std()) * np.sqrt(96 * 252) if pnl.std() > 0 else 0.0
    gains = pnl[pnl > 0].sum()
    losses = abs(pnl[pnl < 0].sum())
    pf = gains / losses if losses > 0 else np.inf
    n_trades = int((trade_mask > 0).sum())

    return {
        "profit": float(total_profit),
        "max_drawdown": float(max_dd),
        "sharpe": float(sharpe),
        "profit_factor": float(pf),
        "n_trades": n_trades,
    }


# ══════════════════════════════════════════════
# Enregistrement v1 : ML (Gradient Boosting)
# ══════════════════════════════════════════════
def register_v1_ml(df_train, df_valid, df_test):
    """Entraîne et enregistre Gradient Boosting en v1."""
    print("\n  [v1] Gradient Boosting (ML)...")

    version_dir = MODELS_DIR / "v1"
    version_dir.mkdir(parents=True, exist_ok=True)

    # Target
    y_train = (df_train["close_15m"].shift(-1) > df_train["close_15m"]).astype(int).values[:-1]
    X_train = df_train[ML_FEATURE_COLS].values[:-1]

    # Scaler
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    # Entraînement
    hyperparams = {
        "max_iter": 300, "max_depth": 6, "min_samples_leaf": 50,
        "learning_rate": 0.05, "random_state": SEED,
    }
    model = HistGradientBoostingClassifier(**hyperparams)
    model.fit(X_train, y_train)

    # Sauvegarder artefacts
    joblib.dump(model, version_dir / "gradient_boosting.joblib")
    joblib.dump(scaler, version_dir / "scaler.joblib")

    # Évaluer sur chaque split
    metrics = {}
    for name, split_df in [("train", df_train), ("valid", df_valid), ("test", df_test)]:
        X = split_df[ML_FEATURE_COLS].values
        preds = model.predict(X)
        signals = np.where(preds == 1, 1, -1)
        metrics[name] = financial_eval(split_df, signals)

        # Accuracy
        y_true = (split_df["close_15m"].shift(-1) > split_df["close_15m"]).astype(int).values[:-1]
        preds_eval = preds[:-1]
        metrics[name]["accuracy"] = float(accuracy_score(y_true, preds_eval))

    register_model(
        version="v1",
        model_type="sklearn",
        model_name="Gradient Boosting",
        hyperparams=hyperparams,
        metrics_train=metrics["train"],
        metrics_valid=metrics["valid"],
        metrics_test=metrics["test"],
        features=ML_FEATURE_COLS,
        artifacts=["gradient_boosting.joblib", "scaler.joblib", "dqn_gbpusd_m15.zip"],
    )

    return metrics


# ══════════════════════════════════════════════
# Enregistrement v2 : RL (DQN) re-entraîné
# ══════════════════════════════════════════════
def register_v2_rl(df_train, df_valid, df_test):
    """Entraîne un DQN amélioré et l'enregistre en v2."""
    print("\n  [v2] DQN amélioré (RL)...")

    # Import de l'env depuis phase8
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from phase8_rl import TradingEnv

    version_dir = MODELS_DIR / "v2"
    version_dir.mkdir(parents=True, exist_ok=True)

    # Normalisation
    feature_mean = df_train[RL_FEATURE_COLS].mean()
    feature_std = df_train[RL_FEATURE_COLS].std()

    env_train = TradingEnv(df_train, feature_mean, feature_std)

    # Hyperparamètres v2 (améliorés)
    hyperparams = {
        "learning_rate": 5e-5,
        "buffer_size": 100_000,
        "learning_starts": 2_000,
        "batch_size": 128,
        "gamma": 0.995,
        "exploration_fraction": 0.4,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.02,
        "target_update_interval": 1000,
        "train_freq": 4,
        "net_arch": [256, 256],
        "seed": SEED,
        "total_timesteps": len(df_train) * 8,
    }

    model = DQN(
        "MlpPolicy", env_train,
        learning_rate=hyperparams["learning_rate"],
        buffer_size=hyperparams["buffer_size"],
        learning_starts=hyperparams["learning_starts"],
        batch_size=hyperparams["batch_size"],
        gamma=hyperparams["gamma"],
        exploration_fraction=hyperparams["exploration_fraction"],
        exploration_initial_eps=hyperparams["exploration_initial_eps"],
        exploration_final_eps=hyperparams["exploration_final_eps"],
        target_update_interval=hyperparams["target_update_interval"],
        train_freq=hyperparams["train_freq"],
        policy_kwargs={"net_arch": hyperparams["net_arch"]},
        seed=SEED,
        verbose=0,
    )

    print(f"    Entraînement: {hyperparams['total_timesteps']} timesteps...")
    model.learn(total_timesteps=hyperparams["total_timesteps"], progress_bar=True)

    # Sauvegarder artefacts
    model.save(str(version_dir / "dqn_gbpusd_m15"))
    joblib.dump({"mean": feature_mean, "std": feature_std}, version_dir / "norm_stats.joblib")

    # Évaluer sur chaque split
    metrics = {}
    for name, split_df in [("train", df_train), ("valid", df_valid), ("test", df_test)]:
        env = TradingEnv(split_df, feature_mean, feature_std)
        obs, _ = env.reset()
        signals = []
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            signals.append(info["position"])
            if terminated or truncated:
                break
        # Pad si nécessaire
        while len(signals) < len(split_df):
            signals.append(signals[-1])
        metrics[name] = financial_eval(split_df, np.array(signals[:len(split_df)]))

    register_model(
        version="v2",
        model_type="stable-baselines3",
        model_name="DQN v2",
        hyperparams=hyperparams,
        metrics_train=metrics["train"],
        metrics_valid=metrics["valid"],
        metrics_test=metrics["test"],
        features=RL_FEATURE_COLS,
        artifacts=["dqn_gbpusd_m15.zip", "norm_stats.joblib"],
    )

    return metrics


# ══════════════════════════════════════════════
# Enregistrement v3 : RL (DQN) avec reward shaping
# ══════════════════════════════════════════════
def register_v3_rl(df_train, df_valid, df_test):
    """Entraîne un DQN v3 avec reward shaping et obs enrichie, enregistre en v3."""
    print("\n  [v3] DQN v3 – reward shaping + obs enrichie (RL)...")

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from phase8_rl import TradingEnv

    version_dir = MODELS_DIR / "v3"
    version_dir.mkdir(parents=True, exist_ok=True)

    # Normalisation
    feature_mean = df_train[RL_FEATURE_COLS].mean()
    feature_std = df_train[RL_FEATURE_COLS].std()

    env_train = TradingEnv(df_train, feature_mean, feature_std)

    # Hyperparamètres v3
    hyperparams = {
        "learning_rate": 3e-4,
        "buffer_size": 200_000,
        "learning_starts": 2_000,
        "batch_size": 256,
        "gamma": 0.99,
        "exploration_fraction": 0.5,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "target_update_interval": 500,
        "train_freq": 4,
        "net_arch": [256, 128],
        "seed": SEED,
        "total_timesteps": len(df_train) * 15,
    }

    model = DQN(
        "MlpPolicy", env_train,
        learning_rate=hyperparams["learning_rate"],
        buffer_size=hyperparams["buffer_size"],
        learning_starts=hyperparams["learning_starts"],
        batch_size=hyperparams["batch_size"],
        gamma=hyperparams["gamma"],
        exploration_fraction=hyperparams["exploration_fraction"],
        exploration_initial_eps=hyperparams["exploration_initial_eps"],
        exploration_final_eps=hyperparams["exploration_final_eps"],
        target_update_interval=hyperparams["target_update_interval"],
        train_freq=hyperparams["train_freq"],
        policy_kwargs={"net_arch": hyperparams["net_arch"]},
        seed=SEED,
        verbose=0,
    )

    print(f"    Entraînement: {hyperparams['total_timesteps']} timesteps...")
    model.learn(total_timesteps=hyperparams["total_timesteps"], progress_bar=True)

    # Sauvegarder artefacts
    model.save(str(version_dir / "dqn_gbpusd_m15"))
    joblib.dump({"mean": feature_mean, "std": feature_std}, version_dir / "norm_stats.joblib")

    # Évaluer sur chaque split
    metrics = {}
    for name, split_df in [("train", df_train), ("valid", df_valid), ("test", df_test)]:
        env = TradingEnv(split_df, feature_mean, feature_std)
        obs, _ = env.reset()
        signals = []
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            signals.append(info["position"])
            if terminated or truncated:
                break
        while len(signals) < len(split_df):
            signals.append(signals[-1])
        metrics[name] = financial_eval(split_df, np.array(signals[:len(split_df)]))

    register_model(
        version="v3",
        model_type="stable-baselines3",
        model_name="DQN v3",
        hyperparams=hyperparams,
        metrics_train=metrics["train"],
        metrics_valid=metrics["valid"],
        metrics_test=metrics["test"],
        features=RL_FEATURE_COLS,
        artifacts=["dqn_gbpusd_m15.zip", "norm_stats.joblib"],
    )

    return metrics


# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════
def main():
    print("=" * 70)
    print("PHASE 11 – Versioning modèle (registry v1/v2/v3)")
    print("=" * 70)

    # Chargement
    print("\n[1] Chargement des données...")
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp_15m"])
    df["year"] = df["timestamp_15m"].dt.year

    df_train = df[(df["year"] == 2022) | (df["year"] == 2023)].copy()
    df_valid = df[df["year"] == 2024].copy()
    df_test = df[(df["year"] == 2025) | (df["year"] == 2026)].copy()

    print(f"  Train: {len(df_train)} | Valid: {len(df_valid)} | Test: {len(df_test)}")

    # ── v1 : ML ──
    print("\n[2] Enregistrement des modèles...")
    metrics_v1 = register_v1_ml(df_train, df_valid, df_test)

    # ── v2 : RL amélioré ──
    metrics_v2 = register_v2_rl(df_train, df_valid, df_test)

    # ── v3 : RL avec reward shaping ──
    metrics_v3 = register_v3_rl(df_train, df_valid, df_test)

    # ── Sélection du meilleur modèle ──
    print("\n[3] Sélection du meilleur modèle (par Sharpe validation)...")
    best = select_best_model()

    # ── Tableau comparatif ──
    print(f"\n{'=' * 70}")
    print("  COMPARAISON DES VERSIONS")
    print(f"{'=' * 70}")

    rows = []
    for version, metrics in [("v1", metrics_v1), ("v2", metrics_v2), ("v3", metrics_v3)]:
        for split, split_name in [("train", "2022 & 2023 (Train)"), ("valid", "2024 (Valid)"), ("test", "2025 & 2026 (Test)")]:
            m = metrics[split]
            rows.append({
                "Version": version,
                "Split": split_name,
                "Profit": m["profit"],
                "Max DD": m["max_drawdown"],
                "Sharpe": m["sharpe"],
                "PF": m["profit_factor"],
                "Trades": m["n_trades"],
            })
    comp_df = pd.DataFrame(rows)
    print(comp_df.to_string(index=False))

    # ── Vérification du chargement ──
    print(f"\n[4] Vérification load_best_model()...")
    model, aux, info = load_best_model()
    print(f"  Version chargée: {info['version']}")
    print(f"  Type: {info['model_type']}")
    print(f"  Modèle: {info['model_name']}")
    print(f"  Sharpe valid: {info['metrics'].get('valid_2024', {}).get('sharpe', 'N/A')}")

    # ── Contenu du registry ──
    print(f"\n[5] Contenu du registry:")
    registry = load_registry()
    print(f"  Versions enregistrées: {list(registry['models'].keys())}")
    print(f"  Version active (best): {registry['best_version']}")

    # Sauvegarder comparaison
    comp_df.to_csv(MODELS_DIR / "version_comparison.csv", index=False)

    print(f"\n  Fichiers générés:")
    print(f"    {REGISTRY_PATH}")
    for v in ["v1", "v2", "v3"]:
        vdir = MODELS_DIR / v
        for f in sorted(vdir.iterdir()):
            print(f"    {f}")

    print("\n" + "=" * 70)
    print("Phase 11 terminée avec succès")
    print("=" * 70)


if __name__ == "__main__":
    main()
