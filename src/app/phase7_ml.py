"""
Phase 7 – Machine Learning
Objectif : predire le mouvement de la prochaine bougie.
  y = 1 si close_{t+1} > close_t, 0 sinon

Split temporel strict :
  - 2022 & 2023 : Entrainement
  - 2024 : Validation
  - 2025 & 2026 : Test final (jamais utilise pour entrainer)

Modeles :
  - Baseline : DummyClassifier (most_frequent)
  - Logistic Regression
  - Random Forest
  - Gradient Boosting (HistGradientBoosting)

Metriques statistiques et financieres.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

INPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "features"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "models"

FEATURE_COLS = [
    "return_1", "return_4",
    "ema_diff", "rsi_14", "rolling_std_20",
    "range_15m", "body", "upper_wick", "lower_wick",
    "distance_to_ema200", "slope_ema50",
    "atr_14", "rolling_std_100", "volatility_ratio",
    "adx_14", "macd", "macd_signal",
]

TRANSACTION_COST = 0.0001
SEED = 42


def load_and_prepare() -> pd.DataFrame:
    """Charge les features et cree la target."""
    path = INPUT_DIR / "gbpusd_m15_features.csv"
    df = pd.read_csv(path, parse_dates=["timestamp_15m"])

    # Target : 1 si close_{t+1} > close_t
    df["target"] = (df["close_15m"].shift(-1) > df["close_15m"]).astype(int)
    df = df.dropna(subset=["target"]).reset_index(drop=True)

    print(f"  Charge: {len(df)} bougies")
    print(f"  Distribution target: {df['target'].value_counts().to_dict()}")
    return df


def split_temporal(df: pd.DataFrame):
    """Split temporel strict par annee."""
    df["year"] = df["timestamp_15m"].dt.year
    train = df[(df["year"] == 2022) | (df["year"] == 2023)].copy()
    val = df[df["year"] == 2024].copy()
    test = df[(df["year"] == 2025) | (df["year"] == 2026)].copy()
    for d in [train, val, test]:
        d.drop(columns=["year"], inplace=True)
    df.drop(columns=["year"], inplace=True)
    return train, val, test


def get_Xy(df: pd.DataFrame):
    """Extrait X (features) et y (target)."""
    return df[FEATURE_COLS].values, df["target"].values


def financial_eval(df: pd.DataFrame, predictions: np.ndarray, name: str) -> dict:
    """Evaluation financiere : convertit predictions en signaux et calcule PnL."""
    signals = pd.Series(0, index=df.index)
    signals[predictions == 1] = 1   # BUY si on predit hausse
    signals[predictions == 0] = -1  # SELL si on predit baisse

    returns = df["close_15m"].pct_change().shift(-1).fillna(0)
    position_changes = signals.diff().abs().fillna(0)
    costs = position_changes * TRANSACTION_COST

    pnl = signals * returns - costs
    cum_pnl = pnl.cumsum()

    # Max drawdown
    peak = cum_pnl.cummax()
    max_dd = (cum_pnl - peak).min()

    # Sharpe
    periods_per_year = 96 * 252
    sharpe = (pnl.mean() / pnl.std()) * np.sqrt(periods_per_year) if pnl.std() > 0 else 0

    # Profit factor
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = abs(pnl[pnl < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

    return {
        "model": name,
        "total_pnl": cum_pnl.iloc[-1],
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "profit_factor": pf,
    }


def main():
    print("=" * 70)
    print("PHASE 7 – Machine Learning GBP/USD")
    print("=" * 70)

    # 1. Chargement
    print("\n[1] Chargement et preparation...")
    df = load_and_prepare()

    # 2. Split
    print("\n[2] Split temporel strict...")
    train, val, test = split_temporal(df)
    print(f"  Train (2022 & 2023): {len(train)}")
    print(f"  Val   (2024): {len(val)}")
    print(f"  Test  (2025 & 2026): {len(test)}")

    X_train, y_train = get_Xy(train)
    X_val, y_val = get_Xy(val)
    X_test, y_test = get_Xy(test)

    # 3. Scaling
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # 4. Modeles
    print("\n[3] Entrainement des modeles...")
    models = {
        "Dummy (most_frequent)": DummyClassifier(strategy="most_frequent", random_state=SEED),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=50, random_state=SEED, n_jobs=-1
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=300, max_depth=6, min_samples_leaf=50, learning_rate=0.05, random_state=SEED
        ),
    }

    results_val = []
    results_test = []
    trained_models = {}

    for name, model in models.items():
        print(f"\n  --- {name} ---")

        # Entrainement
        if "Logistic" in name:
            model.fit(X_train_s, y_train)
            preds_val = model.predict(X_val_s)
            preds_test = model.predict(X_test_s)
        else:
            model.fit(X_train, y_train)
            preds_val = model.predict(X_val)
            preds_test = model.predict(X_test)

        trained_models[name] = model

        # Metriques statistiques - Validation
        acc_val = accuracy_score(y_val, preds_val)
        f1_val = f1_score(y_val, preds_val)
        prec_val = precision_score(y_val, preds_val, zero_division=0)
        rec_val = recall_score(y_val, preds_val, zero_division=0)

        # Metriques statistiques - Test
        acc_test = accuracy_score(y_test, preds_test)
        f1_test = f1_score(y_test, preds_test)
        prec_test = precision_score(y_test, preds_test, zero_division=0)
        rec_test = recall_score(y_test, preds_test, zero_division=0)

        print(f"  Validation : Acc={acc_val:.4f}  Prec={prec_val:.4f}  Rec={rec_val:.4f}  F1={f1_val:.4f}")
        print(f"  Test       : Acc={acc_test:.4f}  Prec={prec_test:.4f}  Rec={rec_test:.4f}  F1={f1_test:.4f}")

        # Metriques financieres
        fin_val = financial_eval(val, preds_val, name)
        fin_test = financial_eval(test, preds_test, name)

        print(f"  Finance Val  : PnL={fin_val['total_pnl']:+.6f}  Sharpe={fin_val['sharpe']:.3f}  MaxDD={fin_val['max_drawdown']:.6f}")
        print(f"  Finance Test : PnL={fin_test['total_pnl']:+.6f}  Sharpe={fin_test['sharpe']:.3f}  MaxDD={fin_test['max_drawdown']:.6f}")

        results_val.append({
            "model": name, "split": "val",
            "accuracy": acc_val, "precision": prec_val, "recall": rec_val, "f1": f1_val,
            **{k: v for k, v in fin_val.items() if k != "model"},
        })
        results_test.append({
            "model": name, "split": "test",
            "accuracy": acc_test, "precision": prec_test, "recall": rec_test, "f1": f1_test,
            **{k: v for k, v in fin_test.items() if k != "model"},
        })

    # 5. Feature importance (Random Forest)
    print("\n[4] Feature importance (Random Forest):")
    rf_model = trained_models["Random Forest"]
    importances = pd.Series(
        rf_model.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)
    for feat, imp in importances.items():
        print(f"    {feat:25s} {imp:.4f}")

    # 6. Confusion matrix meilleur modele sur test
    print("\n[5] Matrice de confusion (Gradient Boosting - Test 2024):")
    gb_preds_test = trained_models["Gradient Boosting"].predict(X_test)  # scaled not needed for HGBC
    cm = confusion_matrix(y_test, gb_preds_test)
    print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}  TP={cm[1][1]}")
    print(f"\n  Classification report:")
    print(classification_report(y_test, gb_preds_test, target_names=["DOWN", "UP"]))

    # 7. Tableau recapitulatif
    print(f"\n{'='*70}")
    print("  TABLEAU RECAPITULATIF")
    print(f"{'='*70}")
    all_results = pd.DataFrame(results_val + results_test)
    all_results = all_results[["model", "split", "accuracy", "f1", "total_pnl", "sharpe", "max_drawdown", "profit_factor"]]
    print(all_results.to_string(index=False))

    # 8. Sauvegarde
    print(f"\n[6] Sauvegarde...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(OUTPUT_DIR / "ml_metrics.csv", index=False)
    print(f"  Sauvegarde: {OUTPUT_DIR / 'ml_metrics.csv'}")

    print("\n" + "=" * 70)
    print("Phase 7 terminee avec succes")
    print("=" * 70)


if __name__ == "__main__":
    main()
