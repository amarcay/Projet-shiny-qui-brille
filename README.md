# Systeme de decision algorithmique GBP/USD

Projet de Master 2 Data Science -- pipeline complet de trading algorithmique sur la paire GBP/USD en resolution M15 (15 minutes), de l'importation des donnees brutes jusqu'au deploiement d'une API de prediction.

**Auteurs** : Alphonse Marcay, Thomas Bourvon

---

## Objectif

Construire et evaluer un systeme de decision de trading (achat / vente / attente) sur GBP/USD, en comparant des approches classiques (regles techniques), du machine learning supervise et du reinforcement learning. Le projet couvre l'ensemble du cycle : collecte, nettoyage, feature engineering, modelisation, evaluation et mise en production via API.

---

## Donnees

- **Source** : HistData.com (bougies M1, format MetaTrader)
- **Paire** : GBP/USD
- **Periode** : janvier 2022 -- janvier 2026
- **Volume brut** : ~1.47 million de bougies M1
- **Apres agregation M15 et nettoyage** : 94 876 bougies M15 propres

### Split temporel (strict, jamais aleatoire)

| Ensemble       | Periode     | Bougies |
|----------------|-------------|---------|
| Entrainement   | 2022--2023  | 45 195  |
| Validation     | 2024        | 23 825  |
| Test           | 2025--2026  | 25 756  |

---

## Architecture du pipeline

```
src/app/
  phase1_import_m1.py          Importation et fusion des CSV bruts M1
  phase2_aggregation_m15.py    Agregation M1 -> M15 (OHLCV + compteur qualite)
  phase3_nettoyage_m15.py      Suppression bougies incompletes, controles OHLC
  phase4_eda.py                Analyse exploratoire
  phase5_feature_engineering.py Construction de 20 features techniques
  phase6_baseline.py           Strategies de reference (Buy&Hold, Random, Regles)
  phase7_ml.py                 Machine Learning supervise (4 modeles)
  phase8_rl.py                 Reinforcement Learning -- DQN (Stable-Baselines3)
  phase8_ql.py                 Reinforcement Learning -- Q-Learning tabulaire
  phase9_evaluation.py         Comparaison finale de toutes les strategies

src/api/                       API FastAPI (core/routers/schemas/services)
```

### Features techniques (20)

| Categorie   | Features |
|-------------|----------|
| Rendements  | return_1, return_4 |
| Tendance    | ema_20, ema_50, ema_200, ema_diff, distance_to_ema200, slope_ema50 |
| Momentum    | rsi_14, macd, macd_signal, adx_14 |
| Volatilite  | rolling_std_20, rolling_std_100, atr_14, volatility_ratio |
| Price action| range_15m, body, upper_wick, lower_wick |

---

## Strategies evaluees

### 1. Baselines (Phase 6)

- **Buy & Hold** : position longue permanente
- **Random** : signaux aleatoires uniformes (BUY/SELL/HOLD)
- **Regles (EMA + RSI + ADX)** : achat si EMA_diff > 0, RSI < 70, ADX > 20 ; vente inverse

### 2. Machine Learning supervise (Phase 7)

- **DummyClassifier** (most_frequent) : reference statistique
- **Logistic Regression** (avec StandardScaler)
- **Random Forest** (200 arbres, max_depth=10)
- **HistGradientBoosting** (300 iterations, lr=0.05)

Target : y = 1 si close_{t+1} > close_t, 0 sinon.

### 3. Reinforcement Learning (Phase 8)

- **DQN** (Deep Q-Network) : reseau 128-128, 19 features normalisees + position, 3 actions discretes. 5 episodes sur le train set (~225k timesteps).
- **Q-Learning tabulaire** : 5 features discretisees en 5 bins (9 375 etats possibles), Q-table classique, 15 episodes.

---

## Resultats sur le test 2025--2026

| Strategie              | Profit cumule | Sharpe | Max drawdown | Profit factor | Trades |
|------------------------|---------------|--------|--------------|---------------|--------|
| Buy & Hold             | +0.0928       | +1.14  | -0.057       | 1.022         | 1      |
| Regles (EMA+RSI+ADX)   | +0.0785       | +0.97  | -0.069       | 1.019         | 324    |
| RL (DQN)               | -0.5158       | -6.29  | -0.549       | 0.888         | 2 868  |
| RL (Q-Learning)         | -1.3430       | -17.33 | -1.355       | 0.716         | 6 701  |
| ML (Gradient Boosting)  | -1.4425       | -17.48 | -1.449       | 0.721         | 7 110  |
| Random                 | -1.8030       | -21.84 | -1.807       | 0.664         | 8 610  |

Seuls Buy & Hold et Regles (EMA+RSI+ADX) sont valides sur le test (Sharpe > 0, profit > 0).

---

## Analyse critique

### Ce qui fonctionne

- Le pipeline de donnees est robuste : 99.6% de regularite M1, nettoyage systematique, pas de look-ahead dans les features.
- Le split temporel strict garantit l'absence de fuite d'information.
- Les baselines simples (Buy & Hold, regles techniques) captent la tendance haussiere du GBP/USD en 2025.

### Ce qui ne fonctionne pas

- **Le ML supervise ne bat pas le hasard** : accuracy ~51%, Sharpe tres negatif. La target binaire (hausse/baisse de la prochaine bougie M15) est essentiellement du bruit. Le marche forex M15 est proche de l'efficience sur ce type de prediction directionnelle a court terme. Le cout de transaction (1 pip) penalise lourdement les modeles qui tradent frequemment (7 000+ trades).

- **Le DQN converge vers un biais long** : 89% du temps en position longue. Il a appris que rester long est moins mauvais que trader activement, mais ne generalise pas -- il subit le drawdown quand la tendance s'inverse. Le reward shaping (penalite drawdown) n'est pas suffisant pour apprendre une politique non-triviale.

- **Le Q-Learning tabulaire est structurellement limite** : discretiser 5 features en 5 bins perd enormement d'information. Avec ~5 500 etats visites sur 9 375 possibles, la Q-table est sous-exploree. L'agent trade presque aleatoirement (50/50 long/short), generant des couts de transaction massifs.

### Limites fondamentales

1. **Resolution M15 et forex** : le marche des changes est l'un des plus efficients au monde. Predire la direction de la prochaine bougie M15 avec des indicateurs techniques publics releve du bruit. Les signaux exploitables existent sur des horizons plus longs ou avec des donnees alternatives (flux d'ordres, sentiment, macro).

2. **Couts de transaction** : avec un spread de 1 pip (~0.01%), tout modele qui trade frequemment doit etre significativement meilleur que le hasard pour etre rentable. Un taux de precision de 51% est insuffisant.

3. **Stationnarite** : les regimes de marche changent. Un modele entraine sur 2022-2023 (forte volatilite post-COVID, hausse des taux) ne generalise pas necessairement a 2025 (contexte macro different). Le walk-forward ou le re-entrainement periodique serait plus adapte.

4. **Reward RL** : le PnL brut est un signal tres sparse et bruise pour l'apprentissage. Des approches avec reward shaping plus elabore (Sharpe incrementiel, risk-adjusted returns) ou des methodes de meta-apprentissage pourraient ameliorer la convergence.

### Pistes d'amelioration

- Walk-forward validation (re-entrainement glissant tous les N mois)
- Features alternatives : sentiment, donnees macro, order flow
- Horizon de prediction plus long (H1, H4, Daily)
- Filtrage des periodes de faible volatilite (ADX < 20 = ne pas trader)
- Methodes d'ensemble ou stacking
- RL avec PPO ou SAC (plus stables que DQN pour les environnements financiers)

---

## API de prediction

API REST (FastAPI) exposant le modele Gradient Boosting pour des predictions en temps reel.

```
GET  /health              Statut du service
GET  /model/info          Version et metadata du modele charge
POST /model/load          Charger une version specifique (v1, v2)
POST /predict             Prediction sur une bougie (17 features)
POST /predict/batch       Prediction sur un lot de bougies
```

### Lancer l'API

```bash
uvicorn src.api.api:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t gbpusd-trading .
docker run -p 5000:5000 -p 8000:8000 gbpusd-trading
```

---

## Execution du pipeline

```bash
# 1. Importation M1
python src/app/phase1_import_m1.py

# 2. Agregation M15
python src/app/phase2_aggregation_m15.py

# 3. Nettoyage
python src/app/phase3_nettoyage_m15.py

# 4. Feature engineering
python src/app/phase5_feature_engineering.py

# 5. Baselines
python src/app/phase6_baseline.py

# 6. Machine Learning
python src/app/phase7_ml.py

# 7. Reinforcement Learning
python src/app/phase8_rl.py     # DQN
python src/app/phase8_ql.py     # Q-Learning tabulaire

# 8. Evaluation finale
python src/app/phase9_evaluation.py
```

---

## Dependances

Python >= 3.10. Principales librairies :

- pandas, numpy, scikit-learn
- stable-baselines3, gymnasium
- matplotlib, seaborn, plotly
- FastAPI, uvicorn, pydantic

Installation : `pip install -e .` ou `uv pip install -e .`

---

## Structure des fichiers

```
Projet-shiny-qui-brille/
  src/
    app/                    Scripts du pipeline (phases 1-9)
    api/                    API FastAPI
      core/                 Configuration, registry des modeles
      routers/              Endpoints (health, model, predict)
      schemas/              Schemas Pydantic (candle, responses)
      services/             Logique metier (prediction)
  models/
    v1/                     Gradient Boosting + scaler + DQN
    v2/                     Q-Learning (Q-table + discretizer)
  data/                     Donnees brutes et traitees (gitignore)
  reports/                  Graphiques et metriques (gitignore)
  Dockerfile
  pyproject.toml
```