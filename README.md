# Projet Shiny Qui Brille – Trading Algorithmique GBP/USD (M15)

Ce projet est une solution complète de trading algorithmique "End-to-End" pour la paire **GBP/USD**. Il part des données brutes (M1), les transforme en indicateurs techniques sophistiqués, entraîne des modèles de Machine Learning (Supervisé et Renforcement), et expose la meilleure stratégie via une API et un Dashboard.

## 📌 Architecture du Pipeline

Le projet est organisé en **11 Phases** séquentielles situées dans `src/app/`. Chaque script est autonome et produit des artefacts pour l'étape suivante.

| Phase | Script | Description |
| :--- | :--- | :--- |
| **1** | `phase1_import_m1.py` | Importation des données brutes, fusion Date+Time, et vérification de la régularité (1 min). |
| **2** | `phase2_aggregation_m15.py` | Agrégation des bougies M1 en **M15** (Open, High, Low, Close, Volume). |
| **3** | `phase3_nettoyage_m15.py` | Nettoyage strict : suppression des bougies incomplètes (<15 min de data) et des aberrations de prix. |
| **4** | `phase4_eda.py` | Analyse exploratoire : distribution des rendements, test de stationnarité (ADF), et autocorrélation. |
| **5** | `phase5_feature_engineering.py` | Création de **20 features techniques** (voir ci-dessous) sans biais futur (look-ahead bias). |
| **6** | `phase6_baseline.py` | Établissement de baselines : *Buy & Hold*, *Random*, et *Règles Fixes* (EMA+RSI+ADX). |
| **7** | `phase7_ml.py` | Entraînement de modèles supervisés (Gradient Boosting, Random Forest) pour prédire la direction du prix. |
| **8** | `phase8_rl.py` | Entraînement d'un agent **RL (Deep Q-Network)** maximisant le PnL sur plusieurs années. |
| **9** | `phase9_evaluation.py` | Comparaison finale de toutes les stratégies (Baselines vs ML vs RL) sur le set de Test (2024). |
| **10** | `src/api/` | API FastAPI exposant le meilleur modèle pour des prédictions en temps réel. |
| **11** | `phase11_model_registry.py` | Versioning automatique (`models/registry.json`) et sélection du champion validé. |

---

## 📊 Feature Engineering (Phase 5)

Le modèle s'appuie sur une combinaison d'indicateurs de momentum, de volatilité et de tendance, calculés sur le passé uniquement :

*   **Momentum / Court Terme** : Retours (1, 4 périodes), RSI (14), EMA (20, 50), Différence EMA.
*   **Volatilité** : Rolling Std (20, 100), ATR (14), Ratio de Volatilité, Range M15, Body, Wicks (mèches).
*   **Tendance / Régime** : EMA (200), Distance à EMA 200, Slope EMA 50, ADX (14), MACD + Signal.

---

## 🧠 Stratégies et Modèles

### 1. Baselines (Phase 6)
*   **Buy & Hold** : Achat au début, vente à la fin (référence de marché).
*   **Règles Fixes** : Stratégie classique "Trend Following" (Achat si EMA court > EMA long + RSI neutre + ADX fort).

### 2. Machine Learning Supervisé (Phase 7 - v1)
*   **Modèle** : HistGradientBoostingClassifier.
*   **Objectif** : Maximiser la précision (Accuracy) de la prédiction Up/Down.
*   **Limitation** : Ne prend pas en compte les coûts de transaction ni l'ampleur des mouvements.

### 3. Reinforcement Learning (Phase 8 - v2)
*   **Modèle** : **DQN (Deep Q-Network)** via Stable-Baselines3.
*   **Architecture** : Réseau de neurones (MlpPolicy) prenant l'état du marché et la position actuelle.
*   **Objectif** : Maximiser directement le **Profit (PnL)** net de frais.
*   **Environnement** : Simulation réaliste incluant spreads et pénalités de drawdown.

---

## 🏆 Résultats et Choix du Modèle

Les modèles sont comparés sur la période de **Test (2025 & 2026)**, totalement inconnue lors de l'entraînement.

| Version | Modèle | Approche | Profit | Sharpe | Max Drawdown | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **v1** | Gradient Boosting | Supervisé | -1.62% | -24.07 | -1.62% | Trop agressif (Overtrading) |
| **v2** | **DQN** | **RL** | **-0.07%** | **-1.10** | **-0.09%** | **Sélectionné** |

### Pourquoi le RL (v2) est-il meilleur ?
L'approche par renforcement a démontré une "intelligence" de gestion supérieure :
1.  **Sélectivité** : Il trade beaucoup moins souvent que le supervisé, évitant d'être mangé par les spreads.
2.  **Gestion du Risque** : Grâce à la pénalité de drawdown dans sa fonction de récompense, il coupe rapidement les pertes ou évite les entrées risquées, divisant le Max Drawdown par 18 par rapport au ML classique.

---

## 💶 Simulation Réaliste (10k€)

Le script `src/app/simulation_10k.py` simule le comportement du modèle v2 sur un portefeuille de **10 000€** en **2025 & 2026** avec :
*   Levier 1:30 (typique retail).
*   Taille de position : 1 mini-lot (10k unités).
*   Spread : 1 pip (coût réaliste).

Les résultats de cette simulation (courbe de capital, drawdown, stats mensuelles) sont générés dans `reports/simulation/`.

---

## 🚀 Guide d'Utilisation

### 1. Installation

```bash
# Via uv (recommandé)
uv sync
```

### 2. Exécution du Pipeline (Entraînement complet)

Pour ré-entraîner les modèles depuis zéro :

```bash
# Génération des features
python src/app/phase5_feature_engineering.py

# Entraînement ML (Supervisé)
python src/app/phase7_ml.py

# Entraînement RL (DQN) - Peut prendre du temps (~10-15 min)
python src/app/phase8_rl.py

# Enregistrement et sélection du champion
python src/app/phase11_model_registry.py
```

### 3. Lancer la Plateforme (Production)

L'architecture sépare le moteur de décision (API) de l'interface utilisateur (Dashboard). Lancez les deux commandes dans deux terminaux séparés :

**Terminal 1 : API FastAPI (Backend)**
```bash
uvicorn src.api.api:app --reload --port 8000
```
*Documentation API : http://localhost:8000/docs*

**Terminal 2 : Dashboard Flask (Frontend)**
```bash
python src/app/app.py
```
*Interface Web : http://localhost:5000*

### 🐳 Docker

Le projet est conteneurisé pour faciliter le déploiement. L'image Docker contient tout l'environnement et lance automatiquement l'API et le Dashboard.

**1. Construire l'image**
```bash
docker build -t gbpusd-trading .
```

**2. Lancer le conteneur**
```bash
docker run -p 5000:5000 -p 8000:8000 gbpusd-trading
```
*L'application sera accessible sur `http://localhost:5000` et l'API sur `http://localhost:8000`.*

---

## 📂 Structure du Projet

```text
.
├── CLAUDE.md           # Guide de développement et conventions
├── Dockerfile          # Configuration Docker image
├── docker-entrypoint.sh # Script de démarrage Docker
├── data/               # Stockage des données (raw, processed, features)
├── models/             # Artefacts des modèles (joblib, zip) et Registry
├── reports/            # Rapports d'évaluation (PNG, CSV)
├── src/
│   ├── api/            # Backend FastAPI (routers, services, schemas)
│   └── app/            # Pipelines de données, Scripts ML/RL, Dashboard
└── pyproject.toml      # Gestion des dépendances
```

---
*Projet scolaire réalisé par Alphonse Marcay et Thomas Bourvon.*
