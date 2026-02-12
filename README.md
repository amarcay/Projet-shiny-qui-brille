# Projet Shiny Qui Brille – Trading GBP/USD

Ce projet est une application de Data Science complète pour le trading automatique sur la paire **GBP/USD** (données M15). Il intègre un pipeline de données, d'analyse exploratoire, de modélisation (ML & RL), et une interface utilisateur web (Flask + FastAPI).

## 📌 Contexte du Projet

Ce projet scolaire a pour but de mettre en œuvre une chaîne de traitement de données financières de bout en bout ("End-to-End"), de la collecte des données brutes jusqu'au déploiement d'un modèle via une API.

L'objectif principal est de maximiser le **Profit cumulé (PnL)** et le **Ratio de Sharpe** sur l'année 2024 (Test), en s'entraînant sur 2022 et en validant sur 2023.

---

## 🏗 Architecture du Projet

Le projet est structuré en plusieurs "Phases" séquentielles situées dans `src/app/` :

1.  **Phases 1-3 (Data)** : Importation, Agrégation (M15) et Nettoyage des données.
2.  **Phase 4 (EDA)** : Analyse exploratoire (Stationnarité, Volatilité, Autocorrélation).
3.  **Phase 5 (Feature Engineering)** : Création d'indicateurs techniques (RSI, MACD, Bandes de Bollinger, etc.).
4.  **Phase 6 (Baseline)** : Modèle naïf pour établir une performance de référence.
5.  **Phase 7 (ML Supervisé)** : Entraînement de modèles classiques (Random Forest, Gradient Boosting).
6.  **Phase 8 (RL)** : Entraînement d'un agent de Reinforcement Learning (DQN).
7.  **Phase 9 (Évaluation)** : Comparaison finale des stratégies.
8.  **Phase 10 (API)** : Exposition du meilleur modèle via FastAPI (`src/api/`).
9.  **Phase 11 (Registry)** : Gestion des versions de modèles (`models/registry.json`).
10. **Application Web** : Dashboard de suivi et de signaux (`src/app/app.py`).

---

## 🧠 Choix du Modèle et Justification

Une partie centrale du projet a été la comparaison entre une approche **Supervisée (v1)** et une approche par **Renforcement (v2)**.

### Comparaison des Versions

Les modèles sont stockés dans le `model registry` avec leurs performances respectives. Voici les résultats obtenus sur le set de Test (2024) :

| Version | Modèle | Type | Profit | Sharpe | Max Drawdown |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **v1** | Gradient Boosting | Supervisé (Sklearn) | -1.62% | -24.07 | -1.62% |
| **v2** | **DQN (Deep Q-Network)** | **Reinforcement Learning** | **-0.07%** | **-1.10** | **-0.09%** |

*(Données issues de `models/version_comparison.csv`)*

### Pourquoi avons-nous choisi le modèle v2 (RL) ?

Bien que les deux modèles aient des difficultés à générer un profit net positif sur la période de test (marché difficile ou coûts de transaction impactants), le modèle **v2 (DQN)** est **nettement supérieur** au modèle v1 pour plusieurs raisons fondamentales :

1.  **Objectif d'Optimisation (La justification clé)** :
    *   **Le modèle v1 (Supervisé)** cherche à maximiser la *précision* (Accuracy) de la prédiction du mouvement futur (Hausse/Baisse). Or, avoir raison 55% du temps ne garantit pas d'être rentable si les gains sont faibles et les pertes importantes.
    *   **Le modèle v2 (RL)** cherche directement à maximiser la **récompense (Reward)**, qui est ici définie comme le **PnL (Profit and Loss)**. L'agent apprend à ne trader que lorsque l'espérance de gain est supérieure aux coûts.

2.  **Gestion des Coûts de Transaction** :
    *   Le modèle RL intègre le coût de transaction (spread) dans son environnement d'entraînement. Il apprend naturellement à éviter le "sur-trading" (trop d'ordres qui grignotent le capital), ce qui explique son nombre de trades beaucoup plus faible et sélectif.
    *   Le modèle Supervisé ne "voit" pas les coûts lors de son entraînement.

3.  **Gestion du Risque (Drawdown)** :
    *   Notre fonction de récompense RL inclut une pénalité pour le **Drawdown** (perte maximale consécutive). Cela force l'agent à être plus prudent pour préserver le capital.

**Conclusion** : Nous avons retenu la version **v2** comme modèle de production car elle démontre une bien meilleure résilience et une "intelligence" de gestion du capital que l'approche supervisée classique ne peut pas capturer.

---

## 🚀 Installation et Utilisation

### 1. Pré-requis

Le projet utilise `poetry` pour la gestion des dépendances, ou peut être installé via `pip`.

```bash
# Via Poetry
poetry install

# Ou via pip (si requirements.txt généré)
pip install -r requirements.txt
```

### 2. Lancer le Pipeline (Entraînement)

Pour régénérer les modèles et mettre à jour le registre :

```bash
# Lance le feature engineering, puis les entraînements ML et RL, et met à jour le registry
python src/app/phase5_feature_engineering.py
python src/app/phase7_ml.py
python src/app/phase8_rl.py
python src/app/phase11_model_registry.py
```

### 3. Lancer l'Application (Production)

L'architecture repose sur deux services qui doivent tourner en parallèle :

**A. L'API (Backend FastAPI)**
Sert les prédictions du meilleur modèle chargé depuis le registry.
```bash
# Depuis la racine du projet
uvicorn src.api.api:app --reload --port 8000
```
*L'API sera accessible sur `http://localhost:8000` (Doc interactive sur `/docs`).*

**B. Le Dashboard (Frontend Flask)**
Interface utilisateur pour visualiser les performances et les signaux.
```bash
python src/app/app.py
```
*L'application sera accessible sur `http://localhost:5000`.*

---

## 📂 Structure des Dossiers

```text
.
├── data/               # Données brutes et processées
├── models/             # Registry et binaires des modèles (v1, v2...)
│   ├── registry.json   # Fichier central de versioning
│   └── version_comparison.csv
├── reports/            # Graphiques et métriques générés
├── src/
│   ├── api/            # Code de l'API FastAPI (backend)
│   └── app/            # Code du Pipeline et du Dashboard Flask (frontend)
│       ├── phase*.py   # Scripts des différentes étapes du projet
│       └── templates/  # Templates HTML pour le dashboard
└── pyproject.toml      # Dépendances du projet
```

---
*Projet réalisé par Alphonse Marcay et Thomas Bourvon.*
