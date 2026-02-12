"""
Phase 10 – API de prediction GBP/USD M15
Point d'entree FastAPI.

Structure :
  src/
    api.py              <- point d'entree (ce fichier)
    core/
      config.py         <- configuration, paths
      registry.py       <- ModelRegistry (chargement/versioning)
    schemas/
      candle.py         <- schemas d'entree (CandleInput, BatchInput)
      responses.py      <- schemas de reponse (PredictionResponse, etc.)
    services/
      prediction.py     <- logique metier (prediction, conversion features)
    routers/
      health.py         <- GET /health
      model.py          <- GET /model/info, POST /model/load
      predict.py        <- POST /predict, POST /predict/batch

Regles :
  - L'API expose uniquement le meilleur modele
  - L'utilisateur ne peut pas relancer l'entrainement
  - Versioning modele obligatoire
  - L'API charge automatiquement la version validee
"""

from fastapi import FastAPI

from core.config import API_TITLE, API_DESCRIPTION, API_VERSION, DEFAULT_VERSION
from core.registry import registry
from routers import health, model, predict

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
)

# Enregistrement des routers
app.include_router(health.router)
app.include_router(model.router)
app.include_router(predict.router)

# Chargement du modele au demarrage
try:
    registry.load(DEFAULT_VERSION)
    print(f"Modele charge: {registry.model_type} (version {registry.version})")
except FileNotFoundError as e:
    print(f"[WARN] Impossible de charger le modele: {e}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
