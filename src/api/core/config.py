"""Configuration globale de l'API."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_VERSION = os.environ.get("MODEL_VERSION", "v1")

API_TITLE = "GBP/USD M15 Trading API"
API_DESCRIPTION = (
    "API de prediction de trading GBP/USD sur bougies M15. "
    "Expose le meilleur modele avec versioning."
)
API_VERSION = "1.0.0"
