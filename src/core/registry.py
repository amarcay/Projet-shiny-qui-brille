"""Model Registry : chargement et versioning des modeles."""

import joblib
import numpy as np
from pathlib import Path
from datetime import datetime

from core.config import MODEL_DIR, DEFAULT_VERSION


class ModelRegistry:
    """Gere le chargement et le versioning des modeles."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.version = None
        self.model_type = None
        self.loaded_at = None

    def list_versions(self) -> list[str]:
        """Liste les versions disponibles."""
        versions = []
        if self.model_dir.exists():
            for d in sorted(self.model_dir.iterdir()):
                if d.is_dir() and (d / "gradient_boosting.joblib").exists():
                    versions.append(d.name)
        return versions

    def load(self, version: str = DEFAULT_VERSION):
        """Charge un modele par version."""
        version_dir = self.model_dir / version

        if not version_dir.exists():
            raise FileNotFoundError(
                f"Version '{version}' introuvable dans {self.model_dir}"
            )

        model_path = version_dir / "gradient_boosting.joblib"
        scaler_path = version_dir / "scaler.joblib"
        features_path = version_dir / "feature_cols.joblib"

        if not model_path.exists():
            raise FileNotFoundError(f"Modele introuvable: {model_path}")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        self.feature_cols = (
            joblib.load(features_path) if features_path.exists() else None
        )
        self.version = version
        self.model_type = type(self.model).__name__
        self.loaded_at = datetime.now().isoformat()

        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Prediction brute (0 ou 1)."""
        if self.model is None:
            raise RuntimeError("Aucun modele charge")
        return self.model.predict(features)

    def predict_proba(self, features: np.ndarray) -> np.ndarray | None:
        """Probabilites de prediction."""
        if self.model is None:
            raise RuntimeError("Aucun modele charge")
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(features)
        return None


# Instance singleton
registry = ModelRegistry()
