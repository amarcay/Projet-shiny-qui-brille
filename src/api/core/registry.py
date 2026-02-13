"""Model Registry : chargement et versioning des modeles."""

import json
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime

from core.config import MODEL_DIR, DEFAULT_VERSION


class ModelRegistry:
    """Gere le chargement et le versioning des modeles (sklearn + stable-baselines3)."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None          # sklearn scaler
        self.norm_stats = None       # DQN norm_stats {"mean": ..., "std": ...}
        self.feature_cols = None
        self.version = None
        self.model_type = None       # "sklearn" ou "stable-baselines3"
        self.model_name = None       # nom affichable (ex: "DQN v2")
        self.loaded_at = None
        self._registry_data = None

    def _load_registry_json(self):
        """Charge registry.json pour connaitre le type de chaque version."""
        reg_path = self.model_dir / "registry.json"
        if reg_path.exists():
            self._registry_data = json.loads(reg_path.read_text())
        else:
            self._registry_data = {"models": {}}

    def list_versions(self) -> list[str]:
        """Liste les versions disponibles."""
        versions = []
        if self.model_dir.exists():
            for d in sorted(self.model_dir.iterdir()):
                if d.is_dir() and d.name.startswith("v"):
                    has_model = (
                        (d / "gradient_boosting.joblib").exists()
                        or (d / "dqn_gbpusd_m15.zip").exists()
                    )
                    if has_model:
                        versions.append(d.name)
        return versions

    def load(self, version: str = DEFAULT_VERSION):
        """Charge un modele par version (sklearn ou DQN)."""
        version_dir = self.model_dir / version

        if not version_dir.exists():
            raise FileNotFoundError(
                f"Version '{version}' introuvable dans {self.model_dir}"
            )

        # Lire registry.json pour connaitre le type
        if self._registry_data is None:
            self._load_registry_json()

        model_meta = self._registry_data.get("models", {}).get(version, {})
        model_type = model_meta.get("model_type", "sklearn")
        self.feature_cols = model_meta.get("features")

        if model_type == "sklearn":
            self._load_sklearn(version_dir)
        elif model_type == "stable-baselines3":
            self._load_sb3(version_dir)
        else:
            raise ValueError(f"Type de modele inconnu: {model_type}")

        self.version = version
        self.model_type = model_type
        self.model_name = model_meta.get("model_name")
        self.loaded_at = datetime.now().isoformat()
        return self

    def _load_sklearn(self, version_dir: Path):
        """Charge un modele sklearn (Gradient Boosting)."""
        model_path = version_dir / "gradient_boosting.joblib"
        scaler_path = version_dir / "scaler.joblib"

        if not model_path.exists():
            raise FileNotFoundError(f"Modele introuvable: {model_path}")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        self.norm_stats = None

    def _load_sb3(self, version_dir: Path):
        """Charge un modele DQN (stable-baselines3)."""
        from stable_baselines3 import DQN

        model_path = version_dir / "dqn_gbpusd_m15.zip"
        norm_path = version_dir / "norm_stats.joblib"

        if not model_path.exists():
            raise FileNotFoundError(f"Modele introuvable: {model_path}")

        self.model = DQN.load(str(model_path))
        self.norm_stats = joblib.load(norm_path) if norm_path.exists() else None
        self.scaler = None

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Prediction brute."""
        if self.model is None:
            raise RuntimeError("Aucun modele charge")

        if self.model_type == "stable-baselines3":
            return self._predict_sb3(features)
        else:
            return self._predict_sklearn(features)

    def _predict_sklearn(self, features: np.ndarray) -> np.ndarray:
        """Prediction sklearn : 0=SELL, 1=BUY."""
        if self.scaler is not None:
            features = self.scaler.transform(features)
        return self.model.predict(features)

    def _predict_sb3(self, features: np.ndarray) -> np.ndarray:
        """Prediction DQN : 0=HOLD, 1=BUY, 2=SELL.

        Le DQN attend 21 dimensions : 19 features normalisees
        + position courante + steps_in_position / 100.
        Pour une prediction ponctuelle on suppose position=0 (flat), steps=0.
        """
        results = []
        for row in features:
            obs = row.astype(np.float32)
            if self.norm_stats is not None:
                mean = self.norm_stats["mean"].values.astype(np.float32)
                std = self.norm_stats["std"].values.astype(np.float32)
                std = np.where(std == 0, 1.0, std)
                obs = (obs - mean) / std
            # Ajouter position (0=flat) et steps_in_position (0)
            obs = np.append(obs, [0.0, 0.0]).astype(np.float32)
            action, _ = self.model.predict(obs, deterministic=True)
            results.append(int(action))
        return np.array(results)

    def predict_proba(self, features: np.ndarray) -> np.ndarray | None:
        """Probabilites de prediction (sklearn only)."""
        if self.model is None:
            raise RuntimeError("Aucun modele charge")
        if self.model_type == "sklearn" and hasattr(self.model, "predict_proba"):
            if self.scaler is not None:
                features = self.scaler.transform(features)
            return self.model.predict_proba(features)
        return None


# Instance singleton
registry = ModelRegistry()
