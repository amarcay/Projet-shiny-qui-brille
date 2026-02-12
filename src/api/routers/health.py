"""Router : health check."""

from datetime import datetime
from fastapi import APIRouter

from core.registry import registry
from schemas.responses import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def health():
    """Statut de l'API."""
    return HealthResponse(
        status="ok" if registry.model is not None else "no_model",
        model_loaded=registry.model is not None,
        model_version=registry.version,
        timestamp=datetime.now().isoformat(),
    )
