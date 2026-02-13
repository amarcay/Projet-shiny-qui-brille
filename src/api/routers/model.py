"""Router : informations et gestion du modele."""

from fastapi import APIRouter, HTTPException, Query

from core.registry import registry
from schemas.responses import ModelInfoResponse

router = APIRouter(prefix="/model", tags=["Model"])


@router.get("/info", response_model=ModelInfoResponse)
def model_info():
    """Informations sur le modele charge."""
    if registry.model is None:
        raise HTTPException(status_code=503, detail="Aucun modele charge")

    return ModelInfoResponse(
        version=registry.version,
        model_type=registry.model_type,
        model_name=registry.model_name,
        features=registry.feature_cols or [],
        available_versions=registry.list_versions(),
        loaded_at=registry.loaded_at,
    )


@router.post("/load")
def load_model(
    version: str = Query(..., description="Version du modele a charger (ex: v1, v2)"),
):
    """Charge une version specifique du modele."""
    try:
        registry.load(version)
        return {"message": f"Modele {version} charge", "model_type": registry.model_type}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
