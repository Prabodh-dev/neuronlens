from fastapi import APIRouter, Depends, Request

from ..deps import get_meta
from ..models import ErrorResponse, Health

router = APIRouter(tags=["health"], responses={500: {"model": ErrorResponse}})


@router.get("/health", response_model=Health)
def health(request: Request, meta: dict = Depends(get_meta)):
    return Health(
        status="ok",
        mode=request.app.state.settings.mode,
        is_synthetic=meta["is_synthetic"] == "true",
        schema_version=meta["schema_version"],
        model_name=meta["model_name"],
        hook_layer=int(meta["hook_layer"]),
        d_model=int(meta["d_model"]),
        d_sae=int(meta["d_sae"]),
        k=int(meta["k"]),
        run_id=meta["run_id"],
    )
