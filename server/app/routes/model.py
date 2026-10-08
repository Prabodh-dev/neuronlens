from fastapi import APIRouter, Depends

from ..db import Database
from ..deps import get_db, get_meta, get_runtime
from ..errors import ApiError
from ..models import (InspectRequest, error_responses, InspectResponse, InspectToken,
                      SteerFeature, SteerRequest, SteerResponse, TokenFeature)

MAX_TOKENS = 256

router = APIRouter(tags=["model"], responses=error_responses(404, 413, 422, 500, 503))


def check_length(runtime, text: str, field: str) -> None:
    n = runtime.count_tokens(text)
    if n > MAX_TOKENS:
        raise ApiError("TEXT_TOO_LONG", f"{field} is {n} tokens; the limit is {MAX_TOKENS}")


def labels_for(db: Database, ids: set[int]) -> dict[int, str | None]:
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    rows = db.query(f"SELECT feature_id, label FROM features WHERE feature_id IN ({marks})",
                    tuple(sorted(ids)))
    return {r["feature_id"]: r["label"] for r in rows}


@router.post("/inspect", response_model=InspectResponse)
def inspect(req: InspectRequest, db: Database = Depends(get_db),
            meta: dict = Depends(get_meta), runtime=Depends(get_runtime)):
    check_length(runtime, req.text, "text")
    raw = runtime.inspect(req.text, req.top_n)
    labels = labels_for(db, {f["feature_id"] for t in raw for f in t["features"]})
    tokens = [
        InspectToken(index=t["index"], text=t["text"], is_outlier_position=t["index"] == 0,
                     features=[TokenFeature(feature_id=f["feature_id"],
                                            activation=f["activation"],
                                            label=labels.get(f["feature_id"]))
                               for f in t["features"]])
        for t in raw]
    return InspectResponse(model_name=meta["model_name"], tokens=tokens)


@router.post("/steer", response_model=SteerResponse)
def steer(req: SteerRequest, db: Database = Depends(get_db), runtime=Depends(get_runtime)):
    row = db.one("SELECT feature_id, label, max_activation, is_dead FROM features "
                 "WHERE feature_id = ?", (req.feature_id,))
    if row is None:
        raise ApiError("FEATURE_NOT_FOUND", f"feature {req.feature_id} not found")
    if row["is_dead"]:
        raise ApiError("INVALID_INPUT", f"feature {req.feature_id} is dead and cannot be steered")
    check_length(runtime, req.prompt, "prompt")
    strength = req.strength if req.mode == "amplify" else 0.0  # ablate ignores strength
    out = runtime.steer(req.prompt, req.feature_id, req.mode, strength,
                        row["max_activation"], req.max_new_tokens, req.seed)
    return SteerResponse(feature=SteerFeature(feature_id=row["feature_id"], label=row["label"]),
                         mode=req.mode, strength=strength, seed=req.seed,
                         baseline_text=out["baseline_text"], steered_text=out["steered_text"])
