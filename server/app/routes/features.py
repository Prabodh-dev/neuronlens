import json
from typing import Literal, Optional

from fastapi import APIRouter, Depends, Query

from ..db import Database
from ..deps import get_db
from ..errors import ApiError
from ..models import (Example, error_responses, FeatureDetail, FeatureList, FeatureListItem,
                      LabelStatus, LogitEntry, Logits)

router = APIRouter(tags=["features"], responses=error_responses(422, 500))

SORT_COLUMNS = {"feature_id": "feature_id", "density": "density",
                "max_activation": "max_activation"}


def list_item(r) -> dict:
    return {"feature_id": r["feature_id"], "label": r["label"],
            "label_status": r["label_status"], "density": r["density"],
            "max_activation": r["max_activation"], "is_dead": bool(r["is_dead"])}


def feature_detail(db: Database, row) -> FeatureDetail:
    fid = row["feature_id"]
    examples = [
        Example(rank=e["rank"], tokens=json.loads(e["tokens_json"]),
                activations=json.loads(e["activations_json"]),
                peak_index=e["peak_index"], max_activation=e["max_activation"])
        for e in db.query("SELECT * FROM feature_examples WHERE feature_id = ? ORDER BY rank",
                          (fid,))]
    logits = {"promoted": [], "suppressed": []}
    for r in db.query("SELECT direction, token, score FROM feature_logits "
                      "WHERE feature_id = ? ORDER BY direction, rank", (fid,)):
        logits[r["direction"]].append(LogitEntry(token=r["token"], score=r["score"]))
    return FeatureDetail(**list_item(row), mean_activation=row["mean_activation"],
                         notes=row["notes"], examples=examples, logits=Logits(**logits))


@router.get("/features", response_model=FeatureList)
def list_features(
    query: str = Query("", description="Case-insensitive label substring; digits also match feature_id"),
    status: Optional[LabelStatus] = None,
    include_dead: bool = False,
    sort: Literal["feature_id", "density", "max_activation"] = "feature_id",
    order: Literal["asc", "desc"] = "asc",
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Database = Depends(get_db),
):
    where, params = [], []
    q = query.strip()
    if q:
        clause = "instr(lower(coalesce(label, '')), lower(?)) > 0"
        params.append(q)
        if q.isascii() and q.isdigit() and len(q) <= 18:
            clause += " OR feature_id = ?"
            params.append(int(q))
        where.append(f"({clause})")
    if status:
        where.append("label_status = ?")
        params.append(status)
    if not include_dead:
        where.append("is_dead = 0")
    wsql = f" WHERE {' AND '.join(where)}" if where else ""
    total = db.one(f"SELECT count(*) AS n FROM features{wsql}", tuple(params))["n"]
    direction = "DESC" if order == "desc" else "ASC"
    rows = db.query(
        f"SELECT * FROM features{wsql} ORDER BY {SORT_COLUMNS[sort]} {direction}, feature_id ASC "
        "LIMIT ? OFFSET ?", tuple(params) + (limit, offset))
    return FeatureList(total=total, limit=limit, offset=offset,
                       items=[FeatureListItem(**list_item(r)) for r in rows])


@router.get("/features/random", response_model=FeatureDetail,
            responses=error_responses(404))
def random_feature(db: Database = Depends(get_db)):
    # Registered before /features/{feature_id} so "random" is not parsed as an id.
    row = db.one("SELECT * FROM features WHERE is_dead = 0 ORDER BY random() LIMIT 1")
    if row is None:
        raise ApiError("NOT_FOUND", "no live features in this database")
    return feature_detail(db, row)


@router.get("/features/{feature_id}", response_model=FeatureDetail)
def get_feature(feature_id: int, db: Database = Depends(get_db)):
    row = db.one("SELECT * FROM features WHERE feature_id = ?", (feature_id,))
    if row is None:
        raise ApiError("FEATURE_NOT_FOUND", f"feature {feature_id} not found")
    return feature_detail(db, row)
