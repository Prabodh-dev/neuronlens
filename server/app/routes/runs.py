from typing import Optional

from fastapi import APIRouter, Depends, Query

from ..db import Database
from ..deps import get_db, get_meta
from ..errors import ApiError
from ..models import (CurvePoint, error_responses, DensityBin, MetricsResponse, MetricValue, Run,
                      RunList, SteeringExample, SteeringExampleList)

router = APIRouter(tags=["runs"], responses=error_responses(404, 422, 500))

METRIC_ORDER = ("fvu", "l0", "dead_pct", "ce_original", "ce_zero_ablation",
                "ce_reconstructed", "ce_recovered")


@router.get("/steering-examples", response_model=SteeringExampleList)
def steering_examples(feature_id: Optional[int] = Query(None, ge=0),
                      db: Database = Depends(get_db)):
    sql, params = "SELECT * FROM steering_examples", ()
    if feature_id is not None:
        sql, params = sql + " WHERE feature_id = ?", (feature_id,)
    rows = db.query(sql + " ORDER BY is_curated DESC, example_id ASC", params)
    return SteeringExampleList(items=[
        SteeringExample(**{k: r[k] for k in r.keys() if k != "is_curated"},
                        is_curated=bool(r["is_curated"])) for r in rows])


@router.get("/runs", response_model=RunList)
def runs(db: Database = Depends(get_db), meta: dict = Depends(get_meta)):
    primary = meta["run_id"]
    rows = db.query("SELECT * FROM runs ORDER BY (run_id = ?) DESC, created_at DESC, run_id",
                    (primary,))
    return RunList(items=[Run(**dict(r), is_primary=r["run_id"] == primary) for r in rows])


@router.get("/metrics", response_model=MetricsResponse)
def metrics(run_id: Optional[str] = None, db: Database = Depends(get_db),
            meta: dict = Depends(get_meta)):
    run_id = run_id or meta["run_id"]
    if db.one("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)) is None:
        raise ApiError("RUN_NOT_FOUND", f"run {run_id!r} not found")
    order = {name: i for i, name in enumerate(METRIC_ORDER)}
    values = sorted(db.query("SELECT name, value FROM metrics WHERE run_id = ?", (run_id,)),
                    key=lambda r: (order.get(r["name"], len(order)), r["name"]))
    hist = db.query("SELECT bin_lo, bin_hi, count FROM density_hist WHERE run_id = ? "
                    "ORDER BY bin_lo", (run_id,))
    curve = db.query("SELECT step, loss, fvu, dead_pct FROM training_curve WHERE run_id = ? "
                     "ORDER BY step", (run_id,))
    return MetricsResponse(
        run_id=run_id,
        metrics=[MetricValue(name=r["name"], value=r["value"]) for r in values],
        density_hist=[DensityBin(**dict(r)) for r in hist],
        training_curve=[CurvePoint(**dict(r)) for r in curve])
