"""Phase 4 steering checks (X1 to X4). The ML functions in ml/neuronlens/steering_checks.py
do the work; this layer does what /steer does (DB lookup, dead-feature and length checks)
and holds the model for the whole check, since each one runs several generations.
Steering is a test of what a feature does, not a fix."""
from fastapi import APIRouter, Depends

from ..db import Database
from ..deps import get_db, get_runtime
from ..errors import ApiError
from ..models import (PromptOnlyRequest, PromptOnlyResponse, SteerCheckResponse,
                      SteerCompareResponse,
                      SteerFeature, SteerRequest, SteerSweepRequest, SteerSweepResponse,
                      error_responses)
from ..services.checks import steering_checks
from .model import check_length

router = APIRouter(tags=["steering checks"], responses=error_responses(404, 413, 422, 500, 503))


def steerable_feature(db: Database, feature_id: int):
    row = db.one("SELECT feature_id, label, max_activation, is_dead FROM features "
                 "WHERE feature_id = ?", (feature_id,))
    if row is None:
        raise ApiError("FEATURE_NOT_FOUND", f"feature {feature_id} not found")
    if row["is_dead"]:
        raise ApiError("INVALID_INPUT", f"feature {feature_id} is dead and cannot be steered")
    return row


def run_check(runtime, fn, *args, **kwargs) -> dict:
    sc = steering_checks()
    try:
        with runtime.exclusive():
            return fn(sc, runtime, *args, **kwargs)
    except sc.SteeringCheckError as e:
        raise ApiError(e.code, str(e)) from e


def with_feature(out: dict, row) -> dict:
    out = dict(out)
    out.pop("feature_id", None)
    out["feature"] = SteerFeature(feature_id=row["feature_id"], label=row["label"])
    return out


@router.post("/steer/check", response_model=SteerCheckResponse)
def check(req: SteerRequest, db: Database = Depends(get_db), runtime=Depends(get_runtime)):
    """X1. Baseline and steered with the same seed, plus the target feature's activation on
    each finished text (re-encoded without steering). Not a quality score."""
    row = steerable_feature(db, req.feature_id)
    check_length(runtime, req.prompt, "prompt")
    out = run_check(runtime, lambda sc, rt: sc.steering_check(
        rt, req.prompt, req.feature_id, req.mode, req.strength, row["max_activation"],
        req.max_new_tokens, req.seed))
    return SteerCheckResponse(**with_feature(out, row))


@router.post("/steer/compare", response_model=SteerCompareResponse)
def compare(req: SteerRequest, db: Database = Depends(get_db), runtime=Depends(get_runtime)):
    """X3. Baseline and steered with the same seed, compared token by token."""
    row = steerable_feature(db, req.feature_id)
    check_length(runtime, req.prompt, "prompt")
    out = run_check(runtime, lambda sc, rt: sc.seed_matched_compare(
        rt, req.prompt, req.feature_id, req.mode, req.strength, row["max_activation"],
        req.max_new_tokens, req.seed))
    return SteerCompareResponse(**with_feature(out, row))


@router.post("/steer/sweep", response_model=SteerSweepResponse)
def sweep(req: SteerSweepRequest, db: Database = Depends(get_db),
          runtime=Depends(get_runtime)):
    """X2. Amplify at several strengths (default 0 to 2) with one seed; flags degenerate text.
    One generation per strength plus the baseline, so this is the slowest call."""
    row = steerable_feature(db, req.feature_id)
    check_length(runtime, req.prompt, "prompt")
    out = run_check(runtime, lambda sc, rt: sc.strength_sweep(
        rt, req.prompt, req.feature_id, row["max_activation"], req.strengths,
        req.max_new_tokens, req.seed))
    return SteerSweepResponse(**with_feature(out, row))


@router.post("/steer/prompt-only", response_model=PromptOnlyResponse)
def prompt_only(req: PromptOnlyRequest, runtime=Depends(get_runtime)):
    """X4. What the model writes from the prompt alone: no steering, same seed. Its text
    equals the baseline_text of the steering calls with the same prompt, seed and length."""
    check_length(runtime, req.prompt, "prompt")
    out = run_check(runtime, lambda sc, rt: sc.prompt_only_baseline(
        rt, req.prompt, req.max_new_tokens, req.seed))
    return PromptOnlyResponse(**out)
