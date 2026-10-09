from fastapi import Request

from .db import Database
from .errors import ApiError


def get_db(request: Request) -> Database:
    return request.app.state.db


def get_meta(request: Request) -> dict:
    return request.app.state.meta


def get_runtime(request: Request):
    runtime = request.app.state.runtime
    if runtime is None:
        reason = request.app.state.runtime_error or "model is not loaded"
        raise ApiError("MODEL_NOT_LOADED", f"Inspect and Steer are unavailable: {reason}")
    return runtime


def get_cofire(request: Request):
    return request.app.state.cofire
