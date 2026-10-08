import logging

from ..config import Settings
from ..db import Database

log = logging.getLogger("neuronlens")


def build_runtime(settings: Settings, db: Database):
    """Returns (runtime, error). In real mode a failed load leaves runtime None so the
    DB-backed pages keep working and /inspect and /steer answer MODEL_NOT_LOADED."""
    if settings.mode == "mock":
        from .mock import MockRuntime
        return MockRuntime(db), None
    try:
        from .real import RealRuntime
        return RealRuntime.load(settings.ckpt), None
    except Exception as e:  # missing torch, checkpoint, GPU memory, ...
        log.error("real Runtime failed to load: %s", e)
        return None, str(e)
