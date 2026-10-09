import logging

from ..config import Settings
from ..db import Database, load_meta

log = logging.getLogger("neuronlens")


def build_runtime(settings: Settings, db: Database):
    """Returns (runtime, error). In real mode a failed load leaves runtime None so the
    DB-backed pages keep working and /inspect and /steer answer MODEL_NOT_LOADED."""
    if settings.mode == "mock":
        from .mock import MockRuntime
        return MockRuntime(db), None
    try:
        from .real import RealRuntime
        rt = RealRuntime.load(settings.ckpt)
        problem = _meta_mismatch(rt, load_meta(db))
        if problem:
            log.error("real Runtime does not match the database: %s", problem)
            return None, problem
        return rt, None
    except Exception as e:  # missing torch, checkpoint, GPU memory, ...
        log.error("real Runtime failed to load: %s", e)
        return None, str(e)


def _meta_mismatch(rt, meta):
    """/health reports the database's model name, layer and width, so the loaded model and
    checkpoint must agree with them. Attributes the runtime does not expose are not checked."""
    pairs = [("model_name", "model_name", str), ("hook_layer", "hook_layer", int),
             ("d_model", "d_model", int)]
    for attr, key, cast in pairs:
        have = getattr(rt, attr, None)
        if have is not None and cast(have) != cast(meta[key]):
            return (f"the loaded model has {attr} {have!r} but the database says "
                    f"{cast(meta[key])!r}; use the database and checkpoint of the same run")
    return None
