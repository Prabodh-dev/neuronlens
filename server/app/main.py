import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import Settings
from .db import Database, load_meta
from .errors import CatchAllMiddleware, install_error_handlers
from .routes import features, health, model, runs, steering
from .services import build_runtime

API_PREFIX = "/api/v1"
log = logging.getLogger("neuronlens")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db = Database(settings.db_path)  # fail fast: missing/unreadable DB
        try:
            app.state.meta = load_meta(db)  # fail fast: SCHEMA_MISMATCH
        except Exception:
            db.close()
            raise
        app.state.db = db
        app.state.runtime, app.state.runtime_error = build_runtime(settings, db)
        log.info("neuronlens ready: mode=%s run=%s", settings.mode, app.state.meta["run_id"])
        try:
            yield
        finally:
            db.close()

    app = FastAPI(title="NeuronLens API", version="1.0", lifespan=lifespan)
    app.state.settings = settings
    install_error_handlers(app)
    app.add_middleware(CatchAllMiddleware)
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                       allow_methods=["*"], allow_headers=["*"])
    app.include_router(health.router, prefix=API_PREFIX)
    app.include_router(features.router, prefix=API_PREFIX)
    app.include_router(model.router, prefix=API_PREFIX)
    app.include_router(steering.router, prefix=API_PREFIX)
    app.include_router(runs.router, prefix=API_PREFIX)
    return app


app = create_app()
