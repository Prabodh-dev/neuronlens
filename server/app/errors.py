import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

log = logging.getLogger("neuronlens")

STATUS = {
    "INVALID_INPUT": 422,
    "TEXT_TOO_LONG": 413,
    "FEATURE_NOT_FOUND": 404,
    "RUN_NOT_FOUND": 404,
    "NOT_FOUND": 404,
    "MODEL_NOT_LOADED": 503,
    "SCHEMA_MISMATCH": 500,
    "INTERNAL": 500,
}


class ApiError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = STATUS[code]


def error_response(code: str, message: str, status: int | None = None) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}},
                        status_code=status or STATUS[code])


def _validation_message(exc: RequestValidationError) -> str:
    parts = []
    for e in exc.errors():
        loc = ".".join(str(x) for x in e.get("loc", ()))
        msg = str(e.get("msg", "invalid")).removeprefix("Value error, ")
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(parts) or "invalid input"


class CatchAllMiddleware(BaseHTTPMiddleware):
    """Turns any unhandled exception into an INTERNAL error. Added inside CORS so the
    error response still carries CORS headers. The traceback is logged, never sent."""

    async def dispatch(self, request, call_next):
        try:
            return await call_next(request)
        except Exception:
            log.exception("unhandled error on %s %s", request.method, request.url.path)
            return error_response("INTERNAL", "Internal server error")


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError):
        return error_response(exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        return error_response("INVALID_INPUT", _validation_message(exc))

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        if exc.status_code == 404:
            return error_response("NOT_FOUND", "Not found")
        return error_response("INVALID_INPUT", str(exc.detail), exc.status_code)
