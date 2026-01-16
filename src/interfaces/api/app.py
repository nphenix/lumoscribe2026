from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.shared.config import get_settings
from src.shared.db import init_db, make_engine, make_session_factory
from src.shared.errors import (
    AppError,
    ERROR_INTERNAL,
    ERROR_VALIDATION,
    error_response,
)
from src.shared.logging import configure_logging, get_logger
from src.shared.request_id import get_request_id, new_request_id, set_request_id

from src.interfaces.api.routes.health import router as health_router
from src.interfaces.api.routes.jobs import router as jobs_router


log = get_logger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(title="Lumoscribe2026 API", version="0.1.0")

    engine = make_engine(settings.sqlite_path)
    init_db(engine)
    app.state.engine = engine
    app.state.session_factory = make_session_factory(engine)

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or new_request_id()
        set_request_id(rid)
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response(
                code=exc.code,
                message=exc.message,
                request_id=get_request_id(),
                details=exc.details,
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content=error_response(
                code=ERROR_VALIDATION,
                message="request validation failed",
                request_id=get_request_id(),
                details={"errors": exc.errors()},
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response(
                code=f"http_{exc.status_code}",
                message=exc.detail if isinstance(exc.detail, str) else "http error",
                request_id=get_request_id(),
            ),
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        log.exception("unhandled_error")
        return JSONResponse(
            status_code=500,
            content=error_response(
                code=ERROR_INTERNAL,
                message="internal server error",
                request_id=get_request_id(),
            ),
        )

    app.include_router(health_router, prefix="/v1")
    app.include_router(jobs_router, prefix="/v1")

    return app
