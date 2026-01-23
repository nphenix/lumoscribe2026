from __future__ import annotations

import os

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
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

log = get_logger(__name__)


def _seed_llm_callsites(session_factory) -> None:
    """从代码种子自动补齐 callsites（仅新增缺失项，不覆盖管理员配置）。"""
    from uuid import uuid4

    from src.application.repositories.llm_call_site_repository import LLMCallSiteRepository
    from src.domain.entities.llm_call_site import LLMCallSite
    from src.shared.constants.llm_callsites import DEFAULT_CALLSITES

    with session_factory() as session:
        repo = LLMCallSiteRepository(session)
        for key, data in DEFAULT_CALLSITES.items():
            expected_model_kind = data.get("expected_model_kind")
            prompt_scope = data.get("prompt_scope")
            description = data.get("description")

            existing = repo.get_by_key(key)
            if existing is None:
                callsite = LLMCallSite(
                    id=str(uuid4()),
                    key=key,
                    expected_model_kind=expected_model_kind,
                    provider_id=None,
                    config_json=None,
                    prompt_scope=prompt_scope,
                    enabled=True,
                    description=description,
                )
                repo.create(callsite)
                continue

            # 只同步“代码侧事实”：expected_model_kind/description；不覆盖管理员配置的 model_id/config_json/enabled
            existing.expected_model_kind = expected_model_kind
            existing.description = description
            if existing.prompt_scope is None and prompt_scope is not None:
                existing.prompt_scope = prompt_scope
            repo.update(existing)


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(title="Lumoscribe2026 API", version="0.1.0")

    # 配置 CORS
    # 注意：`allow_credentials=True` 时，浏览器不接受 `Access-Control-Allow-Origin: *`。
    # 因此默认仅放行本地开发前端（可用环境变量覆盖）。
    raw_origins = os.getenv(
        "LUMO_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    )
    allow_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    engine = make_engine(settings.sqlite_path)
    init_db(engine)
    app.state.engine = engine
    app.state.session_factory = make_session_factory(engine)
    _seed_llm_callsites(app.state.session_factory)

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
                details={"error": str(exc), "type": exc.__class__.__name__},
            ),
        )

    # 路由按模式选择性挂载，避免在 kb_* 模式下导入不必要依赖（例如上传所需 python-multipart）
    api_mode = os.getenv("LUMO_API_MODE", "full").strip().lower()

    # 始终可用：health（用于探活）
    from src.interfaces.api.routes.health import router as health_router

    app.include_router(health_router, prefix="/v1")

    # KB：按模式挂载（支持“建库/查询”两个独立端口部署）
    if api_mode in {"full", "kb_admin", "kb_query"}:
        from src.interfaces.api.routes.kb import router_admin as kb_admin_router
        from src.interfaces.api.routes.kb import router_query as kb_query_router

        if api_mode in {"full", "kb_admin"}:
            app.include_router(kb_admin_router, prefix="/v1")
        if api_mode in {"full", "kb_query"}:
            app.include_router(kb_query_router, prefix="/v1")

    # 其余业务路由：仅 full 模式加载（避免 kb_* 独立服务被无关依赖阻塞）
    if api_mode == "full":
        from src.interfaces.api.routes.intermediates import router as intermediates_router
        from src.interfaces.api.routes.jobs import router as jobs_router
        from src.interfaces.api.routes.llm import router as llm_router
        from src.interfaces.api.routes.prompts import router as prompts_router
        from src.interfaces.api.routes.sources import router as sources_router
        from src.interfaces.api.routes.targets import router as targets_router
        from src.interfaces.api.routes.templates import router as templates_router

        app.include_router(intermediates_router, prefix="/v1")
        app.include_router(jobs_router, prefix="/v1")
        app.include_router(llm_router, prefix="/v1")
        app.include_router(prompts_router, prefix="/v1")
        app.include_router(sources_router, prefix="/v1")
        app.include_router(targets_router, prefix="/v1")
        app.include_router(templates_router, prefix="/v1")

    return app
