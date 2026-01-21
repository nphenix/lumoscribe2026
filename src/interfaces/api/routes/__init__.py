"""API 路由集合。"""

from src.interfaces.api.routes.health import router as health_router
from src.interfaces.api.routes.intermediates import router as intermediates_router
from src.interfaces.api.routes.jobs import router as jobs_router
from src.interfaces.api.routes.llm import router as llm_router
from src.interfaces.api.routes.prompts import router as prompts_router
from src.interfaces.api.routes.sources import router as sources_router
from src.interfaces.api.routes.targets import router as targets_router
from src.interfaces.api.routes.templates import router as templates_router

__all__ = [
    "health_router",
    "intermediates_router",
    "jobs_router",
    "llm_router",
    "prompts_router",
    "sources_router",
    "targets_router",
    "templates_router",
]
