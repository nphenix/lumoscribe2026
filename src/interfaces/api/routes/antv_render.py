from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field

from src.application.services.antv_rendering import AntvRenderingService
from src.shared.errors import AppError

router = APIRouter()


class AntvRenderRequest(BaseModel):
    input_root: str = Field(
        ...,
        min_length=1,
        description="输入目录（例如 data/intermediates/{id}/pic_to_json 或 intermediates/{id}/pic_to_json）",
    )
    chart_ids: list[str] | None = Field(default=None, description="要渲染的 chart_id 列表（为空则渲染全部）")
    theme: str | None = Field(default="whitepaper-default")
    force: bool = Field(default=False, description="是否强制重渲染（忽略 render_meta.json）")


@router.post(
    "/antv/render",
    summary="AntV 预渲染 chart_json（阶段3）",
    description="输入 pic_to_json 目录，读取其 chart_json/*.json 并输出 SVG/PNG 预渲染文件到同目录。",
)
def antv_render(payload: AntvRenderRequest = Body(...)) -> dict[str, Any]:
    raw = (payload.input_root or "").strip()
    p = Path(raw)
    if not p.exists():
        p2 = Path("data") / raw
        if p2.exists():
            p = p2
    if not p.exists() or not p.is_dir():
        raise HTTPException(status_code=400, detail=f"input_root 不存在: {raw}")
    try:
        return AntvRenderingService().render_from_chart_json_dir(
            kb_input_root=p,
            chart_ids=payload.chart_ids,
            theme=payload.theme,
            force=bool(payload.force),
        )
    except AppError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"code": exc.code, "message": exc.message, "details": exc.details})

