from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field, ValidationError

from src.application.services.llm_runtime_service import LLMRuntimeService
from src.shared.constants.prompts import DEFAULT_PROMPTS, SCOPE_CONTENT_GENERATION_SECTION_STRUCTURED
from src.shared.logging import get_logger, log_extra


log = get_logger(__name__)


class StructuredOutlineItemOutput(BaseModel):
    outline_key: str = Field(..., description="大纲条目 key（用于映射到模板骨架）")
    markdown_body: str = Field(..., description="条目正文（不包含 '- <title>' 这一行）")
    chart_ids: list[str] = Field(default_factory=list, description="该条目需要绑定的图表 id 列表（不含 [Chart: ] 包装）")


class SectionStructuredGenerator:
    def __init__(
        self,
        *,
        llm_runtime_service: LLMRuntimeService,
        callsite_scope: str,
        prompt_scope: str = SCOPE_CONTENT_GENERATION_SECTION_STRUCTURED,
    ) -> None:
        self._llm_runtime = llm_runtime_service
        self._callsite_scope = callsite_scope
        self._prompt_scope = prompt_scope

    async def generate_outline_item(
        self,
        *,
        outline_key: str,
        title: str,
        skeleton: str,
        context: str,
        chart_candidates_block: str | None = None,
        document_title: str,
        required_chart_ids: list[str],
        available_chart_ids: list[str],
    ) -> tuple[StructuredOutlineItemOutput, str]:
        prompt = None
        try:
            prompt = self._llm_runtime.prompt_repository.get_active_prompt(self._prompt_scope)
        except Exception:
            prompt = None

        system_prompt = None
        if prompt is not None and (prompt.content or "").strip():
            system_prompt = prompt.content
        else:
            system_prompt = str(DEFAULT_PROMPTS.get(self._prompt_scope, {}).get("content") or "")

        model = self._llm_runtime.get_model_for_callsite(self._callsite_scope)
        if not isinstance(model, BaseChatModel):
            raise TypeError("callsite model is not a chat model")

        required = ", ".join([f"[Chart: {cid}]" for cid in required_chart_ids if str(cid or "").strip()])
        available = ", ".join([f"[Chart: {cid}]" for cid in available_chart_ids if str(cid or "").strip()])

        candidates_txt = (chart_candidates_block or "").strip()
        user_message = "\n".join(
            [
                f"文档标题：{document_title or ''}".strip(),
                f"条目标题：{title or ''}".strip(),
                "",
                ("图表候选（仅供你选择与引用）：\n" + candidates_txt).strip() if candidates_txt else "",
                "" if candidates_txt else "",
                "模板骨架（你必须遵守，不要在输出里重复它）：",
                skeleton or "",
                "",
                f"必须使用的图表锚点列表：{required}".strip(),
                f"可用图表锚点列表：{available}".strip(),
                "",
                "RAG 上下文：",
                context or "",
            ]
        ).strip()

        try:
            agent = create_agent(
                model=model,
                tools=[],
                system_prompt=system_prompt,
                response_format=ToolStrategy(StructuredOutlineItemOutput),
            )
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": user_message,
                    }
                ]
            }
            result = await agent.ainvoke(payload) if hasattr(agent, "ainvoke") else await asyncio.get_running_loop().run_in_executor(None, lambda: agent.invoke(payload))
            structured: StructuredOutlineItemOutput = result["structured_response"]
            return structured, "tool_calling"
        except Exception as exc:
            log.warning(
                "content_generation.structured_output_failed",
                extra=log_extra(scope=self._callsite_scope, prompt_scope=self._prompt_scope, error=str(exc)),
            )

        parser_prompt = "\n".join(
            [
                system_prompt.strip(),
                "",
                "请只输出 JSON，不要输出任何解释文字。",
                "JSON 必须符合以下结构：",
                json.dumps(StructuredOutlineItemOutput.model_json_schema(), ensure_ascii=False),
                "",
                user_message,
            ]
        ).strip()
        try:
            resp = await asyncio.get_running_loop().run_in_executor(None, lambda: model.invoke(parser_prompt))
            raw = "" if resp is None else str(resp)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.strip("`").strip()
            obj = StructuredOutlineItemOutput.model_validate_json(raw)
            return obj, "json_parser"
        except (ValidationError, Exception) as exc:
            raise RuntimeError(f"structured generation failed: {exc}") from exc
