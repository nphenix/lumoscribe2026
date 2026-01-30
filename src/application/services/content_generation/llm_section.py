"""章节内容生成（LLM 调用封装，支持 token 级流式）（T042/T096）。"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Callable

from src.application.services.content_generation.text_utils import (
    strip_leading_title_heading,
    strip_model_think,
)
from src.application.services.llm_runtime_service import LLMRuntimeService


class SectionLLMGenerator:
    """封装章节生成的 LLM 调用细节（同步 invoke + 异步 astream）。"""

    def __init__(self, *, llm_runtime_service: LLMRuntimeService, callsite_scope: str):
        self.llm_runtime_service = llm_runtime_service
        self.callsite_scope = callsite_scope

    async def generate(
        self,
        *,
        section_id: str,
        section_title: str,
        payload: dict[str, Any],
        stream_tokens: bool,
        on_event: Callable[[dict[str, Any]], Any] | None = None,
    ) -> tuple[str, int]:
        """生成章节内容。

        Returns:
            (generated_content, tokens_used_estimate)
        """
        runnable = self.llm_runtime_service.build_runnable_for_callsite(
            self.callsite_scope,
            force_streaming=bool(stream_tokens),
        )

        if stream_tokens:
            parts: list[str] = []
            if on_event is not None:
                await self._emit(
                    on_event,
                    {
                        "type": "llm_stream_start",
                        "section_id": section_id,
                        "section_title": section_title,
                    },
                )
            async with self.llm_runtime_service.acquire_llm_slot(self.callsite_scope):
                async for chunk in runnable.astream(payload):
                    s = "" if chunk is None else str(chunk)
                    if not s:
                        continue
                    parts.append(s)
                    if on_event is not None:
                        await self._emit(
                            on_event,
                            {
                                "type": "token",
                                "section_id": section_id,
                                "section_title": section_title,
                                "content": s,
                            },
                        )
            if on_event is not None:
                await self._emit(
                    on_event,
                    {
                        "type": "llm_stream_done",
                        "section_id": section_id,
                        "section_title": section_title,
                    },
                )
            raw = "".join(parts)
        else:
            loop = asyncio.get_running_loop()

            def _do():
                return runnable.invoke(payload)

            async with self.llm_runtime_service.acquire_llm_slot(self.callsite_scope):
                raw = await loop.run_in_executor(None, _do)

        text = strip_model_think(str(raw or ""))
        text = strip_leading_title_heading(text, title=section_title)
        tokens_used = max(0, len(text) // 4)
        return text, tokens_used

    async def _emit(self, on_event: Callable[[dict[str, Any]], Any], payload: dict[str, Any]) -> None:
        try:
            r = on_event(payload)
            if asyncio.iscoroutine(r):
                await r
        except Exception:
            return


GLOSSARY_PLACEHOLDER_RE = re.compile(r"(概念一|概念二|概念三|制度一|机制一|规划一|指标一|指数一|本概念强调)")

