"""大纲优化服务（基于 LangChain Agent 封装）。

本服务使用 LangChain 1.0 的 Agent 能力，提供大纲润色与优化功能。
最佳实践（基于 Context7 查询结果）：
- 使用 `create_agent()` 函数创建 Agent
- 使用 `ToolStrategy` 约束结构化输出
- 通过 `response_format` 参数定义输出 Schema
- 通过 `system_prompt` 参数传递系统消息
- 通过 `result["structured_response"]` 访问结构化输出

参考: https://docs.langchain.com/oss/python/langchain/structured-output
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from src.application.repositories.llm_call_site_repository import LLMCallSiteRepository
from src.application.repositories.prompt_repository import PromptRepository
from src.application.services.llm_runtime_service import LLMRuntimeService
from src.domain.entities.llm_call_site import LLMCallSite
from src.domain.entities.prompt import Prompt
from src.shared.errors import AppError
from src.shared.logging import logger

from .prompts import (
    SCOPE_OUTLINE_POLISH,
    SYSTEM_PROMPT_TEMPLATE,
    USER_PROMPT_TEMPLATE,
    DEFAULT_INDUSTRY,
    DEFAULT_REPORT_TYPE,
    DEFAULT_LANGUAGE,
    DEFAULT_STYLE,
)
from .schema import (
    OutlinePolishInput,
    OutlinePolishOutput,
    OutlinePolishResult,
)

logger = logging.getLogger(__name__)


class OutlinePolishServiceError(AppError):
    """大纲优化服务错误"""

    pass


class PolishedOutline(BaseModel):
    """大纲润色输出结构（用于 ToolStrategy）。

    符合 LangChain Structured Output 最佳实践：
    - 使用 Pydantic BaseModel 定义输出 Schema
    - 通过 ToolStrategy 约束输出格式
    """

    polished_outline: str = Field(
        ...,
        description="优化后的大纲（Markdown 格式）",
    )
    changes_summary: list[str] = Field(
        default_factory=list,
        description="修改摘要列表",
    )
    structure_integrity: bool = Field(
        default=True,
        description="结构完整性检查结果",
    )
    core_keywords_preserved: bool = Field(
        default=True,
        description="核心关键词是否保留",
    )
    recognized_requirements: list[str] = Field(
        default_factory=list,
        description="识别出的用户要求列表",
    )
    original_structure: list[str] = Field(
        default_factory=list,
        description="原始章节结构（识别出的）",
    )


class OutlinePolishService:
    """大纲优化服务（基于 LangChain Agent）。

    职责：
    1. 使用 create_agent() + ToolStrategy 封装 Agent
    2. 动态加载提示词（从数据库或常量）
    3. 通过 system_prompt 参数传递系统消息
    4. 执行大纲润色并返回结构化输出

    最佳实践参考:
    - https://docs.langchain.com/oss/python/langchain/structured-output
    - https://docs.langchain.com/oss/python/langchain/agents
    """

    def __init__(
        self,
        prompt_service: PromptRepository,
        llm_call_site_repository: LLMCallSiteRepository,
        llm_runtime_service: LLMRuntimeService,
    ):
        """初始化服务。

        Args:
            prompt_service: 提示词仓库（用于动态加载提示词）
            llm_call_site_repository: LLM 调用点仓库（用于获取模型配置）
            llm_runtime_service: LLM 运行时服务（用于获取模型实例）
        """
        self._prompt_service = prompt_service
        self._call_site_repo = llm_call_site_repository
        self._llm_runtime = llm_runtime_service

    async def polish_outline(
        self,
        input_data: OutlinePolishInput,
    ) -> OutlinePolishResult:
        """执行大纲润色。

        Args:
            input_data: 大纲润色输入参数

        Returns:
            OutlinePolishResult: 润色结果
        """
        try:
            # Step 1: 获取提示词（从数据库或常量）
            prompt = await self._get_prompt(SCOPE_OUTLINE_POLISH)
            system_prompt = self._render_system_prompt(
                prompt.content if prompt else SYSTEM_PROMPT_TEMPLATE,
                input_data,
            )

            # Step 2: 获取 LLM 配置（从 CallSite）- 使用 scope 作为 key
            call_site = self._call_site_repo.get_by_scope(SCOPE_OUTLINE_POLISH)
            if not call_site:
                raise OutlinePolishServiceError(
                    code="callsite_not_found",
                    message=f"未找到 CallSite 配置: {SCOPE_OUTLINE_POLISH}",
                    status_code=404,
                )

            # Step 3: 获取模型实例（通过 LLMRuntimeService）
            # 使用 get_model_for_callsite，key 优先从 config 获取，否则回退到 scope
            callsite_key = call_site.key or SCOPE_OUTLINE_POLISH
            model: BaseChatModel = self._llm_runtime.get_model_for_callsite(callsite_key)

            # Step 4: 创建 Agent（使用 create_agent + ToolStrategy）
            # 遵循 LangChain 最佳实践
            agent = create_agent(
                model=model,
                tools=[],  # 当前无自定义工具，可扩展
                system_prompt=system_prompt,
                response_format=ToolStrategy(PolishedOutline),
            )

            # Step 5: 渲染用户消息
            user_message = self._render_user_prompt(
                USER_PROMPT_TEMPLATE,
                input_data,
            )

            # Step 6: 调用 Agent
            # 遵循 LangChain 最佳实践：通过 messages 参数传递用户输入
            result = agent.invoke({
                "messages": [{"role": "user", "content": user_message}],
            })

            # Step 7: 提取结构化输出
            # 遵循 LangChain 最佳实践：通过 structured_response 访问输出
            structured_response: PolishedOutline = result["structured_response"]

            # Step 8: 构建输出
            output = OutlinePolishOutput(
                polished_outline=structured_response.polished_outline,
                changes_summary=structured_response.changes_summary,
                structure_integrity=structured_response.structure_integrity,
                core_keywords_preserved=structured_response.core_keywords_preserved,
                recognized_requirements=structured_response.recognized_requirements,
                original_structure=structured_response.original_structure,
            )

            return OutlinePolishResult(
                success=True,
                input=input_data,
                output=output,
                error=None,
            )

        except OutlinePolishServiceError:
            raise
        except AppError:
            raise
        except Exception as e:
            logger.error(f"大纲润色失败: {e}", exc_info=True)
            return OutlinePolishResult(
                success=False,
                input=input_data,
                output=None,
                error=str(e),
            )

    async def _get_prompt(self, scope: str) -> Prompt | None:
        """获取提示词。

        优先从数据库加载，若不存在则返回 None（使用常量作为 Fallback）。

        Args:
            scope: 提示词 Scope

        Returns:
            Prompt 实体或 None
        """
        try:
            prompt = self._prompt_service.get_active_prompt(scope)
            return prompt
        except Exception:
            # 数据库查询失败，使用常量
            logger.warning(f"无法从数据库加载提示词 {scope}，使用常量")
            return None

    def _render_system_prompt(
        self,
        template: str,
        input_data: OutlinePolishInput,
    ) -> str:
        """渲染系统提示词。

        Args:
            template: 提示词模板
            input_data: 输入参数

        Returns:
            渲染后的提示词
        """
        return template.format(
            industry=input_data.industry or DEFAULT_INDUSTRY,
            report_type=input_data.report_type or DEFAULT_REPORT_TYPE,
            language=input_data.language or DEFAULT_LANGUAGE,
            style=input_data.style or DEFAULT_STYLE,
        )

    def _render_user_prompt(
        self,
        template: str,
        input_data: OutlinePolishInput,
    ) -> str:
        """渲染用户提示词。

        Args:
            template: 提示词模板
            input_data: 输入参数

        Returns:
            渲染后的提示词
        """
        return template.format(
            original_outline=input_data.outline,
            report_type=input_data.report_type or DEFAULT_REPORT_TYPE,
        )


# 依赖注入类型标注
OutlinePolishServiceDep = Annotated[
    OutlinePolishService,
    Field(description="大纲优化服务"),
]
