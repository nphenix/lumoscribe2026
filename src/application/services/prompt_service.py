"""提示词服务层。"""

from __future__ import annotations

import json
from uuid import uuid4

from fastapi import HTTPException

from src.application.repositories.prompt_repository import PromptRepository
from src.domain.entities.prompt import Prompt


class PromptService:
    """提示词服务类。"""

    SUPPORTED_FORMATS = {"text", "chat"}
    SUPPORTED_ROLES = {"system", "user", "assistant"}

    def __init__(self, repository: PromptRepository):
        self.repository = repository

    def _serialize_messages(self, messages: list[dict] | None) -> str | None:
        if messages is None:
            return None
        try:
            return json.dumps(messages, ensure_ascii=True)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"messages 不是有效 JSON: {exc}") from exc

    def _validate_prompt_payload(
        self,
        *,
        format: str,
        content: str | None,
        messages: list[dict] | None,
    ) -> None:
        if format not in self.SUPPORTED_FORMATS:
            raise HTTPException(status_code=400, detail="不支持的提示词格式")

        if format == "text":
            if not content or not content.strip():
                raise HTTPException(status_code=400, detail="text 格式必须提供 content")
            if messages:
                raise HTTPException(status_code=400, detail="text 格式不应包含 messages")
        if format == "chat":
            if not messages:
                raise HTTPException(status_code=400, detail="chat 格式必须提供 messages")
            for msg in messages:
                role = msg.get("role")
                content_value = msg.get("content")
                if role not in self.SUPPORTED_ROLES:
                    raise HTTPException(status_code=400, detail="chat messages 存在无效 role")
                if not content_value or not str(content_value).strip():
                    raise HTTPException(status_code=400, detail="chat messages 内容不能为空")

    def create_prompt(
        self,
        scope: str,
        format: str,
        content: str | None,
        messages: list[dict] | None,
        active: bool = True,
        description: str | None = None,
    ) -> Prompt:
        """创建提示词版本。"""
        if not scope.strip():
            raise HTTPException(status_code=400, detail="scope 不能为空")

        self._validate_prompt_payload(format=format, content=content, messages=messages)

        latest_version = self.repository.get_latest_version(scope)
        next_version = latest_version + 1

        if active:
            self.repository.deactivate_scope(scope)

        prompt = Prompt(
            id=str(uuid4()),
            scope=scope,
            format=format,
            content=content if format == "text" else None,
            messages_json=self._serialize_messages(messages) if format == "chat" else None,
            version=next_version,
            active=active,
            description=description,
        )
        return self.repository.create(prompt)

    def list_prompts(
        self,
        scope: str | None = None,
        active: bool | None = None,
        format: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[Prompt], int]:
        """列出提示词。"""
        items = self.repository.list(
            scope=scope,
            active=active,
            format=format,
            limit=limit,
            offset=offset,
        )
        total = self.repository.count(
            scope=scope,
            active=active,
            format=format,
        )
        return items, total

    def update_prompt(
        self,
        prompt_id: str,
        *,
        active: bool | None = None,
        description: str | None = None,
    ) -> Prompt | None:
        """更新提示词（仅元数据与激活状态）。"""
        prompt = self.repository.get_by_id(prompt_id)
        if prompt is None:
            return None

        if active is not None:
            if active:
                self.repository.deactivate_scope(prompt.scope)
            prompt.active = active

        if description is not None:
            prompt.description = description

        return self.repository.update(prompt)

    def delete_prompt(self, prompt_id: str) -> bool:
        """删除提示词。"""
        return self.repository.delete(prompt_id)

    def get_active_prompt(self, scope: str) -> Prompt | None:
        """获取指定 scope 的激活提示词。"""
        return self.repository.get_active_prompt(scope)
