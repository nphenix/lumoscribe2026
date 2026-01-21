#!/usr/bin/env python3
"""T097 前置：同步更新文档清洗提示词到 SQLite（DB 为单一事实来源）。

说明：
- 脚本不会在此处硬编码一份 Prompt；它从代码侧 `DEFAULT_PROMPTS` 读取最新种子内容。
- 若数据库中 `doc_cleaning:clean_text` 的激活版本与当前默认种子内容不同，则创建一个新版本并设为 active。
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _normalize_text(s: str | None) -> str:
    if not s:
        return ""
    # 统一换行符，避免 Windows/Unix 差异导致误判
    return s.replace("\r\n", "\n").strip()


def main() -> int:
    from src.shared.config import get_settings
    from src.shared.db import init_db, make_engine, make_session_factory
    from src.shared.constants.prompts import DEFAULT_PROMPTS, SCOPE_DOC_CLEANING
    from src.application.repositories.prompt_repository import PromptRepository
    from src.application.services.prompt_service import PromptService

    desired = DEFAULT_PROMPTS.get(SCOPE_DOC_CLEANING)
    if not desired:
        raise RuntimeError(f"DEFAULT_PROMPTS 中未注册 scope: {SCOPE_DOC_CLEANING}")

    settings = get_settings()
    engine = make_engine(settings.sqlite_path)
    init_db(engine)
    session_factory = make_session_factory(engine)

    with session_factory() as session:
        repo = PromptRepository(session)
        service = PromptService(repo)

        active = service.get_active_prompt(SCOPE_DOC_CLEANING)
        active_content = _normalize_text(active.content if active else None)
        desired_content = _normalize_text(desired.get("content"))

        if active_content == desired_content and active is not None:
            print(f"[OK] Active prompt already up-to-date: {SCOPE_DOC_CLEANING} v{active.version}")
            return 0

        created = service.create_prompt(
            scope=SCOPE_DOC_CLEANING,
            format=desired["format"],
            content=desired["content"],
            messages=None,
            active=True,
            description=desired.get("description") or "doc cleaning prompt",
        )
        print(
            f"[UPDATED] {SCOPE_DOC_CLEANING}: created new active version v{created.version} (id={created.id})"
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

