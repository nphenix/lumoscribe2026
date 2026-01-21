#!/usr/bin/env python3
"""数据库初始化脚本。

用于创建所有数据表。适用于开发环境和首次部署。

使用方法：
    python scripts/init-db.py
"""

import sys
from pathlib import Path

# 允许从任意工作目录运行脚本：确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.config import get_settings
from src.shared.db import Base, make_engine, make_session_factory
from src.shared.constants.prompts import DEFAULT_PROMPTS
from src.shared.constants.llm_callsites import DEFAULT_CALLSITES
from src.application.repositories.llm_call_site_repository import LLMCallSiteRepository
from src.application.repositories.llm_capability_repository import LLMCapabilityRepository
from src.application.repositories.prompt_repository import PromptRepository
from src.application.services.prompt_service import PromptService
from src.domain.entities.llm_call_site import LLMCallSite
from src.domain.entities.llm_capability import LLMCapability

# 导入所有实体以注册到 Base.metadata（解决循环导入问题）
from src.domain.entities import (
    SourceFile,
    Template,
    IntermediateArtifact,
    TargetFile,
    LLMProvider,
    LLMCapability,
    LLMCallSite,
    Prompt,
)


def seed_prompts(engine):
    """播种默认提示词。"""
    session_factory = make_session_factory(engine)
    with session_factory() as session:
        repo = PromptRepository(session)
        service = PromptService(repo)
        
        for scope, data in DEFAULT_PROMPTS.items():
            # 检查是否已存在任何版本的提示词
            latest = repo.get_latest_version(scope)
            if latest == 0:
                print(f"Seeding prompt: {scope}")
                service.create_prompt(
                    scope=scope,
                    format=data["format"],
                    content=data["content"],
                    messages=None, # text format doesn't need messages
                    active=True,
                    description=data["description"]
                )
        session.commit()


def seed_llm_callsites(engine):
    """播种默认 LLM 调用点（CallSite）。"""
    session_factory = make_session_factory(engine)
    from uuid import uuid4

    with session_factory() as session:
        repo = LLMCallSiteRepository(session)

        for key, data in DEFAULT_CALLSITES.items():
            expected_model_kind = data.get("expected_model_kind")
            prompt_scope = data.get("prompt_scope")
            description = data.get("description")

            existing = repo.get_by_key(key)
            if existing is None:
                print(f"Seeding callsite: {key}")
                callsite = LLMCallSite(
                    id=str(uuid4()),
                    key=key,
                    expected_model_kind=expected_model_kind,
                    model_id=None,
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


def migrate_capabilities_to_callsites(engine):
    """一次性迁移：将旧 llm_capabilities 的默认映射写入同名 callsite。

    注意：此函数已废弃。LLMModel 表已被移除，模型配置现在存储在 LLMProvider.config_json 中。
    保留此函数签名但不做任何操作，以避免破坏调用方代码。
    """
    session_factory = make_session_factory(engine)
    with session_factory() as session:
        # 原迁移逻辑依赖 LLMModelRepository，现已移除
        # 由于 LLMModel 表已废弃，此迁移不再需要
        pass


def main() -> None:
    """初始化数据库并创建所有表。"""
    settings = get_settings()

    # 确保目录存在
    db_path = Path(settings.sqlite_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # 创建引擎
    engine = make_engine(db_path)

    # 创建所有表
    Base.metadata.create_all(engine)

    print(f"DB initialized: {db_path}")
    print("已创建的表:")
    for table in Base.metadata.tables.values():
        print(f"  - {table.name}")
    
    # 播种数据
    seed_prompts(engine)
    seed_llm_callsites(engine)
    migrate_capabilities_to_callsites(engine)


if __name__ == "__main__":
    main()
