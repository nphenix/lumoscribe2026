#!/usr/bin/env python3
"""检查数据库中的 Provider 数据。

用于确认 llm_providers 表中的数据，特别是 mineru provider。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.config import get_settings
from src.shared.db import make_engine, make_session_factory
from src.domain.entities.llm_provider import LLMProvider


def main() -> None:
    """查询并显示数据库中的 Provider 数据。"""
    settings = get_settings()
    engine = make_engine(Path(settings.sqlite_path))
    session_factory = make_session_factory(engine)

    with session_factory() as session:
        providers = session.query(LLMProvider).order_by(LLMProvider.created_at).all()
        
        print(f"数据库路径: {settings.sqlite_path}")
        print(f"Provider 总数: {len(providers)}")
        print("\n" + "=" * 80)
        
        if len(providers) == 0:
            print("数据库中没有 Provider 记录。")
            return
        
        for provider in providers:
            print(f"\nProvider ID: {provider.id}")
            print(f"  Key: {provider.key}")
            print(f"  名称: {provider.name}")
            print(f"  类型: {provider.provider_type}")
            print(f"  Base URL: {provider.base_url}")
            print(f"  启用: {provider.enabled}")
            print(f"  创建时间: {provider.created_at}")
        
        print("\n" + "=" * 80)
        
        # 特别检查 mineru
        mineru_provider = session.query(LLMProvider).filter(
            LLMProvider.key == "mineru"
        ).first()
        
        if mineru_provider:
            print(f"\n✓ 找到 mineru Provider:")
            print(f"  ID: {mineru_provider.id}")
            print(f"  Key: {mineru_provider.key}")
            print(f"  名称: {mineru_provider.name}")
            print(f"  Base URL: {mineru_provider.base_url}")
        else:
            print("\n✗ 未找到 key='mineru' 的 Provider")


if __name__ == "__main__":
    main()
