#!/usr/bin/env python3
"""检查数据库中的模型数据来源。

用于确认 llm_models 表中的数据是从哪里来的。
"""

import sys
from pathlib import Path

# 允许从任意工作目录运行脚本：确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.config import get_settings
from src.shared.db import make_engine, make_session_factory
from src.domain.entities.llm_model import LLMModel
from src.domain.entities.llm_provider import LLMProvider


def main() -> None:
    """查询并显示数据库中的模型数据。"""
    settings = get_settings()
    engine = make_engine(Path(settings.sqlite_path))
    session_factory = make_session_factory(engine)

    with session_factory() as session:
        # 查询所有模型
        models = session.query(LLMModel).order_by(LLMModel.created_at).all()
        
        print(f"数据库路径: {settings.sqlite_path}")
        print(f"模型总数: {len(models)}")
        print("\n" + "=" * 80)
        
        if len(models) == 0:
            print("数据库中没有模型记录。")
            return
        
        for model in models:
            # 查询关联的 provider
            provider = session.query(LLMProvider).filter(
                LLMProvider.id == model.provider_id
            ).first()
            
            provider_name = provider.name if provider else f"[Provider ID: {model.provider_id} 不存在]"
            provider_key = provider.key if provider else "N/A"
            
            print(f"\n模型 ID: {model.id}")
            print(f"  名称: {model.name}")
            print(f"  类型: {model.model_kind}")
            print(f"  启用: {model.enabled}")
            print(f"  关联 Provider: {provider_name} (key: {provider_key})")
            print(f"  创建时间: {model.created_at}")
            print(f"  更新时间: {model.updated_at}")
            if model.description:
                print(f"  说明: {model.description}")
            if model.config_json:
                print(f"  配置: {model.config_json[:100]}...")
        
        print("\n" + "=" * 80)
        print("\n检查结论:")
        print("1. 如果模型列表为空，说明数据库中没有模型，需要用户在'模型管理'页面创建")
        print("2. 如果模型存在，检查其 provider_id 是否关联到正确的 Provider")
        print("3. 如果模型名称与 DEFAULT_MODELS 中的名称匹配，可能是历史 seed 数据")


if __name__ == "__main__":
    main()
