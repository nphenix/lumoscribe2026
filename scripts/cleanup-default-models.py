#!/usr/bin/env python3
"""清理数据库中的无效默认模型。

删除那些关联到不存在 Provider 的默认模型（历史 seed 数据）。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.config import get_settings
from src.shared.db import make_engine, make_session_factory
from src.domain.entities.llm_model import LLMModel
from src.domain.entities.llm_provider import LLMProvider


def main() -> None:
    """清理无效的默认模型。"""
    settings = get_settings()
    engine = make_engine(Path(settings.sqlite_path))
    session_factory = make_session_factory(engine)

    with session_factory() as session:
        # 获取所有 Provider ID
        provider_ids = {p.id for p in session.query(LLMProvider.id).all()}
        
        # 查询所有模型
        all_models = session.query(LLMModel).all()
        
        # 找出关联到不存在 Provider 的模型
        invalid_models = [
            m for m in all_models
            if m.provider_id not in provider_ids
        ]
        
        print(f"数据库路径: {settings.sqlite_path}")
        print(f"总模型数: {len(all_models)}")
        print(f"有效模型数: {len(all_models) - len(invalid_models)}")
        print(f"无效模型数: {len(invalid_models)}")
        print("\n" + "=" * 80)
        
        if len(invalid_models) == 0:
            print("没有发现无效的模型。")
            return
        
        print("\n将要删除的无效模型:")
        for model in invalid_models:
            print(f"  - {model.name} ({model.model_kind}) [ID: {model.id}]")
            print(f"    关联的 Provider ID: {model.provider_id} (不存在)")
        
        print("\n" + "=" * 80)
        confirm = input("\n确认删除这些无效模型? (yes/no): ")
        
        if confirm.lower() != "yes":
            print("已取消操作。")
            return
        
        # 删除无效模型
        deleted_count = 0
        for model in invalid_models:
            try:
                session.delete(model)
                deleted_count += 1
                print(f"已删除: {model.name} ({model.model_kind})")
            except Exception as e:
                print(f"删除失败 {model.name}: {e}")
        
        session.commit()
        print(f"\n✓ 成功删除 {deleted_count} 个无效模型。")


if __name__ == "__main__":
    main()
