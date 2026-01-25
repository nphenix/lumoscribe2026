"""
Seed MinerU Provider 脚本

此脚本用于手动向数据库中添加 MinerU Provider 配置。
如果数据库中已存在 MinerU 配置，则不会覆盖。

使用方法：
python scripts/seed_mineru.py
"""

import sys
import uuid
from pathlib import Path

# 添加项目根目录到 sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.shared.db import make_engine, make_session_factory
from src.shared.config import get_settings
from src.domain.entities.llm_provider import LLMProvider

def seed_mineru():
    settings = get_settings()
    engine = make_engine(settings.sqlite_path)
    session_factory = make_session_factory(engine)
    
    with session_factory() as session:
        print("检查 MinerU Provider 配置...")
        
        # 检查是否已存在
        from src.application.repositories.llm_provider_repository import LLMProviderRepository
        repo = LLMProviderRepository(session)
        existing = repo.get_by_key("mineru")
        
        if existing:
            print("MinerU Provider 已存在，跳过创建。")
            print(f"ID: {existing.id}")
            print(f"Base URL: {existing.base_url}")
            return

        # 创建新配置
        print("正在创建 MinerU Provider...")
        provider = LLMProvider(
            id=str(uuid.uuid4()),
            key="mineru",
            name="MinerU",
            provider_type="mineru",
            base_url=settings.mineru_api_url,  # 使用配置中的默认 URL
            api_key=settings.mineru_api_key,   # 使用配置中的默认 Key (通常为空)
            api_key_env="MINERU_API_KEY",      # 允许通过环境变量覆盖
            enabled=True,
            description="MinerU PDF 解析服务 (在线 API)",
            config_json='{"timeout": 60, "max_retries": 3}'
        )
        
        session.add(provider)
        session.commit()
        print("MinerU Provider 创建成功！")

if __name__ == "__main__":
    seed_mineru()
