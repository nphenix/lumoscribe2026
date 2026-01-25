"""
检查数据库中 MinerU 配置的脚本
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.shared.db import make_engine, make_session_factory
from src.shared.config import get_settings
from src.application.repositories.llm_provider_repository import LLMProviderRepository

def check_mineru():
    settings = get_settings()
    engine = make_engine(settings.sqlite_path)
    session_factory = make_session_factory(engine)
    
    with session_factory() as session:
        repo = LLMProviderRepository(session)
        
        print(f"Checking database at: {settings.sqlite_path}")
        print("-" * 50)
        
        # 1. 检查所有 Provider
        providers = repo.list(limit=100)
        print(f"Total Providers: {len(providers)}")
        for p in providers:
            print(f"ID: {p.id}")
            print(f"Key: {p.key}")
            print(f"Name: {p.name}")
            print(f"Type: {p.provider_type}")
            print("-" * 20)
            
        # 2. 特别检查 mineru
        mineru = repo.get_by_key("mineru")
        print("-" * 50)
        if mineru:
            print("✅ MinerU found by key 'mineru'")
        else:
            print("❌ MinerU NOT found by key 'mineru'")
            
            # 尝试模糊匹配
            potential = [p for p in providers if "mineru" in p.key.lower() or "mineru" in p.name.lower()]
            if potential:
                print("Found potential matches:")
                for p in potential:
                    print(f" - Key: {p.key}, Name: {p.name}")

if __name__ == "__main__":
    check_mineru()
