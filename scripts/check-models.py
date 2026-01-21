#!/usr/bin/env python3
"""查询数据库中的模型配置。"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from src.shared.config import get_settings
from src.shared.db import make_engine, make_session_factory
from src.domain.entities.llm_provider import LLMProvider

settings = get_settings()
db_path = Path(settings.sqlite_path)
engine = make_engine(db_path)
session_factory = make_session_factory(engine)

print('=' * 60)
print('中台配置的模型供应商列表')
print('=' * 60)

with session_factory() as session:
    providers = session.query(LLMProvider).all()
    
    if not providers:
        print('\n暂无配置的模型供应商')
    else:
        for p in providers:
            config = json.loads(p.config_json) if p.config_json else {}
            status = '启用' if p.enabled else '禁用'
            
            print(f'\n📦 {p.name} ({p.key})')
            print(f'   类型: {p.provider_type}')
            print(f'   状态: {status}')
            print(f'   Base URL: {p.base_url}')
            print(f'   API Key 环境变量: {p.api_key_env}')
            
            # 显示模型相关配置
            if 'model' in config:
                print(f'   模型: {config["model"]}')
            if 'ollama_model' in config:
                print(f'   Ollama 模型: {config["ollama_model"]}')
            
            print(f'   完整配置: {config}')
            print('-' * 60)

print('\n')
