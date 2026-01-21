#!/usr/bin/env python3
"""查询数据库中的 LLM 配置（Provider、Capability、CallSite）。

功能：
1. 查询 llm_providers 表的完整配置
2. 查询 llm_capabilities 表的能力映射
3. 查询 llm_call_sites 表的调用点配置
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.config import get_settings
from src.shared.db import init_db, make_engine, make_session_factory
from src.domain.entities.llm_provider import LLMProvider
from src.domain.entities.llm_capability import LLMCapability
from src.domain.entities.llm_call_site import LLMCallSite


def format_config(config_json: str | None) -> str:
    """格式化配置 JSON。"""
    if not config_json:
        return '(空)'
    try:
        config = json.loads(config_json)
        # 脱敏 api_key
        if 'api_key' in config:
            config['api_key'] = '[已脱敏]'
        return json.dumps(config, indent=2, ensure_ascii=False)
    except:
        return config_json


def main() -> None:
    """查询并显示数据库中的 LLM 配置。"""
    settings = get_settings()
    db_path = Path(settings.sqlite_path)
    engine = make_engine(db_path)
    # 确保数据库结构已迁移到当前代码期望版本
    init_db(engine)
    session_factory = make_session_factory(engine)

    print("=" * 80)
    print("LLM 配置查询")
    print("=" * 80)
    print(f"数据库路径: {db_path}\n")

    with session_factory() as session:
        # 1. 查询 Providers
        print("=" * 80)
        print("1. Providers (供应商)")
        print("=" * 80)
        
        providers = session.query(LLMProvider).order_by(LLMProvider.created_at).all()
        print(f"总数: {len(providers)}\n")
        
        for p in providers:
            print(f"[{p.key}] {p.name}")
            print(f"  类型: {p.provider_type}")
            print(f"  Base URL: {p.base_url}")
            print(f"  启用: {'是' if p.enabled else '否'}")
            
            # API Key
            if p.api_key:
                print(f"  API Key: [已配置 - 明文]")
            elif p.api_key_env:
                print(f"  API Key: [环境变量: {p.api_key_env}]")
            else:
                print(f"  API Key: [未配置]")
            
            # 配置
            config = format_config(p.config_json)
            print(f"  配置:\n{config}")
            print()
        
        # 2. 查询 Capabilities
        print("=" * 80)
        print("2. Capabilities (能力映射)")
        print("=" * 80)
        
        capabilities = session.query(LLMCapability).order_by(LLMCapability.capability).all()
        print(f"总数: {len(capabilities)}\n")
        
        # 按 capability 分组
        caps_by_capability = {}
        for cap in capabilities:
            if cap.capability not in caps_by_capability:
                caps_by_capability[cap.capability] = []
            caps_by_capability[cap.capability].append(cap)
        
        for cap_name, caps in caps_by_capability.items():
            print(f"{cap_name}:")
            for cap in caps:
                status = '启用' if cap.enabled else '禁用'
                print(f"  [{status}] priority={cap.priority} -> provider_id={cap.provider_id}")
            print()
        
        # 3. 查询 CallSites
        print("=" * 80)
        print("3. CallSites (调用点)")
        print("=" * 80)
        
        call_sites = session.query(LLMCallSite).order_by(LLMCallSite.key).all()
        print(f"总数: {len(call_sites)}\n")
        
        for cs in call_sites:
            status = '启用' if cs.enabled else '禁用'
            bound = '已绑定' if cs.provider_id else '未绑定'
            print(f"[{cs.key}]")
            print(f"  expected_model_kind: {cs.expected_model_kind}")
            print(f"  provider_id: {cs.provider_id or '(未绑定)'}")
            print(f"  状态: {status} | 绑定: {bound}")
            print(f"  prompt_scope: {cs.prompt_scope}")
            print(f"  描述: {cs.description}")
            print()

    print("=" * 80)


if __name__ == "__main__":
    main()
