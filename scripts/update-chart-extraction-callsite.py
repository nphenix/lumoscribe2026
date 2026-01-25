#!/usr/bin/env python3
"""T094 前置：同步更新图转 JSON 的 CallSite 配置到 SQLite（DB 为单一事实来源）。

目标：
- 以“调用点绑定的 Provider 类型”为准做配置补齐：
  - Ollama: 补齐 `format=json`（用于强制 JSON 输出）
  - OpenAI-compatible（如火山方舟 Doubao）: 补齐 `response_format={"type":"json_object"}`（结构化输出）
- 不覆盖管理员已有的其他配置项（只做“缺省补齐”/清理明显不兼容字段）。

说明：
- 你们当前已将 SQLite 作为单一事实来源；因此本脚本只在你明确执行时才会改动 DB。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _parse_json(raw: str | None) -> dict:
    if not raw or not raw.strip():
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _dump_json(d: dict) -> str:
    return json.dumps(d, ensure_ascii=False, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只检查并输出建议，不写入 SQLite",
    )
    args = parser.parse_args()

    from src.shared.config import get_settings
    from src.shared.db import init_db, make_engine, make_session_factory
    from src.shared.constants.prompts import SCOPE_CHART_EXTRACTION
    from src.application.repositories.llm_call_site_repository import LLMCallSiteRepository
    from src.application.repositories.llm_provider_repository import LLMProviderRepository

    settings = get_settings()
    engine = make_engine(settings.sqlite_path)
    init_db(engine)
    session_factory = make_session_factory(engine)

    with session_factory() as session:
        callsite_repo = LLMCallSiteRepository(session)
        provider_repo = LLMProviderRepository(session)

        callsite = callsite_repo.get_by_key(SCOPE_CHART_EXTRACTION)
        if callsite is None:
            raise RuntimeError(f"CallSite 不存在: {SCOPE_CHART_EXTRACTION}")
        if not callsite.provider_id:
            raise RuntimeError(f"CallSite 未绑定 Provider: {SCOPE_CHART_EXTRACTION}")

        provider = provider_repo.get_by_id(callsite.provider_id)
        if provider is None:
            raise RuntimeError(f"CallSite 绑定的 Provider 不存在: provider_id={callsite.provider_id}")

        cfg = _parse_json(callsite.config_json)
        changed = False

        provider_type = str(getattr(provider, "provider_type", "") or "").strip().lower()

        # 只做缺省补齐，不覆盖管理员已有配置
        if provider_type == "ollama":
            # Ollama: format=json 可以强制返回 JSON
            if cfg.get("format") != "json":
                cfg.setdefault("format", "json")
                changed = True

        else:
            # OpenAI-compatible（例如 Doubao）不应使用 format（这是 Ollama 语义）
            if "format" in cfg:
                cfg.pop("format", None)
                changed = True

            # 火山方舟结构化输出：json_object（更通用，且你已选定该模式）
            rf = cfg.get("response_format")
            if not (isinstance(rf, dict) and rf.get("type") == "json_object"):
                cfg.setdefault("response_format", {"type": "json_object"})
                changed = True

            # 建议关闭 thinking，避免思考内容混入输出（火山方舟支持 thinking 字段）
            thinking = cfg.get("thinking")
            if not (isinstance(thinking, dict) and thinking.get("type") in {"disabled", "enabled"}):
                cfg.setdefault("thinking", {"type": "disabled"})
                changed = True

        # 图转 JSON 的稳定性关键：默认 temperature=0（若管理员没配置）
        if "temperature" not in cfg:
            cfg["temperature"] = 0
            changed = True

        # 默认调用超时：若未配置，则提升为 300 秒以避免大图/网络波动导致的频繁超时
        if "timeout_seconds" not in cfg and "timeout" not in cfg:
            cfg["timeout_seconds"] = 300
            changed = True

        if not changed:
            print(f"[OK] CallSite already up-to-date: {SCOPE_CHART_EXTRACTION} (provider_type={provider_type})")
            return 0

        if args.check_only:
            print(f"[CHECK] CallSite should be updated: {SCOPE_CHART_EXTRACTION} (provider_type={provider_type})")
            print(_dump_json(cfg))
            return 0

        callsite.config_json = json.dumps(cfg, ensure_ascii=False)
        callsite_repo.update(callsite)
        print(f"[UPDATED] CallSite config_json updated: {SCOPE_CHART_EXTRACTION} (provider_type={provider_type})")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

