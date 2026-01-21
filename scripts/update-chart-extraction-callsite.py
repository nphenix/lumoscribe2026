#!/usr/bin/env python3
"""T094 前置：同步更新图转 JSON 的 CallSite 配置到 SQLite（DB 为单一事实来源）。

目标：
- 为 `chart_extraction:extract_json` 设置 `config_json.format = "json"`，以便在 Ollama 等模型上强制返回严格 JSON。
- 不覆盖管理员已有的其他配置项（只做“缺省补齐”）。
"""

from __future__ import annotations

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


def main() -> int:
    from src.shared.config import get_settings
    from src.shared.db import init_db, make_engine, make_session_factory
    from src.shared.constants.prompts import SCOPE_CHART_EXTRACTION
    from src.application.repositories.llm_call_site_repository import LLMCallSiteRepository

    settings = get_settings()
    engine = make_engine(settings.sqlite_path)
    init_db(engine)
    session_factory = make_session_factory(engine)

    with session_factory() as session:
        repo = LLMCallSiteRepository(session)
        callsite = repo.get_by_key(SCOPE_CHART_EXTRACTION)
        if callsite is None:
            raise RuntimeError(f"CallSite 不存在: {SCOPE_CHART_EXTRACTION}")

        cfg = _parse_json(callsite.config_json)
        changed = False

        # 只做缺省补齐，不覆盖管理员已有配置
        if cfg.get("format") != "json":
            cfg.setdefault("format", "json")
            changed = True

        # 图转 JSON 的稳定性关键：默认 temperature=0（若管理员没配置）
        if "temperature" not in cfg:
            cfg["temperature"] = 0
            changed = True

        if not changed:
            print(f"[OK] CallSite already up-to-date: {SCOPE_CHART_EXTRACTION}")
            return 0

        callsite.config_json = json.dumps(cfg, ensure_ascii=False)
        repo.update(callsite)
        print(f"[UPDATED] CallSite config_json updated: {SCOPE_CHART_EXTRACTION}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

