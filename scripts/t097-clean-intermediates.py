#!/usr/bin/env python3
"""T097：批量对 data/intermediates 下的 Markdown 执行“噪音清洗”，输出到新目录。

约束：
- 必须调用后台清洗能力（DocumentCleaningService + LLMRuntimeService），不使用 mock，不降级。
- LLM/CallSite/Prompt 必须从 SQLite 获取（DB 为单一事实来源）。
- **不做不可逆图片裁剪**：为保证后续 T094 图转 JSON，输出目录会完整复制同目录的 `*.json` 与 `images/`。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


_MD_IMAGE_RE = re.compile(r"!\[[^\]]*]\(([^)]+)\)")


def _normalize_rel_posix(p: str) -> str:
    s = (p or "").strip().strip('"').strip("'")
    s = s.replace("\\", "/")
    if s.startswith("./"):
        s = s[2:]
    return s


def _extract_markdown_image_paths(markdown: str) -> set[str]:
    paths: set[str] = set()
    for m in _MD_IMAGE_RE.finditer(markdown or ""):
        raw = _normalize_rel_posix(m.group(1))
        if not raw:
            continue
        # 只处理 intermediates 里常见的相对图片路径
        if raw.startswith("images/"):
            paths.add(raw)
    return paths


@dataclass
class CleanOneResult:
    md_in: Path
    md_out: Path
    kept_images: int
    removed_images: int


async def _run() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data" / "intermediates"),
        help="输入目录（默认：data/intermediates）",
    )
    parser.add_argument(
        "--output",
        default="",
        help="输出目录（默认：data/intermediates_cleaned/t097_<timestamp>）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="最多处理多少个 md（0 表示不限制）",
    )
    args = parser.parse_args()

    input_root = Path(args.input).resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_root}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (
        Path(args.output).resolve()
        if args.output
        else (PROJECT_ROOT / "data" / "intermediates_cleaned" / f"t097_{ts}")
    )
    output_root.mkdir(parents=True, exist_ok=True)

    from src.shared.config import get_settings
    from src.shared.db import init_db, make_engine, make_session_factory
    from src.application.repositories.llm_provider_repository import LLMProviderRepository
    from src.application.repositories.llm_capability_repository import LLMCapabilityRepository
    from src.application.repositories.llm_call_site_repository import LLMCallSiteRepository
    from src.application.repositories.prompt_repository import PromptRepository
    from src.application.repositories.intermediate_artifact_repository import (
        IntermediateArtifactRepository,
    )
    from src.application.services.llm_runtime_service import LLMRuntimeService
    from src.application.services.prompt_service import PromptService
    from src.application.services.document_cleaning_service import DocumentCleaningService
    from src.application.schemas.document_cleaning import CleaningOptions

    settings = get_settings()
    engine = make_engine(settings.sqlite_path)
    init_db(engine)
    session_factory = make_session_factory(engine)

    md_files = sorted(input_root.rglob("*.md"))
    if args.limit and args.limit > 0:
        md_files = md_files[: args.limit]
    if not md_files:
        raise RuntimeError(f"未找到任何 Markdown 文件: {input_root}")

    results: list[CleanOneResult] = []

    with session_factory() as session:
        llm_runtime = LLMRuntimeService(
            provider_repository=LLMProviderRepository(session),
            capability_repository=LLMCapabilityRepository(session),
            callsite_repository=LLMCallSiteRepository(session),
            prompt_repository=PromptRepository(session),
        )
        prompt_service = PromptService(PromptRepository(session))
        artifact_repo = IntermediateArtifactRepository(session)
        cleaning_service = DocumentCleaningService(
            llm_runtime=llm_runtime,
            artifact_repository=artifact_repo,
            prompt_service=prompt_service,
        )
        options = CleaningOptions()

        for md_path in md_files:
            print(f"[T097] cleaning: {md_path}")
            rel_md = md_path.relative_to(input_root)
            out_md_path = output_root / rel_md
            out_dir = out_md_path.parent
            out_dir.mkdir(parents=True, exist_ok=True)

            original = md_path.read_text(encoding="utf-8")
            print(f"[T097]  - chars: {len(original)}")
            rule_cleaned = cleaning_service.rule_based_clean(original, options)
            cleaned = await cleaning_service.llm_clean(
                rule_cleaned, options, strict=True, original_text=original
            )

            out_md_path.write_text(cleaned, encoding="utf-8")
            print(f"[T097]  - cleaned chars: {len(cleaned)}")

            keep_image_paths = _extract_markdown_image_paths(cleaned)
            print(f"[T097]  - images in md: {len(keep_image_paths)}")

            # 复制同目录 JSON（完整保留，供后续 T094 使用）
            for jf in md_path.parent.glob("*.json"):
                shutil.copy2(jf, out_dir / jf.name)

            # 复制 images/（完整保留，供后续 T094 使用）
            src_images = md_path.parent / "images"
            if src_images.exists() and src_images.is_dir():
                dst_images = out_dir / "images"
                shutil.copytree(src_images, dst_images, dirs_exist_ok=True)

            # 统计（仅基于 Markdown 引用数量估算）
            original_images = _extract_markdown_image_paths(original)
            removed_images = len(original_images - keep_image_paths)
            results.append(
                CleanOneResult(
                    md_in=md_path,
                    md_out=out_md_path,
                    kept_images=len(keep_image_paths),
                    removed_images=removed_images,
                )
            )

    report = {
        "task": "T097",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "total_md": len(results),
        "items": [
            {
                "md_in": str(r.md_in),
                "md_out": str(r.md_out),
                "kept_images": r.kept_images,
                "removed_images": r.removed_images,
            }
            for r in results
        ],
    }
    (output_root / "t097_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[DONE] cleaned md: {len(results)} -> {output_root}")
    return 0


def main() -> int:
    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())

