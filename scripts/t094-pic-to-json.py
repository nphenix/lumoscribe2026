#!/usr/bin/env python3
"""T094：图转 JSON 批处理（正式运行入口）。

约束：
- 先完整复制 `data/intermediates_cleaned/` 到 `data/pic_to_json/` 再处理（输出目录即工作快照）
- LLM/CallSite/Prompt 必须从 SQLite 获取（DB 为单一事实来源）
- 不使用 mock，不降级（strict 模式下无有效 JSON 即失败）

使用示例：
  python scripts/t094-pic-to-json.py --concurrency 4
  python scripts/t094-pic-to-json.py --max-images 50
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import json
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


async def _run() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data" / "intermediates_cleaned"),
        help="输入目录（默认：data/intermediates_cleaned）",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "pic_to_json"),
        help="输出目录（默认：data/pic_to_json）",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="最多处理多少张图片（0 表示全量）",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="并发数（默认 2）",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="每处理多少张图片输出一次进度（默认 10）",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="严格模式：无有效 JSON 直接失败（推荐）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="恢复模式：跳过复制和已处理的文件",
    )
    parser.add_argument(
        "--backfill-state-log",
        default=None,
        help="从历史控制台日志回填 data/pic_to_json/t094_state.jsonl（用于断点续跑，非图表/错误也可跳过）",
    )
    parser.add_argument(
        "--backfill-only",
        action="store_true",
        help="仅回填 t094_state.jsonl 后退出（不执行图转 JSON 主流程）",
    )
    args = parser.parse_args()

    def _ts() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _progress(evt: dict) -> None:
        stage = evt.get("stage")
        ts = evt.get("ts") or _ts()
        if stage == "copy_start":
            print(f"[{ts}] [T094] copy_start: {evt.get('input_root')} -> {evt.get('output_root')}")
            return
        if stage == "copy_done":
            print(f"[{ts}] [T094] copy_done: {evt.get('duration_s')}s")
            return
        if stage == "scan_done":
            print(f"[{ts}] [T094] scan_done: images={evt.get('total_images')} ({evt.get('duration_s')}s)")
            return
        if stage == "chart_to_json_progress":
            done = evt.get("done")
            total = evt.get("total")
            last = evt.get("last_image")
            is_chart = evt.get("last_is_chart")
            status = evt.get("last_status")
            if status:
                print(f"[{ts}] [T094] chart_to_json: {done}/{total} status={status} last_is_chart={is_chart} last={last}")
            else:
                print(f"[{ts}] [T094] chart_to_json: {done}/{total} last_is_chart={is_chart} last={last}")
            return
        if stage == "delete_non_charts_done":
            print(f"[{ts}] [T094] delete_non_charts_done: deleted={evt.get('deleted_images')} ({evt.get('duration_s')}s)")
            return
        if stage == "prune_refs_done":
            print(
                f"[{ts}] [T094] prune_refs_done: md_changed={evt.get('md_files_changed')} "
                f"md_refs_removed={evt.get('md_refs_removed')} json_changed={evt.get('json_files_changed')} "
                f"({evt.get('duration_s')}s)"
            )
            return
        if stage == "validate_done":
            print(f"[{ts}] [T094] validate_done: {evt.get('duration_s')}s")
            return
        if stage == "ollama_unload":
            print(f"[{ts}] [T094] ollama_unload: {evt.get('msg')}")
            return
        # fallback
        print(f"[{ts}] [T094] {stage}: {evt}")

    # 前置：确保 prompt 与 callsite 配置已写入 DB
    print(f"[{_ts()}] [T094] ensure prompt/callsite in SQLite")
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "update-chart-extraction-prompt.py")],
        check=True,
        cwd=str(PROJECT_ROOT),
    )
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "update-chart-extraction-callsite.py")],
        check=True,
        cwd=str(PROJECT_ROOT),
    )

    from src.shared.config import get_settings
    from src.shared.db import init_db, make_engine, make_session_factory
    from src.application.repositories.intermediate_artifact_repository import (
        IntermediateArtifactRepository,
    )
    from src.application.repositories.llm_call_site_repository import LLMCallSiteRepository
    from src.application.repositories.llm_capability_repository import LLMCapabilityRepository
    from src.application.repositories.llm_provider_repository import LLMProviderRepository
    from src.application.repositories.prompt_repository import PromptRepository
    from src.application.services.document_cleaning_service import ChartExtractionService
    from src.application.services.document_cleaning.t094_pic_to_json_pipeline import (
        T094PicToJsonPipeline,
    )
    from src.application.services.llm_runtime_service import LLMRuntimeService
    from src.application.services.prompt_service import PromptService

    settings = get_settings()
    engine = make_engine(settings.sqlite_path)
    init_db(engine)
    session_factory = make_session_factory(engine)

    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()

    max_images = args.max_images if args.max_images and args.max_images > 0 else None
    strict = bool(args.strict)

    def _load_state_latest(state_path: Path) -> dict[str, dict]:
        if not state_path.exists():
            return {}
        latest: dict[str, dict] = {}
        for line in state_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and isinstance(obj.get("source_image"), str):
                latest[obj["source_image"]] = obj
        return latest

    def _image_sig(p: Path) -> dict[str, int] | None:
        try:
            st = p.stat()
            mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000)))
            return {"size": int(st.st_size), "mtime_ns": mtime_ns}
        except OSError:
            return None

    def _append_jsonl(state_path: Path, rec: dict) -> None:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with state_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _backfill_state_from_log(*, log_path: Path) -> None:
        state_path = (output_root / "t094_state.jsonl").resolve()
        latest = _load_state_latest(state_path)

        txt = log_path.read_text(encoding="utf-8", errors="ignore")

        # 1) 先从已有 chart_json 回填 chart（最可靠）
        for jf in sorted(output_root.rglob("chart_json/*.json")):
            try:
                obj = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            src = obj.get("_source_image")
            if not isinstance(src, str) or not src.strip():
                continue
            img_abs = (output_root / Path(src)).resolve()
            rec = {
                "ts": datetime.now().isoformat(),
                "source_image": src,
                "status": "chart",
                "image_sig": _image_sig(img_abs),
                "chart_json": jf.relative_to(output_root).as_posix(),
                "chart_id": obj.get("_chart_id"),
                "chart_name": obj.get("_chart_name"),
                "chart_type": obj.get("chart_type"),
            }
            # 允许重复写入（append-only），但避免同内容反复刷屏
            if latest.get(src) != rec:
                _append_jsonl(state_path, rec)
                latest[src] = rec

        # 2) 从日志回填 non_chart（只需要 False）
        # 形如: [..] [T094] chart_to_json: 1/249 last_is_chart=False last=...
        pat = re.compile(
            r"\[T094\]\s+chart_to_json:\s+\d+/\d+\s+last_is_chart=(True|False)\s+last=([^\s]+)"
        )
        for m in pat.finditer(txt):
            is_chart = m.group(1) == "True"
            src = m.group(2).strip().replace("\\", "/")
            if not src.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            if is_chart:
                continue  # chart 已由 chart_json 覆盖
            img_abs = (output_root / Path(src)).resolve()
            sig = _image_sig(img_abs)
            last = latest.get(src)
            if isinstance(last, dict) and last.get("status") == "non_chart" and last.get("image_sig") == sig:
                continue
            rec = {
                "ts": datetime.now().isoformat(),
                "source_image": src,
                "status": "non_chart",
                "image_sig": sig,
            }
            _append_jsonl(state_path, rec)
            latest[src] = rec

        # 3) 从日志回填 error（用于 --resume 时跳过）
        err_pat = re.compile(r"图表转 JSON 失败:\s+([A-Za-z]:[^\s,]+)")
        for m in err_pat.finditer(txt):
            abs_p = Path(m.group(1))
            try:
                src = abs_p.resolve().relative_to(output_root).as_posix()
            except Exception:
                continue
            sig = _image_sig(output_root / Path(src))
            last = latest.get(src)
            if isinstance(last, dict) and last.get("status") == "error" and last.get("image_sig") == sig:
                continue
            rec = {
                "ts": datetime.now().isoformat(),
                "source_image": src,
                "status": "error",
                "image_sig": sig,
                "error": "chart_to_json_failed",
            }
            _append_jsonl(state_path, rec)
            latest[src] = rec

        print(f"[{_ts()}] [T094] backfill_state_done: {state_path}")

    with session_factory() as session:
        llm_runtime = LLMRuntimeService(
            provider_repository=LLMProviderRepository(session),
            capability_repository=LLMCapabilityRepository(session),
            callsite_repository=LLMCallSiteRepository(session),
            prompt_repository=PromptRepository(session),
        )
        prompt_service = PromptService(PromptRepository(session))
        artifact_repo = IntermediateArtifactRepository(session)

        service = ChartExtractionService(
            llm_runtime=llm_runtime,
            artifact_repository=artifact_repo,
            prompt_service=prompt_service,
        )

        # 可选：先回填 JSONL 状态文件（便于本次 resume 不从头跑）
        if args.backfill_state_log:
            _backfill_state_from_log(log_path=Path(args.backfill_state_log).resolve())
            if args.backfill_only:
                return 0

        pipeline = T094PicToJsonPipeline(
            llm_runtime=llm_runtime,
            prompt_service=prompt_service,
            chart_to_json=service.chart_to_json,
        )
        report = await pipeline.run(
            input_root=input_root,
            output_root=output_root,
            max_images=max_images,
            strict=strict,
            concurrency=max(1, int(args.concurrency)),
            progress_callback=_progress,
            progress_every=max(1, int(args.progress_every)),
            resume=args.resume,
        )

    print(f"[{_ts()}] [DONE] T094 report: {output_root / 'run_report.json'}")
    print(f"[{_ts()}] [DONE] chart index: {output_root / 'chart_index.json'}")
    print(f"[{_ts()}] [DONE] output root (workdir snapshot): {output_root}")
    print(f"[{_ts()}] [STATS] scanned={report.get('total_images_scanned')} deleted={report.get('deleted_images')}")
    return 0


def main() -> int:
    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())

