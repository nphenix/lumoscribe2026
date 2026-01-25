"""摄入任务 Celery Worker。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from celery import shared_task
from sqlalchemy.orm import Session

from src.application.repositories.source_file_repository import SourceFileRepository
from src.application.schemas.mineru import MinerUIngestOptions, MinerUModelVersion, MinerULanguage
from src.application.services.mineru_service import MinerUIngestionService
from src.domain.entities.source_file import SourceFileStatus
from src.shared.config import get_settings
from src.shared.db import Job, make_engine, make_session_factory
from src.shared.logging import configure_logging, get_logger
from src.shared.storage import DATA_ROOT


log = get_logger(__name__)


def _get_session() -> Session:
    """获取数据库会话。"""
    settings = get_settings()
    engine = make_engine(settings.sqlite_path)
    session_factory = make_session_factory(engine)
    return session_factory()


def _update_job(
    job_id: int,
    status: str | None = None,
    progress: int | None = None,
    result_summary: dict[str, Any] | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> None:
    """更新 Job 记录。"""
    db = _get_session()
    try:
        job = db.get(Job, job_id)
        if job is None:
            log.warning("job_not_found", extra={"job_id": job_id})
            return

        if status is not None:
            job.status = status
        if progress is not None:
            job.progress = progress
        if result_summary is not None:
            job.result_summary = result_summary
        if error_code is not None:
            job.error_code = error_code
        if error_message is not None:
            job.error_message = error_message

        if status == "running" and job.started_at is None:
            job.started_at = datetime.utcnow()
        if status in {"succeeded", "failed", "partial"} and job.finished_at is None:
            job.finished_at = datetime.utcnow()

        db.commit()
    finally:
        db.close()


def _update_source_file_status(db: Session, source_file_id: str, status: SourceFileStatus) -> None:
    """更新源文件状态。"""
    repo = SourceFileRepository(db)
    repo.update_status(source_file_id, status)


@shared_task(name="ingest.process_documents", bind=True)
def process_documents(
    self,
    job_id: int,
    source_file_ids: list[str],
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """处理文档摄入任务。

    Args:
        job_id: Job 记录 ID
        source_file_ids: 源文件 ID 列表
        options: 摄入选项

    Returns:
        处理结果摘要
    """
    settings = get_settings()
    configure_logging(settings.log_level)

    db = _get_session()
    try:
        # 1. 获取 Job 记录
        job = db.get(Job, job_id)
        if job is None:
            log.warning("job_not_found", extra={"job_id": job_id})
            return {"success": False, "error": "job_not_found"}

        # 2. 更新状态为 running
        job.status = "running"
        job.started_at = datetime.utcnow()
        job.progress = 0
        db.commit()

        # 3. 查询源文件
        repo = SourceFileRepository(db)
        source_files = []
        for file_id in source_file_ids:
            sf = repo.get_by_id(file_id)
            if sf is not None:
                source_files.append(sf)

        if not source_files:
            job.status = "failed"
            job.error_code = "no_valid_files"
            job.error_message = "没有有效的源文件"
            job.finished_at = datetime.utcnow()
            db.commit()
            return {"success": False, "error": "no_valid_files"}

        # 4. 构建文件路径列表
        file_paths = []
        for sf in source_files:
            # storage_path 是相对路径，需要结合基础目录
            storage_path = Path(sf.storage_path)
            if not storage_path.is_absolute():
                storage_path = (DATA_ROOT / sf.storage_path).resolve()
            file_paths.append(storage_path)

        # 5. 构建摄入选项
        ingest_options = None
        if options:
            ingest_options = MinerUIngestOptions(
                enable_formula=options.get("enable_formula", True),
                enable_table=options.get("enable_table", True),
                language=MinerULanguage(options.get("language", "ch")) if options.get("language") else None,
                model_version=MinerUModelVersion(options.get("model_version")) if options.get("model_version") else None,
                extra_formats=options.get("extra_formats"),
                callback_url=options.get("callback_url"),
            )

        # 6. 更新源文件状态为 processing
        for sf in source_files:
            _update_source_file_status(db, sf.id, SourceFileStatus.MINERU_PROCESSING)

        # 7. 调用 MinerU 摄入服务（异步方法需要同步运行）
        import asyncio
        service = MinerUIngestionService()
        results = asyncio.run(
            service.ingest_batch(
                file_paths=file_paths,
                source_file_ids=[sf.id for sf in source_files],  # 直接传递 UUID 字符串
                options=ingest_options,
                wait_for_completion=True,
                max_concurrent=3,
            )
        )

        # 8. 处理结果并更新状态
        success_count = 0
        failed_count = 0
        errors = []
        result_urls = []

        for i, result in enumerate(results):
            sf = source_files[i]
            if result.success:
                success_count += 1
                # 更新源文件状态为 completed
                _update_source_file_status(db, sf.id, SourceFileStatus.MINERU_COMPLETED)

                # 收集结果 URL
                if result.metadata:
                    download_urls = result.metadata.get("result", {}).get("download_urls", [])
                    result_urls.extend(download_urls)
            else:
                failed_count += 1
                errors.append(f"{sf.original_filename}: {result.error}")

            # 更新进度
            progress = int((i + 1) / len(results) * 100)
            _update_job(job_id, progress=progress)

        # 9. 更新 Job 状态
        result_summary = {
            "processed_count": len(results),
            "success_count": success_count,
            "failed_count": failed_count,
            "errors": errors,
            "result_urls": list(set(result_urls)),  # 去重
        }

        if failed_count == 0:
            _update_job(job_id, status="succeeded", progress=100, result_summary=result_summary)
        elif success_count == 0:
            _update_job(job_id, status="failed", progress=100, result_summary=result_summary, error_code="all_failed", error_message="所有文件处理失败")
        else:
            _update_job(job_id, status="partial", progress=100, result_summary=result_summary, error_code="partial_failed", error_message=f"{failed_count} 个文件处理失败")

        log.info(
            "ingest.process_documents.completed",
            extra={
                "job_id": job_id,
                "total": len(results),
                "success": success_count,
                "failed": failed_count,
            },
        )

        return {
            "success": failed_count == 0,
            "total": len(results),
            "success_count": success_count,
            "failed_count": failed_count,
            "errors": errors[:10],  # 只返回前 10 个错误
        }
    except Exception as exc:
        log.exception("ingest.process_documents.failed", extra={"job_id": job_id})
        _update_job(
            job_id,
            status="failed",
            progress=0,
            error_code="worker_error",
            error_message=str(exc),
        )
        raise
    finally:
        db.close()

@shared_task(name="ingest.process_cleaning", bind=True)
def process_cleaning(
    self,
    job_id: int,
    source_file_ids: list[str],
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """处理文档清洗任务。

    Args:
        job_id: Job 记录 ID
        source_file_ids: 源文件 ID 列表
        options: 清洗选项

    Returns:
        处理结果摘要
    """
    from src.application.services.document_cleaning_service import DocumentCleaningService
    from src.application.services.llm_runtime_service import LLMRuntimeService
    from src.application.services.prompt_service import PromptService
    from src.application.repositories.intermediate_artifact_repository import IntermediateArtifactRepository
    from src.application.repositories.llm_capability_repository import LLMCapabilityRepository
    from src.application.repositories.llm_call_site_repository import LLMCallSiteRepository
    from src.application.repositories.llm_provider_repository import LLMProviderRepository
    from src.application.repositories.prompt_repository import PromptRepository
    from src.application.schemas.document_cleaning import CleaningOptions

    settings = get_settings()
    configure_logging(settings.log_level)

    db = _get_session()
    try:
        # 1. 获取 Job 记录
        job = db.get(Job, job_id)
        if job is None:
            log.warning("job_not_found", extra={"job_id": job_id})
            return {"success": False, "error": "job_not_found"}

        # 2. 更新状态为 running
        job.status = "running"
        job.started_at = datetime.utcnow()
        job.progress = 0
        db.commit()

        # 3. 查询源文件
        repo = SourceFileRepository(db)
        source_files = []
        for file_id in source_file_ids:
            sf = repo.get_by_id(file_id)
            if sf is not None:
                source_files.append(sf)

        if not source_files:
            job.status = "failed"
            job.error_code = "no_valid_files"
            job.error_message = "没有有效的源文件"
            job.finished_at = datetime.utcnow()
            db.commit()
            return {"success": False, "error": "no_valid_files"}

        # 4. 初始化服务
        capability_repo = LLMCapabilityRepository(db)
        callsite_repo = LLMCallSiteRepository(db)
        prompt_repo = PromptRepository(db)
        provider_repo = LLMProviderRepository(db)
        llm_runtime = LLMRuntimeService(provider_repo, capability_repo, callsite_repo, prompt_repo)
        artifact_repo = IntermediateArtifactRepository(db)
        prompt_service = PromptService(prompt_repo)
        cleaning_service = DocumentCleaningService(llm_runtime, artifact_repo, prompt_service)

        # 5. 更新源文件状态为 cleaning_processing
        for sf in source_files:
            _update_source_file_status(db, sf.id, SourceFileStatus.CLEANING_PROCESSING)

        # 6. 文件并发处理
        import asyncio
        success_count = 0
        failed_count = 0
        errors = []
        result_paths = []
        cleaning_opts = CleaningOptions(**(options or {}))
        file_concurrency = 2

        async def process_one(sf):
            mineru_raw_dir = DATA_ROOT / "intermediates" / sf.id / "mineru_raw"
            md_files = list(mineru_raw_dir.glob("*.md"))
            if not md_files:
                raise FileNotFoundError(f"未找到 MinerU 产物 (markdown): {mineru_raw_dir}")
            input_text = md_files[0].read_text(encoding="utf-8")

            s = _get_session()
            try:
                capability_repo2 = LLMCapabilityRepository(s)
                callsite_repo2 = LLMCallSiteRepository(s)
                prompt_repo2 = PromptRepository(s)
                provider_repo2 = LLMProviderRepository(s)
                llm_runtime2 = LLMRuntimeService(provider_repo2, capability_repo2, callsite_repo2, prompt_repo2)
                artifact_repo2 = IntermediateArtifactRepository(s)
                prompt_service2 = PromptService(prompt_repo2)
                cleaning_service2 = DocumentCleaningService(llm_runtime2, artifact_repo2, prompt_service2)

                stream = cleaning_service2.clean_document_stream(
                    text=input_text,
                    source_file_id=sf.id,
                    options=cleaning_opts,
                    mineru_output_dir=mineru_raw_dir,
                )
                final_evt: dict[str, Any] | None = None
                async for evt in stream:
                    if evt.get("type") == "final":
                        final_evt = evt
                if final_evt and final_evt.get("success"):
                    _update_source_file_status(s, sf.id, SourceFileStatus.CLEANING_COMPLETED)
                    stats = final_evt.get("stats") or {}
                    return True, stats
                else:
                    _update_source_file_status(s, sf.id, SourceFileStatus.MINERU_COMPLETED)
                    err = None
                    if final_evt is not None:
                        err = final_evt.get("error") or "stream_final_missing_success"
                    return False, f"{sf.original_filename}: {err or 'stream_no_final'}"
            except Exception as e:
                _update_source_file_status(s, sf.id, SourceFileStatus.MINERU_COMPLETED)
                return False, f"{sf.original_filename}: {str(e)}"
            finally:
                s.close()

        semaphore = asyncio.Semaphore(file_concurrency)
        async def run_with_limit(sf):
            async with semaphore:
                return await process_one(sf)

        async def run_all():
            tasks = [asyncio.create_task(run_with_limit(sf)) for sf in source_files]
            processed = 0
            total = len(source_files)
            results_local = []
            for t in asyncio.as_completed(tasks):
                ok, info = await t
                if ok:
                    success_count_local = 1
                    failed_count_local = 0
                    result_paths.append(info)
                else:
                    success_count_local = 0
                    failed_count_local = 1
                    errors.append(info)
                processed += 1
                _update_job(job_id, progress=int(processed / total * 100))
                nonlocal success_count, failed_count
                success_count += success_count_local
                failed_count += failed_count_local
                results_local.append((ok, info))
            return results_local

        asyncio.run(run_all())

        # 7. 更新 Job 状态
        result_summary = {
            "processed_count": len(source_files),
            "success_count": success_count,
            "failed_count": failed_count,
            "errors": errors,
            "details": result_paths
        }

        if failed_count == 0:
            _update_job(job_id, status="succeeded", progress=100, result_summary=result_summary)
        elif success_count == 0:
            _update_job(job_id, status="failed", progress=100, result_summary=result_summary, error_code="all_failed", error_message="所有文件清洗失败")
        else:
            _update_job(job_id, status="partial", progress=100, result_summary=result_summary, error_code="partial_failed", error_message=f"{failed_count} 个文件清洗失败")

        try:
            log.info(
                "ingest.cleaning.final",
                extra={
                    "job_id": job_id,
                    "processed_count": len(source_files),
                    "success_count": success_count,
                    "failed_count": failed_count,
                    "errors_count": len(errors),
                },
            )
        except Exception:
            pass

        return {
            "success": failed_count == 0,
            "total": len(source_files),
            "success_count": success_count,
            "failed_count": failed_count,
            "errors": errors[:10]
        }

    except Exception as exc:
        log.exception("ingest.process_cleaning.failed", extra={"job_id": job_id})
        _update_job(
            job_id,
            status="failed",
            progress=0,
            error_code="worker_error",
            error_message=str(exc),
        )
        
        # 异常状态回退
        try:
            repo = SourceFileRepository(db)
            for file_id in source_file_ids:
                sf = repo.get_by_id(file_id)
                if sf and sf.status == SourceFileStatus.CLEANING_PROCESSING:
                    _update_source_file_status(db, file_id, SourceFileStatus.MINERU_COMPLETED)
        except Exception:
            pass
            
        raise
    finally:
        db.close()


@shared_task(name="ingest.process_chart_json", bind=True)
def process_chart_json(
    self,
    job_id: int,
    source_file_ids: list[str],
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """执行图转 JSON（T094）。

    - 输入：data/intermediates/<source_id>/cleaned_doc/images/*
    - 输出：data/intermediates/<source_id>/pic_to_json/chart_json/*.json
    - 观测：每张图片都会输出一条 structured log（stage=chart_to_json_progress）。
    """
    settings = get_settings()
    configure_logging(settings.log_level)

    opts = options or {}
    file_concurrency = max(1, int(opts.get("concurrency") or 2))
    max_images = opts.get("max_images")
    max_images = int(max_images) if isinstance(max_images, int) and max_images > 0 else None
    resume = bool(opts.get("resume") or False)
    strict = bool(True if opts.get("strict") is None else opts.get("strict"))
    timeout_seconds = int(opts.get("timeout_seconds") or 300)

    db = _get_session()
    try:
        job = db.get(Job, job_id)
        if job is None:
            log.warning("job_not_found", extra={"job_id": job_id})
            return {"success": False, "error": "job_not_found"}

        job.status = "running"
        job.started_at = datetime.utcnow()
        job.progress = 0
        db.commit()

        # 1) 解析源文件
        repo = SourceFileRepository(db)
        source_files = []
        for file_id in source_file_ids:
            sf = repo.get_by_id(file_id)
            if sf is not None:
                source_files.append(sf)
        if not source_files:
            _update_job(
                job_id,
                status="failed",
                progress=0,
                error_code="no_valid_files",
                error_message="没有有效的源文件",
            )
            return {"success": False, "error": "no_valid_files"}

        # 2) 预扫描图片数（用于计算进度）
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        per_source_total: dict[str, int] = {}
        total_images = 0
        for sf in source_files:
            images_dir = (DATA_ROOT / "intermediates" / sf.id / "cleaned_doc" / "images").resolve()
            cnt = 0
            try:
                if images_dir.exists() and images_dir.is_dir():
                    for p in images_dir.rglob("*"):
                        if p.is_file() and p.suffix.lower() in exts:
                            cnt += 1
            except Exception:
                cnt = 0
            per_source_total[sf.id] = cnt
            total_images += cnt

        if total_images <= 0:
            _update_job(
                job_id,
                status="failed",
                progress=0,
                error_code="no_images",
                error_message="未找到 cleaned_doc/images 下的图片（请先完成文档清洗）",
            )
            return {"success": False, "error": "no_images"}

        # 3) 并发执行（按 source_file 并发；同一 source 内由 T094 保持串行/安全写入）
        import asyncio
        import threading

        lock = threading.Lock()
        done_images = 0
        last_pct = -1

        # 统计（用于结果摘要；最终仍以 report 聚合为准）
        chart_images = 0
        non_chart_images = 0
        error_images = 0

        def _make_progress_cb(source_file_id: str):
            def _cb(evt: dict[str, Any]) -> None:
                nonlocal done_images, last_pct, chart_images, non_chart_images, error_images
                if not isinstance(evt, dict):
                    return
                if evt.get("stage") != "chart_to_json_progress":
                    return

                status = evt.get("last_status")
                is_chart = bool(evt.get("last_is_chart"))
                image_rel = evt.get("last_image")
                elapsed_s = evt.get("elapsed_s")

                with lock:
                    done_images += 1
                    if status == "chart":
                        chart_images += 1
                    elif status == "non_chart":
                        non_chart_images += 1
                    elif status == "error":
                        error_images += 1

                    pct = int(done_images * 100 / max(total_images, 1))
                    should_update = (pct != last_pct) or (done_images >= total_images)
                    if should_update:
                        last_pct = pct

                # 每张图片一条日志（你要看的就是这个）
                log.info(
                    "t094.image_done",
                    extra={
                        "job_id": job_id,
                        "source_file_id": source_file_id,
                        "image": image_rel,
                        "status": status,
                        "is_chart": is_chart,
                        "done_images": done_images,
                        "total_images": total_images,
                        "progress": pct,
                        "elapsed_s": elapsed_s,
                    },
                )

                if should_update:
                    _update_job(
                        job_id,
                        progress=pct,
                        result_summary={
                            "task": "T094",
                            "total_files": len(source_files),
                            "total_images": total_images,
                            "images_done": done_images,
                            "chart_images": chart_images,
                            "non_chart_images": non_chart_images,
                            "error_images": error_images,
                            "updated_at": datetime.utcnow().isoformat(),
                            "last": {
                                "source_file_id": source_file_id,
                                "image": image_rel,
                                "status": status,
                                "is_chart": is_chart,
                            },
                        },
                    )

            return _cb

        async def _run_one(source_file_id: str) -> dict[str, Any]:
            s = _get_session()
            try:
                from src.application.repositories.intermediate_artifact_repository import (
                    IntermediateArtifactRepository,
                )
                from src.application.repositories.llm_call_site_repository import LLMCallSiteRepository
                from src.application.repositories.llm_capability_repository import (
                    LLMCapabilityRepository,
                )
                from src.application.repositories.llm_provider_repository import LLMProviderRepository
                from src.application.repositories.prompt_repository import PromptRepository
                from src.application.services.document_cleaning.t094_pic_to_json_pipeline import (
                    T094PicToJsonPipeline,
                )
                from src.application.services.document_cleaning_service import ChartExtractionService
                from src.application.services.llm_runtime_service import LLMRuntimeService
                from src.application.services.prompt_service import PromptService

                capability_repo = LLMCapabilityRepository(s)
                callsite_repo = LLMCallSiteRepository(s)
                prompt_repo = PromptRepository(s)
                provider_repo = LLMProviderRepository(s)
                llm_runtime = LLMRuntimeService(provider_repo, capability_repo, callsite_repo, prompt_repo)
                artifact_repo = IntermediateArtifactRepository(s)
                prompt_service = PromptService(prompt_repo)

                chart_service = ChartExtractionService(llm_runtime, artifact_repo, prompt_service)
                pipeline = T094PicToJsonPipeline(
                    llm_runtime=llm_runtime,
                    prompt_service=prompt_service,
                    chart_to_json=chart_service.chart_to_json,
                )

                report = await pipeline.run(
                    input_root=DATA_ROOT / "intermediates",
                    # 注意：项目约定 workdir 在 intermediates/<id>/pic_to_json
                    # 这也能避免与其它功能（例如 KB 构建）对 intermediates 的路径约定不一致。
                    output_root=DATA_ROOT / "intermediates",
                    max_images=max_images,
                    strict=bool(strict),
                    concurrency=1,  # 单 source 内保持串行；并发由外层控制
                    progress_callback=_make_progress_cb(source_file_id),
                    progress_every=1,  # 每张图都回调一次
                    timeout_seconds=timeout_seconds,
                    resume=bool(resume),
                    source_id_filter=source_file_id,
                )
                return {"source_file_id": source_file_id, "success": True, "report": report}
            except Exception as exc:  # noqa: BLE001
                log.exception("t094.source_failed", extra={"job_id": job_id, "source_file_id": source_file_id})
                return {"source_file_id": source_file_id, "success": False, "error": str(exc)}
            finally:
                s.close()

        async def _run_all() -> list[dict[str, Any]]:
            sem = asyncio.Semaphore(file_concurrency)

            async def _run_with_limit(sid: str) -> dict[str, Any]:
                async with sem:
                    return await _run_one(sid)

            tasks = [asyncio.create_task(_run_with_limit(sf.id)) for sf in source_files]
            out: list[dict[str, Any]] = []
            for t in asyncio.as_completed(tasks):
                out.append(await t)
            return out

        results = asyncio.run(_run_all())

        # 4) 聚合结果摘要
        success_count = 0
        failed_count = 0
        errors: list[str] = []
        per_file: list[dict[str, Any]] = []
        total_scanned = 0
        total_deleted = 0

        for r in results:
            sid = r.get("source_file_id")
            if r.get("success"):
                success_count += 1
                report = r.get("report") or {}
                total_scanned += int(report.get("total_images_scanned") or 0)
                total_deleted += int(report.get("deleted_images") or 0)
                per_file.append(
                    {
                        "source_file_id": sid,
                        "total_images_scanned": report.get("total_images_scanned"),
                        "deleted_images": report.get("deleted_images"),
                        "workspace_root": report.get("workspace_root"),
                    }
                )
            else:
                failed_count += 1
                err = r.get("error") or "unknown_error"
                errors.append(f"{sid}: {err}")
                per_file.append({"source_file_id": sid, "error": err})

        result_summary = {
            "task": "T094",
            "total_files": len(source_files),
            "success_count": success_count,
            "failed_count": failed_count,
            "total_images_planned": total_images,
            "total_images_scanned": total_scanned,
            "deleted_images": total_deleted,
            "chart_images": chart_images,
            "non_chart_images": non_chart_images,
            "error_images": error_images,
            "output_root": str((DATA_ROOT / "intermediates").resolve()),
            "files": per_file,
            "errors": errors[:20],
        }

        if failed_count == 0:
            _update_job(job_id, status="succeeded", progress=100, result_summary=result_summary)
        elif success_count == 0:
            _update_job(
                job_id,
                status="failed",
                progress=100,
                result_summary=result_summary,
                error_code="all_failed",
                error_message="所有文件图转 JSON 失败",
            )
        else:
            _update_job(
                job_id,
                status="partial",
                progress=100,
                result_summary=result_summary,
                error_code="partial_failed",
                error_message=f"{failed_count} 个文件图转 JSON 失败",
            )

        log.info(
            "ingest.process_chart_json.completed",
            extra={
                "job_id": job_id,
                "total_files": len(source_files),
                "success": success_count,
                "failed": failed_count,
                "total_images": total_images,
            },
        )
        return {"success": failed_count == 0, **result_summary}
    except Exception as exc:  # noqa: BLE001
        log.exception("ingest.process_chart_json.failed", extra={"job_id": job_id})
        _update_job(
            job_id,
            status="failed",
            progress=0,
            error_code="worker_error",
            error_message=str(exc),
        )
        raise
    finally:
        db.close()
