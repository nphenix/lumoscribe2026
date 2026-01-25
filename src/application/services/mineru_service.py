"""MinerU 在线服务接入层。"""

from __future__ import annotations

import json
import re
from datetime import datetime
import zipfile
import io
from pathlib import Path
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.application.schemas.mineru import (
    MinerUConfig,
    MinerUIngestOptions,
    MinerUIngestResult,
    MinerULanguage,
    MinerUModelVersion,
    MinerUTaskInfo,
    MinerUTaskStatus,
)
from src.shared.errors import AppError
from src.shared.logging import get_logger, log_extra

log = get_logger(__name__)


class MinerUClientError(AppError):
    """MinerU 客户端错误"""

    def __init__(self, message: str, status_code: int = 500, code: str = "mineru_client_error"):
        super().__init__(message=message, status_code=status_code, code=code)


class StatusCodeMapper:
    """HTTP 状态码映射到应用错误码"""

    _MAPPING: dict[int, tuple[str, int]] = {
        400: ("invalid_request", 400),
        401: ("unauthorized", 401),
        403: ("forbidden", 403),
        404: ("not_found", 404),
        413: ("file_too_large", 413),
        415: ("unsupported_media_type", 415),
        422: ("unprocessable_entity", 422),
        429: ("rate_limited", 429),
        500: ("internal_server_error", 500),
        502: ("bad_gateway", 502),
        503: ("service_unavailable", 503),
        504: ("gateway_timeout", 504),
    }

    @classmethod
    def get_error(cls, status_code: int) -> tuple[str, int]:
        """获取错误码和 HTTP 状态码"""
        return cls._MAPPING.get(status_code, ("unknown_error", status_code))


class MinerUClient:
    """MinerU 在线服务客户端。

    适配 MinerU 官网 API: https://mineru.net/api/v4/file-urls/batch
    """

    def __init__(self, config: MinerUConfig | None = None):
        """初始化客户端。

        Args:
            config: MinerU 配置，如果为 None 则从 SQLite 数据库加载
        """
        if config is not None:
            self.config = config
        else:
            self.config = self._load_config_from_db()
        self._client: httpx.AsyncClient | None = None

    @staticmethod
    def _load_config_from_db() -> MinerUConfig:
        """从 SQLite 数据库加载 MinerU 配置"""
        try:
            from src.shared.config import get_settings
            from src.shared.db import make_engine, make_session_factory
            from src.application.repositories.llm_call_site_repository import (
                LLMCallSiteRepository,
            )
            from src.application.repositories.llm_provider_repository import (
                LLMProviderRepository,
            )

            settings = get_settings()
            engine = make_engine(settings.sqlite_path)
            session_factory = make_session_factory(engine)
            with session_factory() as session:
                # 1. 先通过 callsite 查找绑定的 Provider
                callsite_repo = LLMCallSiteRepository(session)
                callsite = callsite_repo.get_by_key("mineru:parse_document")
                
                provider = None
                if callsite and callsite.provider_id:
                    # 如果 callsite 绑定了 provider，直接使用绑定的
                    provider_repo = LLMProviderRepository(session)
                    provider = provider_repo.get_by_id(callsite.provider_id)
                
                # 2. 如果 callsite 未绑定或未找到，回退到按 key="mineru" 查找（兼容旧逻辑）
                if provider is None:
                    provider_repo = LLMProviderRepository(session)
                    provider = provider_repo.get_by_key("mineru")

                if provider is None:
                    raise ValueError("mineru provider not found (checked callsite 'mineru:parse_document' and provider key 'mineru')")

                if not provider.enabled:
                    raise ValueError(f"provider {provider.name} is not enabled")

                # 从数据库加载配置
                if not provider.base_url:
                    raise ValueError(f"base_url not configured for provider {provider.name}")

                config_dict = {"base_url": provider.base_url}

                if provider.api_key:
                    config_dict["api_token"] = provider.api_key
                elif provider.api_key_env:
                    import os

                    api_key = os.getenv(provider.api_key_env, "")
                    if api_key:
                        config_dict["api_token"] = api_key

                if provider.config_json:
                    extra_config = json.loads(provider.config_json or "{}")
                    if "model_version" in extra_config:
                        config_dict["model_version"] = extra_config["model_version"]
                    if "timeout" in extra_config:
                        config_dict["timeout"] = extra_config["timeout"]
                    if "upload_timeout" in extra_config:
                        config_dict["upload_timeout"] = extra_config["upload_timeout"]
                    if "max_retries" in extra_config:
                        config_dict["max_retries"] = extra_config["max_retries"]

                return MinerUConfig(**config_dict)

        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to load MinerU config from database: {e}")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout),
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        """关闭客户端"""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "MinerUClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def _sanitize_filename(self, filename: str) -> str:
        """清理文件名，移除不安全字符"""
        sanitized = re.sub(r"[^\w\-.]", "_", filename)
        sanitized = sanitized.strip(".")
        max_length = 200
        if len(sanitized) > max_length:
            name, ext = Path(sanitized).stem, Path(sanitized).suffix
            sanitized = f"{name[:max_length - len(ext) - 1]}{ext}"
        return sanitized

    def _safe_url_for_log(self, url: str) -> str:
        """用于日志：去掉 query/fragment，避免泄露预签名参数。"""
        try:
            from urllib.parse import urlsplit

            parts = urlsplit(url)
            safe = f"{parts.scheme}://{parts.netloc}{parts.path}"
            return safe[:512]
        except Exception:
            return "<invalid_url>"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException)),
    )
    async def apply_upload_urls(
        self,
        files: list[tuple[str, bytes]],
        data_ids: list[str] | None = None,
        options: MinerUIngestOptions | None = None,
    ) -> dict[str, Any]:
        """批量申请上传链接并提交解析任务。

        根据 MinerU 官网 API:
        POST https://mineru.net/api/v4/file-urls/batch

        Args:
            files: 文件列表，格式为 [(filename, file_bytes), ...]
            data_ids: 数据 ID 列表，与 files 一一对应
            options: 解析选项

        Returns:
            包含 batch_id 和 file_urls 的字典

        Raises:
            MinerUClientError: 申请上传链接失败
        """
        client = await self._get_client()
        options = options or MinerUIngestOptions()

        # 构建请求体
        file_list = []
        for idx, (filename, _) in enumerate(files):
            file_item = {"name": filename}
            if data_ids and idx < len(data_ids):
                file_item["data_id"] = data_ids[idx]
            file_list.append(file_item)

        request_body = {
            "files": file_list,
            "model_version": (options.model_version or self.config.model_version).value,
        }

        # 添加可选参数
        if options.enable_formula is not None:
            request_body["enable_formula"] = options.enable_formula
        if options.enable_table is not None:
            request_body["enable_table"] = options.enable_table
        if options.language:
            request_body["language"] = options.language.value if isinstance(options.language, MinerULanguage) else options.language
        if options.extra_formats:
            request_body["extra_formats"] = options.extra_formats
        if options.callback_url:
            request_body["callback"] = options.callback_url
            request_body["seed"] = re.sub(r"[^a-zA-Z0-9_]", "", str(uuid.uuid4()))[:64]

        try:
            log.info(
                "mineru.apply_upload_urls.start",
                extra=log_extra(
                    base_url=self._safe_url_for_log(self.config.base_url),
                    file_count=len(files),
                    model_version=request_body.get("model_version"),
                    has_callback=bool(options.callback_url),
                    extra_formats=options.extra_formats,
                ),
            )
            # 直接使用数据库中的 base_url
            response = await client.post(
                self.config.base_url,
                headers=self.config.get_auth_headers(),
                json=request_body,
            )
            response.raise_for_status()
            data = response.json()

            if data.get("code") != 0:
                raise MinerUClientError(
                    message=data.get("msg", "申请上传链接失败"),
                    status_code=400,
                    code="upload_url_error",
                )

            result = {
                "batch_id": data.get("data", {}).get("batch_id"),
                "file_urls": data.get("data", {}).get("file_urls", []),
            }
            log.info(
                "mineru.apply_upload_urls.ok",
                extra=log_extra(
                    batch_id=result.get("batch_id"),
                    file_urls_count=len(result.get("file_urls") or []),
                    trace_id=data.get("trace_id"),
                ),
            )
            return result

        except httpx.HTTPStatusError as e:
            log.warning(
                "mineru.apply_upload_urls.http_error",
                extra=log_extra(
                    status_code=e.response.status_code,
                    response_text=(e.response.text or "")[:500],
                ),
            )
            code, status_code = StatusCodeMapper.get_error(e.response.status_code)
            raise MinerUClientError(
                message=f"申请上传链接失败: {e.response.text}",
                status_code=status_code,
                code=code,
            ) from e
        except httpx.RequestError as e:
            log.warning(
                "mineru.apply_upload_urls.network_error",
                extra=log_extra(error=repr(e)),
            )
            raise MinerUClientError(
                message=f"网络错误: {repr(e)}",
                status_code=503,
                code="network_error",
            ) from e

    async def parse_document(
        self,
        filename: str,
        data_id: str | None = None,
        options: MinerUIngestOptions | None = None,
    ) -> str:
        """提交文档解析任务。

        Args:
            filename: 文件名
            data_id: 数据 ID
            options: 解析选项

        Returns:
            batch_id（任务 ID）
        """
        # 读取文件内容（这里只是提交任务，不真正上传）
        # 文件上传由 upload_file 方法处理
        upload_result = await self.apply_upload_urls(
            files=[(filename, b"")],  # 空内容，只申请链接
            data_ids=[data_id] if data_id else None,
            options=options,
        )
        return upload_result["batch_id"]

    async def upload_file(self, file_path: Path | str, upload_url: str) -> bool:
        """上传文件到预签名 URL。

        Args:
            file_path: 本地文件路径
            upload_url: 预签名上传 URL

        Returns:
            是否上传成功
        """
        client = await self._get_client()

        file_path = Path(file_path)
        if not file_path.exists():
            raise MinerUClientError(
                message=f"文件不存在: {file_path}",
                status_code=404,
                code="file_not_found",
            )

        try:
            size_bytes = file_path.stat().st_size
            log.info(
                "mineru.upload_file.start",
                extra=log_extra(
                    filename=file_path.name,
                    size_bytes=size_bytes,
                    upload_url=self._safe_url_for_log(upload_url),
                    timeout_s=self.config.upload_timeout,
                ),
            )
            with open(file_path, "rb") as f:
                file_content = f.read()

            def _before_sleep(retry_state) -> None:
                exc = retry_state.outcome.exception() if retry_state.outcome else None
                log.warning(
                    "mineru.upload_file.retry",
                    extra=log_extra(
                        filename=file_path.name,
                        attempt=retry_state.attempt_number,
                        error=repr(exc) if exc is not None else None,
                    ),
                )

            # 仅对网络类异常重试（例如 ReadError / ConnectError / Timeout）
            try:
                async for attempt in AsyncRetrying(
                    retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException)),
                    stop=stop_after_attempt(max(1, int(self.config.max_retries))),
                    wait=wait_exponential(multiplier=1, min=2, max=10),
                    reraise=True,
                    before_sleep=_before_sleep,
                ):
                    with attempt:
                        response = await client.put(
                            upload_url,
                            content=file_content,
                            timeout=httpx.Timeout(self.config.upload_timeout),
                        )
                        response.raise_for_status()
            except RetryError as re:
                # 理论上 reraise=True 不会走到这里；留作保险
                last = re.last_attempt.exception()
                raise last  # type: ignore[misc]

            log.info(
                "mineru.upload_file.ok",
                extra=log_extra(filename=file_path.name, status_code=response.status_code),
            )
            return True

        except httpx.HTTPStatusError as e:
            log.warning(
                "mineru.upload_file.http_error",
                extra=log_extra(
                    filename=file_path.name,
                    status_code=e.response.status_code,
                    response_text=(e.response.text or "")[:500],
                ),
            )
            code, status_code = StatusCodeMapper.get_error(e.response.status_code)
            raise MinerUClientError(
                message=f"上传文件失败: {e.response.text}",
                status_code=status_code,
                code=code,
            ) from e
        except httpx.RequestError as e:
            log.warning(
                "mineru.upload_file.network_error",
                extra=log_extra(filename=file_path.name, error=repr(e)),
            )
            raise MinerUClientError(
                message=f"网络错误: {repr(e)}",
                status_code=503,
                code="network_error",
            ) from e
        except OSError as e:
            log.warning(
                "mineru.upload_file.os_error",
                extra=log_extra(filename=file_path.name, error=repr(e)),
            )
            raise MinerUClientError(
                message=f"读取文件失败: {str(e)}",
                status_code=500,
                code="file_read_error",
            ) from e

    def _get_api_root(self) -> str:
        """从配置的 base_url 提取 API 根域名。

        例如：
        - base_url: https://mineru.net/api/v4/file-urls/batch
        - root: https://mineru.net
        """
        base = (self.config.base_url or "").rstrip("/")
        if "/api/" in base:
            return base.split("/api/")[0]

        # 兜底：避免 base_url 异常时拼出错误 URL
        from urllib.parse import urlparse

        parsed = urlparse(base)
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}"
        return base

    async def get_task_status(self, batch_id: str) -> MinerUTaskInfo:
        """查询任务状态。

        注意：file-urls/batch 返回的是 batch_id（批次 ID），
        官方文档对应的“批量查询解析结果”接口应使用 batch_id 查询，而不是单任务 tasks/{task_id}。

        Args:
            batch_id: 批次 ID

        Returns:
            任务信息
        """
        client = await self._get_client()

        root = self._get_api_root()
        # 批量查询解析结果（batch_id）
        # GET https://mineru.net/api/v4/extract-results/batch/{batch_id}
        batch_status_url = f"{root}/api/v4/extract-results/batch/{batch_id}"
        # 兼容：如服务端仍支持老的单任务查询，再回退到 tasks/{id}
        legacy_task_url = f"{root}/api/v4/tasks/{batch_id}"

        try:
            log.debug(
                "mineru.get_task_status.request",
                extra=log_extra(batch_id=batch_id, url=batch_status_url),
            )
            response = await client.get(batch_status_url, headers=self.config.get_auth_headers())
            response.raise_for_status()
            payload = response.json()

            if payload.get("code") != 0:
                raise MinerUClientError(
                    message=payload.get("msg", "查询任务状态失败"),
                    status_code=400,
                    code="task_query_error",
                )

            data = payload.get("data") or {}

            # 新：批次结果结构（extract_result 列表）
            if "extract_result" in data or "extract_results" in data:
                items = data.get("extract_result") or data.get("extract_results") or []
                if not isinstance(items, list):
                    items = []

                def _map_state_to_status(state: str) -> MinerUTaskStatus:
                    s = (state or "").strip().lower()
                    if s in {"done", "completed", "success", "succeeded"}:
                        return MinerUTaskStatus.COMPLETED
                    if s in {"running", "processing", "in_progress"}:
                        return MinerUTaskStatus.PROCESSING
                    if s in {"failed", "fail", "error"}:
                        return MinerUTaskStatus.FAILED
                    return MinerUTaskStatus.PENDING

                per_file_statuses: list[MinerUTaskStatus] = []
                errors: list[str] = []
                download_url: str | None = None  # 保持兼容：返回首个可下载链接
                download_urls: list[str] = []
                file_results: list[dict[str, Any]] = []
                progress_values: list[int] = []

                for item in items:
                    if not isinstance(item, dict):
                        continue
                    file_name = item.get("file_name") or item.get("filename") or item.get("name")
                    st = _map_state_to_status(str(item.get("state") or item.get("status") or ""))
                    per_file_statuses.append(st)

                    item_download_url = (
                        item.get("full_zip_url")
                        or item.get("download_url")
                        or item.get("zip_url")
                    )

                    if st == MinerUTaskStatus.FAILED:
                        err = item.get("err_msg") or item.get("error_msg") or item.get("error")
                        if err:
                            errors.append(str(err))

                    if st == MinerUTaskStatus.COMPLETED:
                        if item_download_url:
                            download_urls.append(str(item_download_url))
                        if download_url is None and item_download_url:
                            download_url = str(item_download_url)

                    prog = item.get("extract_progress") or {}
                    if isinstance(prog, dict):
                        extracted = prog.get("extracted_pages")
                        total = prog.get("total_pages")
                        if isinstance(extracted, int) and isinstance(total, int) and total > 0:
                            progress_values.append(int(extracted / total * 100))

                    file_results.append(
                        {
                            "file_name": file_name,
                            "state": item.get("state") or item.get("status"),
                            "status": st.value,
                            "download_url": item_download_url,
                            "err_msg": item.get("err_msg") or item.get("error_msg") or item.get("error"),
                        }
                    )

                if per_file_statuses and all(s == MinerUTaskStatus.COMPLETED for s in per_file_statuses):
                    status = MinerUTaskStatus.COMPLETED
                elif any(s == MinerUTaskStatus.FAILED for s in per_file_statuses):
                    status = MinerUTaskStatus.FAILED
                elif any(s == MinerUTaskStatus.PROCESSING for s in per_file_statuses):
                    status = MinerUTaskStatus.PROCESSING
                else:
                    status = MinerUTaskStatus.PENDING

                progress = int(sum(progress_values) / len(progress_values)) if progress_values else 0
                error_msg = "; ".join(errors) if errors else None

                log.info(
                    "mineru.get_task_status.ok",
                    extra=log_extra(
                        batch_id=batch_id,
                        status=status.value,
                        item_count=len(items),
                        progress=progress,
                        has_download_url=bool(download_url),
                    ),
                )
                return MinerUTaskInfo(
                    task_id=batch_id,
                    status=status,
                    status_desc=None,
                    progress=progress,
                    result={
                        "download_url": download_url,
                        "download_urls": download_urls,
                        "file_results": file_results,
                        "batch": data,
                    },
                    error_msg=error_msg,
                    created_at=None,
                    updated_at=None,
                )

            # 旧：单任务结构（data.status / data.result）
            task_data = data
            status_str = task_data.get("status", "pending")

            # 状态映射
            status_map = {
                "pending": MinerUTaskStatus.PENDING,
                "processing": MinerUTaskStatus.PROCESSING,
                "completed": MinerUTaskStatus.COMPLETED,
                "failed": MinerUTaskStatus.FAILED,
            }
            status = status_map.get(status_str, MinerUTaskStatus.PENDING)

            log.info(
                "mineru.get_task_status.ok_legacy",
                extra=log_extra(batch_id=batch_id, status=status.value),
            )
            return MinerUTaskInfo(
                task_id=batch_id,
                status=status,
                status_desc=task_data.get("status_desc"),
                progress=task_data.get("progress", 0),
                result=task_data.get("result"),
                error_msg=task_data.get("error_msg"),
                created_at=task_data.get("created_at"),
                updated_at=task_data.get("updated_at"),
            )

        except httpx.HTTPStatusError as e:
            # 兼容：如果批次接口 404，则回退尝试 legacy tasks 接口
            if e.response.status_code == 404:
                log.debug(
                    "mineru.get_task_status.batch_404_fallback_legacy",
                    extra=log_extra(batch_id=batch_id, url=batch_status_url),
                )
                try:
                    resp2 = await client.get(legacy_task_url, headers=self.config.get_auth_headers())
                    resp2.raise_for_status()
                    data2 = resp2.json()
                    if data2.get("code") != 0:
                        raise MinerUClientError(
                            message=data2.get("msg", "查询任务状态失败"),
                            status_code=400,
                            code="task_query_error",
                        )

                    task_data = data2.get("data", {}) or {}
                    status_str = task_data.get("status", "pending")
                    status_map = {
                        "pending": MinerUTaskStatus.PENDING,
                        "processing": MinerUTaskStatus.PROCESSING,
                        "completed": MinerUTaskStatus.COMPLETED,
                        "failed": MinerUTaskStatus.FAILED,
                    }
                    status = status_map.get(status_str, MinerUTaskStatus.PENDING)
                    log.info(
                        "mineru.get_task_status.ok_legacy_fallback",
                        extra=log_extra(batch_id=batch_id, status=status.value),
                    )
                    return MinerUTaskInfo(
                        task_id=batch_id,
                        status=status,
                        status_desc=task_data.get("status_desc"),
                        progress=task_data.get("progress", 0),
                        result=task_data.get("result"),
                        error_msg=task_data.get("error_msg"),
                        created_at=task_data.get("created_at"),
                        updated_at=task_data.get("updated_at"),
                    )
                except httpx.HTTPStatusError as e2:
                    code, status_code = StatusCodeMapper.get_error(e2.response.status_code)
                    log.warning(
                        "mineru.get_task_status.legacy_http_error",
                        extra=log_extra(
                            batch_id=batch_id,
                            status_code=e2.response.status_code,
                            response_text=(e2.response.text or "")[:500],
                        ),
                    )
                    raise MinerUClientError(
                        message=f"查询任务状态失败: {e2.response.text}",
                        status_code=status_code,
                        code=code,
                    ) from e2
                except httpx.RequestError as e2:
                    log.warning(
                        "mineru.get_task_status.legacy_network_error",
                        extra=log_extra(batch_id=batch_id, error=str(e2)),
                    )
                    raise MinerUClientError(
                        message=f"网络错误: {str(e2)}",
                        status_code=503,
                        code="network_error",
                    ) from e2

            code, status_code = StatusCodeMapper.get_error(e.response.status_code)
            log.warning(
                "mineru.get_task_status.http_error",
                extra=log_extra(
                    batch_id=batch_id,
                    status_code=e.response.status_code,
                    response_text=(e.response.text or "")[:500],
                ),
            )
            raise MinerUClientError(
                message=f"查询任务状态失败: {e.response.text}",
                status_code=status_code,
                code=code,
            ) from e
        except httpx.RequestError as e:
            log.warning(
                "mineru.get_task_status.network_error",
                extra=log_extra(batch_id=batch_id, error=str(e)),
            )
            raise MinerUClientError(
                message=f"网络错误: {str(e)}",
                status_code=503,
                code="network_error",
            ) from e

    async def download_file(self, url: str) -> bytes:
        """下载文件。

        Args:
            url: 下载链接

        Returns:
            文件内容
        """
        client = await self._get_client()
        try:
            response = await client.get(url, timeout=httpx.Timeout(self.config.upload_timeout))
            response.raise_for_status()
            return response.content
        except httpx.HTTPStatusError as e:
            code, status_code = StatusCodeMapper.get_error(e.response.status_code)
            raise MinerUClientError(
                message=f"下载文件失败: {e.response.text}",
                status_code=status_code,
                code=code,
            ) from e
        except httpx.RequestError as e:
            raise MinerUClientError(
                message=f"网络错误: {repr(e)}",
                status_code=503,
                code="network_error",
            ) from e

    async def wait_for_completion(
        self,
        batch_id: str,
        poll_interval: float = 5.0,
        max_wait: float = 600.0,
        expected_item_count: int | None = None,
    ) -> MinerUTaskInfo:
        """等待任务完成。

        Args:
            batch_id: 批次 ID
            poll_interval: 轮询间隔（秒）
            max_wait: 最大等待时间（秒）
            expected_item_count: 期望批次内文件数量（用于避免 MinerU 侧延迟入库导致“只看到少量条目就误判完成”）

        Returns:
            最终任务状态
        """
        import asyncio

        log.info(
            "mineru.wait_for_completion.start",
            extra=log_extra(
                batch_id=batch_id,
                poll_interval=poll_interval,
                max_wait=max_wait,
                expected_item_count=expected_item_count,
            ),
        )
        start_time = asyncio.get_running_loop().time()
        while (asyncio.get_running_loop().time() - start_time) < max_wait:
            try:
                task_info = await self.get_task_status(batch_id)
            except MinerUClientError as e:
                # MinerU 侧异步入库可能出现短暂 404：视为 pending 继续轮询
                if e.code == "not_found":
                    await asyncio.sleep(poll_interval)
                    continue
                raise

            if expected_item_count is not None:
                batch = (task_info.result or {}).get("batch")
                items = []
                if isinstance(batch, dict):
                    items = batch.get("extract_result") or batch.get("extract_results") or []
                if not isinstance(items, list):
                    items = []

                if len(items) < expected_item_count:
                    # 还没看到足够的条目，继续等
                    log.info(
                        "mineru.wait_for_completion.waiting_items",
                        extra=log_extra(
                            batch_id=batch_id,
                            item_count=len(items),
                            expected_item_count=expected_item_count,
                        ),
                    )
                    await asyncio.sleep(poll_interval)
                    continue

            if task_info.status == MinerUTaskStatus.COMPLETED:
                log.info(
                    "mineru.wait_for_completion.done",
                    extra=log_extra(
                        batch_id=batch_id,
                        elapsed_s=round(asyncio.get_running_loop().time() - start_time, 3),
                    ),
                )
                return task_info

            if task_info.status == MinerUTaskStatus.FAILED:
                error_msg = task_info.error_msg or "任务执行失败"
                log.warning(
                    "mineru.wait_for_completion.failed",
                    extra=log_extra(batch_id=batch_id, error_msg=error_msg),
                )
                raise MinerUClientError(
                    message=error_msg,
                    status_code=500,
                    code="task_failed",
                )

            await asyncio.sleep(poll_interval)

        log.warning(
            "mineru.wait_for_completion.timeout",
            extra=log_extra(
                batch_id=batch_id,
                elapsed_s=round(asyncio.get_running_loop().time() - start_time, 3),
            ),
        )
        raise MinerUClientError(
            message="等待任务完成超时",
            status_code=504,
            code="timeout",
        )


# 需要导入 uuid
import uuid


class MinerUIngestionService:
    """MinerU 文档摄入服务"""

    DEFAULT_OUTPUT_DIR = "mineru_raw"

    def __init__(
        self,
        config: MinerUConfig | None = None,
        output_base_dir: Path | str | None = None,
    ):
        """初始化摄入服务。

        Args:
            config: MinerU 配置
            output_base_dir: 输出基础目录
        """
        if config is not None:
            self.config = config
        else:
            self.config = MinerUClient._load_config_from_db()
        self.output_base_dir = Path(output_base_dir or "data/intermediates")
        self._client: MinerUClient | None = None

    @property
    def client(self) -> MinerUClient:
        """获取客户端实例"""
        if self._client is None:
            self._client = MinerUClient(self.config)
        return self._client

    def _get_output_dir(self, source_file_id: int) -> Path:
        """获取指定源文件的输出目录"""
        output_dir = self.output_base_dir / str(source_file_id) / self.DEFAULT_OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _generate_output_filename(self, source_filename: str) -> str:
        """生成输出文件名"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        sanitized = self.client._sanitize_filename(source_filename)
        return f"{sanitized}_{timestamp}.json"

    async def ingest_document(
        self,
        file_path: Path | str,
        source_file_id: int,
        options: MinerUIngestOptions | None = None,
        wait_for_completion: bool = False,
    ) -> MinerUIngestResult:
        """摄入单个文档。

        Args:
            file_path: 文档路径
            source_file_id: 源文件 ID
            options: 摄入选项
            wait_for_completion: 是否等待处理完成

        Returns:
            摄入结果
        """
        path = Path(file_path)
        if not path.exists():
            raise MinerUClientError(
                message=f"文件不存在: {path}",
                status_code=404,
                code="file_not_found",
            )

        output_dir = self._get_output_dir(source_file_id)
        output_filename = self._generate_output_filename(path.name)
        output_path = output_dir / output_filename

        try:
            # 读取文件
            file_content = path.read_bytes()

            # 申请上传链接并提交解析任务
            upload_result = await self.client.apply_upload_urls(
                files=[(path.name, file_content)],
                data_ids=[str(source_file_id)],
                options=options,
            )

            batch_id = upload_result["batch_id"]
            file_urls = upload_result["file_urls"]

            if not file_urls:
                raise MinerUClientError(
                    message="未获取到上传链接",
                    status_code=400,
                    code="upload_url_empty",
                )

            # 上传文件
            await self.client.upload_file(path, file_urls[0])

            if wait_for_completion:
                # 等待任务完成
                task_info = await self.client.wait_for_completion(batch_id)

                if task_info.status == MinerUTaskStatus.COMPLETED:
                    # 下载并解压结果
                    download_url = task_info.result.get("download_url")
                    if download_url:
                        try:
                            zip_content = await self.client.download_file(download_url)
                            with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
                                zf.extractall(output_dir)
                            log.info(
                                "mineru.ingest.extracted",
                                extra=log_extra(
                                    source_file_id=source_file_id,
                                    output_dir=str(output_dir),
                                ),
                            )
                        except Exception as e:
                            log.error(
                                "mineru.ingest.download_failed",
                                extra=log_extra(
                                    source_file_id=source_file_id,
                                    error=str(e),
                                ),
                            )
                            # 即使下载失败，也记录元数据，但标记为部分成功或记录错误
                            # 这里选择继续，但在 result_data 中可能需要体现

                    result_data = {
                        "batch_id": batch_id,
                        "source_file_id": source_file_id,
                        "filename": path.name,
                        "status": "completed",
                        "result": task_info.result,
                        "created_at": task_info.created_at,
                        "updated_at": task_info.updated_at,
                    }
                    self._save_result(output_path, result_data)

                    return MinerUIngestResult(
                        success=True,
                        task_id=batch_id,
                        output_path=output_path,
                        metadata=result_data,
                    )
                else:
                    return MinerUIngestResult(
                        success=False,
                        task_id=batch_id,
                        error=task_info.error_msg or "任务未完成",
                    )
            else:
                # 只提交任务，不等待
                result_data = {
                    "batch_id": batch_id,
                    "source_file_id": source_file_id,
                    "filename": path.name,
                    "status": "pending",
                    "output_path": str(output_path),
                    "file_urls": file_urls,
                    "created_at": datetime.now().isoformat(),
                }
                self._save_result(output_path, result_data)

                return MinerUIngestResult(
                    success=True,
                    task_id=batch_id,
                    output_path=output_path,
                    metadata=result_data,
                )

        except MinerUClientError as e:
            error_data = {
                "source_file_id": source_file_id,
                "filename": path.name,
                "status": "failed",
                "error": e.message,
                "created_at": datetime.now().isoformat(),
            }
            self._save_result(output_path, error_data)

            return MinerUIngestResult(
                success=False,
                error=e.message,
            )

    async def ingest_batch(
        self,
        file_paths: list[Path | str],
        source_file_ids: list[int] | None = None,
        options: MinerUIngestOptions | None = None,
        wait_for_completion: bool = False,
        max_concurrent: int = 5,
    ) -> list[MinerUIngestResult]:
        """批量摄入文档"""
        import asyncio

        if source_file_ids is None:
            source_file_ids = list(range(len(file_paths)))

        if len(file_paths) != len(source_file_ids):
            raise ValueError("file_paths 和 source_file_ids 长度必须一致")

        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(
            file_path: Path | str,
            source_file_id: str | int,
        ) -> MinerUIngestResult:
            async with semaphore:
                return await self.ingest_document(
                    file_path=file_path,
                    source_file_id=source_file_id,
                    options=options,
                    wait_for_completion=wait_for_completion,
                )

        tasks = [
            process_with_semaphore(path, sid)
            for path, sid in zip(file_paths, source_file_ids)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append(
                    MinerUIngestResult(success=False, error=str(result))
                )
            else:
                processed_results.append(result)

        return processed_results

    def _save_result(self, output_path: Path, data: dict[str, Any]) -> None:
        """保存结果到文件"""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            json_str = json.dumps(data, ensure_ascii=False, indent=2)
            json.loads(json_str)
            output_path.write_text(json_str, encoding="utf-8")
        except (OSError, json.JSONDecodeError) as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"保存 MinerU 结果失败: {output_path}, 错误: {e}")

    async def cleanup(self) -> None:
        """清理资源"""
        if self._client is not None:
            await self._client.close()
            self._client = None
