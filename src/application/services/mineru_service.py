"""MinerU 在线服务接入层。"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from tenacity import (
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
    MinerUParseRequest,
    MinerUTaskInfo,
    MinerUTaskResponse,
    MinerUTaskStatus,
    MinerUUploadUrlResponse,
)
from src.shared.errors import AppError


class MinerUClientError(AppError):
    """MinerU 客户端错误。"""

    def __init__(self, message: str, status_code: int = 500, code: str = "mineru_client_error"):
        super().__init__(message=message, status_code=status_code, code=code)


class StatusCodeMapper:
    """HTTP 状态码映射到应用错误码。"""

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
        """获取错误码和 HTTP 状态码。"""
        return cls._MAPPING.get(status_code, ("unknown_error", status_code))


class MinerUClient:
    """MinerU 在线服务客户端。"""

    def __init__(self, config: MinerUConfig | None = None):
        """初始化客户端。

        Args:
            config: MinerU 配置，如果为 None 则从环境变量加载
        """
        self.config = config or MinerUConfig.from_env()
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """获取异步 HTTP 客户端。"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout),
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        """关闭客户端。"""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "MinerUClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def _sanitize_filename(self, filename: str) -> str:
        """清理文件名，移除不安全字符。"""
        # 移除非字母数字、下划线、连字符、点号的字符
        sanitized = re.sub(r"[^\w\-.]", "_", filename)
        # 避免文件名以点号开头或结尾
        sanitized = sanitized.strip(".")
        # 限制长度
        max_length = 200
        if len(sanitized) > max_length:
            name, ext = Path(sanitized).stem, Path(sanitized).suffix
            sanitized = f"{name[:max_length - len(ext) - 1]}{ext}"
        return sanitized

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException)),
    )
    async def apply_upload_urls(
        self,
        files: list[tuple[str, bytes]],
        data_ids: list[str] | None = None,
    ) -> dict[str, str]:
        """批量申请上传链接。

        Args:
            files: 文件列表，格式为 [(filename, file_bytes), ...]
            data_ids: 数据 ID 列表，与 files 一一对应

        Returns:
            上传 URL 字典，格式为 {filename: upload_url}

        Raises:
            MinerUClientError: 申请上传链接失败
        """
        client = await self._get_client()

        # 准备表单数据
        form_data: dict[str, Any] = {}
        for idx, (filename, _) in enumerate(files):
            form_data[f"file_names[{idx}]"] = filename
            if data_ids and idx < len(data_ids):
                form_data[f"data_ids[{idx}]"] = data_ids[idx]

        try:
            response = await client.post(
                f"{self.config.base_url}/upload/pre-signed-url",
                headers=self.config.get_auth_headers(),
                data=form_data,
            )
            response.raise_for_status()
            data = response.json()

            if data.get("code") != 0:
                raise MinerUClientError(
                    message=data.get("message", "申请上传链接失败"),
                    status_code=400,
                    code="upload_url_error",
                )

            # 解析上传 URL
            upload_urls = {}
            upload_data = data.get("data", {})
            for filename, url_info in upload_data.items():
                if isinstance(url_info, dict):
                    upload_urls[filename] = url_info.get("url", "")
                else:
                    upload_urls[filename] = str(url_info)

            return upload_urls

        except httpx.HTTPStatusError as e:
            code, status_code = StatusCodeMapper.get_error(e.response.status_code)
            raise MinerUClientError(
                message=f"申请上传链接失败: {e.response.text}",
                status_code=status_code,
                code=code,
            ) from e
        except httpx.RequestError as e:
            raise MinerUClientError(
                message=f"网络错误: {str(e)}",
                status_code=503,
                code="network_error",
            ) from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException)),
    )
    async def upload_file(
        self,
        file_path: Path | str,
        upload_url: str,
        content_type: str = "application/pdf",
    ) -> bool:
        """上传文件到预签名 URL。

        Args:
            file_path: 本地文件路径
            upload_url: 预签名上传 URL
            content_type: 内容类型

        Returns:
            是否上传成功

        Raises:
            MinerUClientError: 上传失败
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
            with open(file_path, "rb") as f:
                file_content = f.read()

            headers = {"Content-Type": content_type}
            response = await client.put(
                upload_url,
                content=file_content,
                headers=headers,
                timeout=httpx.Timeout(self.config.upload_timeout),
            )
            response.raise_for_status()

            return True

        except httpx.HTTPStatusError as e:
            code, status_code = StatusCodeMapper.get_error(e.response.status_code)
            raise MinerUClientError(
                message=f"上传文件失败: {e.response.text}",
                status_code=status_code,
                code=code,
            ) from e
        except httpx.RequestError as e:
            raise MinerUClientError(
                message=f"网络错误: {str(e)}",
                status_code=503,
                code="network_error",
            ) from e
        except OSError as e:
            raise MinerUClientError(
                message=f"读取文件失败: {str(e)}",
                status_code=500,
                code="file_read_error",
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
            任务 ID

        Raises:
            MinerUClientError: 提交任务失败
        """
        client = await self._get_client()
        options = options or MinerUIngestOptions()

        file_info = {
            "name": filename,
            "is_ocr": options.enable_table,  # 表格识别通常需要 OCR
        }
        if data_id:
            file_info["data_id"] = data_id

        request_data = {
            "enable_formula": options.enable_formula,
            "enable_table": options.enable_table,
            "language": options.language.value if isinstance(options.language, MinerULanguage) else options.language,
            "file": file_info,
        }

        if options.callback_url:
            request_data["callback"] = options.callback_url
            request_data["seed"] = str(uuid.uuid4()).replace("-", "")[:16]

        if options.extra_formats:
            request_data["extra_formats"] = options.extra_formats

        if options.model_version:
            request_data["model_version"] = options.model_version.value
        else:
            request_data["model_version"] = self.config.model_version.value

        try:
            response = await client.post(
                f"{self.config.base_url}/parse",
                headers=self.config.get_auth_headers(),
                json=request_data,
            )
            response.raise_for_status()
            data = response.json()

            if data.get("code") != 0:
                raise MinerUClientError(
                    message=data.get("message", "提交解析任务失败"),
                    status_code=400,
                    code="parse_error",
                )

            return data["data"]["task_id"]

        except httpx.HTTPStatusError as e:
            code, status_code = StatusCodeMapper.get_error(e.response.status_code)
            raise MinerUClientError(
                message=f"提交解析任务失败: {e.response.text}",
                status_code=status_code,
                code=code,
            ) from e
        except httpx.RequestError as e:
            raise MinerUClientError(
                message=f"网络错误: {str(e)}",
                status_code=503,
                code="network_error",
            ) from e

    async def get_task_status(self, task_id: str) -> MinerUTaskInfo:
        """查询任务状态。

        Args:
            task_id: 任务 ID

        Returns:
            任务信息

        Raises:
            MinerUClientError: 查询失败
        """
        client = await self._get_client()

        try:
            response = await client.get(
                f"{self.config.base_url}/tasks/{task_id}",
                headers=self.config.get_auth_headers(),
            )
            response.raise_for_status()
            data = response.json()

            if data.get("code") != 0:
                raise MinerUClientError(
                    message=data.get("message", "查询任务状态失败"),
                    status_code=400,
                    code="task_query_error",
                )

            task_data = data.get("data", {})
            status = task_data.get("status", "pending")
            if status not in MinerUTaskStatus.__members__:
                status = "failed"

            return MinerUTaskInfo(
                task_id=task_id,
                status=MinerUTaskStatus(status),
                status_desc=task_data.get("status_desc"),
                progress=task_data.get("progress", 0),
                result=task_data.get("result"),
                error_msg=task_data.get("error_msg"),
                created_at=task_data.get("created_at"),
                updated_at=task_data.get("updated_at"),
            )

        except httpx.HTTPStatusError as e:
            code, status_code = StatusCodeMapper.get_error(e.response.status_code)
            raise MinerUClientError(
                message=f"查询任务状态失败: {e.response.text}",
                status_code=status_code,
                code=code,
            ) from e
        except httpx.RequestError as e:
            raise MinerUClientError(
                message=f"网络错误: {str(e)}",
                status_code=503,
                code="network_error",
            ) from e

    async def wait_for_completion(
        self,
        task_id: str,
        poll_interval: float = 5.0,
        max_wait: float = 600.0,
    ) -> MinerUTaskInfo:
        """等待任务完成。

        Args:
            task_id: 任务 ID
            poll_interval: 轮询间隔（秒）
            max_wait: 最大等待时间（秒）

        Returns:
            最终任务状态

        Raises:
            MinerUClientError: 等待超时或任务失败
        """
        import asyncio

        start_time = datetime.now()
        while (datetime.now() - start_time).total_seconds() < max_wait:
            task_info = await self.get_task_status(task_id)

            if task_info.status == MinerUTaskStatus.COMPLETED:
                return task_info

            if task_info.status == MinerUTaskStatus.FAILED:
                error_msg = task_info.error_msg or "任务执行失败"
                raise MinerUClientError(
                    message=error_msg,
                    status_code=500,
                    code="task_failed",
                )

            await asyncio.sleep(poll_interval)

        raise MinerUClientError(
            message="等待任务完成超时",
            status_code=504,
            code="timeout",
        )

    async def process_batch(
        self,
        file_paths: list[Path | str],
        options: MinerUIngestOptions | None = None,
    ) -> dict[str, str]:
        """批量处理文件。

        Args:
            file_paths: 文件路径列表
            options: 解析选项

        Returns:
            文件名到任务 ID 的映射

        Raises:
            MinerUClientError: 处理失败
        """
        results = {}
        for file_path in file_paths:
            path = Path(file_path)
            try:
                task_id = await self.parse_document(
                    filename=path.name,
                    data_id=str(path.stat().st_ino) if path.exists() else None,
                    options=options,
                )
                results[path.name] = task_id
            except MinerUClientError as e:
                results[path.name] = f"ERROR: {e.message}"

        return results

    @staticmethod
    def verify_callback(
        content: str,
        checksum: str,
        uid: str,
        seed: str | None = None,
    ) -> bool:
        """验证回调签名。

        Args:
            content: 回调内容（JSON 字符串）
            checksum: 回调中的校验签名
            uid: 用户 ID
            seed: 随机字符串

        Returns:
            签名是否有效
        """
        # 计算期望的 checksum
        message = uid + (seed or "") + content
        expected_checksum = hashlib.sha256(message.encode("utf-8")).hexdigest()
        return hmac.compare_digest(checksum, expected_checksum)


class MinerUIngestionService:
    """MinerU 文档摄入服务。"""

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
        self.config = config or MinerUConfig.from_env()
        self.output_base_dir = Path(output_base_dir or "data/intermediates")
        self._client: MinerUClient | None = None

    @property
    def client(self) -> MinerUClient:
        """获取客户端实例。"""
        if self._client is None:
            self._client = MinerUClient(self.config)
        return self._client

    def _get_output_dir(self, source_file_id: int) -> Path:
        """获取指定源文件的输出目录。"""
        output_dir = self.output_base_dir / str(source_file_id) / self.DEFAULT_OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _generate_output_filename(self, source_filename: str) -> str:
        """生成输出文件名。

        Args:
            source_filename: 源文件名

        Returns:
            带时间戳的输出文件名
        """
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

        Raises:
            MinerUClientError: 摄入失败
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
            # 提交解析任务
            task_id = await self.client.parse_document(
                filename=path.name,
                data_id=str(source_file_id),
                options=options,
            )

            if wait_for_completion:
                # 等待任务完成并获取结果
                task_info = await self.client.wait_for_completion(task_id)

                if task_info.status == MinerUTaskStatus.COMPLETED:
                    # 保存结果到文件
                    result_data = {
                        "task_id": task_id,
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
                        task_id=task_id,
                        output_path=output_path,
                        metadata=result_data,
                    )
                else:
                    return MinerUIngestResult(
                        success=False,
                        task_id=task_id,
                        error=task_info.error_msg or "任务未完成",
                    )
            else:
                # 只提交任务，不等待
                result_data = {
                    "task_id": task_id,
                    "source_file_id": source_file_id,
                    "filename": path.name,
                    "status": "pending",
                    "output_path": str(output_path),
                    "created_at": datetime.now().isoformat(),
                }
                self._save_result(output_path, result_data)

                return MinerUIngestResult(
                    success=True,
                    task_id=task_id,
                    output_path=output_path,
                    metadata=result_data,
                )

        except MinerUClientError as e:
            # 保存错误信息
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
        """批量摄入文档。

        Args:
            file_paths: 文档路径列表
            source_file_ids: 源文件 ID 列表，与 file_paths 一一对应
            options: 摄入选项
            wait_for_completion: 是否等待处理完成
            max_concurrent: 最大并发数

        Returns:
            摄入结果列表

        Raises:
            MinerUClientError: 批量摄入失败
        """
        import asyncio

        if source_file_ids is None:
            source_file_ids = list(range(len(file_paths)))

        if len(file_paths) != len(source_file_ids):
            raise ValueError("file_paths 和 source_file_ids 长度必须一致")

        # 创建信号量控制并发
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(
            file_path: Path | str,
            source_file_id: int,
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

        # 处理异常
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
        """保存结果到文件。

        Args:
            output_path: 输出文件路径
            data: 结果数据
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # 验证 JSON 格式
            json_str = json.dumps(data, ensure_ascii=False, indent=2)
            # 再次解析验证
            json.loads(json_str)
            output_path.write_text(json_str, encoding="utf-8")
        except (OSError, json.JSONDecodeError) as e:
            # 记录错误但不抛出
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"保存 MinerU 结果失败: {output_path}, 错误: {e}")

    async def cleanup(self) -> None:
        """清理资源。"""
        if self._client is not None:
            await self._client.close()
            self._client = None
