"""MinerU Pydantic 模型定义。"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MinerUModelVersion(str, Enum):
    """MinerU 模型版本。"""

    PIPELINE = "pipeline"
    VLM = "vlm"


class MinerUParseMethod(str, Enum):
    """MinerU 解析方法。"""

    AUTO = "auto"
    OCR = "ocr"
    TEXT = "text"
    HYBRID = "hybrid"


class MinerULanguage(str, Enum):
    """MinerU 支持的语言。"""

    CH = "ch"
    EN = "en"
    MULTI = "multi"


class MinerUTaskStatus(str, Enum):
    """MinerU 任务状态。"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class MinerUConfig(BaseModel):
    """MinerU 服务配置类。"""

    base_url: str = Field(..., description="API 地址")
    api_token: str = Field(default="", description="Bearer Token")
    model_version: MinerUModelVersion = Field(
        default=MinerUModelVersion.PIPELINE,
        description="模型版本：pipeline 或 vlm",
    )
    timeout: int = Field(
        default=60,
        ge=1,
        le=600,
        description="请求超时时间（秒）",
    )
    upload_timeout: int = Field(
        default=300,
        ge=1,
        le=1800,
        description="上传超时时间（秒）",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="重试次数",
    )

    @classmethod
    def from_env(cls, env_prefix: str = "MINERU_") -> "MinerUConfig":
        """从环境变量加载配置。"""
        import os

        return cls(
            base_url=os.getenv(f"{env_prefix}API_URL", cls().base_url),
            api_token=os.getenv(f"{env_prefix}API_TOKEN", ""),
            model_version=MinerUModelVersion(
                os.getenv(f"{env_prefix}MODEL_VERSION", MinerUModelVersion.PIPELINE.value)
            ),
            timeout=int(os.getenv(f"{env_prefix}TIMEOUT", "60")),
            upload_timeout=int(os.getenv(f"{env_prefix}UPLOAD_TIMEOUT", "300")),
            max_retries=int(os.getenv(f"{env_prefix}MAX_RETRIES", "3")),
        )

    def get_auth_headers(self) -> dict[str, str]:
        """获取认证请求头。"""
        return {"Authorization": f"Bearer {self.api_token}"} if self.api_token else {}


class MinerUFileInfo(BaseModel):
    """MinerU 文件信息。"""

    name: str = Field(..., description="文件名")
    is_ocr: bool = Field(default=False, description="是否启动 OCR 功能")
    data_id: str | None = Field(None, description="数据 ID")
    page_ranges: str | None = Field(None, description="页码范围")


class MinerUParseRequest(BaseModel):
    """MinerU 解析请求。"""

    enable_formula: bool = Field(default=True, description="是否开启公式识别")
    enable_table: bool = Field(default=True, description="是否开启表格识别")
    language: MinerULanguage = Field(default=MinerULanguage.CH, description="文档语言")
    file: MinerUFileInfo = Field(..., description="文件信息")
    callback: str | None = Field(None, description="回调通知 URL")
    seed: str | None = Field(None, description="随机字符串，用于签名")
    extra_formats: list[str] | None = Field(None, description="额外导出格式")
    model_version: MinerUModelVersion | None = Field(None, description="模型版本")


class MinerUParseResponse(BaseModel):
    """MinerU 解析响应。"""

    task_id: str = Field(..., description="任务 ID")
    status: MinerUTaskStatus = Field(..., description="任务状态")


class MinerUTaskInfo(BaseModel):
    """MinerU 任务信息。"""

    task_id: str = Field(..., description="任务 ID")
    status: MinerUTaskStatus = Field(..., description="任务状态")
    status_desc: str | None = Field(None, description="状态描述")
    progress: int = Field(default=0, ge=0, le=100, description="进度百分比")
    result: dict[str, Any] | None = Field(None, description="解析结果")
    error_msg: str | None = Field(None, description="错误信息")
    created_at: str | None = Field(None, description="创建时间")
    updated_at: str | None = Field(None, description="更新时间")


class MinerUTaskListResponse(BaseModel):
    """MinerU 任务列表响应。"""

    code: int = Field(..., description="状态码")
    message: str = Field(..., description="状态信息")
    data: list[MinerUTaskInfo] = Field(..., description="任务列表")
    total: int = Field(..., description="总数")


class MinerUTaskResponse(BaseModel):
    """MinerU 单任务响应。"""

    code: int = Field(..., description="状态码")
    message: str = Field(..., description="状态信息")
    data: MinerUTaskInfo | None = Field(None, description="任务详情")


class MinerUUploadUrlResponse(BaseModel):
    """MinerU 上传 URL 响应。"""

    code: int = Field(..., description="状态码")
    message: str = Field(..., description="状态信息")
    data: dict[str, Any] | None = Field(None, description="上传 URL 信息")


class MinerURawArtifact(BaseModel):
    """MinerU 原始中间产物。"""

    model_config = ConfigDict(from_attributes=True)

    source_file_id: int = Field(..., description="源文件 ID")
    batch_id: str = Field(..., description="批次 ID")
    output_path: str = Field(..., description="输出路径")
    status: str = Field(
        ...,
        description="状态：pending | processing | completed | failed",
    )
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")


class MinerUIngestOptions(BaseModel):
    """MinerU 摄入选项。"""

    enable_formula: bool = Field(default=True, description="是否开启公式识别")
    enable_table: bool = Field(default=True, description="是否开启表格识别")
    language: MinerULanguage = Field(default=MinerULanguage.CH, description="文档语言")
    model_version: MinerUModelVersion | None = Field(None, description="模型版本覆盖")
    extra_formats: list[str] | None = Field(None, description="额外导出格式")
    callback_url: str | None = Field(None, description="回调 URL")


class MinerUIngestResult(BaseModel):
    """MinerU 摄入结果。"""

    success: bool = Field(..., description="是否成功")
    task_id: str | None = Field(None, description="任务 ID")
    output_path: Path | None = Field(None, description="输出文件路径")
    error: str | None = Field(None, description="错误信息")
    metadata: dict[str, Any] | None = Field(None, description="元数据")


class MinerUProcessResult(BaseModel):
    """MinerU 处理结果。"""

    source_file_id: int
    batch_id: str
    task_id: str
    status: MinerUTaskStatus
    output_path: Path | None
    error: str | None
