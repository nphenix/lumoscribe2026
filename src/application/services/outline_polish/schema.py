"""大纲优化模块 Pydantic 数据模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class OutlinePolishInput(BaseModel):
    """大纲润色输入"""

    outline: str = Field(
        ...,
        description="用户输入的原始大纲（Markdown 格式）",
        min_length=10,
    )
    industry: str = Field(
        default="储能行业",
        description="行业背景配置",
    )
    report_type: str = Field(
        default="市场研究报告",
        description="报告类型",
    )
    language: str = Field(
        default="中文",
        description="语言",
    )
    style: str = Field(
        default="专业、客观、数据驱动",
        description="写作风格",
    )


class OutlinePolishOutput(BaseModel):
    """大纲润色输出"""

    polished_outline: str = Field(
        ...,
        description="优化后的大纲（Markdown 格式）",
    )
    changes_summary: list[str] = Field(
        default_factory=list,
        description="修改摘要列表",
    )
    structure_integrity: bool = Field(
        default=True,
        description="结构完整性检查结果",
    )
    core_keywords_preserved: bool = Field(
        default=True,
        description="核心关键词是否保留",
    )
    recognized_requirements: list[str] = Field(
        default_factory=list,
        description="识别出的用户要求列表",
    )
    original_structure: list[str] = Field(
        default_factory=list,
        description="原始章节结构（识别出的）",
    )


class OutlinePolishResult(BaseModel):
    """大纲润色完整结果"""

    success: bool = Field(
        ...,
        description="处理是否成功",
    )
    input: OutlinePolishInput = Field(
        ...,
        description="输入参数",
    )
    output: OutlinePolishOutput | None = Field(
        default=None,
        description="输出结果",
    )
    error: str | None = Field(
        default=None,
        description="错误信息",
    )
