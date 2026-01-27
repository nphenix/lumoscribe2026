"""内容生成模块类型定义（T042/T096）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.application.schemas.chart_spec import ChartTemplateSnippet


@dataclass
class TemplateSection:
    """模板 section（章节）。"""

    section_id: str
    title: str
    content: str
    order: int


@dataclass
class OutlineItem:
    """白皮书大纲条目（子章节）。"""

    raw: str
    number: str | None
    title: str
    depth: int

    @property
    def display(self) -> str:
        if self.number:
            return f"{self.number} {self.title}".strip()
        return self.title.strip()


@dataclass
class SectionGenerationResult:
    """章节生成结果。"""

    section_id: str
    title: str
    content: str
    rendered_charts: dict[str, ChartTemplateSnippet]
    sources: list[str]
    tokens_used: int
    generation_time_ms: float
    coverage: list[dict] = field(default_factory=list)


@dataclass
class ContentGenerationResult:
    """内容生成最终结果。"""

    template_id: str
    template_name: str
    html_content: str
    sections: list[SectionGenerationResult]
    total_tokens: int
    total_time_ms: float
    generated_at: datetime
    document_title: str | None = None

