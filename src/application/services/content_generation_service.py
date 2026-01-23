"""内容生成服务（T042）。

按模板 section 生成内容，输出单 HTML。
整合 RAG 检索（混合检索 + Rerank）和图表渲染能力。
"""

from __future__ import annotations

import asyncio
import json
import inspect
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal

from src.application.repositories.template_repository import TemplateRepository
from src.application.schemas.chart_spec import (
    ChartConfig,
    ChartDataPoint,
    ChartRenderRequest,
    ChartTemplateSnippet,
    ChartSeries,
)
from src.application.schemas.ingest import HybridSearchOptions, SearchResult
from src.application.schemas.template import PreprocessResponse
from src.application.services.chart_renderer_service import ChartRendererService
from src.application.services.hybrid_search_service import (
    HybridSearchService,
)
from src.application.services.llm_runtime_service import LLMRuntimeService
from src.application.services.template_service import TemplateService
from src.domain.entities.template import Template, TemplateType
from src.shared.errors import AppError
from src.shared.logging import get_logger, log_extra
from src.shared.constants.prompts import (
    SCOPE_CONTENT_GENERATION_SECTION,
    SCOPE_OUTLINE_POLISH,
)
from src.application.services.content_generation.html_renderer import WhitepaperHtmlRenderer

log = get_logger(__name__)

def _read_text_best_effort(path: Path) -> str:
    """尽量读取文本（兼容 Windows 常见编码）。"""
    encodings = ["utf-8", "utf-8-sig", "gb18030", "gbk"]
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    return path.read_text(encoding="utf-8", errors="replace")

_THINK_BLOCK_RE = re.compile(r"(?is)<think>.*?</think>")
_THINK_FENCE_RE = re.compile(r"(?is)```(?:thinking|thought|analysis)[\s\S]*?```")
_REASONING_HEAD_RE = re.compile(
    r"(?im)^(?:#+\s*)?(?:思考|推理|Chain-of-Thought|CoT|Thoughts?)\s*[:：].*$"
)
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _strip_model_think(text: str) -> str:
    """移除模型输出中的“思考/推理”痕迹，保证最终产物为纯正文。"""
    if not text:
        return ""
    s = str(text)
    # 1) 常见 <think>...</think>
    s = _THINK_BLOCK_RE.sub("", s)
    # 2) 常见 ```thinking ...``` / ```analysis ...```
    s = _THINK_FENCE_RE.sub("", s)
    # 3) 行首的“思考/推理”标题行（只移除该行，避免误删正文）
    s = _REASONING_HEAD_RE.sub("", s)
    # 4) 一些模型会输出前缀 “思考：... <正文>”，尝试移除开头到第一个空行
    head = s.lstrip()
    if head.startswith("思考") or head.lower().startswith("thinking") or head.lower().startswith("analysis"):
        parts = re.split(r"\n\s*\n", s, maxsplit=1)
        if len(parts) == 2:
            s = parts[1]
    # 5) 清理多余空行
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s


def _strip_leading_title_heading(text: str, *, title: str) -> str:
    """移除模型重复输出的章节标题（例如开头的 '# 第一章...'），避免与外层 <h2> 重复。"""
    if not text or not title:
        return text or ""
    ttl = str(title).strip()
    if not ttl:
        return text
    # 允许轻微格式差异：去掉空白与中英文冒号后比较
    def _norm(s: str) -> str:
        return re.sub(r"[\s:：]+", "", (s or "").strip())

    lines = str(text).replace("\r\n", "\n").split("\n")
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines):
        return ""
    m = re.match(r"^#{1,6}\s+(.+?)\s*$", lines[i].strip())
    if not m:
        return text
    head = (m.group(1) or "").strip()
    if _norm(head) != _norm(ttl):
        return text
    # 跳过该标题行及其后的空行
    i += 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    return "\n".join(lines[i:]).strip()


class ContentGenerationError(AppError):
    """内容生成错误。"""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        code: str = "content_generation_error",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(
            message=message,
            status_code=status_code,
            code=code,
            details=details,
        )


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
    coverage: list[dict[str, Any]] = field(default_factory=list)


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


class ContentGenerationService:
    """内容生成服务。"""

    # 默认检索选项
    DEFAULT_TOP_K = 10
    DEFAULT_RERANK_TOP = 5
    DEFAULT_PER_QUERY_TOP_N = 2
    DEFAULT_MAX_CONTEXT_CHARS = 15000
    # 默认每个章节最多自动渲染的图表数量（从真实 chart_json 反查）
    # 说明：用户验收中常出现“来源里多个图表，但只渲染 1 个”的情况；此处提高上限以提升命中率。
    DEFAULT_MAX_AUTO_CHARTS_PER_SECTION = 6

    _MD_IMAGE_RE = re.compile(r"!\[[^\]]*]\(([^)]+)\)")
    _GLOSSARY_PLACEHOLDER_RE = re.compile(
        r"(概念一|概念二|概念三|制度一|机制一|规划一|指标一|指数一|本概念强调)"
    )

    async def _emit(
        self,
        on_event: Callable[[dict[str, Any]], Any] | None,
        payload: dict[str, Any],
    ) -> None:
        if on_event is None:
            return
        try:
            r = on_event(payload)
            if inspect.isawaitable(r):
                await r  # type: ignore[misc]
        except Exception:
            # 进度回调失败不影响主流程
            return

    def __init__(
        self,
        template_service: TemplateService,
        hybrid_search_service: HybridSearchService,
        llm_runtime_service: LLMRuntimeService,
        chart_renderer_service: ChartRendererService,
        template_repository: TemplateRepository,
    ):
        self.template_service = template_service
        self.hybrid_search_service = hybrid_search_service
        self.llm_runtime_service = llm_runtime_service
        self.chart_renderer_service = chart_renderer_service
        self.template_repository = template_repository

    def parse_template_sections(
        self, template: Template, *, content_override: str | None = None
    ) -> list[TemplateSection]:
        """解析模板为多个 section。"""
        if content_override is None:
            storage_path = Path("data") / template.storage_path

            if not storage_path.exists():
                raise ContentGenerationError(
                    message="模板文件不存在",
                    code="template_not_found",
                    details={"storage_path": str(storage_path)},
                )

            content = _read_text_best_effort(storage_path)
        else:
            content = content_override

        if template.type == TemplateType.CUSTOM:
            return self._parse_markdown_sections(content, template.id)
        else:
            # Office 文档暂时不支持 section 解析
            return [
                TemplateSection(
                    section_id="main",
                    title=template.original_filename,
                    content=content,
                    order=0,
                )
            ]

    def _parse_markdown_sections(self, content: str, template_id: str) -> list[TemplateSection]:
        """解析 Markdown 模板为章节。"""
        sections = []
        current_order = 0

        # 按 # 标题分割
        lines = content.split("\n")
        current_section_content = []
        current_title = None

        for line in lines:
            if line.startswith("## "):
                # 保存上一个 section
                if current_title is not None:
                    full_content = "\n".join(current_section_content)
                    sections.append(
                        TemplateSection(
                            section_id=f"{template_id}-section-{current_order}",
                            title=current_title,
                            content=full_content,
                            order=current_order,
                        )
                    )
                    current_order += 1

                current_title = line[3:].strip()
                current_section_content = []
            elif line.startswith("# ") and current_title is None:
                # 文档标题，不作为 section
                pass
            else:
                if current_title is not None:
                    current_section_content.append(line)

        # 保存最后一个 section
        if current_title is not None:
            full_content = "\n".join(current_section_content)
            sections.append(
                TemplateSection(
                    section_id=f"{template_id}-section-{current_order}",
                    title=current_title,
                    content=full_content,
                    order=current_order,
                )
            )

        return sections

    def _parse_outline_items_from_section_content(self, section_content: str) -> list[OutlineItem]:
        """从章节 section 内容中解析大纲条目。

        约定：白皮书大纲子章节使用 Markdown 列表：`- 1.1 标题`，嵌套层级使用缩进 `    - 2.2.1 标题`。
        """
        items: list[OutlineItem] = []
        lines = (section_content or "").replace("\r\n", "\n").split("\n")
        for line in lines:
            m = re.match(r"^(\s*)-\s+(.*)$", line)
            if not m:
                continue
            indent = m.group(1) or ""
            raw = (m.group(2) or "").strip()
            if not raw:
                continue

            # depth: 1 表示无缩进；每 4 个空格增加一级
            depth = 1 + max(0, len(indent) // 4)

            m2 = re.match(r"^(\d+(?:\.\d+)+)\s+(.*)$", raw)
            if m2:
                number = m2.group(1).strip()
                title = (m2.group(2) or "").strip()
            else:
                # 兼容：- 2.2 标题（只有一层点也可）
                m3 = re.match(r"^(\d+(?:\.\d+)*)\s+(.*)$", raw)
                if m3:
                    number = m3.group(1).strip() or None
                    title = (m3.group(2) or "").strip()
                else:
                    number = None
                    title = raw

            if title:
                items.append(OutlineItem(raw=raw, number=number, title=title, depth=depth))

        return items

    def _parse_number_maybe(self, v: Any) -> float | None:
        if v is None:
            return None
        try:
            if isinstance(v, (int, float)):
                return float(v)
            s = str(v).strip()
            if not s:
                return None
            # 54.3% / 54.3
            s = s.replace(",", "")
            is_pct = s.endswith("%")
            if is_pct:
                s = s[:-1].strip()
            return float(s)
        except Exception:
            return None

    def _parse_number_from_cell(self, v: Any) -> float | None:
        """从“单元格”解析数值（兼容 {'value': ...} 结构）。"""
        if isinstance(v, dict) and "value" in v:
            return self._parse_number_maybe(v.get("value"))
        return self._parse_number_maybe(v)

    def _try_table_like_to_grouped_bar(self, *, title: str, obj: dict[str, Any]) -> ChartConfig | None:
        """将“对比表/多指标列表”类 chart_data 尝试转为分组柱状图。"""
        chart_data = obj.get("chart_data")
        if chart_data is None:
            chart_data = obj.get("data")
        # canonical(table): {"columns":[...], "rows":[{...}]}
        if isinstance(chart_data, dict) and isinstance(chart_data.get("rows"), list):
            chart_data = chart_data.get("rows")
        if not isinstance(chart_data, list) or not chart_data:
            return None

        rows: list[dict[str, Any]] = []
        for it in chart_data:
            if not isinstance(it, dict):
                continue
            # 跳过分隔符（多图场景）
            if "_chart_separator" in it:
                continue
            rows.append(it)
        if not rows:
            return None

        prefer_cat_keys = [
            "technology",
            "tech",
            "name",
            "category",
            "label",
            "item",
            "类型",
            "技术",
            "路线",
            "场景",
        ]
        cat_key: str | None = None
        for k in prefer_cat_keys:
            if k in rows[0] and isinstance(rows[0].get(k), (str, int, float)):
                cat_key = k
                break
        if cat_key is None:
            # 兜底：选第一个“看起来像字符串”的列
            for k, v in rows[0].items():
                if isinstance(v, str) and v.strip():
                    cat_key = k
                    break
        if cat_key is None:
            return None

        categories: list[str] = []
        for r in rows:
            v = r.get(cat_key)
            s = str(v).strip() if v is not None else ""
            if not s:
                continue
            categories.append(s)
        if not categories:
            return None

        # 找出数值列：至少 1 行可解析为数值
        numeric_keys: list[str] = []
        for k in rows[0].keys():
            if k == cat_key:
                continue
            ok = False
            for r in rows:
                if self._parse_number_from_cell(r.get(k)) is not None:
                    ok = True
                    break
            if ok:
                numeric_keys.append(k)
        if not numeric_keys:
            # 常见：[{category,value,unit}] 形式
            if "value" in rows[0] and self._parse_number_from_cell(rows[0].get("value")) is not None:
                numeric_keys = ["value"]
            else:
                return None

        # 控制系列数量，避免图表过载（仍不引入 mock，只是渲染选择）
        numeric_keys = numeric_keys[:6]

        series_list: list[ChartSeries] = []
        for nk in numeric_keys:
            pts: list[ChartDataPoint] = []
            for r in rows:
                x = str(r.get(cat_key)).strip() if r.get(cat_key) is not None else ""
                if not x:
                    continue
                y = self._parse_number_from_cell(r.get(nk))
                pts.append(ChartDataPoint(x=x, y=y, name=x))
            series_list.append(ChartSeries(name=str(nk).strip() or "数值", chart_type="bar", data=pts))

        if not series_list:
            return None

        height = 520 if len(categories) > 10 else 460
        return ChartConfig(
            chart_type="bar",
            title=title,
            x_axis_title="",
            y_axis_title="",
            series=series_list,
            height=height,
        )

    def _normalize_chart_type_for_render(self, raw: Any) -> str:
        s = str(raw).strip().lower() if raw is not None else ""
        aliases = {
            # legacy (T094/T097)
            "bar_chart": "bar",
            "line_chart": "line",
            "pie_chart": "pie",
            "stacked_area_chart_with_legend": "stacked_area",
            "flow_chart_with_categories": "sankey",
            # canonical (prompt v1)
            "bar": "bar",
            "line": "line",
            "pie": "pie",
            "stacked_area": "stacked_area",
            "sankey": "sankey",
            "table": "table",
            "scatter": "scatter",
            "heatmap": "heatmap",
            "radar": "radar",
            "unknown": "unknown",
            "none": "none",
            # common variants
            "donut": "pie",
            "donut_chart": "pie",
            "process_flow_diagram": "sankey",
            "process_flow_chart": "sankey",
            "process_flow": "sankey",
            "flow_diagram": "sankey",
            "comparison_table": "table",
            "comparison_chart": "table",
            "energy_storage_comparison_chart": "table",
        }
        return aliases.get(s, s or "unknown")

    def _chart_json_to_bar_configs(self, *, title: str, obj: dict[str, Any]) -> list[ChartConfig]:
        # legacy: chart_data=[{category,value}]
        pts: list[ChartDataPoint] = []
        cd = obj.get("chart_data")
        if isinstance(cd, list):
            for it in cd:
                if not isinstance(it, dict):
                    continue
                if "_chart_separator" in it:
                    continue
                x = it.get("category")
                y = self._parse_number_maybe(it.get("value"))
                if x is None:
                    continue
                pts.append(ChartDataPoint(x=str(x), y=y, name=str(x)))
            if pts:
                return [
                    ChartConfig(
                        chart_type="bar",
                        title=title,
                        x_axis_title="",
                        y_axis_title="",
                        series=[ChartSeries(name="数值", chart_type="bar", data=pts)],
                        height=460,
                    )
                ]

        # canonical: chart_data={"categories":[...], "series":[{"name", "values":[...]}]}
        if isinstance(cd, dict):
            categories = cd.get("categories")
            series = cd.get("series")
            if isinstance(categories, list) and isinstance(series, list) and categories:
                x_order = [str(x).strip() for x in categories if str(x).strip()]
                if not x_order:
                    return []
                series_list: list[ChartSeries] = []
                for s in series[:8]:
                    if not isinstance(s, dict):
                        continue
                    name = str(s.get("name") or "系列").strip()
                    vals = s.get("values")
                    if not isinstance(vals, list):
                        continue
                    pts2: list[ChartDataPoint] = []
                    for i, x in enumerate(x_order):
                        y = self._parse_number_maybe(vals[i]) if i < len(vals) else None
                        pts2.append(ChartDataPoint(x=x, y=y, name=x))
                    series_list.append(ChartSeries(name=name, chart_type="bar", data=pts2))
                if series_list:
                    return [
                        ChartConfig(
                            chart_type="bar",
                            title=title,
                            x_axis_title="",
                            y_axis_title="",
                            series=series_list,
                            height=460,
                        )
                    ]
        return []

    def _chart_json_to_line_configs(self, *, title: str, obj: dict[str, Any]) -> list[ChartConfig]:
        cd = obj.get("chart_data")

        # legacy: [{series, data_points:[{time,value}]}]
        if isinstance(cd, list):
            x_order: list[str] = []
            seen_x: set[str] = set()
            for s in cd:
                if not isinstance(s, dict):
                    continue
                for dp in s.get("data_points") or []:
                    if not isinstance(dp, dict):
                        continue
                    t = dp.get("time")
                    if t is None:
                        continue
                    tx = str(t).strip()
                    if tx and tx not in seen_x:
                        seen_x.add(tx)
                        x_order.append(tx)

            series_list: list[ChartSeries] = []
            for s in cd:
                if not isinstance(s, dict):
                    continue
                s_name = str(s.get("series") or s.get("_chart_separator") or "系列").strip()
                mp: dict[str, float] = {}
                for dp in s.get("data_points") or []:
                    if not isinstance(dp, dict):
                        continue
                    t = dp.get("time")
                    y = self._parse_number_maybe(dp.get("value"))
                    if t is None or y is None:
                        continue
                    mp[str(t).strip()] = y
                pts = [ChartDataPoint(x=x, y=mp.get(x), name=x) for x in x_order]
                series_list.append(ChartSeries(name=s_name or "系列", chart_type="line", data=pts))

            if series_list and x_order:
                return [
                    ChartConfig(
                        chart_type="line",
                        title=title,
                        x_axis_title="",
                        y_axis_title="",
                        series=series_list,
                        height=460,
                    )
                ]

        # canonical: {"x":[...], "series":[{"name","values":[...]}]}
        if isinstance(cd, dict):
            x = cd.get("x")
            series = cd.get("series")
            if isinstance(x, list) and isinstance(series, list) and x:
                x_order = [str(xx).strip() for xx in x if str(xx).strip()]
                if not x_order:
                    return []
                series_list2: list[ChartSeries] = []
                for s in series[:10]:
                    if not isinstance(s, dict):
                        continue
                    name = str(s.get("name") or "系列").strip()
                    vals = s.get("values")
                    if not isinstance(vals, list):
                        continue
                    pts2: list[ChartDataPoint] = []
                    for i, xx in enumerate(x_order):
                        y = self._parse_number_maybe(vals[i]) if i < len(vals) else None
                        pts2.append(ChartDataPoint(x=xx, y=y, name=xx))
                    series_list2.append(ChartSeries(name=name, chart_type="line", data=pts2))
                if series_list2:
                    return [
                        ChartConfig(
                            chart_type="line",
                            title=title,
                            x_axis_title="",
                            y_axis_title="",
                            series=series_list2,
                            height=460,
                        )
                    ]
        return []

    def _chart_json_to_pie_configs(self, *, title: str, obj: dict[str, Any]) -> list[ChartConfig]:
        cd = obj.get("chart_data")

        # canonical: {"items":[{"name","value","unit"}]}
        if isinstance(cd, dict):
            items = cd.get("items")
            if isinstance(items, list) and items:
                pts: list[ChartDataPoint] = []
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    name = it.get("name")
                    val = self._parse_number_maybe(it.get("value"))
                    if name is None or val is None:
                        continue
                    pts.append(ChartDataPoint(x=None, y=val, name=str(name).strip()))
                if pts:
                    return [
                        ChartConfig(
                            chart_type="pie",
                            title=title,
                            series=[ChartSeries(name="占比", chart_type="pie", data=pts)],
                            height=460,
                        )
                    ]

        # legacy: chart_data: 可能包含多个 pie；默认取前 1 个
        if isinstance(cd, list):
            for group in cd[:1]:
                if not isinstance(group, dict):
                    continue
                g_title = str(group.get("_chart_separator") or group.get("description") or title).strip()
                pts2: list[ChartDataPoint] = []
                for dp in group.get("data") or []:
                    if not isinstance(dp, dict):
                        continue
                    label = dp.get("label")
                    val = self._parse_number_maybe(dp.get("value"))
                    if label is None or val is None:
                        continue
                    pts2.append(ChartDataPoint(x=None, y=val, name=str(label)))
                if pts2:
                    return [
                        ChartConfig(
                            chart_type="pie",
                            title=g_title,
                            series=[ChartSeries(name="占比", chart_type="pie", data=pts2)],
                            height=460,
                        )
                    ]
        return []

    def _chart_json_to_stacked_area_configs(self, *, title: str, obj: dict[str, Any]) -> list[ChartConfig]:
        cd = obj.get("chart_data")

        # canonical: {"x":[...], "series":[{"name","values":[...]}]}
        if isinstance(cd, dict):
            x = cd.get("x")
            series = cd.get("series")
            if isinstance(x, list) and isinstance(series, list) and x:
                x_order = [str(xx).strip() for xx in x if str(xx).strip()]
                if not x_order:
                    return []
                series_list: list[ChartSeries] = []
                for s in series[:10]:
                    if not isinstance(s, dict):
                        continue
                    name = str(s.get("name") or "系列").strip()
                    vals = s.get("values")
                    if not isinstance(vals, list):
                        continue
                    pts: list[ChartDataPoint] = []
                    for i, xx in enumerate(x_order):
                        y = self._parse_number_maybe(vals[i]) if i < len(vals) else None
                        pts.append(ChartDataPoint(x=xx, y=y, name=xx))
                    series_list.append(
                        ChartSeries(
                            name=name,
                            chart_type="line",
                            data=pts,
                            extra={
                                "stack": "total",
                                "areaStyle": {},
                                "smooth": True,
                                "emphasis": {"focus": "series"},
                            },
                        )
                    )
                if series_list:
                    return [
                        ChartConfig(
                            chart_type="line",
                            title=title,
                            x_axis_title="时间",
                            y_axis_title="",
                            series=series_list,
                            height=460,
                        )
                    ]

        # legacy: stacked_area_chart_with_legend
        if isinstance(cd, list) and cd:
            group = cd[0] if isinstance(cd[0], dict) else None
            if not isinstance(group, dict):
                return []
            legend = group.get("legend") or {}
            if not isinstance(legend, dict) or not legend:
                return []

            x_order2: list[str] = []
            seen_x: set[str] = set()
            for _, sv in legend.items():
                if not isinstance(sv, dict):
                    continue
                for dp in sv.get("data") or []:
                    if not isinstance(dp, dict):
                        continue
                    t = dp.get("time")
                    if t is None:
                        continue
                    tx = str(t).strip()
                    if tx and tx not in seen_x:
                        seen_x.add(tx)
                        x_order2.append(tx)
            if not x_order2:
                return []

            series_list2: list[ChartSeries] = []
            colors: list[str] = []
            for s_name, sv in legend.items():
                if not isinstance(sv, dict):
                    continue
                name = str(s_name).strip() or "系列"
                mp: dict[str, float] = {}
                for dp in sv.get("data") or []:
                    if not isinstance(dp, dict):
                        continue
                    t = dp.get("time")
                    y = self._parse_number_maybe(dp.get("value"))
                    if t is None or y is None:
                        continue
                    mp[str(t).strip()] = y
                pts = [ChartDataPoint(x=x, y=mp.get(x), name=x) for x in x_order2]
                color = sv.get("color")
                color_s = str(color).strip() if isinstance(color, str) and color.strip() else None
                if color_s:
                    colors.append(color_s)
                series_list2.append(
                    ChartSeries(
                        name=name,
                        chart_type="line",
                        data=pts,
                        color=color_s,
                        extra={
                            "stack": "total",
                            "areaStyle": {},
                            "smooth": True,
                            "emphasis": {"focus": "series"},
                        },
                    )
                )

            y_axis_label = ""
            y_axis = group.get("y_axis")
            if isinstance(y_axis, dict) and isinstance(y_axis.get("label"), str):
                y_axis_label = y_axis.get("label") or ""

            return [
                ChartConfig(
                    chart_type="line",
                    title=title,
                    x_axis_title="时间",
                    y_axis_title=y_axis_label,
                    series=series_list2,
                    colors=colors or None,
                    height=460,
                )
            ]

        return []

    def _chart_json_to_sankey_configs(self, *, title: str, obj: dict[str, Any]) -> list[ChartConfig]:
        chart_data = obj.get("chart_data")
        if chart_data is None:
            chart_data = obj.get("data")
        if chart_data is None:
            chart_data = []

        nodes: dict[str, dict[str, Any]] = {}
        links: list[dict[str, Any]] = []

        def _add_node(name: str) -> None:
            n = name.strip()
            if not n:
                return
            if n not in nodes:
                nodes[n] = {"name": n}

        def _add_link(src: str, dst: str, value: int = 1) -> None:
            s = src.strip()
            d = dst.strip()
            if not s or not d or s == d:
                return
            _add_node(s)
            _add_node(d)
            links.append({"source": s, "target": d, "value": int(value)})

        # canonical: {"nodes":[{name}], "links":[{source,target,value}]}
        if isinstance(chart_data, dict):
            raw_nodes = chart_data.get("nodes")
            raw_links = chart_data.get("links") or chart_data.get("edges")
            if isinstance(raw_nodes, list):
                for n in raw_nodes:
                    if isinstance(n, dict) and isinstance(n.get("name"), str):
                        _add_node(n["name"])
                    elif isinstance(n, str):
                        _add_node(n)
            if isinstance(raw_links, list):
                for e in raw_links:
                    if not isinstance(e, dict):
                        continue
                    src = e.get("source") or e.get("from") or e.get("src")
                    dst = e.get("target") or e.get("to") or e.get("dst")
                    if src is None or dst is None:
                        continue
                    v = self._parse_number_maybe(e.get("value"))
                    _add_link(str(src), str(dst), int(v) if v is not None else 1)

        # legacy/other: list[edge] or flow_chart_with_categories
        if isinstance(chart_data, list) and chart_data and not links:
            edge_like = True
            for e in chart_data[:5]:
                if not isinstance(e, dict):
                    edge_like = False
                    break
                if not (
                    ("source" in e and "target" in e)
                    or ("from" in e and "to" in e)
                    or ("src" in e and "dst" in e)
                ):
                    edge_like = False
                    break
            if edge_like:
                for e in chart_data:
                    if not isinstance(e, dict):
                        continue
                    src = e.get("source") or e.get("from") or e.get("src")
                    dst = e.get("target") or e.get("to") or e.get("dst")
                    if src is None or dst is None:
                        continue
                    v = self._parse_number_maybe(e.get("value"))
                    _add_link(str(src), str(dst), int(v) if v is not None else 1)

        if isinstance(chart_data, list) and chart_data and not links:
            for group in chart_data:
                if not isinstance(group, dict):
                    continue
                category = str(group.get("category") or "").strip()
                if category:
                    _add_node(category)
                steps = group.get("steps") or []
                if not isinstance(steps, list):
                    continue
                first_stage: str | None = None
                for st in steps:
                    if not isinstance(st, dict):
                        continue
                    stage = str(st.get("stage") or "").strip()
                    next_stage = str(st.get("next_stage") or "").strip()
                    if stage and first_stage is None:
                        first_stage = stage
                    if stage and next_stage:
                        _add_link(stage, next_stage, 1)
                if category and first_stage:
                    _add_link(category, first_stage, 1)

        if not nodes or not links:
            return []

        echarts_option = {
            "title": {"text": title, "left": "center"} if title else None,
            "tooltip": {"trigger": "item", "triggerOn": "mousemove"},
            "series": [
                {
                    "type": "sankey",
                    "layout": "none",
                    "data": list(nodes.values()),
                    "links": links,
                    "emphasis": {"focus": "adjacency"},
                    "lineStyle": {"color": "source", "curveness": 0.5},
                    "label": {"color": "#333", "fontSize": 12},
                }
            ],
        }
        echarts_option = {k: v for k, v in echarts_option.items() if v is not None}

        return [
            ChartConfig(
                chart_type="sankey",
                title=title,
                series=[],
                height=520,
                extra={"echarts_option": echarts_option},
            )
        ]

    def _chart_json_to_scatter_configs(self, *, title: str, obj: dict[str, Any]) -> list[ChartConfig]:
        cd = obj.get("chart_data")
        if not isinstance(cd, dict):
            return []
        pts_raw = cd.get("points")
        if not isinstance(pts_raw, list) or not pts_raw:
            return []
        data: list[list[float]] = []
        for it in pts_raw[:2000]:
            if not isinstance(it, dict):
                continue
            x = self._parse_number_maybe(it.get("x"))
            y = self._parse_number_maybe(it.get("y"))
            if x is None or y is None:
                continue
            data.append([x, y])
        if not data:
            return []
        opt = {
            "title": {"text": title, "left": "center"} if title else None,
            "tooltip": {"trigger": "item"},
            "xAxis": {"type": "value"},
            "yAxis": {"type": "value"},
            "series": [{"type": "scatter", "data": data, "symbolSize": 8}],
        }
        opt = {k: v for k, v in opt.items() if v is not None}
        return [
            ChartConfig(
                chart_type="scatter",
                title=title,
                series=[],
                height=460,
                extra={"echarts_option": opt},
            )
        ]

    def _chart_json_to_heatmap_configs(self, *, title: str, obj: dict[str, Any]) -> list[ChartConfig]:
        cd = obj.get("chart_data")
        if not isinstance(cd, dict):
            return []
        xs = cd.get("x_labels")
        ys = cd.get("y_labels")
        vals = cd.get("values")
        if not (isinstance(xs, list) and isinstance(ys, list) and isinstance(vals, list)):
            return []
        x_labels = [str(x).strip() for x in xs if str(x).strip()]
        y_labels = [str(y).strip() for y in ys if str(y).strip()]
        if not x_labels or not y_labels:
            return []
        xi = {x: i for i, x in enumerate(x_labels)}
        yi = {y: i for i, y in enumerate(y_labels)}
        data: list[list[Any]] = []
        for it in vals[:20000]:
            if not isinstance(it, dict):
                continue
            x = str(it.get("x") or "").strip()
            y = str(it.get("y") or "").strip()
            v = self._parse_number_maybe(it.get("value"))
            if x in xi and y in yi and v is not None:
                data.append([xi[x], yi[y], v])
        if not data:
            return []
        opt = {
            "title": {"text": title, "left": "center"} if title else None,
            "tooltip": {"position": "top"},
            "grid": {"top": 72 if title else 18, "left": 80, "right": 20, "bottom": 60},
            "xAxis": {"type": "category", "data": x_labels, "splitArea": {"show": True}},
            "yAxis": {"type": "category", "data": y_labels, "splitArea": {"show": True}},
            "visualMap": {
                "min": min([d[2] for d in data]),
                "max": max([d[2] for d in data]),
                "calculable": True,
                "orient": "horizontal",
                "left": "center",
                "bottom": 10,
            },
            "series": [{"type": "heatmap", "data": data, "label": {"show": False}}],
        }
        opt = {k: v for k, v in opt.items() if v is not None}
        return [
            ChartConfig(
                chart_type="heatmap",
                title=title,
                series=[],
                height=520,
                extra={"echarts_option": opt},
            )
        ]

    def _chart_json_to_radar_configs(self, *, title: str, obj: dict[str, Any]) -> list[ChartConfig]:
        cd = obj.get("chart_data")
        if not isinstance(cd, dict):
            return []
        indicators = cd.get("indicators")
        series = cd.get("series")
        if not (isinstance(indicators, list) and isinstance(series, list) and indicators):
            return []
        inds: list[dict[str, Any]] = []
        for it in indicators[:20]:
            if not isinstance(it, dict):
                continue
            name = str(it.get("name") or "").strip()
            if not name:
                continue
            mx = self._parse_number_maybe(it.get("max"))
            inds.append({"name": name, "max": mx} if mx is not None else {"name": name})
        if not inds:
            return []
        ser: list[dict[str, Any]] = []
        for s in series[:10]:
            if not isinstance(s, dict):
                continue
            name = str(s.get("name") or "系列").strip()
            vals = s.get("values")
            if not isinstance(vals, list):
                continue
            vv: list[float] = []
            for i in range(min(len(inds), len(vals))):
                x = self._parse_number_maybe(vals[i])
                vv.append(x if x is not None else 0.0)
            ser.append({"name": name, "value": vv})
        if not ser:
            return []
        opt = {
            "title": {"text": title, "left": "center"} if title else None,
            "tooltip": {},
            "legend": {"top": 44, "left": "center"} if len(ser) > 1 else None,
            "radar": {"indicator": inds},
            "series": [{"type": "radar", "data": ser}],
        }
        opt = {k: v for k, v in opt.items() if v is not None}
        return [
            ChartConfig(
                chart_type="radar",
                title=title,
                series=[],
                height=520,
                extra={"echarts_option": opt},
            )
        ]

    def _chart_json_to_configs(self, obj: dict[str, Any]) -> list[ChartConfig]:
        """将 chart_json 转为可渲染 ChartConfig（正式渲染入口）。

        设计目标：
        - chart_type 不是开放字符串：上游 prompt 已约束为受控集合，但仍兼容旧数据/别名；
        - 优先按 chart_type 分发到策略函数；类型缺失/异常时再做结构兜底推断；
        - 不做 mock：只使用 JSON 内现有字段进行转换。
        """
        if not obj or obj.get("is_chart") is not True:
            return []
        ctype_raw = obj.get("chart_type")
        ctype = self._normalize_chart_type_for_render(ctype_raw)
        desc = str(obj.get("description") or "").strip()
        chart_name = str(obj.get("_chart_name") or "").strip()
        title = desc or chart_name or "图表"

        # none/非图表：不渲染
        if ctype == "none":
            return []

        handlers = {
            "bar": self._chart_json_to_bar_configs,
            "line": self._chart_json_to_line_configs,
            "pie": self._chart_json_to_pie_configs,
            "stacked_area": self._chart_json_to_stacked_area_configs,
            "sankey": self._chart_json_to_sankey_configs,
            "scatter": self._chart_json_to_scatter_configs,
            "heatmap": self._chart_json_to_heatmap_configs,
            "radar": self._chart_json_to_radar_configs,
            "table": lambda **kwargs: [cfg]
            if (cfg := self._try_table_like_to_grouped_bar(title=title, obj=obj)) is not None
            else [],
        }

        fn = handlers.get(ctype)
        if fn is not None:
            cfgs = fn(title=title, obj=obj)
            if cfgs:
                return cfgs

        # 兜底：即便 chart_type=unknown，也尽量把“表格/对比”渲染成柱状图
        cfg2 = self._try_table_like_to_grouped_bar(title=title, obj=obj)
        if cfg2 is not None:
            return [cfg2]

        return []

    def _build_outline_skeleton_markdown(self, items: list[OutlineItem]) -> str:
        """将 outline items 生成 Markdown 骨架（用于约束 LLM 输出结构）。"""
        if not items:
            return ""
        out: list[str] = []
        for it in items:
            # 严格保持白皮书大纲的列表格式：缩进层级 + "- 1.1 标题"
            indent = "    " * max(0, it.depth - 1)
            out.append(f"{indent}- {it.display}".rstrip())
            # 给正文留出空间：正文应缩进到 list item 内（额外 4 个空格）
            out.append(f"{indent}    ")
        return "\n".join(out).rstrip()

    async def assemble_context_for_section(
        self,
        section: TemplateSection,
        collection_name: str = "default",
        options: HybridSearchOptions | None = None,
        *,
        coverage_score_threshold: float | None = None,
        kb_input_root: Path | None = None,
        max_auto_charts_per_section: int | None = None,
        on_event: Callable[[dict[str, Any]], Any] | None = None,
    ) -> tuple[
        str,
        list[dict[str, Any]],
        str | None,
        list[OutlineItem],
        list[str],
        list[ChartConfig],
        dict[str, str],
    ]:
        """为指定 section 组装 RAG 检索上下文，并返回覆盖评估。

        返回：(context, coverage, template_skeleton, outline_items, sources)
        - template_skeleton：当 section 符合白皮书大纲格式时返回骨架 Markdown，否则为 None
        """
        options = options or HybridSearchOptions(
            top_k=self.DEFAULT_TOP_K,
            use_rerank=True,
            rerank_top_n=self.DEFAULT_RERANK_TOP,
        )

        coverage: list[dict[str, Any]] = []
        outline_items: list[OutlineItem] = []
        template_skeleton: str | None = None
        sources_set: set[str] = set()
        chart_configs: list[ChartConfig] = []
        seen_chart_json: set[str] = set()
        max_auto = max_auto_charts_per_section or self.DEFAULT_MAX_AUTO_CHARTS_PER_SECTION
        item_context_map: dict[str, str] = {}

        # 生成阶段只关注“生成 HTML”：
        # - 有知识库就检索（全量基于 collection，不做单文档限制）
        # - 没有知识库就当作无召回，让 LLM 保守填充
        # 不依赖“建库流程”的中间产物来决定是否生成。

        # 自动图表渲染需要能定位到 chart_json：应由调用方显式传入 kb_input_root
        # 说明：不在这里做“隐式默认路径”硬编码。
        # - 正式链路应由调用方（API/任务编排）显式传入 kb_input_root
        # - 若未传入，则自动图表渲染会自然降级为“不渲染”（仍可正常生成文本内容）

        outline_items = self._parse_outline_items_from_section_content(section.content)
        if outline_items:
            template_skeleton = self._build_outline_skeleton_markdown(outline_items)
            log.info(
                "content_generation.section_outline_parsed",
                extra=log_extra(
                    section_id=section.section_id,
                    section_title=section.title,
                    outline_items=len(outline_items),
                    collection_name=collection_name,
                ),
            )

            def _is_bad_title(t: str) -> bool:
                s = (t or "").strip()
                if not s:
                    return True
                if _UUID_RE.match(s):
                    return True
                if re.fullmatch(r"\d{4}", s) or re.fullmatch(r"\d+", s):
                    return True
                if s in {"证券研究报告", "研究报告", "年度报告", "报告"}:
                    return True
                if len(s) < 4:
                    return True
                return False

            async def _search_one(item: OutlineItem):
                # 查询文本：尽量避免把编号（8.2.1）当成检索噪声
                base = f"{section.title} {item.title}".strip()
                if section.title.strip() == "附录":
                    base = item.title.strip() or item.display.strip()
                # 针对“名词解释”类条目，补充检索意图词，提升命中概率
                if "名词解释" in base:
                    base = f"{base} 术语 定义".strip()
                q = base
                try:
                    resp = await self.hybrid_search_service.search(
                        query=q,
                        collection_name=collection_name,
                        options=options,
                    )
                    return item, q, resp
                except Exception as exc:  # noqa: BLE001
                    return item, q, exc

            # 控制并发，避免对向量库/重排模型造成冲击
            sem = asyncio.Semaphore(6)

            async def _guarded(it: OutlineItem):
                async with sem:
                    return await _search_one(it)

            # 用 as_completed 便于输出“进行到哪一步”，避免看起来像卡住
            tasks = [asyncio.create_task(_guarded(it)) for it in outline_items]
            results: list[tuple[OutlineItem, str, Any]] = []
            done = 0
            total = len(tasks)
            for fut in asyncio.as_completed(tasks):
                results.append(await fut)
                done += 1
                if done == total or (done % 10) == 0:
                    log.info(
                        "content_generation.section_recall_progress",
                        extra=log_extra(
                            section_id=section.section_id,
                            section_title=section.title,
                            done=done,
                            total=total,
                            collection_name=collection_name,
                        ),
                    )
                    await self._emit(
                        on_event,
                        {
                            "type": "section_recall_progress",
                            "section_id": section.section_id,
                            "section_title": section.title,
                            "done": done,
                            "total": total,
                        },
                    )

            # 组装上下文（按条目分组，去重 chunk）
            seen_chunk_ids: set[str] = set()
            context_parts: list[str] = []
            for item, q, resp in results:
                if isinstance(resp, Exception):
                    coverage.append(
                        {
                            "type": "outline_item",
                            "item": item.display,
                            "depth": item.depth,
                            "query": q,
                            "hits": 0,
                            "max_score": None,
                            "uncovered": True,
                            "error": {"type": resp.__class__.__name__, "message": str(resp)},
                        }
                    )
                    item_context_map[item.display] = ""
                    continue

                hits = resp.results or []
                try:
                    max_score = max((r.score for r in hits), default=None)
                except Exception:
                    max_score = None
                uncovered = not bool(hits)
                if coverage_score_threshold is not None and max_score is not None:
                    if float(max_score) < float(coverage_score_threshold):
                        uncovered = True

                coverage.append(
                    {
                        "type": "outline_item",
                        "item": item.display,
                        "depth": item.depth,
                        "query": q,
                        "hits": len(hits),
                        "max_score": max_score,
                        "uncovered": uncovered,
                        "bm25_index_used": getattr(resp.metrics, "bm25_index_used", None),
                        "rerank_applied": getattr(resp.metrics, "rerank_applied", None),
                    }
                )

                # 上下文按条目附上少量 top 结果
                per_top = min(self.DEFAULT_PER_QUERY_TOP_N, len(hits))
                if per_top <= 0:
                    item_context_map[item.display] = ""
                    continue
                block_lines = [f"###{'#' * max(0, item.depth - 1)} {item.display}".strip()]

                # 尽量让每个条目覆盖不同文档，避免“只看到一篇来源”
                picked: list[SearchResult] = []
                seen_docs: set[str] = set()
                for r in hits:
                    doc_rel = r.metadata.get("doc_rel_path") if isinstance(r.metadata, dict) else None
                    src_id = r.metadata.get("source_file_id") if isinstance(r.metadata, dict) else None
                    key = str(doc_rel or src_id or r.chunk_id or "").strip()
                    if key and key in seen_docs:
                        continue
                    if key:
                        seen_docs.add(key)
                    picked.append(r)
                    if len(picked) >= per_top:
                        break

                for r in picked:
                    if r.chunk_id in seen_chunk_ids:
                        continue
                    seen_chunk_ids.add(r.chunk_id)

                    doc_title = r.metadata.get("doc_title") if isinstance(r.metadata, dict) else None
                    doc_rel = r.metadata.get("doc_rel_path") if isinstance(r.metadata, dict) else None
                    ofn = r.metadata.get("original_filename") if isinstance(r.metadata, dict) else None

                    dt_key = str(doc_title or "").strip()
                    ofn_key = str(ofn or "").strip()
                    dr_key = str(doc_rel or "").strip()

                    display = ""
                    if not _is_bad_title(dt_key):
                        display = dt_key
                    elif ofn_key:
                        display = Path(ofn_key).stem
                    elif dr_key:
                        display = Path(dr_key).name
                    display = re.sub(r"\s+", " ", (display or "").strip())

                    header = f"【来源】{display}" if display else "【来源】"
                    text = (r.content or "").strip()
                    if len(text) > 1200:
                        text = text[:1200] + "…"
                    block_lines.append(header)
                    block_lines.append(text)
                    block_lines.append("")

                    if display:
                        sources_set.add(display)

                    # 自动图表：从命中 chunk 中识别 images/<stem> 并定位 chart_json/<stem>.json
                    if kb_input_root is not None and doc_rel and len(chart_configs) < max_auto:
                        for mimg in self._MD_IMAGE_RE.finditer(r.content or ""):
                            raw_path = (mimg.group(1) or "").strip().strip('"').strip("'")
                            raw_path = raw_path.replace("\\", "/").split("#", 1)[0].split("?", 1)[0]
                            if "images/" not in raw_path:
                                continue
                            stem = Path(raw_path).stem
                            if not stem:
                                continue
                            cj = (kb_input_root / str(doc_rel) / "chart_json" / f"{stem}.json").resolve()
                            key = str(cj).replace("\\", "/")
                            if key in seen_chart_json:
                                continue
                            seen_chart_json.add(key)
                            if not cj.exists():
                                continue
                            try:
                                obj = json.loads(_read_text_best_effort(cj))
                            except Exception:
                                continue
                            if not isinstance(obj, dict) or obj.get("is_chart") is not True:
                                continue
                            cfgs = self._chart_json_to_configs(obj)
                            if cfgs:
                                cn = str(obj.get("_chart_name") or "").strip()
                                if cn:
                                    sources_set.add(f"图表来源: {cn}")
                                for cfg in cfgs:
                                    if len(chart_configs) >= max_auto:
                                        break
                                    chart_configs.append(cfg)

                block = "\n".join(block_lines).strip()
                item_context_map[item.display] = block
                context_parts.append(block)

            context = "\n\n".join([p for p in context_parts if p]).strip()
            if len(context) > self.DEFAULT_MAX_CONTEXT_CHARS:
                context = context[: self.DEFAULT_MAX_CONTEXT_CHARS]

            uncovered_count = sum(
                1 for c in coverage if c.get("type") == "outline_item" and c.get("uncovered") is True
            )
            log.info(
                "content_generation.section_context_ready",
                extra=log_extra(
                    section_id=section.section_id,
                    section_title=section.title,
                    context_chars=len(context),
                    coverage_items=len(coverage),
                    uncovered_items=uncovered_count,
                ),
            )
            await self._emit(
                on_event,
                {
                    "type": "section_context_ready",
                    "section_id": section.section_id,
                    "section_title": section.title,
                    "context_chars": len(context),
                    "coverage_items": len(coverage),
                },
            )
            if chart_configs:
                coverage.append({"type": "auto_charts", "count": len(chart_configs)})

            return (
                context,
                coverage,
                template_skeleton,
                outline_items,
                sorted(sources_set),
                chart_configs,
                item_context_map,
            )

        # 无大纲条目：仍尝试用章节标题做一次检索；无知识库时会降级为空上下文
        try:
            resp = await self.hybrid_search_service.search(
                query=section.title,
                collection_name=collection_name,
                options=options,
            )
            hits = resp.results or []
        except Exception:
            hits = []
        context = "\n\n".join([(h.content or "").strip() for h in hits[: min(3, len(hits))]]).strip()
        if len(context) > self.DEFAULT_MAX_CONTEXT_CHARS:
            context = context[: self.DEFAULT_MAX_CONTEXT_CHARS]
        return context, coverage, None, [], sorted(sources_set), [], {}

    def generate_section_content(
        self,
        section: TemplateSection,
        context: str,
        chart_configs: dict[str, ChartConfig],
        collection_name: str = "default",
        *,
        template_content_override: str | None = None,
        coverage: list[dict[str, Any]] | None = None,
        outline_items: list[OutlineItem] | None = None,
        document_title: str | None = None,
        sources: list[str] | None = None,
        auto_chart_configs: list[ChartConfig] | None = None,
    ) -> SectionGenerationResult:
        """生成单个 section 的内容。"""
        import time

        start_time = time.time()

        # 替换文本占位符
        generated_content = template_content_override or section.content
        tokens_used = 0

        # 准备上下文
        truncated_context = context[: self.DEFAULT_MAX_CONTEXT_CHARS] if len(context) > self.DEFAULT_MAX_CONTEXT_CHARS else context

        # 调用 LLM 生成内容
        try:
            runnable = self.llm_runtime_service.build_runnable_for_callsite(
                SCOPE_CONTENT_GENERATION_SECTION
            )
            result = runnable.invoke(
                {
                    "title": section.title,
                    "template_content": template_content_override or section.content,
                    "context": truncated_context,
                    "document_title": document_title or "",
                    "outline_items": "\n".join([it.display for it in (outline_items or [])]).strip(),
                }
            )
            generated_content_raw = result if result else (template_content_override or section.content)
            generated_content = _strip_model_think(str(generated_content_raw))
            generated_content = _strip_leading_title_heading(generated_content, title=section.title)
            # 附录名词解释：避免占位符模板内容污染最终产物（无召回时宁可留空）
            if section.title.strip() == "附录" and "名词解释" in generated_content:
                if self._GLOSSARY_PLACEHOLDER_RE.search(generated_content):
                    lines = generated_content.replace("\r\n", "\n").split("\n")
                    out: list[str] = []
                    in_glossary = False
                    for line in lines:
                        if re.match(r"^#{1,6}\s*C\.\s*名词解释\s*$", line.strip()):
                            in_glossary = True
                            out.append(line.rstrip())
                            continue
                        if in_glossary:
                            # 直接丢弃后续占位符内容
                            continue
                        out.append(line.rstrip())
                    generated_content = "\n".join([x for x in out]).strip()
            # 估算 token 使用量
            tokens_used = len(generated_content) // 4
        except Exception as e:
            raise ContentGenerationError(
                message=f"LLM 调用失败: {str(e)}",
                code="llm_invocation_failed",
                details={"section_id": section.section_id},
            )

        # 兜底：如果要求“严格按骨架”但模型未完全遵守，补齐缺失标题，避免真实产物缺章
        if template_content_override:
            required_items: list[str] = []
            for line in template_content_override.replace("\r\n", "\n").split("\n"):
                s = line.strip()
                if s.startswith("- "):
                    # list item 行（保留原样）
                    required_items.append(line.rstrip("\n"))
            # 说明：
            # - 当 RAG 上下文为空时，不做“补齐标题”追加，避免把骨架又列一遍（用户体验更差）。
            # - 当模型把 list item 输出成 heading/plain line 时，视作“已覆盖”，不再追加。
            def _has_item(item_line: str) -> bool:
                plain = (item_line or "").strip()
                if plain.startswith("- "):
                    plain = plain[2:].strip()
                if not plain:
                    return False
                # 允许：- item / ## item / item
                pat = re.compile(rf"(?m)^(?:[-*]\s+|#{1,6}\s+)?{re.escape(plain)}\s*$")
                return pat.search(generated_content) is not None

            missing = [h for h in required_items if h and not _has_item(h)]
            if missing:
                log.warning(
                    "content_generation.section_missing_headings",
                    extra=log_extra(
                        section_id=section.section_id,
                        section_title=section.title,
                        missing_count=len(missing),
                    ),
                )
                if truncated_context.strip():
                    # 仅在确实有召回上下文时才追加缺失条目，避免“无召回时把骨架又列一遍”
                    parts = [generated_content.rstrip(), ""]
                    for h in missing:
                        parts.append(h)
                        parts.append("")
                    generated_content = "\n".join(parts).strip()

        # 渲染图表
        rendered_charts: dict[str, ChartTemplateSnippet] = {}

        # 自动图表：从检索命中反查 chart_json 并渲染（不依赖模板占位符）
        for idx, cfg in enumerate(auto_chart_configs or []):
            try:
                snippet = self.chart_renderer_service.render_template_snippet(
                    cfg,
                    library="echarts",
                )
                rendered_charts[f"auto_chart_{idx+1}"] = snippet
            except Exception:
                continue

        # 来源（由检索阶段确定性产出；不依赖 LLM 是否引用）
        sources = sources or []

        generation_time_ms = (time.time() - start_time) * 1000

        return SectionGenerationResult(
            section_id=section.section_id,
            title=section.title,
            content=generated_content,
            rendered_charts=rendered_charts,
            sources=sources,
            coverage=coverage or [],
            tokens_used=tokens_used,
            generation_time_ms=generation_time_ms,
        )

    async def generate_section_content_async(
        self,
        *,
        section: TemplateSection,
        context: str,
        chart_configs: dict[str, ChartConfig],
        collection_name: str = "default",
        template_content_override: str | None = None,
        coverage: list[dict[str, Any]] | None = None,
        outline_items: list[OutlineItem] | None = None,
        document_title: str | None = None,
        sources: list[str] | None = None,
        auto_chart_configs: list[ChartConfig] | None = None,
        item_context_map: dict[str, str] | None = None,
        on_event: Callable[[dict[str, Any]], Any] | None = None,
        stream_tokens: bool = False,
    ) -> SectionGenerationResult:
        """生成单个 section（支持 token 级别流式输出）。

        约定：
        - stream_tokens=True：通过 runnable.astream() 获取增量输出，并用 on_event 推送 token 事件（SSE）
        - stream_tokens=False：仍使用一次性 invoke（放线程池避免阻塞 event loop）
        """
        import time

        start_time = time.time()
        generated_content = template_content_override or section.content
        tokens_used = 0

        truncated_context = (
            context[: self.DEFAULT_MAX_CONTEXT_CHARS]
            if len(context) > self.DEFAULT_MAX_CONTEXT_CHARS
            else context
        )

        payload = {
            "title": section.title,
            "template_content": template_content_override or section.content,
            "context": truncated_context,
            "document_title": document_title or "",
            "outline_items": "\n".join([it.display for it in (outline_items or [])]).strip(),
        }

        try:
            runnable = self.llm_runtime_service.build_runnable_for_callsite(
                SCOPE_CONTENT_GENERATION_SECTION,
                force_streaming=bool(stream_tokens),
            )

            if stream_tokens:
                parts: list[str] = []
                if on_event is not None:
                    await self._emit(
                        on_event,
                        {
                            "type": "llm_stream_start",
                            "section_id": section.section_id,
                            "section_title": section.title,
                        },
                    )
                async for chunk in runnable.astream(payload):
                    s = "" if chunk is None else str(chunk)
                    if not s:
                        continue
                    parts.append(s)
                    if on_event is not None:
                        await self._emit(
                            on_event,
                            {
                                "type": "token",
                                "section_id": section.section_id,
                                "section_title": section.title,
                                "content": s,
                            },
                        )
                if on_event is not None:
                    await self._emit(
                        on_event,
                        {
                            "type": "llm_stream_done",
                            "section_id": section.section_id,
                            "section_title": section.title,
                        },
                    )
                generated_content_raw = "".join(parts)
            else:
                loop = asyncio.get_running_loop()

                def _do():
                    return runnable.invoke(payload)

                generated_content_raw = await loop.run_in_executor(None, _do)

            generated_content = _strip_model_think(str(generated_content_raw or ""))
            generated_content = _strip_leading_title_heading(generated_content, title=section.title)

            # 附录名词解释：避免占位符模板内容污染最终产物（无召回时宁可留空）
            if section.title.strip() == "附录" and "名词解释" in generated_content:
                if self._GLOSSARY_PLACEHOLDER_RE.search(generated_content):
                    lines = generated_content.replace("\r\n", "\n").split("\n")
                    out: list[str] = []
                    in_glossary = False
                    for line in lines:
                        if re.match(r"^#{1,6}\s*C\.\s*名词解释\s*$", line.strip()):
                            in_glossary = True
                            out.append(line.rstrip())
                            continue
                        if in_glossary:
                            continue
                        out.append(line.rstrip())
                    generated_content = "\n".join([x for x in out]).strip()

            tokens_used = len(generated_content) // 4
        except Exception as e:
            raise ContentGenerationError(
                message=f"LLM 调用失败: {str(e)}",
                code="llm_invocation_failed",
                details={"section_id": section.section_id},
            )

        # 兜底：如果要求“严格按骨架”但模型未完全遵守，补齐缺失标题，避免真实产物缺章
        if template_content_override:
            required_items: list[str] = []
            for line in template_content_override.replace("\r\n", "\n").split("\n"):
                s = line.strip()
                if s.startswith("- "):
                    required_items.append(line.rstrip("\n"))

            def _has_item(item_line: str) -> bool:
                plain = (item_line or "").strip()
                if plain.startswith("- "):
                    plain = plain[2:].strip()
                if not plain:
                    return False
                pat = re.compile(rf"(?m)^(?:[-*]\s+|#{1,6}\s+)?{re.escape(plain)}\s*$")
                return pat.search(generated_content) is not None

            missing = [h for h in required_items if h and not _has_item(h)]
            if missing:
                log.warning(
                    "content_generation.section_missing_headings",
                    extra=log_extra(
                        section_id=section.section_id,
                        section_title=section.title,
                        missing_count=len(missing),
                    ),
                )
                if truncated_context.strip():
                    parts = [generated_content.rstrip(), ""]
                    for h in missing:
                        parts.append(h)
                        parts.append("")
                    generated_content = "\n".join(parts).strip()

        # 渲染图表（auto）
        rendered_charts: dict[str, ChartTemplateSnippet] = {}
        for idx, cfg in enumerate(auto_chart_configs or []):
            try:
                snippet = self.chart_renderer_service.render_template_snippet(
                    cfg,
                    library="echarts",
                )
                rendered_charts[f"auto_chart_{idx+1}"] = snippet
            except Exception:
                continue

        generation_time_ms = (time.time() - start_time) * 1000
        return SectionGenerationResult(
            section_id=section.section_id,
            title=section.title,
            content=generated_content,
            rendered_charts=rendered_charts,
            sources=sources or [],
            coverage=coverage or [],
            tokens_used=tokens_used,
            generation_time_ms=generation_time_ms,
        )

    async def generate_content(
        self,
        template_id: str,
        collection_name: str = "default",
        chart_configs: dict[str, ChartConfig] | None = None,
        search_options: HybridSearchOptions | None = None,
        *,
        template_content_override: str | None = None,
        document_title: str | None = None,
        coverage_score_threshold: float | None = None,
        kb_input_root: Path | None = None,
        max_auto_charts_per_section: int | None = None,
        on_event: Callable[[dict[str, Any]], Any] | None = None,
        stream_tokens: bool = False,
    ) -> ContentGenerationResult:
        """生成完整内容。"""
        import time

        start_time = time.time()

        # 获取模板
        template = self.template_service.get_template(template_id)
        if template is None:
            raise ContentGenerationError(
                message="模板不存在",
                code="template_not_found",
                details={"template_id": template_id},
            )

        # 预处理校验
        preprocess_result = self.template_service.preprocess_template(template_id)
        if preprocess_result["overall_status"] == "error":
            raise ContentGenerationError(
                message=f"模板预处理失败: {preprocess_result['message']}",
                code="template_preprocess_failed",
                details=preprocess_result,
            )

        # 解析 sections（允许使用内存中的大纲文本覆盖文件内容）
        sections = self.parse_template_sections(template, content_override=template_content_override)

        # 生成每个 section
        section_results: list[SectionGenerationResult] = []
        total_tokens = 0

        chart_configs = chart_configs or {}

        total_sections = len(sections)
        for idx, section in enumerate(sections, start=1):
            log.info(
                "content_generation.section_start",
                extra=log_extra(
                    section_index=idx,
                    section_total=total_sections,
                    section_id=section.section_id,
                    section_title=section.title,
                    collection_name=collection_name,
                ),
            )
            await self._emit(
                on_event,
                {
                    "type": "section_start",
                    "section_index": idx,
                    "section_total": total_sections,
                    "section_id": section.section_id,
                    "section_title": section.title,
                },
            )
            # 检索上下文
            (
                context,
                coverage,
                skeleton,
                outline_items,
                sources,
                auto_charts,
                item_context_map,
            ) = await self.assemble_context_for_section(
                section,
                collection_name,
                search_options,
                coverage_score_threshold=coverage_score_threshold,
                kb_input_root=kb_input_root,
                max_auto_charts_per_section=max_auto_charts_per_section,
                on_event=on_event,
            )

            # 生成内容（可选 token 级别流式）
            result = await self.generate_section_content_async(
                section=section,
                context=context,
                chart_configs=chart_configs,
                collection_name=collection_name,
                template_content_override=skeleton,
                coverage=coverage,
                outline_items=outline_items,
                document_title=document_title,
                sources=sources,
                auto_chart_configs=auto_charts,
                item_context_map=item_context_map,
                on_event=on_event,
                stream_tokens=bool(stream_tokens),
            )
            section_results.append(result)
            total_tokens += result.tokens_used
            log.info(
                "content_generation.section_done",
                extra=log_extra(
                    section_index=idx,
                    section_total=total_sections,
                    section_id=result.section_id,
                    section_title=result.title,
                    tokens_used=result.tokens_used,
                    generation_time_ms=result.generation_time_ms,
                ),
            )
            await self._emit(
                on_event,
                {
                    "type": "section_done",
                    "section_index": idx,
                    "section_total": total_sections,
                    "section_id": result.section_id,
                    "section_title": result.title,
                    "tokens_used": result.tokens_used,
                    "generation_time_ms": result.generation_time_ms,
                },
            )

        # 渲染最终 HTML
        html_content = WhitepaperHtmlRenderer().render(
            template=template,
            section_results=section_results,
            document_title=document_title,
        )

        total_time_ms = (time.time() - start_time) * 1000

        return ContentGenerationResult(
            template_id=template_id,
            template_name=template.original_filename,
            html_content=html_content,
            sections=section_results,
            total_tokens=total_tokens,
            total_time_ms=total_time_ms,
            generated_at=datetime.now(),
            document_title=document_title,
        )

    def polish_outline(self, outline: str) -> str:
        """对用户自定义大纲进行 LLM 润色。

        Args:
            outline: 原始大纲（Markdown 格式）

        Returns:
            润色后的大纲（Markdown 格式）

        Raises:
            ContentGenerationError: 当 LLM 调用失败时
        """
        if not outline or not outline.strip():
            raise ContentGenerationError(
                message="大纲内容不能为空",
                code="outline_empty",
                status_code=400,
            )

        try:
            runnable = self.llm_runtime_service.build_runnable_for_callsite(
                SCOPE_OUTLINE_POLISH
            )
            result = runnable.invoke({"outline": outline})
            polished_outline_raw = result if result else outline
            polished_outline = _strip_model_think(str(polished_outline_raw))
            return polished_outline.strip()
        except Exception as e:
            raise ContentGenerationError(
                message=f"LLM 润色失败: {str(e)}",
                code="outline_polish_failed",
                details={"original_outline": outline[:200]},  # 只记录前200字符
            )
