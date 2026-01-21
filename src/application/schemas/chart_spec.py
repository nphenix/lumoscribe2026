"""图表规格 Pydantic 模型。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ChartDataPoint(BaseModel):
    """图表数据点。"""

    x: str | int | float | None = Field(None, description="X轴值")
    y: str | int | float | None = Field(None, description="Y轴值")
    name: str | None = Field(None, description="数据点名称")
    extra: dict[str, Any] | None = Field(None, description="额外数据")


class ChartSeries(BaseModel):
    """图表系列。"""

    name: str = Field(..., description="系列名称")
    data: list[ChartDataPoint] = Field(default_factory=list, description="数据点列表")
    chart_type: str = Field(default="line", description="图表类型: line, bar, pie, scatter")
    color: str | None = Field(None, description="系列颜色")
    extra: dict[str, Any] | None = Field(None, description="额外配置")


class ChartConfig(BaseModel):
    """图表配置。"""

    chart_type: str = Field(..., description="图表类型: bar, line, pie, scatter, radar, heatmap")
    title: str | None = Field(None, description="图表标题")
    subtitle: str | None = Field(None, description="副标题")
    x_axis_title: str | None = Field(None, description="X轴标题")
    y_axis_title: str | None = Field(None, description="Y轴标题")
    series: list[ChartSeries] = Field(default_factory=list, description="数据系列")
    legend_position: str = Field(default="bottom", description="图例位置: top, bottom, left, right")
    tooltip_enabled: bool = Field(default=True, description="是否启用提示框")
    grid_enabled: bool = Field(default=True, description="是否显示网格")
    animation_enabled: bool = Field(default=True, description="是否启用动画")
    colors: list[str] | None = Field(None, description="配色方案")
    width: int | None = Field(None, description="图表宽度")
    height: int | None = Field(None, description="图表高度")
    theme: str = Field(default="default", description="图表主题")
    extra: dict[str, Any] | None = Field(None, description="额外配置")


class ChartRenderRequest(BaseModel):
    """图表渲染请求。"""

    chart_config: ChartConfig = Field(..., description="图表配置")
    output_format: Literal["svg", "png", "html", "base64"] = Field(
        default="html", description="输出格式"
    )
    library: Literal["echarts", "chartjs", "d3"] = Field(
        default="echarts", description="渲染库"
    )


class ChartRenderResponse(BaseModel):
    """图表渲染响应。"""

    chart_id: str = Field(..., description="图表ID")
    output_format: str = Field(..., description="输出格式")
    content: str = Field(..., description="渲染结果")
    content_type: str = Field(..., description="内容类型")
    width: int = Field(..., description="宽度")
    height: int = Field(..., description="高度")
    rendered_at: datetime = Field(default_factory=datetime.now, description="渲染时间")


class ChartTemplateSnippet(BaseModel):
    """图表模板片段（用于嵌入HTML）。"""

    chart_id: str = Field(..., description="图表ID")
    container_html: str = Field(..., description="容器HTML")
    script_html: str = Field(..., description="脚本HTML")
    styles: str = Field(default="", description="样式")
    dependencies: list[str] = Field(default_factory=list, description="依赖库")


class ChartLibraryInfo(BaseModel):
    """图表库信息。"""

    library: str = Field(..., description="库名称")
    version: str = Field(..., description="版本号")
    supported_types: list[str] = Field(default_factory=list, description="支持的图表类型")
    features: list[str] = Field(default_factory=list, description="特性列表")
