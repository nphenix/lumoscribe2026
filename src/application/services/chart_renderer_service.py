"""图表渲染服务（T043）。

将图表 JSON 配置渲染为 SVG、PNG 或 HTML snippet。
支持 ECharts、Chart.js、D3.js 等主流图表库。
"""

from __future__ import annotations

import base64
import hashlib
import json
import uuid
from io import BytesIO
from typing import Any, Literal

from src.application.schemas.chart_spec import (
    ChartConfig,
    ChartRenderRequest,
    ChartRenderResponse,
    ChartSeries,
    ChartTemplateSnippet,
)
from src.shared.errors import AppError


class ChartRenderError(AppError):
    """图表渲染错误。"""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        code: str = "chart_render_error",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(
            message=message,
            status_code=status_code,
            code=code,
            details=details,
        )


class ChartRendererService:
    """图表渲染服务。"""

    # 默认配色方案
    DEFAULT_COLORS = [
        "#5470c6", "#91cc75", "#fac858", "#ee6666", "#73c0de",
        "#3ba272", "#fc8452", "#9a60b4", "#ea7ccc", "#5470c6",
    ]

    # ECharts CDN
    ECHARTS_CDN = "https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"

    # Chart.js CDN
    CHARTJS_CDN = "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"

    # D3.js CDN
    D3_CDN = "https://cdn.jsdelivr.net/npm/d3@7.8.5/dist/d3.min.js"

    def __init__(self):
        self._chart_cache: dict[str, dict] = {}

    def _generate_chart_id(self, config: ChartConfig) -> str:
        """生成图表ID。"""
        content = json.dumps(config.model_dump(), sort_keys=True)
        hash_val = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"chart-{uuid.uuid4().hex[:8]}-{hash_val}"

    def _get_default_colors(self, count: int) -> list[str]:
        """获取默认配色。"""
        return self.DEFAULT_COLORS[:count] if count <= len(self.DEFAULT_COLORS) else self.DEFAULT_COLORS

    def _convert_series_for_echarts(self, series: list[ChartSeries]) -> list[dict[str, Any]]:
        """转换系列数据为 ECharts 格式。"""
        result = []
        for s in series:
            result.append({
                "name": s.name,
                "type": s.chart_type,
                "data": [
                    {"value": [dp.x, dp.y], "name": dp.name, **(dp.extra or {})}
                    if dp.x is not None and dp.y is not None
                    else {"value": dp.y, "name": dp.name, **(dp.extra or {})}
                    for dp in s.data
                ],
                "itemStyle": {"color": s.color} if s.color else None,
                **(s.extra or {}),
            })
        return result

    def _convert_series_for_chartjs(self, series: list[ChartSeries]) -> list[dict[str, Any]]:
        """转换系列数据为 Chart.js 格式。"""
        result = []
        colors = self._get_default_colors(len(series))
        for idx, s in enumerate(series):
            result.append({
                "label": s.name,
                "data": [dp.y for dp in s.data],
                "backgroundColor": s.color or colors[idx % len(colors)],
                "borderColor": s.color or colors[idx % len(colors)],
                "borderWidth": 1,
                "fill": s.chart_type == "line",
                **(s.extra or {}),
            })
        return result

    def render_echarts(
        self,
        config: ChartConfig,
        output_format: Literal["svg", "png", "html", "base64"] = "html",
    ) -> ChartRenderResponse:
        """使用 ECharts 渲染图表。"""
        chart_id = self._generate_chart_id(config)

        width = config.width or 800
        height = config.height or 400
        colors = config.colors or self._get_default_colors(len(config.series))

        # 构建 ECharts 配置
        series_data = self._convert_series_for_echarts(config.series)

        echarts_config = {
            "title": {
                "text": config.title,
                "subtext": config.subtitle,
                "left": "center",
            }
            if config.title
            else None,
            "tooltip": {
                "trigger": "item",
                "enabled": config.tooltip_enabled,
            }
            if config.tooltip_enabled
            else None,
            "legend": {
                "orient": config.legend_position,
                "left": "center" if config.legend_position in ("top", "bottom") else "left",
                "top" if config.legend_position in ("top", "bottom") else "left": "5%",
            }
            if config.series else None,
            "grid": {
                "left": "3%",
                "right": "4%",
                "bottom": "3%",
                "containLabel": True,
            }
            if config.grid_enabled else None,
            "xAxis": {
                "type": "category",
                "name": config.x_axis_title,
                "data": [dp.x for dp in config.series[0].data] if config.series and config.series[0].data else [],
                "axisLabel": {"rotate": 45} if config.series and len(config.series[0].data) > 10 else None,
            }
            if config.chart_type not in ("pie", "donut") and config.series else None,
            "yAxis": {
                "type": "value",
                "name": config.y_axis_title,
            }
            if config.chart_type not in ("pie", "donut") and config.series else None,
            "series": series_data,
            "color": colors,
            "animation": config.animation_enabled,
        }

        # 清理空值
        echarts_config = {k: v for k, v in echarts_config.items() if v is not None}

        if output_format in ("svg", "png"):
            raise ChartRenderError(
                message=f"ECharts 原生不支持 {output_format} 格式输出，请使用 html 或 base64",
                code="unsupported_format",
            )

        if output_format == "base64":
            html_content = self._build_echarts_html(chart_id, echarts_config, width, height)
            import base64 as b64

            return ChartRenderResponse(
                chart_id=chart_id,
                output_format=output_format,
                content=b64.b64encode(html_content.encode()).decode(),
                content_type="image/svg+xml" if output_format == "svg" else "text/html",
                width=width,
                height=height,
            )

        # 返回 HTML
        html_content = self._build_echarts_html(chart_id, echarts_config, width, height)

        return ChartRenderResponse(
            chart_id=chart_id,
            output_format=output_format,
            content=html_content,
            content_type="text/html",
            width=width,
            height=height,
        )

    def _build_echarts_html(
        self,
        chart_id: str,
        config: dict[str, Any],
        width: int,
        height: int,
    ) -> str:
        """构建 ECharts HTML。"""
        config_json = json.dumps(config, ensure_ascii=False)

        return f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <script src="{self.ECHARTS_CDN}"></script>
</head>
<body>
    <div id="{chart_id}" style="width: {width}px; height: {height}px;"></div>
    <script>
        var chartDom = document.getElementById('{chart_id}');
        var myChart = echarts.init(chartDom);
        var option = {config_json};
        myChart.setOption(option);
    </script>
</body>
</html>'''

    def render_chartjs(
        self,
        config: ChartConfig,
        output_format: Literal["svg", "png", "html", "base64"] = "html",
    ) -> ChartRenderResponse:
        """使用 Chart.js 渲染图表。"""
        chart_id = self._generate_chart_id(config)

        width = config.width or 800
        height = config.height or 400
        colors = config.colors or self._get_default_colors(len(config.series))

        # 构建 Chart.js 配置
        series_data = self._convert_series_for_chartjs(config.series)

        labels = [dp.x for dp in config.series[0].data] if config.series and config.series[0].data else []

        chart_type_map = {
            "line": "line",
            "bar": "bar",
            "pie": "pie",
            "scatter": "scatter",
        }
        js_chart_type = chart_type_map.get(config.chart_type, "bar")

        chartjs_config = {
            "type": js_chart_type,
            "data": {
                "labels": labels,
                "datasets": series_data,
            },
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {
                    "title": {
                        "display": bool(config.title),
                        "text": config.title,
                    },
                    "legend": {
                        "position": config.legend_position,
                    },
                },
                "scales": {
                    "x": {"title": {"display": bool(config.x_axis_title), "text": config.x_axis_title}}
                    if config.x_axis_title and config.chart_type not in ("pie", "donut")
                    else None,
                    "y": {"title": {"display": bool(config.y_axis_title), "text": config.y_axis_title}}
                    if config.y_axis_title and config.chart_type not in ("pie", "donut")
                    else None,
                },
            },
        }

        config_json = json.dumps(chartjs_config, ensure_ascii=False)

        html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <script src="{self.CHARTJS_CDN}"></script>
</head>
<body>
    <div style="width: {width}px; height: {height}px;">
        <canvas id="{chart_id}"></canvas>
    </div>
    <script>
        var ctx = document.getElementById('{chart_id}');
        new Chart(ctx, {config_json});
    </script>
</body>
</html>'''

        return ChartRenderResponse(
            chart_id=chart_id,
            output_format=output_format,
            content=html_content,
            content_type="text/html",
            width=width,
            height=height,
        )

    def render(
        self,
        request: ChartRenderRequest,
    ) -> ChartRenderResponse:
        """渲染图表（通用入口）。"""
        config = request.chart_config

        if request.library == "echarts":
            return self.render_echarts(config, request.output_format)
        elif request.library == "chartjs":
            return self.render_chartjs(config, request.output_format)
        elif request.library == "d3":
            raise ChartRenderError(
                message="D3.js 渲染器暂未实现，请使用 echarts 或 chartjs",
                code="not_implemented",
            )
        else:
            raise ChartRenderError(
                message=f"不支持的图表库: {request.library}",
                code="unsupported_library",
            )

    def render_template_snippet(
        self,
        config: ChartConfig,
        library: Literal["echarts", "chartjs"] = "echarts",
    ) -> ChartTemplateSnippet:
        """渲染为可嵌入 HTML 的模板片段。"""
        chart_id = self._generate_chart_id(config)

        width = config.width or 800
        height = config.height or 400

        if library == "echarts":
            series_data = self._convert_series_for_echarts(config.series)
            colors = config.colors or self._get_default_colors(len(config.series))

            echarts_config = {
                "title": {"text": config.title, "left": "center"} if config.title else None,
                "tooltip": {"trigger": "item"} if config.tooltip_enabled else None,
                "legend": {"orient": config.legend_position, "left": "center"}
                if config.series and config.legend_position
                else None,
                "xAxis": {
                    "type": "category",
                    "name": config.x_axis_title,
                    "data": [dp.x for dp in config.series[0].data] if config.series and config.series[0].data else [],
                }
                if config.chart_type not in ("pie", "donut") and config.series else None,
                "yAxis": {"type": "value", "name": config.y_axis_title}
                if config.chart_type not in ("pie", "donut") and config.series else None,
                "series": series_data,
                "color": colors,
                "animation": config.animation_enabled,
            }
            echarts_config = {k: v for k, v in echarts_config.items() if v is not None}

            container_html = f'<div id="{chart_id}" style="width: 100%; height: {height}px;"></div>'
            script_html = f'''
            <script src="{self.ECHARTS_CDN}"></script>
            <script>
                (function() {{
                    var chartDom = document.getElementById('{chart_id}');
                    if (chartDom) {{
                        var myChart = echarts.init(chartDom);
                        var option = {json.dumps(echarts_config, ensure_ascii=False)};
                        myChart.setOption(option);
                        window.addEventListener('resize', function() {{ myChart.resize(); }});
                    }}
                }})();
            </script>'''
            dependencies = ["echarts"]

        else:
            series_data = self._convert_series_for_chartjs(config.series)
            colors = config.colors or self._get_default_colors(len(config.series))
            labels = [dp.x for dp in config.series[0].data] if config.series and config.series[0].data else []

            chartjs_config = {
                "type": config.chart_type,
                "data": {"labels": labels, "datasets": series_data},
                "options": {
                    "responsive": True,
                    "maintainAspectRatio": False,
                    "plugins": {
                        "title": {"display": bool(config.title), "text": config.title},
                        "legend": {"position": config.legend_position},
                    },
                },
            }

            container_html = f'<div style="width: 100%; height: {height}px;"><canvas id="{chart_id}"></canvas></div>'
            script_html = f'''
            <script src="{self.CHARTJS_CDN}"></script>
            <script>
                (function() {{
                    var ctx = document.getElementById('{chart_id}');
                    if (ctx) {{
                        new Chart(ctx, {json.dumps(chartjs_config, ensure_ascii=False)});
                    }}
                }})();
            </script>'''
            dependencies = ["chartjs"]

        return ChartTemplateSnippet(
            chart_id=chart_id,
            container_html=container_html,
            script_html=script_html,
            styles="",
            dependencies=dependencies,
        )

    def get_supported_libraries(self) -> list[dict]:
        """获取支持的图表库信息。"""
        return [
            {
                "library": "echarts",
                "version": "5.4.3",
                "supported_types": ["bar", "line", "pie", "scatter", "area", "radar", "heatmap", "donut"],
                "features": ["丰富的图表类型", "交互性强", "支持大数据", "响应式布局"],
            },
            {
                "library": "chartjs",
                "version": "4.4.1",
                "supported_types": ["bar", "line", "pie", "doughnut", "scatter", "polarArea", "radar"],
                "features": ["简单易用", "轻量级", "Canvas 渲染", "动画效果好"],
            },
            {
                "library": "d3",
                "version": "7.8.5",
                "supported_types": ["custom"],
                "features": ["高度可定制", "底层控制", "SVG 渲染", "适合复杂可视化"],
            },
        ]
