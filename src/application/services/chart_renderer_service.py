"""图表渲染服务（T043）。

将图表 JSON 配置渲染为 SVG、PNG 或 HTML snippet。
支持 ECharts、Chart.js、D3.js 等主流图表库。
"""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import uuid
from io import BytesIO
from pathlib import Path
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

    # ECharts CDN（兜底：当本地无可用静态资源时使用）
    ECHARTS_CDN = "https://cdn.jsdelivr.net/npm/echarts@6/dist/echarts.min.js"

    # Chart.js CDN
    CHARTJS_CDN = "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"

    # D3.js CDN
    D3_CDN = "https://cdn.jsdelivr.net/npm/d3@7.8.5/dist/d3.min.js"

    def __init__(self):
        self._chart_cache: dict[str, dict] = {}

    def _ensure_local_static_asset(self, *, library: str) -> str | None:
        """确保生成的 HTML 在离线场景也能渲染图表。

        约定：
        - 输出 HTML 存放在 data/targets/<workspace>/<id>.html
        - 静态资源统一落到 data/targets/assets/
        - HTML 中使用相对路径 ../assets/<file>
        """
        lib = (library or "").strip().lower()
        if lib not in {"echarts", "chartjs", "d3"}:
            return None

        # 目标：data/targets/assets/<file>
        targets_assets_dir = Path("data") / "targets" / "assets"
        targets_assets_dir.mkdir(parents=True, exist_ok=True)

        if lib == "echarts":
            filename = "echarts.min.js"
            dest = targets_assets_dir / filename
            if dest.exists() and dest.stat().st_size > 0:
                return f"../assets/{filename}"

            # 优先从前端依赖（node_modules）拷贝（开发/一体化部署常见）
            candidates = [
                Path("node_modules") / "echarts" / "dist" / "echarts.min.js",
                Path("node_modules") / "echarts" / "dist" / "echarts.js",
            ]
            for src in candidates:
                try:
                    if src.exists() and src.stat().st_size > 0:
                        shutil.copyfile(src, dest)
                        return f"../assets/{filename}"
                except Exception:
                    continue
            return None

        if lib == "chartjs":
            filename = "chart.umd.min.js"
            dest = targets_assets_dir / filename
            if dest.exists() and dest.stat().st_size > 0:
                return f"../assets/{filename}"
            candidates = [
                Path("node_modules") / "chart.js" / "dist" / "chart.umd.min.js",
                Path("node_modules") / "chart.js" / "dist" / "chart.umd.js",
            ]
            for src in candidates:
                try:
                    if src.exists() and src.stat().st_size > 0:
                        shutil.copyfile(src, dest)
                        return f"../assets/{filename}"
                except Exception:
                    continue
            return None

        if lib == "d3":
            filename = "d3.min.js"
            dest = targets_assets_dir / filename
            if dest.exists() and dest.stat().st_size > 0:
                return f"../assets/{filename}"
            candidates = [
                Path("node_modules") / "d3" / "dist" / "d3.min.js",
                Path("node_modules") / "d3" / "dist" / "d3.js",
            ]
            for src in candidates:
                try:
                    if src.exists() and src.stat().st_size > 0:
                        shutil.copyfile(src, dest)
                        return f"../assets/{filename}"
                except Exception:
                    continue
            return None

        return None

    def _script_src_for(self, library: str) -> str:
        local = self._ensure_local_static_asset(library=library)
        if local:
            return local
        if library == "echarts":
            return self.ECHARTS_CDN
        if library == "chartjs":
            return self.CHARTJS_CDN
        if library == "d3":
            return self.D3_CDN
        return ""

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
            st = (s.chart_type or "").strip().lower()
            result.append({
                "name": s.name,
                "type": s.chart_type,
                "data": [
                    # bar/line/pie：使用 y 值；x 轴由 xAxis.data 提供（避免把 [x,y] 当成散点）
                    {"value": dp.y, "name": (dp.name or dp.x), **(dp.extra or {})}
                    if st in {"bar", "line", "area", "pie", "donut"} or dp.x is None
                    else (
                        # scatter：保留 (x,y)
                        {"value": [dp.x, dp.y], "name": dp.name, **(dp.extra or {})}
                        if dp.x is not None and dp.y is not None
                        else {"value": dp.y, "name": (dp.name or dp.x), **(dp.extra or {})}
                    )
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

    def _build_echarts_option(self, config: ChartConfig) -> dict[str, Any]:
        """构建 ECharts option（与 render_echarts / render_template_snippet 保持一致）。"""
        colors = config.colors or self._get_default_colors(len(config.series))
        series_data = self._convert_series_for_echarts(config.series)

        # legend position：ChartConfig.legend_position 表示位置（top/bottom/left/right）
        legend_pos = (config.legend_position or "").strip().lower() or "bottom"
        legend_orient = "horizontal" if legend_pos in ("top", "bottom") else "vertical"
        legend: dict[str, Any] | None = None
        show_legend = bool(config.series) and (
            len(config.series) > 1 or config.chart_type in ("pie", "donut", "sankey")
        )
        if show_legend:
            legend = {
                "orient": legend_orient,
                "left": "center" if legend_pos in ("top", "bottom") else "left",
                **(
                    {"top": 44}
                    if legend_pos == "top"
                    else ({"bottom": 0} if legend_pos == "bottom" else {"top": "middle"})
                ),
                "textStyle": {"fontSize": 12, "color": "#444"},
            }

        grid_top = 18
        if config.title:
            # title 占用空间
            grid_top = 56
            # 若 legend 也在 top，再额外下移
            if show_legend and legend_pos == "top":
                grid_top = 72

        opt: dict[str, Any] = {
            "title": {
                "text": config.title,
                "subtext": config.subtitle,
                "left": "center",
                "top": 8,
                "textStyle": {"fontSize": 14, "fontWeight": "bold", "color": "#222"},
                "subtextStyle": {"fontSize": 12, "color": "#666"},
            }
            if config.title
            else None,
            "tooltip": {"trigger": "item", "enabled": config.tooltip_enabled}
            if config.tooltip_enabled
            else None,
            "legend": legend,
            "grid": {
                "left": "3%",
                "right": "4%",
                "bottom": "3%",
                "top": grid_top,
                "containLabel": True,
            }
            if config.grid_enabled
            else None,
            "xAxis": {
                "type": "category",
                "name": config.x_axis_title,
                # 兜底：有些 chart_json 可能缺失 categories，但数据点里有 name
                "data": [
                    (dp.x if dp.x is not None else (dp.name or ""))
                    for dp in config.series[0].data
                    if (dp.x is not None) or (dp.name is not None)
                ]
                if config.series and config.series[0].data
                else [],
                "axisLabel": {"rotate": 45}
                if config.series and len(config.series[0].data) > 10
                else None,
            }
            if config.chart_type not in ("pie", "donut") and config.series
            else None,
            "yAxis": {"type": "value", "name": config.y_axis_title}
            if config.chart_type not in ("pie", "donut") and config.series
            else None,
            "series": series_data,
            "color": colors,
            "animation": config.animation_enabled,
        }
        return {k: v for k, v in opt.items() if v is not None}

    def render_echarts(
        self,
        config: ChartConfig,
        output_format: Literal["svg", "png", "html", "base64"] = "html",
    ) -> ChartRenderResponse:
        """使用 ECharts 渲染图表。"""
        chart_id = self._generate_chart_id(config)

        width = config.width or 800
        height = config.height or 400
        echarts_config = self._build_echarts_option(config)

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
        src = self._script_src_for("echarts")

        return f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <script src="{src}"></script>
</head>
<body>
    <div id="{chart_id}" style="width: {width}px; height: {height}px;"></div>
    <script>
        var chartDom = document.getElementById('{chart_id}');
        var myChart = echarts.init(chartDom);
        var option = {config_json};
        try {{
            myChart.setOption(option);
        }} catch (e) {{
            console.error("ECharts render failed", e);
            try {{
                if (chartDom) {{
                    var errDiv = document.createElement('div');
                    errDiv.style.cssText = "padding:12px;border:1px solid #f5c2c7;background:#f8d7da;color:#842029;border-radius:8px;font-size:12px;line-height:1.5;";
                    var msg = (e && e.message) ? e.message : String(e);
                    errDiv.textContent = "图表渲染失败：" + msg;
                    chartDom.innerHTML = "";
                    chartDom.appendChild(errDiv);
                }}
            }} catch (_e2) {{}}
        }}
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
        src = self._script_src_for("chartjs")

        html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <script src="{src}"></script>
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
            return self.render_d3(config, request.output_format)
        else:
            raise ChartRenderError(
                message=f"不支持的图表库: {request.library}",
                code="unsupported_library",
            )


    def _build_d3_bar_chart(
        self,
        chart_id: str,
        labels: list[str],
        values: list[float],
        colors: list[str],
        width: int,
        height: int,
    ) -> dict[str, str]:
        """构建 D3.js 柱状图。"""
        data_json = json.dumps([{"label": l, "value": v} for l, v in zip(labels, values)], ensure_ascii=False)
        colors_json = json.dumps(colors, ensure_ascii=False)
        src = self._script_src_for("d3")

        svg = f'''<svg id="{chart_id}" width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
    <style>
        .{chart_id}-bar {{ fill: steelblue; }}
        .{chart_id}-bar:hover {{ fill: darkblue; }}
        .{chart_id}-axis {{ font-size: 12px; }}
        .{chart_id}-title {{ font-size: 16px; font-weight: bold; text-anchor: middle; }}
    </style>
</svg>'''

        html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <script src="{src}"></script>
</head>
<body>
    <svg id="{chart_id}" width="{width}" height="{height}"></svg>
    <script>
        (function() {{
            const data = {data_json};
            const colors = {colors_json};
            const margin = {{top: 40, right: 20, bottom: 60, left: 60}};
            const chartWidth = {width} - margin.left - margin.right;
            const chartHeight = {height} - margin.top - margin.bottom;

            const svg = d3.select("#{chart_id}")
                .attr("width", {width})
                .attr("height", {height})
                .append("g")
                .attr("transform", `translate(${{margin.left}},${{margin.top}})`);

            const x = d3.scaleBand()
                .domain(data.map(d => d.label))
                .range([0, chartWidth])
                .padding(0.2);

            const y = d3.scaleLinear()
                .domain([0, d3.max(data, d => d.value)])
                .nice()
                .range([chartHeight, 0]);

            svg.append("g")
                .attr("transform", `translate(0,${{chartHeight}})`)
                .call(d3.axisBottom(x))
                .selectAll("text")
                .attr("transform", "rotate(-45)")
                .style("text-anchor", "end");

            svg.append("g").call(d3.axisLeft(y));

            svg.selectAll(".bar")
                .data(data)
                .enter()
                .append("rect")
                .attr("class", "bar")
                .attr("x", d => x(d.label))
                .attr("y", d => y(d.value))
                .attr("width", x.bandwidth())
                .attr("height", d => chartHeight - y(d.value))
                .attr("fill", (d, i) => colors[i % colors.length]);
        }})();
    </script>
</body>
</html>'''

        return {"svg": svg, "html": html}

    def _build_d3_line_chart(
        self,
        chart_id: str,
        labels: list[str],
        values: list[float],
        colors: list[str],
        width: int,
        height: int,
    ) -> dict[str, str]:
        """构建 D3.js 折线图。"""
        data_json = json.dumps([{"label": l, "value": v} for l, v in zip(labels, values)], ensure_ascii=False)
        colors_json = json.dumps(colors, ensure_ascii=False)
        src = self._script_src_for("d3")

        svg = f'''<svg id="{chart_id}" width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
    <style>
        .{chart_id}-line {{ fill: none; stroke: steelblue; stroke-width: 2; }}
        .{chart_id}-dot {{ fill: steelblue; }}
        .{chart_id}-axis {{ font-size: 12px; }}
    </style>
</svg>'''

        html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <script src="{src}"></script>
</head>
<body>
    <svg id="{chart_id}" width="{width}" height="{height}"></svg>
    <script>
        (function() {{
            const data = {data_json};
            const colors = {colors_json};
            const margin = {{top: 40, right: 20, bottom: 60, left: 60}};
            const chartWidth = {width} - margin.left - margin.right;
            const chartHeight = {height} - margin.top - margin.bottom;

            const svg = d3.select("#{chart_id}")
                .attr("width", {width})
                .attr("height", {height})
                .append("g")
                .attr("transform", `translate(${{margin.left}},${{margin.top}})`);

            const x = d3.scalePoint()
                .domain(data.map(d => d.label))
                .range([0, chartWidth]);

            const y = d3.scaleLinear()
                .domain([0, d3.max(data, d => d.value)])
                .nice()
                .range([chartHeight, 0]);

            svg.append("g")
                .attr("transform", `translate(0,${{chartHeight}})`)
                .call(d3.axisBottom(x))
                .selectAll("text")
                .attr("transform", "rotate(-45)")
                .style("text-anchor", "end");

            svg.append("g").call(d3.axisLeft(y));

            const line = d3.line()
                .x(d => x(d.label))
                .y(d => y(d.value));

            svg.append("path")
                .datum(data)
                .attr("class", "line")
                .attr("d", line)
                .attr("stroke", colors[0]);

            svg.selectAll(".dot")
                .data(data)
                .enter()
                .append("circle")
                .attr("class", "dot")
                .attr("cx", d => x(d.label))
                .attr("cy", d => y(d.value))
                .attr("r", 5)
                .attr("fill", colors[0]);
        }})();
    </script>
</body>
</html>'''

        return {"svg": svg, "html": html}

    def _build_d3_pie_chart(
        self,
        chart_id: str,
        labels: list[str],
        values: list[float],
        colors: list[str],
        width: int,
        height: int,
        chart_type: str,
    ) -> dict[str, str]:
        """构建 D3.js 饼图/环形图。"""
        data_json = json.dumps([{"label": l, "value": v} for l, v in zip(labels, values)], ensure_ascii=False)
        colors_json = json.dumps(colors, ensure_ascii=False)
        src = self._script_src_for("d3")
        inner_radius = 0 if chart_type == "pie" else width * 0.3

        svg = f'''<svg id="{chart_id}" width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
    <style>
        .{chart_id}-arc {{ stroke: white; }}
        .{chart_id}-label {{ font-size: 12px; text-anchor: middle; }}
    </style>
</svg>'''

        html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <script src="{src}"></script>
</head>
<body>
    <svg id="{chart_id}" width="{width}" height="{height}"></svg>
    <script>
        (function() {{
            const data = {data_json};
            const colors = {colors_json};
            const margin = 20;
            const radius = Math.min({width}, {height}) / 2 - margin;
            const innerRadius = {inner_radius};

            const svg = d3.select("#{chart_id}")
                .attr("width", {width})
                .attr("height", {height})
                .append("g")
                .attr("transform", `translate(${{{width}/2}},${{{height}/2}})`);

            const pie = d3.pie()
                .value(d => d.value)
                .sort(null);

            const arc = d3.arc()
                .innerRadius(innerRadius)
                .outerRadius(radius);

            const arcs = svg.selectAll(".arc")
                .data(pie(data))
                .enter()
                .append("g")
                .attr("class", "arc");

            arcs.append("path")
                .attr("d", arc)
                .attr("fill", (d, i) => colors[i % colors.length])
                .attr("stroke", "white")
                .style("stroke-width", "2px");

            arcs.append("text")
                .attr("transform", d => `translate(${{arc.centroid(d)}})`)
                .attr("class", "label")
                .text(d => d.data.label);
        }})();
    </script>
</body>
</html>'''

        return {"svg": svg, "html": html}


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
            src = self._script_src_for("echarts")
            # 高级用法：允许直接传入完整的 ECharts option（用于复杂图，如 sankey/graph）
            if isinstance(config.extra, dict):
                raw_opt = config.extra.get("echarts_option")
                if isinstance(raw_opt, dict) and raw_opt:
                    container_html = (
                        f'<div id="{chart_id}" style="width: 100%; height: {height}px;"></div>'
                    )
                    script_html = f'''
            <script src="{src}"></script>
            <script>
                (function() {{
                    var chartDom = document.getElementById('{chart_id}');
                    if (chartDom) {{
                        var myChart = echarts.init(chartDom);
                        var option = {json.dumps(raw_opt, ensure_ascii=False)};
                        try {{
                            myChart.setOption(option);
                        }} catch (e) {{
                            console.error("ECharts setOption failed", e);
                            try {{
                                var errDiv = document.createElement('div');
                                errDiv.style.cssText = "padding:12px;border:1px solid #f5c2c7;background:#f8d7da;color:#842029;border-radius:8px;font-size:12px;line-height:1.5;";
                                var msg = (e && e.message) ? e.message : String(e);
                                errDiv.textContent = "图表渲染失败：" + msg;
                                chartDom.innerHTML = "";
                                chartDom.appendChild(errDiv);
                            }} catch (_e2) {{}}
                        }}
                        window.addEventListener('resize', function() {{ myChart.resize(); }});
                    }}
                }})();
            </script>'''
                    return ChartTemplateSnippet(
                        chart_id=chart_id,
                        container_html=container_html,
                        script_html=script_html,
                        styles="",
                        dependencies=["echarts"],
                    )

            echarts_config = self._build_echarts_option(config)

            container_html = f'<div id="{chart_id}" style="width: 100%; height: {height}px;"></div>'
            script_html = f'''
            <script src="{src}"></script>
            <script>
                (function() {{
                    var chartDom = document.getElementById('{chart_id}');
                    if (chartDom) {{
                        var myChart = echarts.init(chartDom);
                        var option = {json.dumps(echarts_config, ensure_ascii=False)};
                        try {{
                            myChart.setOption(option);
                        }} catch (e) {{
                            console.error("ECharts setOption failed", e);
                            try {{
                                var errDiv = document.createElement('div');
                                errDiv.style.cssText = "padding:12px;border:1px solid #f5c2c7;background:#f8d7da;color:#842029;border-radius:8px;font-size:12px;line-height:1.5;";
                                var msg = (e && e.message) ? e.message : String(e);
                                errDiv.textContent = "图表渲染失败：" + msg;
                                chartDom.innerHTML = "";
                                chartDom.appendChild(errDiv);
                            }} catch (_e2) {{}}
                        }}
                        window.addEventListener('resize', function() {{ myChart.resize(); }});
                    }}
                }})();
            </script>'''
            dependencies = ["echarts"]

        else:
            src = self._script_src_for("chartjs")
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
            <script src="{src}"></script>
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
                "supported_types": ["bar", "line", "pie", "donut", "histogram", "area"],
                "features": ["高度可定制", "底层控制", "SVG 渲染", "支持 SVG/PNG/HTML 输出"],
            },
        ]

    def render_d3(
        self,
        config: ChartConfig,
        output_format: Literal["svg", "png", "html", "base64"] = "html",
    ) -> ChartRenderResponse:
        """使用 D3.js 渲染图表。"""
        chart_id = self._generate_chart_id(config)

        width = config.width or 800
        height = config.height or 400

        # 构建 D3.js 可视化配置
        labels = [dp.x for dp in config.series[0].data] if config.series and config.series[0].data else []
        values = [dp.y for dp in config.series[0].data] if config.series and config.series[0].data else []
        colors = config.colors or self._get_default_colors(len(config.series))

        # 根据图表类型生成不同的 D3 代码
        chart_type = (config.chart_type or "").strip().lower()

        if chart_type in ("bar", "histogram"):
            d3_code = self._build_d3_bar_chart(chart_id, labels, values, colors, width, height)
        elif chart_type in ("line", "area"):
            d3_code = self._build_d3_line_chart(chart_id, labels, values, colors, width, height)
        elif chart_type in ("pie", "donut"):
            d3_code = self._build_d3_pie_chart(chart_id, labels, values, colors, width, height, chart_type)
        else:
            # 默认使用柱状图
            d3_code = self._build_d3_bar_chart(chart_id, labels, values, colors, width, height)

        if output_format == "svg":
            # D3.js 原生支持 SVG 输出
            svg_content = d3_code["svg"]
            return ChartRenderResponse(
                chart_id=chart_id,
                output_format=output_format,
                content=svg_content,
                content_type="image/svg+xml",
                width=width,
                height=height,
            )

        if output_format == "png":
            # 将 SVG 转换为 PNG
            svg_content = d3_code["svg"]
            import base64 as b64
            from src.shared.errors import ChartRenderError

            try:
                from cairosvg import svg2png
                png_bytes = svg2png(bytestring=svg_content.encode("utf-8"), output_width=width, output_height=height)
                return ChartRenderResponse(
                    chart_id=chart_id,
                    output_format=output_format,
                    content=b64.b64encode(png_bytes).decode(),
                    content_type="image/png",
                    width=width,
                    height=height,
                )
            except ImportError:
                raise ChartRenderError(
                    message="PNG 格式需要 cairosvg 库，请安装: pip install cairosvg",
                    code="missing_dependency",
                )

        if output_format == "base64":
            html_content = d3_code["html"]
            import base64 as b64

            return ChartRenderResponse(
                chart_id=chart_id,
                output_format=output_format,
                content=b64.b64encode(html_content.encode()).decode(),
                content_type="text/html",
                width=width,
                height=height,
            )

        # 返回 HTML
        html_content = d3_code["html"]
        return ChartRenderResponse(
            chart_id=chart_id,
            output_format=output_format,
            content=html_content,
            content_type="text/html",
            width=width,
            height=height,
        )
