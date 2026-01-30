from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


Engine = Literal["g2", "s2", "infographic"]


@dataclass(frozen=True)
class AntvRenderPayload:
    engine: Engine
    width: int
    height: int
    theme: str
    spec: dict[str, Any]


def _normalize_chart_type(raw: Any) -> str:
    s = str(raw).strip().lower() if raw is not None else ""
    aliases = {
        "bar_chart": "bar",
        "line_chart": "line",
        "pie_chart": "pie",
        "stacked_area_chart_with_legend": "stacked_area",
        "flow_chart_with_categories": "sankey",
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


def _g2_theme_whitepaper_default() -> dict[str, Any]:
    """白皮书默认主题 - 专业电力行业配色方案"""
    return {
        "type": "classic",
        "fontFamily": "PingFang SC, Microsoft YaHei, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
        "palette": [
            "#3b82f6",  # 蓝色 - 火电/主要系列
            "#10b981",  # 绿色 - 水电/环保
            "#f59e0b",  # 琥珀色 - 储能放电
            "#ef4444",  # 红色 - 储能充电/警示
            "#8b5cf6",  # 紫色 - 新能源
            "#06b6d4",  # 青色 - 核电
            "#ec4899",  # 粉色 - 特殊系列
            "#84cc16",  # 青柠色 - 可选
        ],
        "background": "#ffffff",
    }


def chart_json_to_antv_payload(
    *,
    chart_id: str,
    chart_json: dict[str, Any],
    theme: str = "whitepaper-default",
) -> AntvRenderPayload:
    chart_type = _normalize_chart_type(chart_json.get("chart_type"))
    title = str(chart_json.get("_chart_name") or chart_json.get("description") or "").strip()
    chart_data = chart_json.get("chart_data") if isinstance(chart_json, dict) else None

    if chart_type == "sankey":
        links = []
        if isinstance(chart_data, dict):
            raw_links = chart_data.get("links")
            if isinstance(raw_links, list):
                for it in raw_links:
                    if not isinstance(it, dict):
                        continue
                    src = str(it.get("source") or "").strip()
                    tgt = str(it.get("target") or "").strip()
                    val = it.get("value")
                    if not src or not tgt:
                        continue
                    try:
                        v = float(val) if val is not None else 1.0
                    except Exception:
                        v = 1.0
                    links.append({"source": src, "target": tgt, "value": v})

        spec = {
            "type": "sankey",
            "data": {"value": {"links": links}},
            "layout": {"nodeAlign": "justify", "nodePadding": 0.03, "nodeStrokeWidth": 0},
            "style": {
                "labelSpacing": 6,
                "labelFontSize": 12,
                "labelFontWeight": 500,
                "nodeStrokeWidth": 1,
                "nodeStroke": "#fff",
                "linkFillOpacity": 0.4,
                "linkCurveOffset": 8,
            },
            "edge": {
                "style": {
                    "endArrow": True,
                    "lineWidth": 2,
                }
            },
            "axis": False,
            "title": {
                "text": title or chart_id,
                "fontSize": 16,
                "fontWeight": 600,
                "paddingBottom": 16,
            },
            "theme": _g2_theme_whitepaper_default(),
        }
        return AntvRenderPayload(engine="g2", width=980, height=520, theme=theme, spec=spec)

    if chart_type == "stacked_area":
        area_records: list[dict[str, Any]] = []
        if isinstance(chart_data, dict):
            xs = chart_data.get("x")
            series = chart_data.get("series")
            if isinstance(xs, list) and isinstance(series, list):
                # 去除重复的时间点（最后一个如果与第一个相同则移除）
                x_list = [str(x).strip() for x in xs if str(x).strip()]
                if len(x_list) > 1 and x_list[0] == x_list[-1]:
                    x_list = x_list[:-1]
                x_order = x_list
                for s in series:
                    if not isinstance(s, dict):
                        continue
                    name = str(s.get("name") or "系列").strip()
                    vals = s.get("values")
                    if not isinstance(vals, list):
                        continue
                    for i, x in enumerate(x_order):
                        if i >= len(vals):
                            continue
                        y = vals[i]
                        try:
                            yv = float(y) if y is not None else None
                        except Exception:
                            yv = None
                        if yv is None:
                            continue
                        area_records.append({"x": x, "series": name, "y": yv})

        spec = {
            "type": "area",
            "data": area_records,
            "encode": {"x": "x", "y": "y", "color": "series"},
            "transform": [{"type": "stackY"}],
            "axis": {
                "x": {"grid": True, "label": {"autoRotate": True}, "title": False},
                "y": {"grid": True, "title": False},
            },
            "legend": {"color": {"position": "bottom", "layout": {"justifyContent": "center"}}},
            "title": {
                "text": title or chart_id,
                "fontSize": 16,
                "fontWeight": 600,
                "paddingBottom": 8,
            },
            "style": {
                "fillOpacity": 0.85,
                "lineWidth": 1.5,
            },
            "theme": _g2_theme_whitepaper_default(),
        }
        return AntvRenderPayload(engine="g2", width=980, height=520, theme=theme, spec=spec)

    if chart_type == "bar":
        bar_records: list[dict[str, Any]] = []
        if isinstance(chart_data, list):
            for it in chart_data:
                if not isinstance(it, dict) or "_chart_separator" in it:
                    continue
                x = str(it.get("category") or "").strip()
                y = it.get("value")
                if not x:
                    continue
                try:
                    yv = float(y) if y is not None else None
                except Exception:
                    yv = None
                if yv is None:
                    continue
                bar_records.append({"x": x, "series": "数值", "y": yv})
        elif isinstance(chart_data, dict):
            categories = chart_data.get("categories")
            series = chart_data.get("series")
            if isinstance(categories, list) and isinstance(series, list):
                x_order = [str(x).strip() for x in categories if str(x).strip()]
                for s in series[:8]:
                    if not isinstance(s, dict):
                        continue
                    name = str(s.get("name") or "系列").strip()
                    vals = s.get("values")
                    if not isinstance(vals, list):
                        continue
                    for i, x in enumerate(x_order):
                        y = vals[i] if i < len(vals) else None
                        try:
                            yv = float(y) if y is not None else None
                        except Exception:
                            yv = None
                        if yv is None:
                            continue
                        bar_records.append({"x": x, "series": name, "y": yv})
        spec = {
            "type": "interval",
            "data": bar_records,
            "encode": {"x": "x", "y": "y", "color": "series"},
            "axis": {
                "x": {"grid": False, "title": False},
                "y": {"grid": True, "title": False},
            },
            "legend": {"color": {"position": "bottom", "layout": {"justifyContent": "center"}}},
            "title": {
                "text": title or chart_id,
                "fontSize": 16,
                "fontWeight": 600,
                "paddingBottom": 8,
            },
            "theme": _g2_theme_whitepaper_default(),
        }
        return AntvRenderPayload(engine="g2", width=980, height=520, theme=theme, spec=spec)

    if chart_type == "line":
        line_records: list[dict[str, Any]] = []
        if isinstance(chart_data, list):
            for s in chart_data[:8]:
                if not isinstance(s, dict):
                    continue
                name = str(s.get("series") or s.get("name") or "系列").strip()
                dps = s.get("data_points")
                if not isinstance(dps, list):
                    continue
                for dp in dps:
                    if not isinstance(dp, dict):
                        continue
                    x = str(dp.get("time") or "").strip()
                    y = dp.get("value")
                    if not x:
                        continue
                    try:
                        yv = float(y) if y is not None else None
                    except Exception:
                        yv = None
                    if yv is None:
                        continue
                    line_records.append({"x": x, "series": name, "y": yv})
        elif isinstance(chart_data, dict):
            x_axis = chart_data.get("x") or chart_data.get("x_axis") or chart_data.get("categories") or []
            series = chart_data.get("series") or []
            if isinstance(x_axis, list) and isinstance(series, list):
                x_order = [str(x).strip() for x in x_axis if str(x).strip()]
                for s in series[:8]:
                    if not isinstance(s, dict):
                        continue
                    name = str(s.get("name") or "系列").strip()
                    vals = s.get("values")
                    if not isinstance(vals, list):
                        continue
                    for i, x in enumerate(x_order):
                        y = vals[i] if i < len(vals) else None
                        try:
                            yv = float(y) if y is not None else None
                        except Exception:
                            yv = None
                        if yv is None:
                            continue
                        line_records.append({"x": x, "series": name, "y": yv})
        spec = {
            "type": "line",
            "data": line_records,
            "encode": {"x": "x", "y": "y", "color": "series"},
            "axis": {
                "x": {"grid": True, "label": {"autoRotate": True}, "title": False},
                "y": {"grid": True, "title": False},
            },
            "legend": {"color": {"position": "bottom", "layout": {"justifyContent": "center"}}},
            "title": {
                "text": title or chart_id,
                "fontSize": 16,
                "fontWeight": 600,
                "paddingBottom": 8,
            },
            "style": {"lineWidth": 2},
            "theme": _g2_theme_whitepaper_default(),
        }
        return AntvRenderPayload(engine="g2", width=980, height=520, theme=theme, spec=spec)

    if chart_type == "pie":
        pie_records: list[dict[str, Any]] = []
        if isinstance(chart_data, list):
            for it in chart_data:
                if not isinstance(it, dict) or "_chart_separator" in it:
                    continue
                x = str(it.get("category") or it.get("name") or "").strip()
                y = it.get("value")
                if not x:
                    continue
                try:
                    yv = float(y) if y is not None else None
                except Exception:
                    yv = None
                if yv is None:
                    continue
                pie_records.append({"category": x, "value": yv})
        spec = {
            "type": "interval",
            "data": pie_records,
            "encode": {"y": "value", "color": "category"},
            "transform": [{"type": "stackY"}],
            "coordinate": {"type": "theta", "outerRadius": 0.75},
            "legend": {"color": {"position": "right", "layout": {"justifyContent": "center"}}},
            "title": {
                "text": title or chart_id,
                "fontSize": 16,
                "fontWeight": 600,
                "paddingBottom": 8,
            },
            "theme": _g2_theme_whitepaper_default(),
        }
        return AntvRenderPayload(engine="g2", width=760, height=520, theme=theme, spec=spec)

    if chart_type == "table":
        if not isinstance(chart_data, dict):
            chart_data = {}
        columns = chart_data.get("columns") if isinstance(chart_data.get("columns"), list) else []
        rows = chart_data.get("rows") if isinstance(chart_data.get("rows"), list) else []
        spec = {"columns": columns, "rows": rows}
        return AntvRenderPayload(engine="s2", width=980, height=520, theme=theme, spec=spec)

    raise ValueError(f"unsupported chart_type for antv render: {chart_type}")
