"""chart_json -> ChartConfig 转换（T042）。

说明：
- 该文件只负责把 T094 的 chart_json 结构映射为可渲染的 `ChartConfig`
- 不做文件 IO，不做检索，不做渲染
"""

from __future__ import annotations

import re
from typing import Any

from src.application.schemas.chart_spec import ChartConfig, ChartDataPoint, ChartSeries


class ChartJsonConverter:
    """将 chart_json 转为可渲染 ChartConfig（与历史实现保持兼容）。"""

    _NUM_RE = re.compile(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?")

    def _parse_number_maybe(self, x: Any) -> float | None:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            try:
                return float(x)
            except Exception:
                return None
        s = str(x).strip()
        if not s:
            return None
        s2 = s.replace(",", "")
        try:
            return float(s2)
        except Exception:
            m = self._NUM_RE.search(s2)
            if not m:
                return None
            try:
                return float(m.group(0))
            except Exception:
                return None

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

    def _try_table_like_to_grouped_bar(self, *, title: str, obj: dict[str, Any]) -> ChartConfig | None:
        """把“表格/对比类”结构尽量映射为分组柱状图（兜底增强可视化命中率）。"""
        cd = obj.get("chart_data")
        if not isinstance(cd, dict):
            return None
        categories = cd.get("categories")
        series = cd.get("series")
        if not isinstance(categories, list) or not isinstance(series, list) or not categories:
            return None
        x_order = [str(x).strip() for x in categories if str(x).strip()]
        if not x_order:
            return None
        series_list: list[ChartSeries] = []
        for s in series[:8]:
            if not isinstance(s, dict):
                continue
            name = str(s.get("name") or "系列").strip()
            vals = s.get("values")
            if not isinstance(vals, list):
                continue
            pts: list[ChartDataPoint] = []
            for i, x in enumerate(x_order):
                y = self._parse_number_maybe(vals[i]) if i < len(vals) else None
                pts.append(ChartDataPoint(x=x, y=y, name=x))
            series_list.append(ChartSeries(name=name, chart_type="bar", data=pts))
        if not series_list:
            return None
        return ChartConfig(
            chart_type="bar",
            title=title,
            x_axis_title="",
            y_axis_title="",
            series=series_list,
            height=460,
        )

    def chart_json_to_bar_configs(self, *, title: str, obj: dict[str, Any]) -> list[ChartConfig]:
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

    def chart_json_to_line_configs(self, *, title: str, obj: dict[str, Any]) -> list[ChartConfig]:
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
            if not x_order:
                return []

            series_list: list[ChartSeries] = []
            for s in cd[:8]:
                if not isinstance(s, dict):
                    continue
                name = str(s.get("series") or s.get("name") or "系列").strip()
                dps = s.get("data_points") or []
                if not isinstance(dps, list):
                    continue
                mp: dict[str, float | None] = {}
                for dp in dps:
                    if not isinstance(dp, dict):
                        continue
                    t = dp.get("time")
                    tx = str(t).strip() if t is not None else ""
                    if not tx:
                        continue
                    mp[tx] = self._parse_number_maybe(dp.get("value"))
                pts: list[ChartDataPoint] = []
                for x in x_order:
                    pts.append(ChartDataPoint(x=x, y=mp.get(x), name=x))
                series_list.append(ChartSeries(name=name, chart_type="line", data=pts))
            if not series_list:
                return []

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

        # canonical: chart_data={"x":[...], "series":[{"name","values":[...]}]}
        if isinstance(cd, dict):
            x_axis = cd.get("x") or cd.get("x_axis") or cd.get("categories") or []
            series = cd.get("series") or []
            if isinstance(x_axis, list) and isinstance(series, list) and x_axis:
                x_order = [str(x).strip() for x in x_axis if str(x).strip()]
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
                    pts: list[ChartDataPoint] = []
                    for i, x in enumerate(x_order):
                        y = self._parse_number_maybe(vals[i]) if i < len(vals) else None
                        pts.append(ChartDataPoint(x=x, y=y, name=x))
                    series_list.append(ChartSeries(name=name, chart_type="line", data=pts))
                if series_list:
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

        return []

    def chart_json_to_pie_configs(self, *, title: str, obj: dict[str, Any]) -> list[ChartConfig]:
        cd = obj.get("chart_data")

        # legacy: [{label,value}]
        pts: list[ChartDataPoint] = []
        if isinstance(cd, list):
            for it in cd:
                if not isinstance(it, dict):
                    continue
                if "_chart_separator" in it:
                    continue
                x = it.get("label") or it.get("category")
                y = self._parse_number_maybe(it.get("value"))
                if x is None:
                    continue
                pts.append(ChartDataPoint(x=str(x), y=y, name=str(x)))
            if pts:
                return [
                    ChartConfig(
                        chart_type="pie",
                        title=title,
                        series=[ChartSeries(name="占比", chart_type="pie", data=pts)],
                        height=460,
                    )
                ]

        # canonical: chart_data={"labels":[...], "values":[...]}
        if isinstance(cd, dict):
            labels = cd.get("labels") or cd.get("categories") or []
            values = cd.get("values") or []
            if isinstance(labels, list) and isinstance(values, list) and labels:
                pts2: list[ChartDataPoint] = []
                for i, lab in enumerate(labels):
                    x = str(lab).strip()
                    if not x:
                        continue
                    y = self._parse_number_maybe(values[i]) if i < len(values) else None
                    pts2.append(ChartDataPoint(x=x, y=y, name=x))
                if pts2:
                    return [
                        ChartConfig(
                            chart_type="pie",
                            title=title,
                            series=[ChartSeries(name="占比", chart_type="pie", data=pts2)],
                            height=460,
                        )
                    ]
        return []

    def chart_json_to_stacked_area_configs(self, *, title: str, obj: dict[str, Any]) -> list[ChartConfig]:
        cd = obj.get("chart_data")
        if not isinstance(cd, dict):
            return []
        categories = cd.get("categories") or cd.get("x") or []
        series = cd.get("series") or []
        if not isinstance(categories, list) or not isinstance(series, list) or not categories:
            return []
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
            pts: list[ChartDataPoint] = []
            for i, x in enumerate(x_order):
                y = self._parse_number_maybe(vals[i]) if i < len(vals) else None
                pts.append(ChartDataPoint(x=x, y=y, name=x))
            series_list.append(
                ChartSeries(name=name, chart_type="area", data=pts, extra={"stack": "total", "areaStyle": {}})
            )
        if not series_list:
            return []
        return [
            ChartConfig(
                chart_type="stacked_area",
                title=title,
                x_axis_title="",
                y_axis_title="",
                series=series_list,
                height=460,
            )
        ]

    def chart_json_to_sankey_configs(self, *, title: str, obj: dict[str, Any]) -> list[ChartConfig]:
        cd = obj.get("chart_data")
        if not isinstance(cd, dict):
            return []
        nodes = cd.get("nodes") or obj.get("nodes") or []
        links = cd.get("links") or obj.get("links") or []
        if not isinstance(nodes, list) or not isinstance(links, list):
            return []

        def _norm_node_name(x: Any) -> str | None:
            if x is None:
                return None
            s = str(x).strip()
            return s or None

        def _extract_edges(links0: list[Any]) -> list[tuple[str, str]]:
            edges: list[tuple[str, str]] = []
            for lk in links0:
                if not isinstance(lk, dict):
                    continue
                s = _norm_node_name(lk.get("source"))
                t = _norm_node_name(lk.get("target"))
                if not s or not t:
                    continue
                edges.append((s, t))
            return edges

        def _detect_cycle_nodes(edges: list[tuple[str, str]]) -> list[str] | None:
            """若存在环，返回一个环路径节点序列（首尾同节点）；否则返回 None。"""
            if not edges:
                return None
            adj: dict[str, list[str]] = {}
            nodes_set: set[str] = set()
            for s, t in edges:
                adj.setdefault(s, []).append(t)
                nodes_set.add(s)
                nodes_set.add(t)
            # 保持确定性：邻接表与节点都排序
            for k in list(adj.keys()):
                adj[k] = sorted(adj[k])
            nodes_list = sorted(nodes_set)

            visited: set[str] = set()
            in_stack: set[str] = set()
            stack: list[tuple[str, int]] = []

            for start in nodes_list:
                if start in visited:
                    continue
                stack = [(start, 0)]
                in_stack.add(start)
                while stack:
                    node, idx = stack[-1]
                    neigh = adj.get(node, [])
                    if idx >= len(neigh):
                        stack.pop()
                        in_stack.discard(node)
                        visited.add(node)
                        continue
                    nxt = neigh[idx]
                    stack[-1] = (node, idx + 1)
                    if nxt in in_stack:
                        path_nodes = [n for n, _ in stack]
                        try:
                            j = path_nodes.index(nxt)
                        except ValueError:
                            j = 0
                        cycle = path_nodes[j:] + [nxt]
                        return cycle
                    if nxt in visited:
                        continue
                    stack.append((nxt, 0))
                    in_stack.add(nxt)
            return None

        edges = _extract_edges(links)
        cycle_nodes = _detect_cycle_nodes(edges)
        if cycle_nodes:
            # 降级：用 graph 兜底（强标注，避免误导）
            subtext = "示意图（原 Sankey 数据存在环，无法按 Sankey 展示）"
            # nodes/links 形态尽量兼容 ECharts graph
            graph_nodes: list[dict[str, Any]] = []
            for n in nodes:
                if isinstance(n, dict):
                    nn = _norm_node_name(n.get("name") or n.get("id") or n.get("label"))
                    if not nn:
                        continue
                    d = dict(n)
                    d.setdefault("name", nn)
                    graph_nodes.append(d)
                else:
                    nn = _norm_node_name(n)
                    if nn:
                        graph_nodes.append({"name": nn})

            graph_links: list[dict[str, Any]] = []
            for lk in links:
                if not isinstance(lk, dict):
                    continue
                s = _norm_node_name(lk.get("source"))
                t = _norm_node_name(lk.get("target"))
                if not s or not t:
                    continue
                d = dict(lk)
                d["source"] = s
                d["target"] = t
                graph_links.append(d)

            opt = {
                "title": {"text": title, "subtext": subtext, "left": "center"} if title or subtext else None,
                "tooltip": {"trigger": "item"},
                "series": [
                    {
                        "type": "graph",
                        "layout": "force",
                        "roam": True,
                        "data": graph_nodes,
                        "links": graph_links,
                        "label": {"show": True, "position": "right"},
                        "edgeSymbol": ["none", "arrow"],
                        "lineStyle": {"opacity": 0.6, "curveness": 0.15},
                        "force": {"repulsion": 120, "edgeLength": 80},
                        "emphasis": {"focus": "adjacency"},
                    }
                ],
            }
            opt = {k: v for k, v in opt.items() if v is not None}

            cycle_edges_sample = [
                {"source": cycle_nodes[i], "target": cycle_nodes[i + 1]}
                for i in range(max(0, len(cycle_nodes) - 1))
            ][:8]

            return [
                ChartConfig(
                    chart_type="graph",
                    title=title,
                    series=[],
                    height=520,
                    extra={
                        "echarts_option": opt,
                        "original_chart_type": "sankey",
                        "cycle_detected": True,
                        "cycle_edges_sample": cycle_edges_sample,
                        "original_links_count": len(edges),
                        "nodes_count": len(graph_nodes),
                    },
                )
            ]

        opt = {
            "title": {"text": title, "left": "center"} if title else None,
            "tooltip": {"trigger": "item"},
            "series": [
                {
                    "type": "sankey",
                    "data": nodes,
                    "links": links,
                    "emphasis": {"focus": "adjacency"},
                }
            ],
        }
        opt = {k: v for k, v in opt.items() if v is not None}
        return [
            ChartConfig(
                chart_type="sankey",
                title=title,
                series=[],
                height=520,
                extra={"echarts_option": opt},
            )
        ]

    def chart_json_to_scatter_configs(self, *, title: str, obj: dict[str, Any]) -> list[ChartConfig]:
        cd = obj.get("chart_data")
        if not isinstance(cd, dict):
            return []
        pts = cd.get("points") or cd.get("data") or []
        if not isinstance(pts, list) or not pts:
            return []
        data: list[ChartDataPoint] = []
        for p in pts:
            if isinstance(p, dict):
                x = self._parse_number_maybe(p.get("x"))
                y = self._parse_number_maybe(p.get("y"))
                name = p.get("name")
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                x = self._parse_number_maybe(p[0])
                y = self._parse_number_maybe(p[1])
                name = None
            else:
                continue
            if x is None or y is None:
                continue
            data.append(ChartDataPoint(x=x, y=y, name=str(name) if name else None))
        if not data:
            return []
        return [
            ChartConfig(
                chart_type="scatter",
                title=title,
                x_axis_title="",
                y_axis_title="",
                series=[ChartSeries(name="散点", chart_type="scatter", data=data)],
                height=460,
            )
        ]

    def chart_json_to_heatmap_configs(self, *, title: str, obj: dict[str, Any]) -> list[ChartConfig]:
        cd = obj.get("chart_data")
        if not isinstance(cd, dict):
            return []
        matrix = cd.get("matrix") or cd.get("data") or obj.get("matrix") or []
        x_labels = cd.get("x_labels") or cd.get("xAxis") or obj.get("x_labels") or []
        y_labels = cd.get("y_labels") or cd.get("yAxis") or obj.get("y_labels") or []
        if not isinstance(matrix, list):
            return []
        opt = {
            "title": {"text": title, "left": "center"} if title else None,
            "tooltip": {},
            "xAxis": {"type": "category", "data": x_labels},
            "yAxis": {"type": "category", "data": y_labels},
            "visualMap": {"min": 0, "max": 1, "calculable": True, "orient": "horizontal", "left": "center"},
            "series": [{"type": "heatmap", "data": matrix}],
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

    def chart_json_to_radar_configs(self, *, title: str, obj: dict[str, Any]) -> list[ChartConfig]:
        cd = obj.get("chart_data")
        if not isinstance(cd, dict):
            return []
        indicator = cd.get("indicator") or cd.get("indicators") or obj.get("indicator") or []
        series = cd.get("series") or obj.get("series") or []
        if not isinstance(indicator, list) or not isinstance(series, list):
            return []
        inds = []
        for it in indicator:
            if not isinstance(it, dict):
                continue
            name = str(it.get("name") or "").strip()
            if not name:
                continue
            mx = self._parse_number_maybe(it.get("max"))
            inds.append({"name": name, "max": mx if mx is not None else 1})
        if not inds:
            return []
        ser = []
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

    def chart_json_to_configs(self, obj: dict[str, Any]) -> list[ChartConfig]:
        """将 chart_json 转为可渲染 ChartConfig（正式渲染入口）。"""
        if not obj or obj.get("is_chart") is not True:
            return []
        ctype_raw = obj.get("chart_type")
        ctype = self._normalize_chart_type_for_render(ctype_raw)
        desc = str(obj.get("description") or "").strip()
        chart_name = str(obj.get("_chart_name") or "").strip()
        title = desc or chart_name or "图表"

        if ctype == "none":
            return []

        handlers = {
            "bar": self.chart_json_to_bar_configs,
            "line": self.chart_json_to_line_configs,
            "pie": self.chart_json_to_pie_configs,
            "stacked_area": self.chart_json_to_stacked_area_configs,
            "sankey": self.chart_json_to_sankey_configs,
            "scatter": self.chart_json_to_scatter_configs,
            "heatmap": self.chart_json_to_heatmap_configs,
            "radar": self.chart_json_to_radar_configs,
            "table": lambda **kwargs: [cfg]
            if (cfg := self._try_table_like_to_grouped_bar(title=title, obj=obj)) is not None
            else [],
        }

        fn = handlers.get(ctype)
        if fn is not None:
            cfgs = fn(title=title, obj=obj)
            if cfgs:
                return cfgs

        cfg2 = self._try_table_like_to_grouped_bar(title=title, obj=obj)
        if cfg2 is not None:
            return [cfg2]
        return []

