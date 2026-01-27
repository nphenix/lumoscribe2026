from __future__ import annotations

from src.application.services.content_generation.chart_json_converter import ChartJsonConverter


def test_sankey_with_cycle_is_downgraded_to_graph_with_audit_fields():
    converter = ChartJsonConverter()
    obj = {
        "is_chart": True,
        "chart_type": "sankey",
        "_chart_name": "cycle_demo",
        "description": "cycle demo",
        "chart_data": {
            "nodes": [{"name": "A"}, {"name": "B"}, {"name": "C"}],
            "links": [
                {"source": "A", "target": "B", "value": 1},
                {"source": "B", "target": "C", "value": 1},
                {"source": "C", "target": "A", "value": 1},
            ],
        },
    }

    cfgs = converter.chart_json_to_configs(obj)
    assert len(cfgs) == 1
    cfg = cfgs[0]
    assert cfg.chart_type == "graph"
    assert isinstance(cfg.extra, dict)
    assert cfg.extra.get("original_chart_type") == "sankey"
    assert cfg.extra.get("cycle_detected") is True
    assert isinstance(cfg.extra.get("cycle_edges_sample"), list)
    assert cfg.extra.get("original_links_count") == 3
    assert cfg.extra.get("nodes_count") == 3

    opt = cfg.extra.get("echarts_option")
    assert isinstance(opt, dict)
    assert opt.get("series") and opt["series"][0].get("type") == "graph"
    assert "示意图" in (opt.get("title", {}) or {}).get("subtext", "")


def test_sankey_without_cycle_keeps_sankey():
    converter = ChartJsonConverter()
    obj = {
        "is_chart": True,
        "chart_type": "sankey",
        "_chart_name": "dag_demo",
        "description": "dag demo",
        "chart_data": {
            "nodes": [{"name": "A"}, {"name": "B"}, {"name": "C"}],
            "links": [
                {"source": "A", "target": "B", "value": 1},
                {"source": "B", "target": "C", "value": 1},
            ],
        },
    }

    cfgs = converter.chart_json_to_configs(obj)
    assert len(cfgs) == 1
    cfg = cfgs[0]
    assert cfg.chart_type == "sankey"
    assert isinstance(cfg.extra, dict)
    opt = cfg.extra.get("echarts_option")
    assert isinstance(opt, dict)
    assert opt.get("series") and opt["series"][0].get("type") == "sankey"

